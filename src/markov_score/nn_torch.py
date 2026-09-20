"""Neural score/denoiser baselines in PyTorch (CPU), under ONE training recipe.

Why a second neural stack. The legacy baselines (`nnet.py`, `local_head.py`,
`seq_nets.py`) are hand-written NumPy with hand-derived backward passes; adding a
transformer or a U-Net there means deriving and testing attention/LayerNorm gradients
by hand, and the audit found the legacy screen's recipe unequal across architectures
(one noise level per step for the long-range arms, twelve for the window head; eight
restarts vs one). This module gives every architecture the same recipe, so that a
difference between arms is a difference between architectures.

Architectures (all map (B, L) noisy chains + a per-example level t to (B, L) outputs)
    MLP           fully connected on [x, t-features]; no sharing, no locality
    WindowHead    weight-shared window of radius r + pointwise MLP (a 1-D CNN, one layer)
    TCN           residual dilated 1-D convolutions (non-causal), receptive radius sum(d)
    BiGRU         bidirectional gated recurrence (global propagation, shared weights)
    Transformer   pre-LN encoder, learned relative-position bias (length-agnostic), global
    UNet1D        two-level 1-D U-Net with skip connections (multiscale, shared weights)

Inputs. `t` enters through a small Fourier feature vector (sin/cos of log t at a few
frequencies) concatenated per site; the MLP sees it once, the others at every site.
Both output parameterisations of the legacy code are kept: `eps` predicts the noise
z and the posterior mean is (x - sqrt(Delta) eps)/alpha; `x0` predicts the mean.

Losses
    dsm      denoising score matching on clean targets (the legacy recipe):
             E ||f(x_t, t) - target||^2 with target = z (eps) or a (x0).
    ism      implicit (Hyvärinen) score matching from noisy samples ONLY:
             E [ 1/2 ||s(x)||^2 + div s(x) ], s = -(x - alpha m(x))/Delta,
             divergence by exact Jacobian trace (L backward passes), by Hutchinson's
             one-probe estimator, or---for a map whose output at site i depends on the
             inputs within a radius r only (the window head)---by 2r+1 coloured probes,
             which is the exact trace at the cost of Hutchinson (`divergence`).
             This is the information-matched arm: it never sees a clean chain.

Device. `TrainConfig.device` ("cpu", "cuda", "cuda:k", or "auto" through `pick_device`)
places the model and every batch; state dicts are saved on the CPU, and `Denoiser`
moves its inputs to the model's device, so a caller never sees a device tensor. On the
CPU the code path is the one every frozen run used, operation for operation.

Recipe (`TrainConfig`): batch 64, Adam, grad-clip 1.0, per-example t drawn uniformly
from the twelve frozen levels, fresh corruption each step, checkpoint ladder, both
parameterisations; validation selection outside this module (the experiments select on
the validation bundle with the true-kernel BP reference, exactly as the legacy arms).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)


# --------------------------------------------------------------------------- time features


class TimeFeatures(nn.Module):
    def __init__(self, n_freq: int = 4):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float())
        self.dim = 2 * n_freq + 1

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (B,)
        u = torch.log(t)[:, None] * self.freqs[None]
        return torch.cat([torch.log(t)[:, None], torch.sin(u), torch.cos(u)], dim=1)  # (B, dim)


# --------------------------------------------------------------------------- fast 1-D convolution


class FastConv1d(nn.Module):
    """`nn.Conv1d` semantics (weight (O, C, K), bias (O,), zero padding, dilation, stride 1)
    computed as gather + matmul. On this CPU build of PyTorch (no oneDNN on arm64) the
    native `conv1d` kernel is 10–100× slower than a matmul of the same size for these
    shapes (64 channels, 32 sites); the two are equal to floating-point round-off
    (`tests/test_nn_torch.py`). Parameters and initialisation are `nn.Conv1d`'s."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding: int = 0, dilation: int = 1, bias: bool = True):
        super().__init__()
        ref = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=bias)
        self.weight = ref.weight
        self.bias = ref.bias
        self.in_ch, self.out_ch, self.k, self.pad, self.dil = in_ch, out_ch, kernel_size, padding, dilation

    def forward(self, x):  # x: (B, C, L) -> (B, O, L_out)
        B, C, L = x.shape
        if self.k == 1 and self.pad == 0:
            y = torch.einsum("bcl,oc->bol", x, self.weight[:, :, 0])
        else:
            xp = F.pad(x, (self.pad, self.pad)) if self.pad else x
            L_out = xp.shape[-1] - self.dil * (self.k - 1)
            idx = torch.arange(L_out, device=x.device)[:, None] + self.dil * torch.arange(self.k, device=x.device)[None, :]
            cols = xp[:, :, idx]                                            # (B, C, L_out, K)
            y = torch.einsum("bclk,ock->bol", cols, self.weight)
        if self.bias is not None:
            y = y + self.bias[None, :, None]
        return y


# --------------------------------------------------------------------------- architectures


class MLP(nn.Module):
    """Global fully-connected denoiser: (B, L) + t -> (B, L). No weight sharing."""

    def __init__(self, L: int, hidden=(128, 128), n_freq: int = 4):
        super().__init__()
        self.tf = TimeFeatures(n_freq)
        dims = [L + self.tf.dim, *hidden, L]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.length_agnostic = False

    def forward(self, x, t):
        h = torch.cat([x, self.tf(t)], dim=1)
        for lin in self.layers[:-1]:
            h = torch.tanh(lin(h))
        return self.layers[-1](h)


class WindowHead(nn.Module):
    """Weight-shared window of radius r: a single conv of width 2r+1 (zero-padded) followed
    by a pointwise MLP. Equals the legacy local head up to the time features."""

    def __init__(self, radius: int = 2, width: int = 64, n_freq: int = 4):
        super().__init__()
        self.tf = TimeFeatures(n_freq)
        self.radius = radius
        self.receptive_radius = radius      # output i depends on inputs |j - i| <= radius only
        self.conv = FastConv1d(1, width, 2 * radius + 1, padding=radius)
        self.tproj = nn.Linear(self.tf.dim, width)
        self.mlp = nn.Sequential(nn.Tanh(), FastConv1d(width, width, 1), nn.Tanh(), FastConv1d(width, 1, 1))
        self.length_agnostic = True

    def forward(self, x, t):
        h = self.conv(x[:, None, :]) + self.tproj(self.tf(t))[:, :, None]
        return self.mlp(h)[:, 0, :]


class TCN(nn.Module):
    """Residual stack of dilated 3-tap non-causal convolutions; receptive radius sum(dilations)."""

    def __init__(self, channels: int = 64, dilations=(1, 2, 4, 8), n_freq: int = 4):
        super().__init__()
        self.tf = TimeFeatures(n_freq)
        self.stem = FastConv1d(1, channels, 1)
        self.tproj = nn.Linear(self.tf.dim, channels)
        self.blocks = nn.ModuleList([FastConv1d(channels, channels, 3, padding=d, dilation=d) for d in dilations])
        self.proj = nn.ModuleList([FastConv1d(channels, channels, 1) for _ in dilations])
        self.head = FastConv1d(channels, 1, 1)
        self.receptive_radius = sum(dilations)
        self.length_agnostic = True

    def forward(self, x, t):
        h = self.stem(x[:, None, :]) + self.tproj(self.tf(t))[:, :, None]
        for conv, proj in zip(self.blocks, self.proj):
            h = h + proj(torch.tanh(conv(h)))
        return self.head(h)[:, 0, :]


class BiGRU(nn.Module):
    """Bidirectional gated recurrence: global propagation with shared local computation."""

    def __init__(self, hidden: int = 64, n_freq: int = 4):
        super().__init__()
        self.tf = TimeFeatures(n_freq)
        self.inp = nn.Linear(1 + self.tf.dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.length_agnostic = True

    def forward(self, x, t):
        B, L = x.shape
        tfeat = self.tf(t)[:, None, :].expand(B, L, -1)
        h = torch.tanh(self.inp(torch.cat([x[:, :, None], tfeat], dim=2)))
        h, _ = self.gru(h)
        return self.head(torch.cat([h, x[:, :, None]], dim=2))[:, :, 0]


class RelPosAttention(nn.Module):
    """Multi-head self-attention with a learned relative-position bias (clipped at max_dist),
    so the model is length-agnostic and translation-equivariant in expectation."""

    def __init__(self, d: int, heads: int, max_dist: int = 32):
        super().__init__()
        self.d, self.h, self.max_dist = d, heads, max_dist
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.bias = nn.Parameter(torch.zeros(heads, 2 * max_dist + 1))

    def forward(self, h):
        B, L, d = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, self.h, d // self.h).transpose(1, 2)
        k = k.view(B, L, self.h, d // self.h).transpose(1, 2)
        v = v.view(B, L, self.h, d // self.h).transpose(1, 2)
        rel = torch.arange(L, device=h.device)[:, None] - torch.arange(L, device=h.device)[None, :]
        rel = rel.clamp(-self.max_dist, self.max_dist) + self.max_dist
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.h) + self.bias[:, rel][None]
        att = att.softmax(dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, L, d)
        return self.out(o)


class Transformer(nn.Module):
    """Pre-LN encoder with relative-position attention; global receptive field."""

    def __init__(self, d: int = 64, heads: int = 4, layers: int = 2, n_freq: int = 4, max_dist: int = 32):
        super().__init__()
        self.tf = TimeFeatures(n_freq)
        self.inp = nn.Linear(1 + self.tf.dim, d)
        self.attn = nn.ModuleList([RelPosAttention(d, heads, max_dist) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.length_agnostic = True

    def forward(self, x, t):
        B, L = x.shape
        tfeat = self.tf(t)[:, None, :].expand(B, L, -1)
        h = self.inp(torch.cat([x[:, :, None], tfeat], dim=2))
        for a, f, l1, l2 in zip(self.attn, self.ff, self.ln1, self.ln2):
            h = h + a(l1(h))
            h = h + f(l2(h))
        return self.head(h)[:, :, 0]


class UNet1D(nn.Module):
    """Two-level 1-D U-Net: conv blocks at full and half resolution with skip connections."""

    def __init__(self, channels: int = 32, n_freq: int = 4):
        super().__init__()
        c = channels
        self.tf = TimeFeatures(n_freq)
        self.tproj = nn.Linear(self.tf.dim, c)
        self.enc1 = nn.Sequential(FastConv1d(1, c, 3, padding=1), nn.GELU(), FastConv1d(c, c, 3, padding=1), nn.GELU())
        self.down = nn.Conv1d(c, 2 * c, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(FastConv1d(2 * c, 2 * c, 3, padding=1), nn.GELU(), FastConv1d(2 * c, 2 * c, 3, padding=1), nn.GELU())
        self.up = nn.ConvTranspose1d(2 * c, c, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(FastConv1d(2 * c, c, 3, padding=1), nn.GELU(), FastConv1d(c, c, 3, padding=1), nn.GELU())
        self.head = FastConv1d(c, 1, 1)
        self.length_agnostic = True

    def forward(self, x, t):
        tp = self.tproj(self.tf(t))[:, :, None]
        h1 = self.enc1(x[:, None, :]) + tp
        h2 = self.enc2(self.down(h1))
        u = self.up(h2)
        if u.shape[-1] != h1.shape[-1]:
            u = F.pad(u, (0, h1.shape[-1] - u.shape[-1]))
        return self.head(self.dec1(torch.cat([u, h1], dim=1)))[:, 0, :]


ARCHS = {"mlp": MLP, "window": WindowHead, "tcn": TCN, "bigru": BiGRU, "transformer": Transformer, "unet": UNet1D}


def build(arch: str, L: int, **hp) -> nn.Module:
    if arch == "mlp":
        return MLP(L, **hp)
    return ARCHS[arch](**hp)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- denoiser wrapper


class Denoiser:
    """A trained network plus its parameterisation; exposes posterior_mean(X, t) in NumPy."""

    def __init__(self, model: nn.Module, parameterization: str):
        self.model = model
        self.parameterization = parameterization

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def posterior_mean(self, X: np.ndarray, t: float) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        tt = torch.full((x.shape[0],), float(t), device=self.device)
        out = self.model(x, tt)
        alpha, delta = math.exp(-t), 1.0 - math.exp(-2.0 * t)
        m = out if self.parameterization == "x0" else (x - math.sqrt(delta) * out) / alpha
        return m.cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- training


@dataclass
class TrainConfig:
    steps: int = 64_000
    batch_size: int = 64
    lr: float = 1e-3
    grad_clip: float = 1.0
    parameterization: str = "eps"      # "eps" | "x0"
    loss: str = "dsm"                  # "dsm" | "ism"
    checkpoints: tuple = (500, 1000, 2000, 4000, 8000, 16000, 32000, 64000)
    hutchinson: bool = False           # ism only: Hutchinson trace instead of exact
    trace: str | None = None           # ism only: "exact" | "hutchinson" | "coloured"; None -> from `hutchinson`
    seed: int = 0
    t_levels: tuple = field(default_factory=tuple)
    device: str = "cpu"                # "cpu" | "cuda" | "cuda:k" | "auto" (see pick_device)


def pick_device(spec: str = "auto") -> str:
    """Resolve a device spec. "auto" is CUDA when available---spread over the visible devices by
    process id, so a worker pool on a multi-GPU node shares them evenly---and the CPU otherwise."""
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return f"cuda:{os.getpid() % torch.cuda.device_count()}"
    return "cpu"


def trace_mode(cfg: TrainConfig) -> str:
    return cfg.trace or ("hutchinson" if cfg.hutchinson else "exact")


def receptive_radius(model: nn.Module):
    """The radius r such that output i depends on inputs i-r..i+r only, when the module declares
    it (the window head); None for a map with a dense Jacobian."""
    return getattr(model, "receptive_radius", None)


def divergence(s: torch.Tensor, x: torch.Tensor, mode: str, rng: np.random.Generator | None = None,
               radius: int | None = None, create_graph: bool = True) -> torch.Tensor:
    """sum_i ds_i/dx_i for every row of `s` (B, L), computed from the leaf `x` that `s` came from.

    exact       L backward passes, one per output coordinate.
    hutchinson  one standard-normal probe v: E[v^T J v] = tr J (unbiased, high variance).
    coloured    for a map whose output i depends only on inputs |j - i| <= radius, the p = 2r+1
                probes v_k = 1{j = k mod p} give the exact trace at Hutchinson's cost per probe:
                (J^T v_k)_j = sum_i v_k[i] J_ij picks out the ONE i in [j-r, j+r] with i = k mod p,
                and that i is j itself exactly when j = k mod p, so sum_k v_k . (J^T v_k) = tr J.
                (Curtis, Powell and Reid, 1974: colouring of a banded Jacobian.) Zero padding at
                the ends only removes dependencies, so the argument covers the boundary sites.
    """
    B, L = x.shape
    if mode == "hutchinson":
        v = torch.as_tensor(rng.standard_normal(x.shape), dtype=x.dtype, device=x.device)
        jv = torch.autograd.grad((s * v).sum(), x, create_graph=create_graph, retain_graph=True)[0]
        return (jv * v).sum(dim=1)
    if mode == "coloured":
        if radius is None:
            raise ValueError("coloured trace: the model declares no receptive radius (dense Jacobian)")
        p = 2 * int(radius) + 1
        idx = torch.arange(L, device=x.device)
        div = torch.zeros(B, dtype=x.dtype, device=x.device)
        for k in range(p):
            v = (idx % p == k).to(x.dtype)[None, :].expand(B, L)
            jv = torch.autograd.grad((s * v).sum(), x, create_graph=create_graph, retain_graph=True)[0]
            div = div + (jv * v).sum(dim=1)
        return div
    if mode == "exact":
        div = torch.zeros(B, dtype=x.dtype, device=x.device)
        for i in range(L):
            g = torch.autograd.grad(s[:, i].sum(), x, create_graph=create_graph, retain_graph=True)[0]
            div = div + g[:, i]
        return div
    raise ValueError(f"unknown divergence estimator {mode!r}")


def _draw_batch(A: np.ndarray, cfg: TrainConfig, rng: np.random.Generator):
    dev = torch.device(cfg.device)
    idx = rng.integers(0, len(A), size=min(cfg.batch_size, len(A)))
    a = torch.as_tensor(A[idx], dtype=torch.float32, device=dev)
    t = torch.as_tensor(rng.choice(cfg.t_levels, size=len(idx)), dtype=torch.float32, device=dev)
    alpha = torch.exp(-t)[:, None]
    delta = 1.0 - torch.exp(-2.0 * t)[:, None]
    z = torch.as_tensor(rng.standard_normal(a.shape), dtype=torch.float32, device=dev)
    x = alpha * a + torch.sqrt(delta) * z
    return a, x, z, t, alpha, delta


def dsm_loss(model, a, x, z, t, alpha, delta, cfg: TrainConfig):
    out = model(x, t)
    target = z if cfg.parameterization == "eps" else a
    return ((out - target) ** 2).mean()


def _score_from_out(out, x, alpha, delta, cfg):
    m = out if cfg.parameterization == "x0" else (x - torch.sqrt(delta) * out) / alpha
    return -(x - alpha * m) / delta


def ism_loss(model, x, t, alpha, delta, cfg: TrainConfig, rng: np.random.Generator):
    """Implicit score matching: E[ ½||s||² + div s ] on the noisy sample only."""
    x = x.detach().requires_grad_(True)
    out = model(x, t)
    s = _score_from_out(out, x, alpha, delta, cfg)
    div = divergence(s, x, trace_mode(cfg), rng=rng, radius=receptive_radius(model), create_graph=True)
    return (0.5 * (s**2).sum(dim=1) + div).mean()


def ism_objective(den: "Denoiser", groups, trace: str = "exact") -> float:
    """Mean implicit-score-matching objective E[1/2 |s|^2 + div s] over held-out noisy (X, t)
    groups---the model-selection criterion of the information-matched arm, which needs no clean
    chain and no true score. `trace` names the divergence estimator: "exact" (L passes) for any
    map, "coloured" (exact, 2r+1 passes) for a map that declares its receptive radius."""
    dev = den.device
    tot, n = 0.0, 0
    for X, t in groups:
        x = torch.as_tensor(X, dtype=torch.float32, device=dev).requires_grad_(True)
        tt = torch.full((x.shape[0],), float(t), device=dev)
        al, de = math.exp(-t), 1.0 - math.exp(-2.0 * t)
        out = den.model(x, tt)
        m = out if den.parameterization == "x0" else (x - math.sqrt(de) * out) / al
        s = -(x - al * m) / de
        div = divergence(s, x, trace, radius=receptive_radius(den.model), create_graph=False)
        tot += float((0.5 * (s ** 2).sum(dim=1) + div).detach().sum()); n += x.shape[0]
    return tot / n


def train(model: nn.Module, A_train: np.ndarray, cfg: TrainConfig, X_noisy_groups=None) -> dict:
    """Train under the common recipe; returns {step: state_dict copy} at the checkpoints.

    For loss == "ism" the training data are `X_noisy_groups`: a list of (X, t) with one
    noised view per chain at one level (the EM-BP interface); `A_train` is ignored.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = torch.device(cfg.device)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    saved = {}
    model.train()
    if cfg.loss == "ism":
        Xs = np.concatenate([X for X, _ in X_noisy_groups])
        Ts = np.concatenate([np.full(len(X), t) for X, t in X_noisy_groups])
    for step in range(1, cfg.steps + 1):
        if cfg.loss == "dsm":
            a, x, z, t, alpha, delta = _draw_batch(A_train, cfg, rng)
            loss = dsm_loss(model, a, x, z, t, alpha, delta, cfg)
        else:
            idx = rng.integers(0, len(Xs), size=min(cfg.batch_size, len(Xs)))
            x = torch.as_tensor(Xs[idx], dtype=torch.float32, device=dev)
            t = torch.as_tensor(Ts[idx], dtype=torch.float32, device=dev)
            alpha = torch.exp(-t)[:, None]; delta = 1.0 - torch.exp(-2.0 * t)[:, None]
            loss = ism_loss(model, x, t, alpha, delta, cfg, rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step in cfg.checkpoints:
            saved[step] = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    return saved


def load_checkpoint(model: nn.Module, state: dict) -> nn.Module:
    model.load_state_dict(state)
    return model
