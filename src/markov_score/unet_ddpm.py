"""The standard diffusion network for sequences: a DDPM U-Net and its standard training recipe (E55).

Why a second U-Net. `nn_torch.UNet1D` is a two-level toy: no residual blocks, no normalisation, the noise level
entering once. E55 asks how the network diffusion models actually use compares with a kernel learned inside BP, so this
one follows the U-Net of Ho, Jain & Abbeel (2020, "Denoising Diffusion Probabilistic Models"), written in one dimension:

    residual blocks      GroupNorm -> SiLU -> conv(3), plus the noise-level embedding, GroupNorm -> SiLU -> conv(3),
                         and a 1x1 skip when the channel count changes
    noise level          sinusoidal embedding of 1000 t through a two-layer MLP, fed to EVERY residual block
    resolutions          L, L/2, L/4 with channel multipliers (1, 2, 2), two residual blocks per resolution,
                         stride-2 convolutions down, nearest-neighbour upsampling + conv(3) up
    attention            self-attention after every block at the coarsest resolution and in the middle
    skips                every encoder activation is concatenated into the decoder (the "U")
    output               GroupNorm -> SiLU -> conv(3), initialised at ZERO: the untrained network predicts no noise

The recipe is DDPM's: predict the noise eps of x_t = alpha a + sqrt(Delta) eps (alpha = e^-t, Delta = 1 - e^-2t, the
project's OU channel), a fresh clean sequence, level and noise for every example of every step (standard denoising score
matching, Vincent 2011), Adam with a linear warm-up, gradient clipping at 1, and an exponential moving average (EMA) of the
weights. The EMA network is the one saved at every checkpoint and the one that is evaluated. The EMA decay ramps as
min(decay, (1 + step) / (10 + step)) so that an early checkpoint is not dominated by the random initialisation.

The noise levels are the twelve of the frozen schedule (`frozen_config.FROZEN.t_grid`), passed in by the caller: the
estimand is scored at those levels, so training on them is the network-favourable choice.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DOUBLING_CHECKPOINTS = tuple(2**k for k in range(10, 19))  # 1024 ... 262144


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of 1000 t (the transformer's position code, as in DDPM), shape (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=t.dtype, device=t.device) / half)
    args = 1000.0 * t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int, groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, c_in)
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out)
        self.norm2 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, h, emb):
        y = self.conv1(F.silu(self.norm1(h)))
        y = y + self.emb(F.silu(emb))[:, :, None]
        y = self.conv2(F.silu(self.norm2(y)))
        return self.skip(h) + y


class Attention(nn.Module):
    """Multi-head self-attention over the sites, with a residual connection."""

    def __init__(self, c: int, groups: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(groups, c)
        self.qkv = nn.Conv1d(c, 3 * c, 1)
        self.proj = nn.Conv1d(c, c, 1)

    def forward(self, h):
        b, c, n = h.shape
        q, k, v = self.qkv(self.norm(h)).reshape(b, 3, self.heads, c // self.heads, n).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return h + self.proj(out.transpose(-1, -2).reshape(b, c, n))


class UNetDDPM(nn.Module):
    """(B, L) noisy sequences and (B,) levels t -> (B, L) predicted noise. L must be a multiple of 2**(len(mults) - 1)."""

    def __init__(self, base: int = 64, mults: tuple = (1, 2, 2), n_res: int = 2, groups: int = 8, heads: int = 4):
        super().__init__()
        if base % groups or (base * min(mults)) % heads:
            raise ValueError("channel counts must divide by the GroupNorm groups and the attention heads")
        self.factor = 2 ** (len(mults) - 1)
        self.base = base
        emb = 4 * base
        self.temb = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.conv_in = nn.Conv1d(1, base, 3, padding=1)
        coarsest = len(mults) - 1
        # encoder: a list of ("res", block) / ("attn", block) / ("down", conv); every res and down output is a skip
        self.enc = nn.ModuleList()
        self.enc_kind = []
        skips = [base]
        c = base
        for lvl, m in enumerate(mults):
            for _ in range(n_res):
                self.enc.append(ResBlock(c, base * m, emb, groups)); self.enc_kind.append("res")
                c = base * m
                if lvl == coarsest:
                    self.enc.append(Attention(c, groups, heads)); self.enc_kind.append("attn")
                skips.append(c)
            if lvl < coarsest:
                self.enc.append(nn.Conv1d(c, c, 3, stride=2, padding=1)); self.enc_kind.append("down")
                skips.append(c)
        self.mid1 = ResBlock(c, c, emb, groups)
        self.mid_attn = Attention(c, groups, heads)
        self.mid2 = ResBlock(c, c, emb, groups)
        # decoder: n_res + 1 blocks per level, each consuming one skip
        self.dec = nn.ModuleList()
        self.dec_kind = []
        for lvl in reversed(range(len(mults))):
            for _ in range(n_res + 1):
                self.dec.append(ResBlock(c + skips.pop(), base * mults[lvl], emb, groups)); self.dec_kind.append("res")
                c = base * mults[lvl]
                if lvl == coarsest:
                    self.dec.append(Attention(c, groups, heads)); self.dec_kind.append("attn")
            if lvl > 0:
                self.dec.append(nn.Conv1d(c, c, 3, padding=1)); self.dec_kind.append("up")
        assert not skips
        self.norm_out = nn.GroupNorm(groups, c)
        self.conv_out = nn.Conv1d(c, 1, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
        self.length_agnostic = True

    def forward(self, x, t):
        if x.shape[-1] % self.factor:
            raise ValueError(f"sequence length {x.shape[-1]} is not a multiple of {self.factor}")
        emb = self.temb(timestep_embedding(t, self.base))
        h = self.conv_in(x[:, None, :])
        hs = [h]
        for kind, layer in zip(self.enc_kind, self.enc):
            if kind == "res":
                h = layer(h, emb); hs.append(h)
            elif kind == "attn":
                h = layer(h); hs[-1] = h
            else:
                h = layer(h); hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for kind, layer in zip(self.dec_kind, self.dec):
            if kind == "res":
                h = layer(torch.cat([h, hs.pop()], dim=1), emb)
            elif kind == "attn":
                h = layer(h)
            else:
                h = layer(F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.conv_out(F.silu(self.norm_out(h)))[:, 0, :]


def build(seed: int, **hp) -> UNetDDPM:
    """A U-Net whose initial weights depend on `seed` only (the global torch generator is left as it was)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return UNetDDPM(**hp)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass(frozen=True)
class DSMConfig:
    steps: int = DOUBLING_CHECKPOINTS[-1]
    batch_size: int = 128
    lr: float = 2e-4
    warmup: int = 1000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    checkpoints: tuple = DOUBLING_CHECKPOINTS
    seed: int = 0
    device: str = "cpu"
    t_range: tuple | None = None   # (t_min, t_max): draw t log-uniformly instead of from `t_levels` (E55's generation nets)
    compile: bool = False          # torch.compile the training forward/backward (CUDA only; the same model and loss)
    compile_mode: str = "reduce-overhead"   # CUDA graphs: the step is replayed, not relaunched kernel by kernel


def train_dsm(model: nn.Module, A: np.ndarray, t_levels, cfg: DSMConfig, clock: dict | None = None) -> dict:
    """Standard DSM on the clean sequences A (n, L). Returns {step: EMA state dict on the CPU} at cfg.checkpoints.

    Every step draws a batch of clean sequences (with replacement), a level from `t_levels` and fresh noise for each,
    on the device, from one generator seeded by cfg.seed. On the CPU two runs with the same seed are identical; on a
    GPU the convolutions are not bitwise deterministic. `clock`, if given, receives {step: seconds since the start} at
    every checkpoint (the steady-state step time is a difference of two of them, free of compilation time).
    """
    import time
    dev = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed)
    model.to(dev).train()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    data = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=dev)
    levels = torch.as_tensor(tuple(t_levels), dtype=torch.float32, device=dev)
    params, ema_params = list(model.parameters()), list(ema.parameters())
    # fused Adam and one multi-tensor EMA update: the same arithmetic as the per-tensor loops, a few GPU launches per
    # step instead of hundreds (a small U-Net on an H200 is launch-bound: 18-37 ms/step measured, job 720363)
    opt = torch.optim.Adam(params, lr=cfg.lr, fused=dev.type == "cuda")
    ckpts = set(int(c) for c in cfg.checkpoints)
    # Compilation fuses the U-Net's ~1000 small kernels per step (eager mode is launch-bound on an H200: 16-37 ms/step,
    # timing jobs 720363/723188); the compiled module shares its parameters with `model`, so the optimiser, the EMA and
    # every saved state are untouched. Off on the CPU.
    compiled = bool(cfg.compile and dev.type == "cuda")
    if compiled:
        # a worker trains many models in a row: without a reset each new model is one more recompilation (dynamo falls back
        # to eager, silently, past its cache limit of 8) and the CUDA-graph pools of finished fits stay allocated
        torch._dynamo.reset()
    fwd = torch.compile(model, mode=cfg.compile_mode) if compiled else model
    saved = {}
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, data.shape[0], (cfg.batch_size,), generator=gen, device=dev)
        a = data[idx]
        if cfg.t_range is None:
            t = levels[torch.randint(0, len(levels), (cfg.batch_size,), generator=gen, device=dev)]
        else:                                   # continuous levels, uniform in log t
            u = torch.rand((cfg.batch_size,), generator=gen, device=dev)
            t = torch.exp(math.log(cfg.t_range[0]) + u * (math.log(cfg.t_range[1]) - math.log(cfg.t_range[0])))
        eps = torch.randn(a.shape, generator=gen, device=dev)
        x = torch.exp(-t)[:, None] * a + torch.sqrt(1.0 - torch.exp(-2.0 * t))[:, None] * eps
        loss = ((fwd(x, t) - eps) ** 2).mean()
        for g in opt.param_groups:
            g["lr"] = cfg.lr * min(1.0, step / max(cfg.warmup, 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        decay = min(cfg.ema_decay, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, params, 1.0 - decay)
            for be, b in zip(ema.buffers(), model.buffers()):
                be.copy_(b)
        if step in ckpts:
            saved[step] = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
            if clock is not None:
                clock[step] = time.time() - t0
    if compiled:
        del fwd
        torch._dynamo.reset()
    return saved


@torch.no_grad()
def posterior_mean(model: nn.Module, X: np.ndarray, t: float, chunk: int = 512) -> np.ndarray:
    """E[a | x_t] from the predicted noise, (x - sqrt(Delta) eps_hat) / alpha, in float64 on the host; chunked rows."""
    model.eval()
    dev = next(model.parameters()).device
    alpha, delta = math.exp(-t), 1.0 - math.exp(-2.0 * t)
    out = np.empty(np.shape(X), dtype=np.float64)
    for s in range(0, len(X), chunk):
        x = torch.as_tensor(np.asarray(X[s:s + chunk]), dtype=torch.float32, device=dev)
        eps = model(x, torch.full((x.shape[0],), float(t), device=dev))
        out[s:s + chunk] = ((x - math.sqrt(delta) * eps) / alpha).double().cpu().numpy()
    return out
