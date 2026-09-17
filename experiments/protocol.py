"""The confirmatory protocol: every factor the registered experiments share, frozen once.

Why a second frozen object next to `frozen_config.FROZEN`. `FROZEN` fixes the *world*
(the chain, the channel, the quadrature grid, the noise schedule, the seed count).
`PROTOCOL` fixes the *comparison*: bundle sizes and their random streams, the sizes M,
the seeds, the EM ladder and initialisation, the network recipe and screen budget, the
estimand and the regions. Every registered experiment reads these from here and varies
exactly one declared factor (`PROTOCOL.factors`), so two arms differ only in that factor
-- the ceteris-paribus design that lets a difference be attributed. A number that is
not produced under `PROTOCOL` (or a narrowed copy recorded in its params file) is not a
confirmatory number.

Versions. `V1` is the protocol the laptop runs of 15 September 2026 started under (EM
ladder to 400, capacities to 8, a two-seed one-run screen); `PROTOCOL` (version 2) is
the registered protocol for the cluster runs: the EM ladder extends to 1200 because at
M = 2048 the C >= 4 arms selected the top of the 400-ladder in every seed, capacity 16 is
added to show the plateau, and the screen is the full pre-registered design. Every
params file records the resolved settings, so a run's version is read from the file,
not remembered.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace

from frozen_config import FROZEN


@dataclass(frozen=True)
class Protocol:
    version: str = "2"

    # --- bundles: the same random streams for every arm and every experiment ---------
    sizes: tuple = (32, 128, 512, 2048)
    """Training-set sizes M (sequences). One arm's fit at (seed, M) sees exactly the
    sequences another arm's fit at (seed, M) sees: the stream is keyed by (tag, seed, M)."""
    seeds: tuple = tuple(range(16))
    """Confirmatory seeds. Never used for any selection."""
    dev_seeds: tuple = (10_000, 10_001, 10_002, 10_003)
    """Development seeds: architecture and hyperparameter selection only."""
    n_val: int = 1024
    n_test: int = 2048
    bundle_tags: tuple = (("train", "exp31-train"), ("val", "exp31-val"), ("test", "exp31-test"))
    """Stream tags. `rng_for(tag, seed[, M])` makes the bundles identical across experiments,
    which is what lets E02a, E03 and E09 rows be paired through the bundle hash."""

    # --- estimand ----------------------------------------------------------------
    estimand: str = "schedule-pooled relative score error: (sum_t sum_j,i (s_hat - s)^2 / sum_t sum_j,i s^2)^(1/2)"
    headline_region: str = "all"
    selection_region: str = "bulk"
    bulk_margin: int = 4
    """Bulk = sites 5..L-4 (four sites trimmed at each end)."""

    # --- EM arm ------------------------------------------------------------------
    capacities: tuple = (1, 2, 4, 8, 16)
    em_checkpoints: tuple = (10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200)
    """Iteration ladder; the checkpoint is selected on the validation bundle (bulk), for every
    capacity identically. EM also carries the frozen likelihood tolerance (1e-9 nats per
    transition, `FrozenConfig.em_loglik_tol`), which can end a run before the last rung; what
    changed relative to the archival sweep is not that the tolerance is gone but that the reported
    checkpoint is chosen by validation among the rungs reached, rather than being whichever rung the
    tolerance happened to stop on."""
    em_init_rho: float = 0.3
    em_init_var: float = 0.8
    one_view: bool = True
    """Each training sequence contributes one noised view at one level; sequences are
    split evenly across the twelve levels."""

    # --- network arms ------------------------------------------------------------
    net_steps: int = 32_000
    net_checkpoints: tuple = (500, 1000, 2000, 4000, 8000, 16000, 32000)
    batch_size: int = 64
    lrs: tuple = (1e-3, 3e-3)
    parameterizations: tuple = ("eps", "x0")
    grad_clip: float = 1.0
    screen_sizes: tuple = (32, 2048)
    screen_restarts: int = 3
    screen_steps: int = 32_000

    # --- which experiment varies which factor (everything else held at this protocol) --
    factors: tuple = (
        ("E02a", "mixture capacity C"),
        ("E02b", "innovation shape (generalised-Gaussian beta) at fixed covariance"),
        ("E03", "estimator family: architecture, recipe, information interface"),
        ("E04", "quadrature resolution (N_g, A) at fixed fitted models"),
        ("E05", "the score driving one reverse SDE (same Brownian path)"),
        ("E06", "observation level t of the noised training data"),
        ("E07", "correlation rho and innovation family (oracles only)"),
        ("E08", "misspecification mechanism and strength"),
        ("E09", "capacity C at infinite data"),
    )

    def as_dict(self) -> dict:
        d = asdict(self)
        d["world"] = FROZEN.as_dict()
        return d


PROTOCOL = Protocol()

V1 = replace(PROTOCOL, version="1", capacities=(1, 2, 4, 8), em_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400),
             dev_seeds=(10_000, 10_001), screen_restarts=1, screen_steps=16_000)
"""The laptop protocol of 15 September 2026 (E02a relaunch 07:58, E03 screen 07:58)."""


def protocol_hash(p: Protocol = PROTOCOL) -> str:
    body = "\n".join(f"{k}={v!r}" for k, v in sorted(p.as_dict().items()))
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def v1_overrides() -> list[str]:
    """`--set` assignments that turn a version-2 script into the version-1 run."""
    return ["components=(1, 2, 4, 8)", "em_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400)",
            "dev_seeds=(10000, 10001)", "screen_restarts=1", "screen_steps=16000"]
