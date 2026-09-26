"""No driver may re-create its random generator inside the loop that draws its data.

E14's first runs (N-22) drew the training set as

    np.stack([prior.sample(rng_for("e14-train", seed, L, rho), L) for _ in range(n)])

-- `rng_for` re-seeded on every iteration, so all 2048 "chains" were one draw and every arm
fitted 2048 copies of a single sequence. Four cluster jobs were void before it was found. The
correct form creates the generator once (`rng = rng_for(...)`) and draws from it.

Two shapes that match the same text are sound and are not flagged: a single call that draws
every chain at once (`truth.sample(rng_for(...), n)`, E08, no loop), and a per-chain seed
whose loop variable is inside `rng_for(...)` (`for k in range(n)`, exp_05, exp_14). The
degenerate form is exactly a freshly seeded generator inside a `for _ in` loop: the seed
cannot depend on a variable the loop throws away.

exp_32, exp_33 and exp_34 were found to carry the pattern on 21 September 2026 and were
ALLOWLISTED rather than fixed, so this gate stayed green while the pattern remained. On 25
September exp_34 was then submitted as E25 and all four jobs were void (amendment N-23). The
allowlist is now empty and must stay so; see KNOWN_ARCHIVAL below.

A second, stronger gate on the same defect class lives in tests/test_no_degenerate_bundles.py:
it walks the AST instead of matching text, so it covers any comprehension that re-seeds
rng_for per iteration rather than only `.sample(rng_for(...))` inside `for _ in`, and it has no
allowlist. Both are kept deliberately — this one pins the exact original line and the shapes
that must NOT be flagged, which is a different job from a general sweep.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALL = re.compile(r"\.sample\(\s*rng_for\(")
THROWAWAY_LOOP = re.compile(r"\bfor\s+_\s+in\b")
WINDOW = 3          # lines after the call within which the loop header may sit

KNOWN_ARCHIVAL: set[str] = set()
"""Empty since 26 September 2026, and it should stay empty.

It used to list exp_32, exp_33 and exp_34, on the reasoning below that they were archival
copies the paper does not build from, so listing them kept this gate green while leaving the
pattern visible. That reasoning failed in the only way that mattered: on 25 September exp_34
was submitted to the cluster as experiment E25, four jobs and about thirty CPU-hours, and
every training set was one chain replicated M times. The allowlist WAS the warning and it was
walked straight past, because nothing connects "this driver is on a known-broken list" to
"this driver is about to be run".

All three are now fixed, so the list is empty. Do not add to it. A driver with this defect
should be repaired, not registered as acceptably broken — an entry here is an invitation to
run it.
"""


def degenerate_hits(text: str, name: str = "<src>") -> list[str]:
    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines):
        if CALL.search(line) and not line.lstrip().startswith("#"):
            if THROWAWAY_LOOP.search(" ".join(lines[i:i + WINDOW + 1])):
                hits.append(f"{name}:{i + 1}: {line.strip()}")
    return hits


def _files():
    return sorted(p for d in ("experiments", "tools") for p in (ROOT / d).glob("*.py"))


def test_live_drivers_do_not_sample_from_a_freshly_seeded_generator():
    hits = []
    for p in _files():
        rel = str(p.relative_to(ROOT))
        if rel not in KNOWN_ARCHIVAL:
            hits += degenerate_hits(p.read_text(), rel)
    assert not hits, ("generator re-created inside the sampling loop: every draw identical (N-22):\n"
                      + "\n".join(hits))


def test_known_archival_list_is_current():
    """The list must shrink when a file is fixed, and must not shelter a file that is clean."""
    still = {str(p.relative_to(ROOT)) for p in _files() if degenerate_hits(p.read_text())}
    stale = KNOWN_ARCHIVAL - still
    assert not stale, f"clean now; delete from KNOWN_ARCHIVAL: {sorted(stale)}"


def test_the_gate_catches_the_original_line_and_passes_the_sound_shapes():
    bug = 'A = np.stack([prior.sample(rng_for("e14-train", seed, L, rho), L) for _ in range(n)])'
    assert degenerate_hits(bug)
    split = 'A = np.stack([\n    prior.sample(rng_for("exp32-train", seed, n_chains), N_SITES)\n    for _ in range(n_chains)\n])'
    assert degenerate_hits(split)
    one_call = 'A = truth.sample(rng_for(("e08-train", mech, eps), seed, n), n)'
    assert not degenerate_hits(one_call)
    per_chain = 'clean = np.stack([prior.sample(rng_for("exp05-clean", k), N_SITES)\n                  for k in range(n_batch)])'
    assert not degenerate_hits(per_chain)
