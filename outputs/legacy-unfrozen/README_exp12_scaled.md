# exp_12_scaled — the structured-baseline table's source, and why it is flagged

`outputs/exp_12_scaled/seed*/efficiency_val.csv` is what
`tools/make_tab_structured.py` reads to produce `tab:structured` in
`overleaf/paper/appendix.tex` — the "how much of the headline margin is
architectural" table, CNN/EM-BP ratios 1.82–5.54 across four sizes.

## The defect (round-two review, §4; also documented in
## `outputs/README_m801_resolve.md`)

Every `params_efficiency_efficiency_val.json` under this directory records:

    git_commit = 286b305a419b741e3965d119a8404955017510a1
    git_dirty  = 207 paths
    overrides  = ["eff_seed0=0", ...]

At commit `286b305`, `experiments/exp_12_receptive_field.py` did **not**
declare an `eff_seed0` setting, and `apply_overrides` raises `SystemExit` on
an unknown key. The command that produced these files could not have run at
the commit it names. The dirty list does not include
`exp_12_receptive_field.py` either, so the stamp was already stale when it was
written — deployed source and recorded revision are two different programs,
and neither is fully recoverable from what was written down.

This is the same defect class as `exp_07_n8192_seed*` (withdrawn from the
headline table, see `tools/make_tab_efficiency.py`), found independently and
first, in this session, before the n=8192 row.

## Three further defects, layered on top of the unreproducible source
## (round-two review §3, closed by `exp_31_structured_baseline.py`)

Even if the source were reproducible, the protocol it ran has three problems
the replacement experiment (`experiments/exp_31_structured_baseline.py`,
running as of this note) was built to fix:

1. **Centre-site only.** `interior_slice(33, 16)` returns `slice(16, 17)` — a
   single coordinate of a 33-site chain. The table reads as a whole-sequence
   comparison and is not one.
2. **Fixed training budget (8000 steps).** Presentations per chain fall from
   8000 at n=32 to 250 at n=2048, so the widening ratio cannot be separated
   from the per-datum budget shrinking. `exp_31` checkpoints and selects on
   validation for both arms instead.
3. **Protocol drift.** 33 sites (not 32), five noise levels (not twelve),
   C=4 (not C=8) — not directly comparable to the headline table's protocol.

## Current status

The paper's main text no longer cites a specific ratio from this table (see
the "How much of this is the architecture?" paragraph, scoped to qualitative
language: "a small multiple", "bounds what remains to be explained"). The
table itself remains in Appendix `app:structured` with the historical
numbers, but the caption and surrounding prose now say plainly that its
provenance is not independently verifiable and point to `exp_31` as the
properly-scoped, clean-deployed replacement.

**Do not** read 1.82–5.54 as a certified number. **Do not** re-derive it by
rerunning `exp_12_receptive_field.py` under the old `stamp_revision.sh` +
`sync_to_cluster.sh` deploy path — that path is what produced this defect.
Use `hpc/deploy_clean.sh`.

When `exp_31`'s confirmatory results land, `make_tab_structured.py` (or its
replacement) should be pointed at them and this table retired.
