# exp_07_m801_seed* — informative, NOT certifiable

Job 634478, 16 seeds, all COMPLETED 2026-08-23. This is the review's Priority 4
/ §10.7: refit the cells the headline table excludes at `M=801`, so that
exclusion is replaced by resolution.

## The result

576 cells, **569 resolved, 7 still unresolved** at `M=801` — down from 16
unresolved at `M=401`, but not to zero. Refining the grid shrinks the excluded
set by about half rather than eliminating it, so the drop-and-disclose argument
is weakened, not retired. Reaching zero would need `M=1201` on the remainder,
which is what §10.7 anticipates.

## Why these files cannot go in a table

`tools/provenance_gate.py` refuses them, correctly:

    params_sample_efficiency_val.json
      git_commit = 286b305a419b741e3965d119a8404955017510a1
      git_dirty  = 207 paths

They were produced by the old `stamp_revision.sh` + `sync_to_cluster.sh` path —
the same deployment whose stamp is provably wrong for `exp_12_scaled`, where the
recorded command passes an override the recorded commit does not define. The
dirty list here includes `exp_07_em_vs_score_network.py`, `src/em.py`,
`frozen_config.py` and `tools/make_tab_efficiency.py`: every file that
determines these numbers. The exact program is not recoverable.

Unlike `exp_12_scaled`, nothing here is self-contradictory — the stamp is
unverifiable rather than demonstrably false. That is not a meaningful
distinction for a published table. It is the difference between "we cannot show
this is right" and "we can show this is wrong", and only the second is worse.

## What to do with them

Keep as an indication of what the clean rerun should find: 7 unresolved cells,
not 0 and not 16. If a clean rerun disagrees materially, that disagreement is
itself worth investigating, because it would mean the uncommitted source
differed from `286b305` in a way that moved results.

Rerun with:

    hpc/deploy_clean.sh                       # from a committed tree
    # then submit the resolve mode against ~/$NGBP_SRC with NGBP_REQUIRE_CLEAN=1

Do not merge these into `outputs/frozen/`, and do not pass `allow_legacy=True`
to the gate to make a table build. The gate refusing them is the gate working.
