# exp_07_certified_clean_seed* — the certified table, rerun through the clean gate

Job 634911, 16 array tasks, all COMPLETED, commit `569a67c`/`53b95a2`
(deploy_clean.sh, `NGBP_REQUIRE_CLEAN=1`, 0 dirty paths on every seed).

## Why this run exists

Two things, both from the round-two review:

1. **Independent reproducibility evidence for the headline table.** The
   certified table (`outputs/frozen/exp_07_certified_seed*/`, commit
   `2ae68ac3`, 0 dirty paths) was never in question — unlike `exp_12_scaled`
   and the withdrawn n=8192 row — but it had also never been rerun through
   `hpc/deploy_clean.sh`. **It reproduces exactly**: 1344/1344 matched cells,
   `ratio_selected` identical to zero difference (`max_abs_diff = 0.00000`).
2. **The parent-law-weighted Hellinger numbers** (review, "Hellinger"
   section). `src/metrics.py` gained `hellinger_weighted_mean` after the
   original certified run, so getting the real number meant rerunning, not
   recomputing from what was on disk. Real values, mean over 16 seeds:

   | n_seq | unweighted median | weighted mean |
   |------:|-------------------:|---------------:|
   |    32 |             0.1030 |         0.0994 |
   |    64 |             0.0896 |         0.0867 |
   |   128 |             0.0757 |         0.0741 |
   |   256 |             0.0637 |         0.0630 |
   |   512 |             0.0598 |         0.0592 |
   |  1024 |             0.0522 |         0.0521 |
   |  2048 |             0.0500 |         0.0499 |

   The weighted mean is consistently a little below the unweighted median —
   consistent with the parent law downweighting tail parents, where the fit
   is noisier — and both fall with n_seq, so the qualitative story
   (transition genuinely recovered, not just fitted) is unchanged.

## Not merged into `outputs/frozen/`

Deliberately. `frozen_config.py`'s whole point is that a number in the paper
traces to exactly one file, and silently swapping the source directory a
generator reads from — even for a bit-identical rerun — reintroduces the
"which run actually produced this" question the frozen convention exists to
foreclose. `tools/make_tab_hellinger.py` reads this directory explicitly and
says so in its own header.
