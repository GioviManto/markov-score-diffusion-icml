# Commit map of 1 October 2026: the heavy history and the pushed one

GitHub refused the branches because two fetched cluster tables committed in 86c3f50 on 26
September exceed its 100 MB file limit (`outputs/hpc_E22c/shapes.csv`, 1.28 GB;
`outputs/hpc_E23/representation.csv`, 122 MB). Neither is a frozen output (E22.json and
E23.json pin nothing), both are `export-ignore` for the cluster archive, and both stay on the
laptop, now git-ignored. The 24 commits from 86c3f50 to the tip were replayed without them
(`git cherry-pick`, the first with the two files unstaged); every other file of every commit
is byte-identical (`git diff --stat` against the heavy tips lists only the two tables).

Everything that cites a commit hash -- `frozen_commit` / `frozen_run_commits` in the registry,
`git_commit` in the params files under `outputs/`, the cluster deployment tags
`9a55030-20260929T212520Z`, `9b1c940-20261001T084745Z`, `52cf980-20261001T094107Z`, the
memory and docs -- names the HEAVY hash. This table resolves it. The heavy history is kept
locally under the original branch names until the owner retargets them (N-32).

| heavy | pushed | subject |
|---|---|---|
| `86c3f50` | `b4176a3` | N-24/N-25: the MDN arm never trained — three defects, and a correction |
| `10d8d73` | `8ea830a` | Deploy: stop shipping 1.5 GB of fetched cluster results back to the cl |
| `2889d91` | `dfc0bba` | E27: Q8's strong form -- generation with no score and no reverse SDE |
| `9af53fb` | `e5aadf7` | hpc: E27's timing part gets its own job |
| `d82ee7b` | `2303e98` | Q10: the information curves in cumulant coordinates -- a recompute, no |
| `1bd5eec` | `eded176` | E28: Q9's second half -- the pooled scalar hides a crossover |
| `515f529` | `c489baa` | E29: score nets AT the kernel's parameter count -- the refutable arm |
| `9783a8f` | `63834c6` | Q5 analytically: a closed form for the BP message, and a theorem for w |
| `f396d5a` | `c123468` | N-26: E28's first run was void, and my first diagnosis of it was wrong |
| `7908703` | `10b6e1c` | Report: everything since the call with Jerome, in nine pages |
| `91d55ae` | `215154c` | Answer Q6 and Q9: the ledger is complete at 11 of 11 |
| `b08e4ba` | `543a910` | E30: put the MDN kernel on the same ladder as the score nets |
| `101a772` | `f1ee55e` | Close N-23's open question; register E31 for E29's budget confound |
| `4aa497e` | `f201bd7` | E30 and E31: the neural kernel is the worst of the three arms |
| `9a55030` | `e487406` | Queue the uncontended timing reruns; no E24/E27 cost number is citable |
| `9b1c940` | `cb1a993` | The phi^4 field as a second laboratory: model, learned pair energy, E3 |
| `4c24b1f` | `1c20c74` | E33 submitted: eight jobs from deployment 9b1c940; E32 laptop run star |
| `2ddd56c` | `81a9234` | E32 frozen: the field laboratory's spectrum, cells, oracle ladder, ref |
| `789fb1b` | `4ee0947` | E34 registered (physics, clean-data interface, learned samples); the f |
| `70647cf` | `4960405` | Report figure 2: legend in the trough between the wells, off the curve |
| `8d5b32e` | `b2b809c` | Report figure 2: one-column legend inside the trough |
| `f9c804b` | `95dc27f` | Two review nits; the first physics run is superseded (its kink claim w |
| `811eb06` | `3e69474` | E34: the coarse kink statistic; N-31 (sign changes are not kinks under |
| `52cf980` | `ff28263` | E34 physics_truth frozen: C(r) within 1.8 s.e. everywhere; raw kink de |
