# outputs — experiment results

CSVs and manifests written by `experiments/`. `frozen/` holds the runs the documents cite; everything else is scratch from development runs and may be stale.

Each run directory carries a `params_*.json` with the configuration, seeds, git commit (or the deploy stamp on the cluster) and environment. `CLUSTER_JOBS_FROZEN.md` is the job ledger: which Slurm job produced what, and which produced nothing usable — three of nine did.

Not tracked in git beyond the frozen set; these are regenerable and large.
