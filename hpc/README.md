# hpc — running registered experiments on the Bocconi cluster

Scripts for the SLURM cluster (needs the university VPN). Cluster account, paths and hosts come from the
environment, never from the repository: set `MSD_HPC_*` (see the header of each script).

| script | what it does |
|---|---|
| [icml_deploy.sh](icml_deploy.sh) | ships a **committed** revision (`git archive HEAD`, never the working tree) to the cluster, verifies its SHA-256, builds the shared venv, writes `hpc/.last_deploy.env` |
| [icml_submit.sh](icml_submit.sh) | submits one registered experiment or a chained set (`hpc/icml_submit.sh e56-screen`); resumable, outputs outside the source tree. `MSD_HPC_GRES` for a GPU, `MSD_HPC_VENV=venv-cuda` for the CUDA build, `MSD_HPC_THREADS` for jobs that measure wall-clock |
| [icml_fetch.sh](icml_fetch.sh) | copies a finished experiment's CSV/JSON into `outputs/<id>/` (size-guarded: nothing above 100 MB), then shows the freeze status |
| [icml_gpu_env.sh](icml_gpu_env.sh) | the CUDA environment used by GPU jobs |
| [deploy_clean.sh](deploy_clean.sh), [sync_to_cluster.sh](sync_to_cluster.sh) | the older clean-deploy path that `experiments/common.py` and `tools/provenance_gate.py` still name in their provenance messages |

Workflow: commit → `icml_deploy.sh` → `icml_submit.sh <case>` → wait → `icml_fetch.sh <id>` → read with the matching
`tools/core/read_eNN.py` → `tools/freeze.py <id>`. Pass a variable *name* to `sbatch --export`, never a value
containing a comma (commas silently truncate it).

The SLURM scripts of the thesis-era runs are in `archive/hpc-legacy/`.
