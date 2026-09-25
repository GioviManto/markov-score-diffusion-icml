#!/usr/bin/env bash
# Submit one registered experiment (or a chained set) to SLURM on the Bocconi HPC.
#
#   hpc/icml_submit.sh e02a                     # capacity, protocol v2
#   hpc/icml_submit.sh e03-screen               # then: e03-confirm --after <jobid>, e03-matched, e03-length
#   hpc/icml_submit.sh e06 | e02b | e08 | e09 | e04 | e05
#   hpc/icml_submit.sh all                      # everything, with the dependencies wired
#
# Env (from hpc/.last_deploy.env, written by icml_deploy.sh) plus optional
#   MSD_HPC_PARTITION  (default: first non-GPU partition sinfo lists)
#   MSD_HPC_CPUS       (default 32)   MSD_HPC_MEM (default 64G)   MSD_HPC_TIME (default 12:00:00)
# Every job: one node, --workers = $SLURM_CPUS_PER_TASK, single-threaded BLAS, outputs under
# $MSD_OUT/<E>/ (outside the source tree), logs under $MSD_HPC_ROOT/logs/. Resumable: a
# resubmitted job skips the cells whose files exist (experiments/common.py: run_cells).
set -euo pipefail
cd "$(dirname "$0")/.."
# MSD_DEPLOY_ENV=hpc/.deploy_<shortsha>.env resubmits into an OLDER deployment (a resumed run
# must land in the output directory its finished cells already live in); default = newest.
DEPLOY_ENV=${MSD_DEPLOY_ENV:-hpc/.last_deploy.env}
[[ -f "$DEPLOY_ENV" ]] || { echo "run hpc/icml_deploy.sh first (no $DEPLOY_ENV)" >&2; exit 2; }
source "$DEPLOY_ENV"
CPUS=${MSD_HPC_CPUS:-32}; MEM=${MSD_HPC_MEM:-64G}; TIME=${MSD_HPC_TIME:-12:00:00}
# Optional: MSD_HPC_WORKERS caps --workers below the cpu count (memory-bound cells; GPU jobs),
# MSD_HPC_GRES adds a --gres line (e.g. gpu:4g.40gb:1), MSD_HPC_VENV picks another venv under
# $MSD_HPC_ROOT (default venv; venv-cuda carries the CUDA torch build for the GPU partitions).
WORKERS_CAP=${MSD_HPC_WORKERS:-}; GRES=${MSD_HPC_GRES:-}; VENV=${MSD_HPC_VENV:-venv}
# MSD_HPC_THREADS sets the BLAS/OMP thread count inside the job. Default 1, which is right
# for the cell-parallel drivers (one thread per worker, no oversubscription). E24 is the
# exception: it measures wall-clock, so it must be told how many threads it is allowed and
# must record that -- a timing at an unstated thread count is not a reproducible number.
THREADS=${MSD_HPC_THREADS:-1}
AFTER=""; WHAT=${1:-}; shift || true
while [[ $# -gt 0 ]]; do case "$1" in --after) AFTER="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[[ -n "$WHAT" ]] || { echo "usage: $0 <e24-cpu|e24-gpu|e06c|e07b|e07c|e11b|e12b|e14|e14-seeds|e14b-screen|e14b-confirm|e13x|e02a|e02b|e03-screen|e03-confirm|e03-matched|e03-length|e06|e06b|e08|e09|e04|e05|all> [--after JOBID]" >&2; exit 2; }

partition() {
  if [[ -n "${MSD_HPC_PARTITION:-}" ]]; then echo "$MSD_HPC_PARTITION"; return; fi
  ssh -o BatchMode=yes "$MSD_HPC_HOST" 'sinfo -h -o "%P" | sed "s/\*//" | grep -v -i gpu | head -1'
}
PART=$(partition); [[ -n "$PART" ]] || { echo "no partition found; set MSD_HPC_PARTITION" >&2; exit 2; }
# SLURM does not expand $HOME inside #SBATCH directives: resolve the remote home once and use absolute paths
RHOME=$(ssh -o BatchMode=yes "$MSD_HPC_HOST" 'echo $HOME'); [[ -n "$RHOME" ]] || { echo "cannot resolve remote HOME" >&2; exit 2; }

# name -> "script args" (relative to experiments/); {OUT} is replaced by the remote outputs dir
cmd_for() {
  case "$1" in
    e02a)        echo "e02_capacity.py --only capacity --output-dir {OUT}/E02a";;
    e02b)        echo "e02_capacity.py --only family --output-dir {OUT}/E02b";;
    e03-screen)  echo "e03_baselines.py --only screen --output-dir {OUT}/E03";;
    e03-confirm) echo "e03_baselines.py --only confirm --set winners=\"'auto:{OUT}/E03/screening.csv'\" --output-dir {OUT}/E03";;
    e03-matched) echo "e03_baselines.py --only matched --set winners=\"'auto:{OUT}/E03/screening.csv'\" --output-dir {OUT}/E03";;
    e03-length)  echo "e03_baselines.py --only length --set winners=\"'auto:{OUT}/E03/screening.csv'\" --output-dir {OUT}/E03";;
    e06)         echo "e06_learning_through_noise.py --output-dir {OUT}/E06";;
    e06b)        echo "e06_learning_through_noise.py --set rho=0.5 --set n_chains=2048 --set levels=\"(0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.4)\" --output-dir {OUT}/E06b";;
    e08)         echo "e08_misspecification.py --output-dir {OUT}/E08";;
    e09)         echo "e09_capacity_floor.py --output-dir {OUT}/E09";;
    e11)         echo "e11_grid_validation.py --output-dir {OUT}/E11";;
    e13)         echo "e13_regime_sweep.py --output-dir {OUT}/E13";;
    e12)         echo "e12_sample_complexity.py --output-dir {OUT}/E12";;
    # --- budget and quadrature controls, and the regime robustness of the headline (18 Sep) ---
    e06c)        echo "e06_learning_through_noise.py --set em_checkpoints=\"(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1600, 2200, 3000, 4000, 6000, 8000)\" --output-dir {OUT}/E06c";;
    e07b)        echo "e07_phase_diagram.py --set grid_size=801 --set half_width=12.0 --output-dir {OUT}/E07b";;
    e07c)        echo "e07_phase_diagram.py --set grid_size=1601 --set half_width=16.0 --output-dir {OUT}/E07c";;
    e11b)        echo "e11_grid_validation.py --set lengths=\"(64, 128)\" --output-dir {OUT}/E11b";;
    e12b)        echo "e12_sample_complexity.py --set levels=\"(0.6, 0.7, 0.8, 0.9, 1.0)\" --set seeds=\"(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)\" --set em_checkpoints=\"(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1600, 2200, 3000, 4000, 6000, 8000)\" --output-dir {OUT}/E12b";;
    e14-screen)  echo "e14_regime_baselines.py --only screen --output-dir {OUT}/E14";;
    e14-confirm) echo "e14_regime_baselines.py --only confirm --set winners=\"'auto:{OUT}/E14/screening.csv'\" --output-dir {OUT}/E14";;
    # --- 21 Sep: the protocol's sixteen seeds for the regime comparisons, and the bimodal member ---
    e14-seeds)   echo "e14_regime_baselines.py --only confirm --set seeds=\"(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)\" --set winners=\"'auto:{OUT}/E14/screening.csv'\" --output-dir {OUT}/E14";;
    e14b-screen) echo "e14_regime_baselines.py --only screen --set family=\"'bimodal'\" --set rhos=\"(0.85,)\" --set lengths=\"(32,)\" --output-dir {OUT}/E14b";;
    e14b-confirm) echo "e14_regime_baselines.py --only confirm --set family=\"'bimodal'\" --set rhos=\"(0.85,)\" --set lengths=\"(32,)\" --set seeds=\"(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)\" --set winners=\"'auto:{OUT}/E14b/screening.csv'\" --output-dir {OUT}/E14b";;
    e13x)        echo "e13_regime_sweep.py --set seeds=\"(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)\" --output-dir {OUT}/E13x";;
    e10)         echo "e10_capacity_evidence.py --set em_source=\"'{OUT}/E02a/capacity.csv'\" --output-dir {OUT}/E10";;
    e04)         echo "e04_refinement.py --set em_source=\"'{OUT}/E02a/capacity.csv'\" --output-dir {OUT}/E04";;
    e05)         echo "e05_generation.py --set em_source=\"'{OUT}/E02a/capacity.csv'\" --set net_dir=\"'{OUT}/E03/ckpts'\" --output-dir {OUT}/E05";;
    # --- 25 Sep, after the Jerome call: inference cost and scaling (docs/JEROME_CALL_2026-09-25.md Q7/Q11) ---
    # Cost is the whole measurement here, so both arms must get the same hardware. The CPU job
    # asks for a whole node's threads; the GPU job runs the SAME driver with devices=("cpu","gpu")
    # so the cross-device ratio comes from one process on one machine rather than two runs.
    e24-cpu)     echo "e24_inference_cost.py --set devices=\"('cpu',)\" --output-dir {OUT}/E24";;
    e24-gpu)     echo "e24_inference_cost.py --set devices=\"('cpu', 'gpu')\" --output-dir {OUT}/E24gpu";;
    *) echo "unknown experiment $1" >&2; exit 2;;
  esac
}
submit_one() {
  local name=$1 dep=${2:-}
  local cmd; cmd=$(cmd_for "$name"); cmd=${cmd//\{OUT\}/\$HOME/$MSD_OUT}
  local depflag=""; [[ -n "$dep" ]] && depflag="--dependency=afterok:$dep"     # dep may be "id1:id2"
  # --workers only for drivers whose source declares it: the serial oracle ladders (e01, e07)
  # do not, and passing it to them failed E07b at launch on 18 Sep (argparse: unrecognized
  # arguments). Decided here, at submission, from the driver's own source.
  local driver=${cmd%% *} workers_flag='' cpus=$CPUS
  if grep -q -- '"--workers"' "experiments/$driver"; then workers_flag="--workers ${WORKERS_CAP:-\$SLURM_CPUS_PER_TASK}"; else cpus=4; fi
  local gres_line=''; [[ -n "$GRES" ]] && gres_line="#SBATCH --gres=$GRES"
  local jid
  jid=$(ssh -o BatchMode=yes "$MSD_HPC_HOST" "sbatch --parsable $depflag" <<SB
#!/bin/bash
#SBATCH --job-name=msd-$name
#SBATCH --partition=$PART
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=$cpus --mem=$MEM --time=$TIME
#SBATCH --propagate=NONE
$gres_line
#SBATCH --output=$RHOME/$MSD_HPC_ROOT/logs/%x_%j.out --error=$RHOME/$MSD_HPC_ROOT/logs/%x_%j.err
set -euo pipefail
# The login shell carries RLIMIT_CPU = 600 s. Without --propagate=NONE SLURM hands that
# limit to every process in the job, which kills a multi-worker run partway through with
# 'CPU time limit exceeded' (E05, 15 Sep) or leaves the job holding an idle node (E06b).
ulimit -t unlimited 2>/dev/null || true
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export PYTHONPATH=\$HOME/$MSD_SRC/src
cd \$HOME/$MSD_SRC/experiments
echo "start=\$(date -Is) host=\$(hostname) job=\$SLURM_JOB_ID cpus=\$SLURM_CPUS_PER_TASK"; cat ../REVISION
\$HOME/$MSD_HPC_ROOT/$VENV/bin/python $cmd $workers_flag
echo "done=\$(date -Is)"
SB
)
  echo "$name -> job $jid (partition $PART, $cpus cpus, dep=${dep:-none})"; LAST_JOB=$jid
}
case "$WHAT" in
  all)
    submit_one e02a "$AFTER"; J_E02A=$LAST_JOB
    submit_one e09 "$AFTER"
    submit_one e03-screen "$AFTER"; J_SCREEN=$LAST_JOB
    submit_one e03-confirm "$J_SCREEN"; J_CONF=$LAST_JOB
    submit_one e03-matched "$J_SCREEN"
    submit_one e03-length "$J_CONF"
    submit_one e06 "$AFTER"; submit_one e02b "$AFTER"; submit_one e08 "$AFTER"
    submit_one e04 "$J_E02A"
    submit_one e05 "$J_E02A:$J_CONF";;
  *) submit_one "$WHAT" "$AFTER";;
esac
