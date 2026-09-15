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
[[ -f hpc/.last_deploy.env ]] || { echo "run hpc/icml_deploy.sh first" >&2; exit 2; }
source hpc/.last_deploy.env
CPUS=${MSD_HPC_CPUS:-32}; MEM=${MSD_HPC_MEM:-64G}; TIME=${MSD_HPC_TIME:-12:00:00}
AFTER=""; WHAT=${1:-}; shift || true
while [[ $# -gt 0 ]]; do case "$1" in --after) AFTER="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[[ -n "$WHAT" ]] || { echo "usage: $0 <e02a|e02b|e03-screen|e03-confirm|e03-matched|e03-length|e06|e06b|e08|e09|e04|e05|all> [--after JOBID]" >&2; exit 2; }

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
    e04)         echo "e04_refinement.py --set em_source=\"'{OUT}/E02a/capacity.csv'\" --output-dir {OUT}/E04";;
    e05)         echo "e05_generation.py --set em_source=\"'{OUT}/E02a/capacity.csv'\" --output-dir {OUT}/E05";;
    *) echo "unknown experiment $1" >&2; exit 2;;
  esac
}
submit_one() {
  local name=$1 dep=${2:-}
  local cmd; cmd=$(cmd_for "$name"); cmd=${cmd//\{OUT\}/\$HOME/$MSD_OUT}
  local depflag=""; [[ -n "$dep" ]] && depflag="--dependency=afterok:$dep"     # dep may be "id1:id2"
  local workers_flag='--workers $SLURM_CPUS_PER_TASK' cpus=$CPUS            # every driver takes --workers
  local jid
  jid=$(ssh -o BatchMode=yes "$MSD_HPC_HOST" "sbatch --parsable $depflag" <<SB
#!/bin/bash
#SBATCH --job-name=msd-$name
#SBATCH --partition=$PART
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=$cpus --mem=$MEM --time=$TIME
#SBATCH --output=$RHOME/$MSD_HPC_ROOT/logs/%x_%j.out --error=$RHOME/$MSD_HPC_ROOT/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=\$HOME/$MSD_SRC/src
cd \$HOME/$MSD_SRC/experiments
echo "start=\$(date -Is) host=\$(hostname) job=\$SLURM_JOB_ID cpus=\$SLURM_CPUS_PER_TASK"; cat ../REVISION
\$HOME/$MSD_HPC_ROOT/venv/bin/python $cmd $workers_flag
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
