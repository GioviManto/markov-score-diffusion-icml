#!/bin/zsh
# Overnight local queue on the 8-core laptop: runs the confirmatory experiments sequentially,
# each with `workers` processes, single-threaded BLAS (Apple Accelerate ignores OMP; one core
# per process is what it actually delivers). Logs under outputs/scratch/queue_*.log.
set -u
cd "$(dirname "$0")/../experiments"
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-6}
run() { name=$1; shift; echo "[$(date '+%F %T')] START $name" >> ../outputs/scratch/queue.log; "$@" > "../outputs/scratch/queue_$name.log" 2>&1; echo "[$(date '+%F %T')] END $name exit $?" >> ../outputs/scratch/queue.log; }
run E02a ../.venv/bin/python e02_capacity.py --only capacity --workers $W --output-dir ../outputs/E02a
run E06  ../.venv/bin/python e06_learning_through_noise.py --workers $W --output-dir ../outputs/E06
run E02b ../.venv/bin/python e02_capacity.py --only family --workers $W --output-dir ../outputs/E02b
