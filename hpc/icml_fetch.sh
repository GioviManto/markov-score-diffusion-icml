#!/usr/bin/env bash
# Fetch a finished experiment's registered files from the cluster into outputs/<E>/ (CSV and
# JSON only; per-cell files and checkpoints stay on the cluster), then show the freeze status.
#   hpc/icml_fetch.sh E02a [E03 ...]
set -euo pipefail
cd "$(dirname "$0")/.."
source hpc/.last_deploy.env
for E in "$@"; do
  mkdir -p "outputs/$E"
  rsync -az --include='*.csv' --include='*.json' --exclude='*' "$MSD_HPC_HOST:$MSD_OUT/$E/" "outputs/$E/"
  echo "fetched outputs/$E:"; ls -la "outputs/$E" | grep -E "csv|json"
done
echo; echo "next: python tools/registry.py --check && python tools/freeze.py <id>"
