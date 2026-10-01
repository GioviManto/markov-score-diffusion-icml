#!/usr/bin/env bash
# Fetch a finished experiment's registered files from the cluster into outputs/<E>/ (CSV and
# JSON only; per-cell files and checkpoints stay on the cluster), then show the freeze status.
#   hpc/icml_fetch.sh E02a [E03 ...]
#   MSD_FETCH_DEST=hpc_E33k_s0 hpc/icml_fetch.sh E33k_s0     # land in a differently named local dir
#
# Size guard (N-32): GitHub refuses files above 100 MB, and on 26 Sep a 1.28 GB table fetched
# by this script was committed and kept the branch off GitHub for five days. Files above
# MSD_FETCH_MAX_MB (default 95) are NOT transferred; they are listed with their sizes, and the
# script exits non-zero so the omission cannot pass unnoticed. Fetch such a file by hand, keep it
# out of git (.gitignore), and record where it lives.
set -euo pipefail
cd "$(dirname "$0")/.."
source "${MSD_DEPLOY_ENV:-hpc/.last_deploy.env}"   # MSD_DEPLOY_ENV=hpc/.deploy_<tag>.env fetches from an older deployment
MAX_MB=${MSD_FETCH_MAX_MB:-95}
skipped=0
for E in "$@"; do
  DEST=${MSD_FETCH_DEST:-$E}
  mkdir -p "outputs/$DEST"
  big=$(ssh -o BatchMode=yes "$MSD_HPC_HOST" "cd '$MSD_OUT/$E' 2>/dev/null && find . -maxdepth 1 -type f \( -name '*.csv' -o -name '*.json' \) -size +${MAX_MB}M -exec ls -l {} \; " || true)
  if [[ -n "$big" ]]; then
    echo "REFUSED (above ${MAX_MB} MB, not transferred) in $E:" >&2
    echo "$big" | awk '{printf "    %s  %.0f MB\n", $NF, $5/1048576}' >&2
    skipped=1
  fi
  rsync -az --max-size="${MAX_MB}m" --include='*.csv' --include='*.json' --exclude='*' "$MSD_HPC_HOST:$MSD_OUT/$E/" "outputs/$DEST/"
  echo "fetched outputs/$DEST:"; ls -la "outputs/$DEST" | grep -E "csv|json" || echo "    (no csv/json files yet)"
done
echo; echo "next: python tools/registry.py --check && python tools/freeze.py <id>"
exit $skipped
