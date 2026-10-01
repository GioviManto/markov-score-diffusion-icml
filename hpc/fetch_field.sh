#!/usr/bin/env bash
# Fetch every field-laboratory shard (E33, E33 amendment 1, E34 clean) from the deployment it was
# submitted from, into the local directory the registry expects, then check coverage and read the
# predeclared predictions. Safe to rerun: rsync only transfers what changed, and an unfinished
# shard is reported as such rather than read.
#
#   hpc/fetch_field.sh            # needs the Bocconi VPN
#
# Where each job ran (experiments/registry/E33.json and E34.json):
#   9b1c940-20261001T084745Z  E33o (678084), E33k_s0..s12 (678087-678090), E33n (678091), E33n_s0/s8 (678092/678094)
#   52cf980-20261001T094107Z  E34c (678166)
#   32448fe-20261001T112539Z  E33q_s0..s12 (678260-678263)
set -uo pipefail
cd "$(dirname "$0")/.."
status=0
fetch() {  # env remote_dir
  MSD_DEPLOY_ENV="hpc/.deploy_$1.env" MSD_FETCH_DEST="hpc_$2" hpc/icml_fetch.sh "$2" >/tmp/msd_fetch_$$.log 2>&1 || status=1
  grep -E "REFUSED|MB$" /tmp/msd_fetch_$$.log || true
  ls outputs/hpc_$2/*.csv >/dev/null 2>&1 && echo "  $2: $(ls outputs/hpc_$2/*.csv | xargs -n1 basename | tr '\n' ' ')" || echo "  $2: nothing yet"
}
ssh -o BatchMode=yes -o ConnectTimeout=15 3164542@lnode02-da.hpc.unibocconi.it true 2>/dev/null \
  || { echo "cluster unreachable: connect the Bocconi VPN (GlobalProtect) first" >&2; exit 2; }
echo "--- fetching"
for s in E33o E33k_s0 E33k_s4 E33k_s8 E33k_s12 E33n E33n_s0 E33n_s8; do fetch 9b1c940 "$s"; done
for s in E33q_s0 E33q_s4 E33q_s8 E33q_s12; do fetch 32448fe "$s"; done
fetch 52cf980 E34c
rm -f /tmp/msd_fetch_$$.log
echo "--- coverage"
.venv/bin/python tools/check_field_coverage.py || status=1
echo "--- predictions (confirmatory shards present so far)"
.venv/bin/python tools/make_field_numbers.py 2>&1 | grep -v "timestamp seems" | tail -40
exit $status
