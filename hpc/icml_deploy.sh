#!/usr/bin/env bash
# Deploy THIS repository (a committed revision, never the working tree) to the Bocconi HPC.
#
#   hpc/icml_deploy.sh                # deploy HEAD; refuses a dirty tree
#   hpc/icml_deploy.sh --allow-dirty  # scratch deployment: stamps source_is_clean=false
#
# What it does, in order: git-archive HEAD -> tarball -> SHA-256 -> scp -> verify the digest
# on the other side -> extract under $MSD_HPC_ROOT/src/<short>-<stamp>/ -> write REVISION at
# the package root (experiments/common.py reads it when .git is absent) -> create the shared
# virtual environment once ($MSD_HPC_ROOT/venv; numpy/scipy/pandas/matplotlib/torch-cpu/
# pytest, no editable install: jobs set PYTHONPATH to the deployment) -> print the env file
# hpc/.last_deploy.env that icml_submit.sh sources.
#
# Coordinates: MSD_HPC_HOST (default lnode01-da.hpc.unibocconi.it; the user comes from
# ~/.ssh/config), MSD_HPC_ROOT (default msd-icml, under $HOME on the cluster).
set -euo pipefail
HOST=${MSD_HPC_HOST:-lnode01-da.hpc.unibocconi.it}
REMOTE_ROOT=${MSD_HPC_ROOT:-msd-icml}
cd "$(dirname "$0")/.."
ALLOW_DIRTY=0; [[ "${1:-}" == "--allow-dirty" ]] && ALLOW_DIRTY=1
dirty=$(git status --porcelain | grep -v "^?? outputs/" || true)
if [[ -n "$dirty" && $ALLOW_DIRTY -eq 0 ]]; then
  echo "FATAL: working tree is dirty; commit first (or --allow-dirty for a scratch deployment)." >&2
  echo "$dirty" | sed 's/^/    /' >&2; exit 1
fi
commit=$(git rev-parse HEAD); short=$(git rev-parse --short HEAD); stamp=$(date -u +%Y%m%dT%H%M%SZ); tag="${short}-${stamp}"
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
git archive --format=tar -o "$work/src.tar" HEAD
digest=$(shasum -a 256 "$work/src.tar" | cut -d' ' -f1)
printf 'commit=%s\narchive_sha256=%s\ndeployed_at=%s\ndirty=%s\n' "$commit" "$digest" "$stamp" "$(echo "$dirty" | tr '\n' ';' | sed 's/;$//')" > "$work/REVISION"
SRC="${REMOTE_ROOT}/src/${tag}"; OUT="${REMOTE_ROOT}/outputs/${tag}"
echo "commit $commit"; echo "digest $digest"; echo "remote $HOST:~/$SRC"
ssh -o BatchMode=yes "$HOST" "mkdir -p '$SRC' '$OUT' '$REMOTE_ROOT/logs'"
scp -q "$work/src.tar" "$HOST:$SRC/src.tar"; scp -q "$work/REVISION" "$HOST:$SRC/REVISION"
remote_digest=$(ssh -o BatchMode=yes "$HOST" "sha256sum '$SRC/src.tar' | cut -d' ' -f1")
[[ "$remote_digest" == "$digest" ]] || { echo "FATAL: digest mismatch after transfer ($remote_digest)" >&2; exit 1; }
ssh -o BatchMode=yes "$HOST" bash -s "$SRC" "$REMOTE_ROOT" <<'REMOTE'
set -euo pipefail
SRC=$1; ROOT=$2
cd "$HOME/$SRC" && tar xf src.tar && rm src.tar
echo "extracted $(find . -name '*.py' | wc -l) python files"
# one shared environment per cluster account; created once, reused by every deployment
if [[ ! -x "$HOME/$ROOT/venv/bin/python" ]]; then
  (module load python/3.12 2>/dev/null || module load python 2>/dev/null || true)
  PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
  echo "creating venv with $PY ($($PY --version))"
  $PY -m venv "$HOME/$ROOT/venv"
  "$HOME/$ROOT/venv/bin/pip" install --quiet --upgrade pip
  "$HOME/$ROOT/venv/bin/pip" install --quiet "numpy>=1.26" "scipy>=1.11" "pandas>=2.0" "matplotlib>=3.8" "pytest>=8.0"
  "$HOME/$ROOT/venv/bin/pip" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
fi
"$HOME/$ROOT/venv/bin/python" -c "import numpy, scipy, pandas, torch; print('venv ok: numpy', numpy.__version__, 'torch', torch.__version__)"
# Import check only. The quick experiment run used to happen here and was the wrong thing to do
# on a shared login node (4 cores, CPU-limited: it was killed mid-numpy-import on 15 Sep). The
# real smoke is the first submitted job, which runs on a compute node.
cd "$HOME/$SRC/experiments" && OMP_NUM_THREADS=1 PYTHONPATH="$HOME/$SRC/src" "$HOME/$ROOT/venv/bin/python" -c "
import sys; sys.path.insert(0, '.')
import e02_capacity, e03_baselines, e04_refinement, e05_generation, e06_learning_through_noise, e08_misspecification, e09_capacity_floor
from protocol import PROTOCOL, protocol_hash
print('imports ok; protocol', PROTOCOL.version, protocol_hash())" || echo "WARNING: import check failed or was killed on the login node; the submitted job will report the real error"
REMOTE
cat > hpc/.last_deploy.env <<ENVEOF
MSD_HPC_HOST=$HOST
MSD_HPC_ROOT=$REMOTE_ROOT
MSD_SRC=$SRC
MSD_OUT=$OUT
MSD_TAG=$tag
MSD_COMMIT=$commit
ENVEOF
echo "wrote hpc/.last_deploy.env (SRC=$SRC OUT=$OUT)"
