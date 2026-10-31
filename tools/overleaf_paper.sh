#!/usr/bin/env bash
# The Overleaf-linked repository GioviManto/markov-score-diffusion-icml-paper.
#
# Since 31 Oct 2026 that repository holds the hand-written paper (Jerome's call of 29 Oct). It is the paper's source of
# truth and is edited in Overleaf. The 6 Oct ICML paper that this script used to mirror from paper/ is in its old/
# folder, and paper/ here is archival. The old `push` (rsync --delete of paper/) would have wiped the new paper, so it
# is retired; only the figures travel from this repository now.
#
#   tools/overleaf_paper.sh figures   # reports/paper-figures/twocol/ -> figures/ there, commit, push
#   tools/overleaf_paper.sh status    # the remote head, and which figures differ from the generated ones
#
# Before `figures`, if you edited in Overleaf: Menu -> GitHub -> Push Overleaf changes to GitHub (the script only
# fast-forwards and refuses to merge). After it: Menu -> GitHub -> Pull GitHub changes into Overleaf.
# The local clone lives in .overleaf-paper/ (git-ignored).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="GioviManto/markov-score-diffusion-icml-paper"
CLONE="$ROOT/.overleaf-paper"
FIGS="$ROOT/reports/paper-figures/twocol"

ensure_clone() {
  if [ ! -d "$CLONE/.git" ]; then
    gh repo clone "$REPO" "$CLONE" -- --quiet
  fi
  git -C "$CLONE" fetch --quiet origin
}

case "${1:-status}" in
  figures)
    ensure_clone
    if [ -n "$(git -C "$CLONE" status --porcelain)" ]; then
      echo "refusing: $CLONE has uncommitted changes" >&2
      exit 2
    fi
    git -C "$CLONE" checkout --quiet main
    if ! git -C "$CLONE" merge --quiet --ff-only origin/main; then
      echo "refusing: the local clone has diverged from origin/main" >&2
      exit 2
    fi
    mkdir -p "$CLONE/figures"
    cp "$FIGS"/*.pdf "$FIGS"/*.tikz "$CLONE/figures/"
    git -C "$CLONE" add figures
    if git -C "$CLONE" diff --cached --quiet; then
      echo "figures already up to date"
      exit 0
    fi
    src="$(git -C "$ROOT" rev-parse --short HEAD)"
    git -C "$CLONE" commit --quiet -m "figures from markov-score-diffusion-icml@$src (tools/paper_figs)"
    git -C "$CLONE" push --quiet origin HEAD:main
    echo "pushed figures; in Overleaf: Menu -> GitHub -> Pull GitHub changes into Overleaf"
    ;;
  status)
    ensure_clone
    echo "remote head: $(git -C "$CLONE" rev-parse --short origin/main); local: $(git -C "$CLONE" rev-parse --short HEAD)"
    for f in "$FIGS"/*.pdf "$FIGS"/*.tikz; do
      g="$CLONE/figures/$(basename "$f")"
      if [ ! -f "$g" ]; then
        echo "new:     $(basename "$f")"
      elif ! cmp -s "$f" "$g"; then
        echo "changed: $(basename "$f")"
      fi
    done
    ;;
  push|pull)
    echo "'$1' is retired: it mirrored the old paper/ (now old/ in the Overleaf repository), and push would delete" >&2
    echo "the new paper. Edit the paper in Overleaf; send figures with: $0 figures" >&2
    exit 1
    ;;
  *) echo "usage: $0 figures|status" >&2; exit 1 ;;
esac
