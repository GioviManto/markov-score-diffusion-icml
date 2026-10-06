#!/usr/bin/env bash
# Keep paper/ and the Overleaf-linked repository in step.
#
#   tools/overleaf_paper.sh push   # paper/ -> GioviManto/markov-score-diffusion-icml-paper
#   tools/overleaf_paper.sh pull   # that repository -> paper/ (review with `git diff paper/`, then commit)
#   tools/overleaf_paper.sh status # what each side has that the other has not
#
# The Overleaf project "markov-score-diffusion-icml-paper" was imported from that repository and is linked to it by
# Overleaf's GitHub Sync (Menu -> GitHub): after a `push` here, press "Pull GitHub changes into Overleaf" there;
# before a `pull` here, press "Push Overleaf changes to GitHub" there.
#
# Why a copy and not the folder itself: Overleaf caps a project at 2,000 files (this repository has ~2,500) and does
# not follow symbolic links, and paper/ has three (icml2026.sty, icml2026.bst, fancyhdr.sty -> style/icml2026/).
# The copy dereferences them and leaves out main.pdf (Overleaf compiles its own) and legacy-neurips/.
#
# Safety: `push` refuses when the remote has commits that were never pulled (Overleaf edits would be overwritten);
# `pull` never deletes a local file and never touches the three symlinked style files. The local clone lives in
# .overleaf-paper/ (git-ignored).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="GioviManto/markov-score-diffusion-icml-paper"
CLONE="$ROOT/.overleaf-paper"
MARK="$CLONE/.git/last_sync"
EXCLUDE=(--exclude .git --exclude main.pdf --exclude legacy-neurips/ --exclude '*.aux' --exclude '*.log'
         --exclude '*.out' --exclude '*.bbl' --exclude '*.blg' --exclude '*.synctex.gz')
STYLE=(icml2026.sty icml2026.bst fancyhdr.sty)

ensure_clone() {
  if [ ! -d "$CLONE/.git" ]; then
    gh repo clone "$REPO" "$CLONE" -- --quiet
  fi
  git -C "$CLONE" fetch --quiet origin
}

remote_head() { git -C "$CLONE" rev-parse origin/main 2>/dev/null || echo none; }

case "${1:-status}" in
  push)
    ensure_clone
    if [ -f "$MARK" ] && [ "$(remote_head)" != "$(cat "$MARK")" ]; then
      echo "refusing: the remote has commits not yet pulled (Overleaf edits?). Run: tools/overleaf_paper.sh pull" >&2
      exit 2
    fi
    git -C "$CLONE" checkout --quiet main 2>/dev/null || true
    git -C "$CLONE" reset --quiet --hard origin/main 2>/dev/null || true
    rsync -a --copy-links --delete "${EXCLUDE[@]}" "$ROOT/paper/" "$CLONE/"
    git -C "$CLONE" add -A
    if git -C "$CLONE" diff --cached --quiet; then
      echo "nothing to push"
    else
      src="$(git -C "$ROOT" rev-parse --short HEAD)"
      git -C "$CLONE" commit --quiet -m "paper/ from markov-score-diffusion-icml@$src"
      git -C "$CLONE" push --quiet origin HEAD:main
      echo "pushed paper/ at $src; in Overleaf: Menu -> GitHub -> Pull GitHub changes into Overleaf"
    fi
    git -C "$CLONE" rev-parse HEAD > "$MARK"
    ;;
  pull)
    ensure_clone
    git -C "$CLONE" checkout --quiet main 2>/dev/null || true
    git -C "$CLONE" reset --quiet --hard origin/main
    excl=("${EXCLUDE[@]}")
    for f in "${STYLE[@]}"; do excl+=(--exclude "/$f"); done
    rsync -a "${excl[@]}" "$CLONE/" "$ROOT/paper/"
    git -C "$CLONE" rev-parse HEAD > "$MARK"
    echo "pulled $(git -C "$CLONE" rev-parse --short HEAD); review with: git -C \"$ROOT\" diff --stat paper/"
    ;;
  status)
    ensure_clone
    echo "remote head: $(remote_head); last sync: $(cat "$MARK" 2>/dev/null || echo never)"
    tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
    rsync -a --copy-links "${EXCLUDE[@]}" "$ROOT/paper/" "$tmp/"
    git -C "$CLONE" reset --quiet --hard origin/main 2>/dev/null || true
    diff -rq -x .git "$tmp" "$CLONE" || true
    ;;
  *) echo "usage: $0 push|pull|status" >&2; exit 1 ;;
esac
