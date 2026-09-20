#!/usr/bin/env bash
# Compile ONE part of the compendium inside the real preamble, so a chapter can be checked
# on its own while the others are still being written.
#   reports/compendium/check_part.sh parts/04-fisher-cramer-rao.tex
# Prints the tectonic errors (if any) and the page count. Leaves nothing behind but
# build/<part>.pdf.
set -euo pipefail
cd "$(dirname "$0")"
part=${1:?usage: check_part.sh parts/NN-name.tex}
[[ -f "$part" ]] || { echo "no such part: $part" >&2; exit 2; }
mkdir -p build
stem=$(basename "$part" .tex)
# the preamble is everything of compendium.tex before \begin{document}; the body is one \input
{ sed -n '1,/^\\begin{document}/p' compendium.tex; echo "\\input{$part}"; echo "\\end{document}"; } > "build/$stem.tex"
if tectonic -X compile -Z search-path=. --outdir build "build/$stem.tex" >"build/$stem.log" 2>&1; then
  pages=$(pdfinfo "build/$stem.pdf" 2>/dev/null | awk '/^Pages/ {print $2}')
  echo "OK  $part -> build/$stem.pdf (${pages:-?} pages)"
  grep -n "Overfull \\\\hbox\|undefined\|Undefined" "build/$stem.log" | head -20 || true
else
  echo "FAIL $part"; grep -n "^error\|! \|Error" "build/$stem.log" | head -30; exit 1
fi
