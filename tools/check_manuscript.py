#!/usr/bin/env python
"""Consistency checks between generated results and the manuscript.

Three checks, each of which failed silently in the previous manuscript's pipeline
(docs/CURRENT_STATE_AUDIT.md §2.2–2.3):

1. --regenerate: run every generator into a temporary directory and diff the result
   against the committed paper/sections/*.tex. A difference means a generator or an
   input changed without the fragment being regenerated (or a fragment was hand-edited).
2. --diff is implied by --regenerate; without --regenerate only checks 3–4 run.
3. Typed literals: every decimal number in paper/main.tex and paper/sections/*.tex body
   files that is not inside a generated fragment, a \\newcommand, a citation key, a
   label, a URL or an equation number must be listed in paper/LITERALS_ALLOWED.txt with
   its justification (analytic constant, definition, or a registered output path).
4. Every \\ref and \\cite in the manuscript resolves (checked from the .aux if present).

Exit code 1 on any failure; the report names each offending line.
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
GENERATORS = [
    "make_tab_efficiency.py", "make_tab_structured.py", "make_tab_screening.py",
    "make_aggregation_structured.py", "make_aggregation_robustness.py",
    "make_convergence_numbers.py", "make_rate_analysis.py", "make_tab_hellinger.py",
    "make_theory_numbers.py", "make_oracle_numbers.py",
]
# generators that need registered outputs which may not exist yet; skipped if the source is absent
OPTIONAL = {"make_capacity_numbers.py": "outputs/E02a/capacity.csv", "make_learning_numbers.py": "outputs/E06/levels.csv",
            "make_generation_numbers.py": "outputs/E05/generation.csv", "make_refinement_numbers.py": "outputs/E04/refinement.csv",
            "make_fig_phase.py": "outputs/E07/ladder.csv", "make_misspec_numbers.py": "outputs/E08/misspecification.csv",
            "make_baseline_numbers.py": "outputs/E03/confirm.csv", "make_family_numbers.py": "outputs/E02b/family.csv",
            "make_audit_numbers.py": "outputs/E16/edgeworth.csv", "make_fig_theory.py": "outputs/E18/closure_wide.csv",
            "make_fig_locality.py": "outputs/E15/locality.csv", "make_exceedance.py": "outputs/E07/ladder.csv",
            "make_grid_envelope.py": "outputs/E11/grid.csv",
            "make_gaussian_sanity.py": "outputs/E39/sanity.csv",
            # regime-numbers.tex is generated from E13x, the sixteen-seed run, NOT from the frozen
            # eight-seed E13: both would write the same fragment and the later one would silently win.
            # E13 stays frozen as history and feeds nothing.
            # E12b supersedes E12 the moment its ladder exists (see SUPERSEDES): until then the
            # fragment is generated from E12 and checked, rather than going unchecked.
            "make_sample_complexity.py": "outputs/E12/ladder.csv",
            "make_sample_complexity.py --src outputs/E12b": "outputs/E12b/ladder.csv",
            "make_regime_baselines.py": "outputs/E14/regimes.csv",
            "make_regime_baselines.py --src outputs/E14b": "outputs/E14b/regimes.csv",
            "make_regime_numbers.py --src outputs/E13x": "outputs/E13x/regimes.csv"}
# Two OPTIONAL entries that write the SAME fragment would both run, and the later one would win
# silently -- the trap E13/E13x sprang on 23 Sep, where the checker compared a fragment made from
# one run against numbers recomputed from another. Where a longer run supersedes a shorter one,
# say so here: while the right-hand key's source is absent the left-hand one generates the
# fragment and the gate checks it; once the new run lands the old key is dropped, so exactly one
# generator ever writes it. The fragments also carry a source stamp that audit_claims asserts,
# which catches the same mistake from the other side.
SUPERSEDES = {"make_sample_complexity.py": "make_sample_complexity.py --src outputs/E12b"}
# Fragments the legacy generators emit that the manuscript may or may not input.
FRAGMENT_GLOB = "sections/*.tex"
NUMBER = re.compile(r"(?<![\w\\{])(?:\d+\.\d+|\d+e-?\d+|\d+\s*\\times\s*10\^\{-?\d+\})(?![\w}])")
SKIP_LINE = re.compile(r"\\(newcommand|renewcommand|label|ref|cite|citep|citet|url|href|input|include|bibitem)")


def regenerate_and_diff() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "paper" / "sections").mkdir(parents=True)
        # Run the generators from a scratch tree that symlinks everything except paper/sections,
        # so what they emit can be diffed against the committed fragments. The legacy generators
        # write to paper/sections relative to the cwd; every later one resolves its repository
        # root from its own __file__ (ROOT = Path(__file__).resolve().parents[1]). For those the
        # symlinked tools/ defeated the scratch tree: __file__ resolved to the real path, ROOT to
        # the real repository, and the fragment was rewritten IN PLACE while the scratch tree
        # stayed empty and the diff passed vacuously (found 21 Sep 2026 -- a gated generator run
        # this way wrote 0 fragments to the scratch tree and touched the real one). tools/ is
        # therefore COPIED, so __file__ resolves inside the scratch tree and both conventions
        # land in tmp/paper/sections. Figures stay a symlink: a generator that also draws a
        # figure writes it in place, as before.
        for item in ROOT.iterdir():
            if item.name in {"paper", ".git", ".venv", "tools"}:
                continue
            (tmp / item.name).symlink_to(item)
        shutil.copytree(ROOT / "tools", tmp / "tools", ignore=shutil.ignore_patterns("__pycache__"))
        (tmp / "paper" / "figures").symlink_to(PAPER / "figures")
        gens = list(GENERATORS) + [g for g, src in OPTIONAL.items() if (ROOT / src).exists()]
        for old_gen, new_gen in SUPERSEDES.items():
            if new_gen in gens and old_gen in gens:
                gens.remove(old_gen)
        for g in gens:
            # a key may carry arguments ("make_x.py --src outputs/E14b"): one generator, several
            # registered outputs, each fragment gated on its own source
            name, *argv = g.split()
            r = subprocess.run([sys.executable, str(tmp / "tools" / name), *argv], cwd=tmp, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[FAIL] {g} exited {r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
                failures += 1
        for frag in sorted((tmp / "paper" / "sections").glob("*.tex")):
            committed = PAPER / "sections" / frag.name
            if not committed.exists():
                print(f"[WARN] generator emitted {frag.name} but the manuscript has no such fragment")
                continue
            a, b = committed.read_text().splitlines(), frag.read_text().splitlines()
            if a != b:
                failures += 1
                print(f"[FAIL] {frag.name} differs from the committed fragment:")
                for line in difflib.unified_diff(a, b, "committed", "regenerated", lineterm="", n=1):
                    print("   " + line)
    return failures


def typed_literals() -> int:
    allowed_path = PAPER / "LITERALS_ALLOWED.txt"
    allowed = set()
    if allowed_path.exists():
        for line in allowed_path.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                allowed.add(line.split()[0])
    generated = {p.name for p in (PAPER / "sections").glob("*.tex")
                 if "generated" in "\n".join(p.read_text().splitlines()[:8]).lower()}
    failures = 0
    for tex in [PAPER / "main.tex", *sorted((PAPER / "body").glob("*.tex")), *sorted((PAPER / "sections").glob("*.tex"))]:
        if tex.name in generated or not tex.exists():
            continue
        in_comment_block = False
        for no, line in enumerate(tex.read_text().splitlines(), 1):
            stripped = line.split("%")[0]
            if not stripped.strip() or SKIP_LINE.search(stripped):
                continue
            for m in NUMBER.finditer(stripped):
                tok = m.group(0).replace(" ", "")
                if tok in allowed:
                    continue
                failures += 1
                print(f"[FAIL] typed literal {tok!r} at {tex.relative_to(ROOT)}:{no}: {stripped.strip()[:90]}")
    return failures


def unresolved_refs() -> int:
    aux = PAPER / "main.aux"
    log = PAPER / "main.log"
    failures = 0
    if log.exists():
        txt = log.read_text(errors="replace")
        for pat in ("Reference `", "Citation `", "multiply defined", "undefined"):
            for line in txt.splitlines():
                if pat in line and ("Warning" in line or "undefined" in line):
                    failures += 1
                    print(f"[FAIL] {line.strip()[:120]}")
    return failures


def frozen_sources() -> int:
    """Every registered output a generator reads must be frozen (tools/freeze.py) and unchanged:
    the manuscript's numbers come from immutable artefacts, not from whatever the directory holds."""
    sys.path.insert(0, str(ROOT / "tools"))
    from freeze import check, is_frozen
    failures = check(verbose=False)
    for gen, src in OPTIONAL.items():
        if (ROOT / src).exists() and not is_frozen(src):
            print(f"[FAIL] {src} exists and feeds {gen} but is not frozen (python tools/freeze.py <id>)"); failures += 1
    return failures


_OVERFULL = re.compile(r"warning: ([^:\s]+):(\d+): Overfull \\hbox \(([\d.]+)pt too wide\)")
BODY_TOLERANCE_PT = 10.0     # a body paragraph a few points over is LaTeX's ordinary noise


def overfull_boxes() -> int:
    """Build the paper and fail on any table or figure that does not fit its environment.

    tectonic reports every Overfull \\hbox with the file it came from. Any such box in a
    generated table (sections/tab-*) or a figure file is a violation of
    docs/FIGURE_TABLE_STANDARDS.md T1/F4 -- the table runs past its column or the figure
    past the text width -- and fails the check. Elsewhere, a box more than BODY_TOLERANCE_PT
    too wide is also a failure; smaller ones in prose are reported and tolerated.
    """
    r = subprocess.run(["tectonic", "-X", "compile", "main.tex"], cwd=ROOT / "paper",
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("[FAIL] the paper does not build:\n" + "\n".join(r.stderr.splitlines()[-8:]))
        return 1
    bad = 0
    for m in _OVERFULL.finditer(r.stderr):
        f, line, pt = m.group(1), int(m.group(2)), float(m.group(3))
        hard = f.startswith("sections/tab-") or f.startswith("figures/") or pt > BODY_TOLERANCE_PT
        print(f"[{'FAIL' if hard else 'note'}] Overfull \\hbox {pt:.1f}pt too wide at {f}:{line}"
              + ("  -- a table or figure must fit its environment (FIGURE_TABLE_STANDARDS T1/F4)" if hard else ""))
        bad += int(hard)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    # Regeneration is ON by default: on 18 Sep a generated table shipped with numbers from
    # a superseded run while the paragraph beside it carried the frozen ones, because the
    # table file had not been rebuilt after the fetch. A stale generated file is a science
    # error, not a formatting one, and the check that catches it must not be optional.
    ap.add_argument("--no-regenerate", action="store_true", help="skip regenerating every generated file and diffing it (development only)")
    ap.add_argument("--regenerate", action="store_true", help="(kept for compatibility; regeneration is the default)")
    ap.add_argument("--diff", action="store_true", help="(kept for compatibility; the diff always runs with regeneration)")
    ap.add_argument("--no-literals", action="store_true")
    ap.add_argument("--no-freeze", action="store_true", help="skip the frozen-source check (development only)")
    ap.add_argument("--no-build", action="store_true", help="skip the build and its Overfull-box gate (development only)")
    args = ap.parse_args()
    failures = 0
    if not args.no_regenerate:
        failures += regenerate_and_diff()
    if not args.no_literals:
        failures += typed_literals()
    failures += unresolved_refs()
    if not args.no_freeze:
        failures += frozen_sources()
    if not args.no_build:
        failures += overfull_boxes()
    print("OK" if failures == 0 else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
