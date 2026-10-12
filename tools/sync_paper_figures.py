#!/usr/bin/env python
"""Copy the figures-first figures from reports/figures/ into paper/figures/, where the manuscript reads them.

    python tools/sync_paper_figures.py           # copy what differs
    python tools/sync_paper_figures.py --check   # exit 1 if any paper copy differs from its source (tests/test_paper_figures_synced.py)

The sources are drawn by tools/make_figures_first.py, make_field_sweep_figs.py, make_report_field_figs.py and
make_field_runtime_fig.py (house rules: tools/figstyle); the paper carries byte-identical copies so that the report and
the paper can never show two versions of the same figure. The two figures that exist only for the paper
(fig_grid_coarse, fig_speciation) are drawn into paper/figures/ directly and are not listed here.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "reports" / "figures", ROOT / "paper" / "figures"
FIGURES = ("ff-summary", "ff-nongaussian-gain", "ff-sample-efficiency", "ff-misspec-repair",
           "fig-field-chain", "fig-field-sweep", "fig-field-runtime")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bad = 0
    for name in FIGURES:
        for ext in (".pdf", ".png"):
            s, d = SRC / f"{name}{ext}", DST / f"{name}{ext}"
            if not s.exists():
                print(f"[FAIL] missing source {s.relative_to(ROOT)}"); bad += 1; continue
            if d.exists() and _sha(s) == _sha(d):
                continue
            if a.check:
                print(f"[FAIL] {d.relative_to(ROOT)} differs from {s.relative_to(ROOT)}"); bad += 1
            else:
                shutil.copyfile(s, d); print(f"copied {d.relative_to(ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
