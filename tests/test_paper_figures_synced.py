"""The paper's copies of the figures-first figures are byte-identical to the report's (tools/sync_paper_figures.py)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_paper_figures_match_report_figures():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "sync_paper_figures.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
