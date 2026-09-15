"""The file-based cell runner: rows are written by the computing process, the CSV is
rebuilt from the cell files, finished cells are skipped on resume, and a raising cell
stops the run with its traceback on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from common import default_cell_key, run_cells


def _cell(cell):
    cfg, seed, n = cell
    if cfg.get("boom") and seed == 1:
        raise ValueError("boom")
    return [dict(seed=seed, n=n, region=r, risk=0.1 * seed + n / 1000, s="x" * cfg.get("pad", 0)) for r in ("all", "bulk")]


@pytest.mark.parametrize("workers", [1, 2])
def test_rows_come_back_through_files(tmp_path, workers):
    cfg = {"pad": 3000}
    cells = [(cfg, s, n) for s in range(3) for n in (32, 128)]
    rows = run_cells(cells, _cell, workers, tmp_path, "thing.csv")
    assert len(rows) == 12
    df = pd.read_csv(tmp_path / "thing.csv")
    assert sorted(df.seed.unique()) == [0, 1, 2] and len(df) == 12
    files = sorted(p.name for p in (tmp_path / "cells" / "thing").glob("*.json"))
    assert files == sorted(f"{s}_{n}.json" for s in range(3) for n in (32, 128))


def test_resume_skips_finished_cells(tmp_path, capsys):
    cfg = {}
    cells = [(cfg, s, 32) for s in range(4)]
    run_cells(cells[:2], _cell, 1, tmp_path, "r.csv")
    stamp = (tmp_path / "cells" / "r" / "0_32.json").stat().st_mtime_ns
    rows = run_cells(cells, _cell, 1, tmp_path, "r.csv")
    assert len(rows) == 8
    assert (tmp_path / "cells" / "r" / "0_32.json").stat().st_mtime_ns == stamp
    assert "2 already on disk, 2 to run" in capsys.readouterr().out


def test_failure_is_loud_and_on_disk(tmp_path):
    cfg = {"boom": True}
    cells = [(cfg, s, 32) for s in range(3)]
    with pytest.raises((ValueError, RuntimeError)):
        run_cells(cells, _cell, 2, tmp_path, "f.csv")
    failed = tmp_path / "cells" / "f" / "1_32.failed"
    assert failed.exists() and "boom" in failed.read_text()


def test_keys_are_stable_and_filename_safe():
    assert default_cell_key(({}, 0, 32)) == "0_32"
    assert default_cell_key(({}, 10000, 2048, "window", {"radius": 2, "width": 64}, 0.001, "eps")) == "10000_2048_window_radius=2_width=64_0.001_eps"
    assert default_cell_key(({}, "ar2", 0.1, 3, 512)) == "ar2_0.1_3_512"
    assert default_cell_key(({}, 0, 1.25)) == "0_1.25"
