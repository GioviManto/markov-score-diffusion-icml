"""markov_score.utils.write_csv with rows whose keys differ.

A resumed run mixes cell files written before and after a column was added (E33's rerun of
3 Oct 2026 reuses 76 cells written without `failed` beside new cells that carry it).
csv.DictWriter took its columns from the first row and raised ValueError on the first later row
with an extra key, which would have stopped the rerun at its first new cell. Pinned: the header is
the union of the keys in first-seen order, a missing value is blank, and rows that all share one
key set are written exactly as before.
"""

from __future__ import annotations

import csv

import pytest

from markov_score.utils import write_csv


def test_uneven_rows_get_the_union_of_keys(tmp_path):
    p = tmp_path / "t.csv"
    write_csv(p, [{"a": 1, "b": 2}, {"a": 3, "b": 4, "failed": 1}, {"b": 5, "a": 6}])
    lines = p.read_text().splitlines()
    assert lines[0] == "a,b,failed"
    assert lines[1:] == ["1,2,", "3,4,1", "6,5,"]


def test_uniform_rows_are_written_as_before(tmp_path):
    rows = [{"x": 0.5, "y": "s"}, {"x": 1.5, "y": "t"}]
    p, q = tmp_path / "new.csv", tmp_path / "old.csv"
    write_csv(p, rows)
    with q.open("w", newline="", encoding="utf-8") as f:  # the writer as it was until 3 Oct 2026
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    assert p.read_bytes() == q.read_bytes()


def test_no_rows_is_still_an_error(tmp_path):
    with pytest.raises(ValueError):
        write_csv(tmp_path / "e.csv", [])
