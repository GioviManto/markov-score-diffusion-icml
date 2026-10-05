"""tools/freeze.py: which registry entry owns an output directory.

On 5 Oct 2026 four confirmatory field directories could not be frozen: E33's oracles and screen
(hpc_E33o, hpc_E33n: a one-letter part, which the shard pattern does not cover) and E34's two
laptop runs of physics_learned (E34_learned, E34_learned_pair: no main CSV and no script on file,
so the provenance gate had no code path to check). They are mapped explicitly, not by a wider
pattern, because a wider pattern would also have accepted the PREVIEW directories
(E33_preview_laptop, ...) into the confirmatory entry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import freeze
import registry


@pytest.mark.parametrize("out_id,owner", [("hpc_E33o", "E33"), ("hpc_E33n", "E33"), ("hpc_E33k_s4", "E33"),
                                           ("hpc_E34c", "E34"), ("E34_learned", "E34"), ("E34_learned_pair", "E34"),
                                           ("hpc_E24x", "E24"), ("hpc_E24x_gpu", "E24"), ("hpc_E27x", "E27"), ("hpc_E27big_excl", "E27")])
def test_confirmatory_field_directories_have_an_owner(out_id, owner):
    assert freeze._registry_file(out_id).name == f"{owner}.json"


@pytest.mark.parametrize("out_id", ["E33_preview_laptop", "E33_preview_laptop_pair", "E34_preview_laptop_learned"])
def test_preview_directories_are_never_owned(out_id):
    with pytest.raises(SystemExit):
        freeze._registry_file(out_id)


@pytest.mark.parametrize("out_id", ["E34_learned", "E34_learned_pair"])
def test_laptop_runs_have_a_main_csv_and_a_code_path(out_id):
    assert registry.MAIN_CSV[out_id] == "learned.csv"
    assert registry.SCRIPTS[out_id] == "e34_field_physics.py"
