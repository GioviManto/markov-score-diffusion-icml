"""The confirmatory protocol is one object, every registered script reads it, and its digest
is stable across processes (so a params file can say which protocol produced a run)."""

from __future__ import annotations

import importlib

from protocol import PROTOCOL, V1, protocol_hash


def test_hash_is_short_and_distinguishes_versions():
    assert len(protocol_hash()) == 12 and protocol_hash() != protocol_hash(V1)


def test_scripts_read_the_protocol():
    e02 = importlib.import_module("e02_capacity"); e03 = importlib.import_module("e03_baselines")
    e06 = importlib.import_module("e06_learning_through_noise"); e08 = importlib.import_module("e08_misspecification")
    assert e02.SETTINGS["components"] == PROTOCOL.capacities and e02.SETTINGS["em_checkpoints"] == PROTOCOL.em_checkpoints
    assert e02.SETTINGS["seeds"] == PROTOCOL.seeds and e02.SETTINGS["n_test"] == PROTOCOL.n_test
    assert e03.SETTINGS["dev_seeds"] == PROTOCOL.dev_seeds and e03.SETTINGS["net_checkpoints"] == PROTOCOL.net_checkpoints
    assert e03.SETTINGS["screen_restarts"] == PROTOCOL.screen_restarts and e03.SETTINGS["screen_steps"] == PROTOCOL.screen_steps
    assert e06.SETTINGS["em_checkpoints"] == PROTOCOL.em_checkpoints and e08.SETTINGS["em_checkpoints"] == PROTOCOL.em_checkpoints


def test_v1_is_the_laptop_run_of_15_september():
    assert V1.capacities == (1, 2, 4, 8) and V1.em_checkpoints[-1] == 400 and V1.screen_restarts == 1 and V1.screen_steps == 16_000
    assert V1.sizes == PROTOCOL.sizes and V1.seeds == PROTOCOL.seeds and V1.n_test == PROTOCOL.n_test
