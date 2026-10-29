"""E55's driver against E33's: the same data, array for array, and the same kernel fit, number for number, on the CPU;
the registered screen rule; the site-energy fits E55 adds are exactly those E33 lacks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import e14_regime_baselines as e14  # noqa: E402
import e33_field_learning as e33  # noqa: E402
import e55_unet_energy as e55  # noqa: E402

SMALL = dict(n_val=24, n_test=12, grid_size=81)          # >= 12 chains: one-view groups span the 12 levels


def _cfg(**kw):
    c = dict(e33.SETTINGS); c.update(SMALL); c.update(kw)
    return c


def test_common_reproduces_e33_common_on_an_e33_cell():
    cfg = _cfg()
    a = e33._common(cfg, "T0.5_iso_L32", 3, 16)
    b = e55.common(cfg, "T0.5_iso_L32", 3, 16, np)
    for i in (2, 3, 4, 5, 6, 7):                       # w, log_k, log_mu, sigma0, A, Av
        np.testing.assert_array_equal(a[i], b[i])
    for bundle_a, bundle_b in ((a[8], b[8]), (a[9], b[9])):
        for t in e33.T_SCHEDULE:
            np.testing.assert_array_equal(bundle_a[t][0], bundle_b[t][0])
            np.testing.assert_array_equal(bundle_a[t][1], bundle_b[t][1])
    assert {k: v for k, v in b[10].items() if k in a[10]} == a[10]
    assert set(b[10]) - set(a[10]) == {"val_draws", "test_draws"}
    assert e55.draws_hash(b[9]) == b[10]["test_draws"] != b[10]["test_bundle"]


def test_the_longer_chains_are_the_base_field():
    chain_a, info_a = e55.resolve_cell(e55.find_cell("T0.3_h1_L32"))
    chain_b, info_b = e55.resolve_cell(e55.find_cell("T0.3_h1_L512"))
    assert (info_a["T"], info_a["h"], info_a["rho1"]) == (info_b["T"], info_b["h"], info_b["rho1"])
    assert info_b["n_sites"] == 512


def test_fit_kernel_reproduces_e33_fit_arm_for_the_site_energy():
    cfg = _cfg(em_checkpoints=(2, 4))
    cell, seed, n = "T0.3_h1_L32", 2, 24
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = e33._common(cfg, cell, seed, n)
    L = base["n_sites"]
    groups = e14._one_view_groups(A, f"e33-em-{cell}-M{n}", seed, L)
    val_groups = e14._one_view_groups(Av, f"e33-emval-{cell}", seed, L)
    ref = e33._fit_arm(cfg, "em_pair_quartic", grid, w, groups, val, val_groups, test, seed, cell, L, n)
    k0 = e33._init_kernel("em_pair_quartic", cfg, grid, seed, cell, n)
    got = e55.fit_kernel(cfg, "em_pair_quartic", k0, grid, w, groups, val, val_groups, test, L, cfg["em_checkpoints"], np)
    assert [r["selection"] for r in ref] == [r["selection"] for r in got]
    for r, g in zip(ref, got):
        assert r["checkpoint"] == g["checkpoint"]
        assert abs(r["risk"] - g["risk"]) <= 1e-12 * abs(r["risk"])
        assert abs(r["val_selected"] - g["val_selected"]) <= 1e-12 * abs(r["val_selected"])


def _row(cell, arm, conf, n, v):
    import json
    return dict(cell=cell, screen_arm=arm, conf_key=json.dumps(conf, sort_keys=True), n_chains=n, val_selected=v)


def test_screen_rule_mean_log_risk_complete_and_finite_ties_to_the_smaller():
    small, mid, big = (dict(base=32, lr=2e-4), dict(base=64, lr=2e-4), dict(base=128, lr=2e-4))
    rows = [_row("c", "unet", small, 32, 0.4), _row("c", "unet", small, 2048, 0.1),     # mean log = log 0.2
            _row("c", "unet", mid, 32, 0.8), _row("c", "unet", mid, 2048, 0.05),        # mean log = log 0.2 (tie)
            _row("c", "unet", big, 2048, 0.01)]                                         # missing a size: cannot win
    w = e55.choose_winners(rows)
    assert w["c|unet"]["conf"] == small
    rows += [_row("c", "unet", big, 32, float("nan"))]                                   # complete but non-finite
    assert e55.choose_winners(rows)["c|unet"]["conf"] == small
    rows = [_row("c", "unet", small, 32, 0.4), _row("c", "unet", small, 2048, 0.1),
            _row("c", "unet", mid, 32, 0.3), _row("c", "unet", mid, 2048, 0.1)]
    assert e55.choose_winners(rows)["c|unet"]["conf"] == mid


def test_pair_targets_are_exactly_what_e33_lacks():
    t = set(e55.pair_targets(dict(e55.SETTINGS)))
    for c in e55.E33_CELLS:
        for n in (32, 128, 512, 2048):
            assert (c, n) not in t
        assert (c, 8192) in t
    assert ("T0.3_h1_L32", 32768) in t
    assert {n for c, n in t if c == "T0.3_h1_L512"} == {32, 128, 512, 2048}
    assert {n for c, n in t if c == "T0.3_h1_L128"} == {32, 128, 512, 2048, 8192}


def test_the_largest_fits_have_about_a_million_transitions():
    for c, sizes in e55.SIZES.items():
        L = e55.find_cell(c)["L"]
        assert max(sizes) * (L - 1) <= 1.1e6


def test_the_window_sizes_are_e33s_frozen_winners():
    import json
    f = Path(__file__).resolve().parents[1] / "outputs" / "hpc_E33n_s0" / "winners.json"
    if not f.exists():
        pytest.skip("frozen E33 outputs not present (cluster deployment)")
    frozen = {k.split("|")[0]: v["hp"] for k, v in json.loads(f.read_text()).items() if k.endswith("|window")}
    assert frozen == e55.E33_WINDOW_HP
    assert e55._window_hp("T0.3_h1_L512") == e55.E33_WINDOW_HP["T0.3_h1_L32"]


def test_every_set_key_in_the_e55_job_commands_is_a_known_setting():
    """apply_overrides refuses an unknown key (job 723189 died on 'timing_probes' in its first second)."""
    import re
    sub = (Path(__file__).resolve().parents[1] / "hpc" / "icml_submit.sh").read_text()
    start = sub.index("# E55 (26 Oct 2026)"); block = sub[start:sub.index("*) echo \"unknown experiment", start)]
    lines = block.split("\n")
    keys = {k for l in lines for k in re.findall(r'--set \\?"?([A-Za-z_]+)=', l)}
    assert {"cells", "winners", "screen_glob", "timing_probes"} <= keys
    assert keys <= set(e55.SETTINGS), keys - set(e55.SETTINGS)


def test_every_worker_compiles_into_its_own_cache():
    import os
    root = e55._private_compile_caches()
    assert str(os.getpid()) in root.name
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith(str(root))
    assert os.environ["TRITON_CACHE_DIR"].startswith(str(root))


def test_pooled_levels_is_e14_pooled_plus_the_per_level_errors():
    cfg = _cfg()
    *_, sigma0, A, Av, val, test, base = e55.common(cfg, "T1_iso_L32", 1, 16, np)
    from markov_score import theory
    f = lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t)  # noqa: E731
    pooled, per = e55.pooled_levels(f, test)
    assert pooled == e14._pooled(f, test)                       # identical, not merely close
    assert set(per) == set(e33.T_SCHEDULE) and all(0 < v < 1 for v in per.values())
    assert min(per.values()) <= pooled <= max(per.values())


def test_the_dsm_validation_loss_ranks_the_exact_denoiser_first():
    cfg = _cfg()
    *_, sigma0, A, Av, val, test, base = e55.common(cfg, "T2_iso_L32", 0, 16, np)
    from markov_score import theory
    dset = e55.dsm_validation_set(Av, "T2_iso_L32", 0)
    good = e55.dsm_loss(lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t), dset)
    bad = e55.dsm_loss(lambda X, t: X / np.exp(-t), dset)       # predicts no noise at all
    assert good < bad and abs(bad - 1.0) < 0.1                    # eps_hat = 0 costs E eps^2 = 1
    assert e55.dsm_validation_set(Av, "T2_iso_L32", 0)[3][2].tobytes() == dset[3][2].tobytes()   # fixed draws


def test_generation_stats_see_exact_samples_as_close_and_noise_as_far():
    chain, info = e55.resolve_cell(e55.find_cell("T0.3_h1_L32"))
    L = 32
    R = chain.sample_batch(np.random.default_rng(0), 2048, L)
    R2 = chain.sample_batch(np.random.default_rng(1), 2048, L)
    near = e55.generation_stats(R2, R, chain, info, L)
    far = e55.generation_stats(np.random.default_rng(2).standard_normal((2048, L)), R, chain, info, L)
    assert near["swd"] < 0.5 * far["swd"] and near["corr_max_err"] < 0.1 < far["corr_max_err"]
    assert abs(near["coarse_kink_density"] / e55.generation_stats(R, R, chain, info, L)["coarse_kink_density"] - 1) < 0.1


def test_kernel_row_picks_the_risk_selected_unfailed_fit(tmp_path):
    import pandas as pd
    rows = [dict(arm="em_energy", selection="risk", cell="c", n_chains=128, seed=0, theta="1;2", failed=0, hp='{"hidden": 16}'),
            dict(arm="em_energy", selection="evidence", cell="c", n_chains=128, seed=0, theta="9;9", failed=0, hp='{"hidden": 16}'),
            dict(arm="em_energy", selection="risk", cell="c", n_chains=128, seed=1, theta=None, failed=1, hp='{"hidden": 16}')]
    pd.DataFrame(rows).to_csv(tmp_path / "k.csv", index=False)
    assert e55._kernel_row([str(tmp_path / "missing.csv"), str(tmp_path / "k.csv")], "em_energy", "c", 128, 0).theta == "1;2"
    assert e55._kernel_row([str(tmp_path / "k.csv")], "em_energy", "c", 128, 1) is None


def test_each_network_uses_the_ladder_up_to_its_own_budget():
    lad = e55.SETTINGS["net_checkpoints"]
    assert lad[0] == 16 and lad[-1] == 2**20 == e55.SETTINGS["window_steps"] and e55.SETTINGS["steps"] == 2**18
    assert all(b / a < 1.45 for a, b in zip(lad, lad[1:]))           # half-octave rungs (rounded: 23/16 = 1.44)
    assert 2**18 in lad
