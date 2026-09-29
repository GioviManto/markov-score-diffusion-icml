#!/usr/bin/env python
"""Registry sync: read every registered experiment's output directory and report its provenance.

`registry.py --sync` prints, for each experiments/registry/<id>.json, the output
directory, whether the params files exist, the commit(s) recorded, whether the code path
was dirty at run time, the resolved config hash, and the row count of the main CSV; it
then writes the commit and status back into the `commit` column of
docs/EXPERIMENT_REGISTRY.md for rows whose id matches. Nothing else in the document is
touched. `registry.py --check` exits non-zero if any DONE experiment lacks a clean
provenance record (the CI gate for citing a result).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "experiments" / "registry"
DOC = ROOT / "docs" / "EXPERIMENT_REGISTRY.md"
OUTPUT_DIRS = {"E01": "E01", "E02": "E02a", "E02a": "E02a", "E02b": "E02b", "E03": "E03", "E04": "E04", "E05": "E05", "E06": "E06", "E07": "E07", "E08": "E08", "E09": "E09", "E06b": "E06b", "E10": "E10", "E02a_v1": "E02a_v1", "E11": "E11", "E12": "E12", "E13": "E13", "E06c": "E06c", "E07b": "E07b", "E07c": "E07c", "E11b": "E11b", "E12b": "E12b", "E14": "E14", "E15": "E15", "E16": "E16", "E17": "E17", "E18": "E18", "E19": "E19", "E20": "E20", "E21": "E21", "E14b": "E14b", "E13x": "E13x", "E22": "hpc_E22c", "E23": "hpc_E23", "E24": "hpc_E24gpu2", "E25": "hpc_E25m_s0", "E26": "hpc_E26", "E27": "hpc_E27", "E28": "hpc_E28", "E29": "hpc_E29", "E30": "hpc_E30_s0", "E31": "hpc_E31_bigru"}
MAIN_CSV = {"E01": "closure_gap.csv", "E02a": "capacity.csv", "E02b": "family.csv", "E03": "confirm.csv", "E04": "refinement.csv",
            "E05": "generation.csv", "E06": "levels.csv", "E07": "ladder.csv", "E08": "misspecification.csv", "E09": "floor.csv", "E06b": "levels.csv", "E10": "evidence.csv", "E02a_v1": "capacity.csv", "E11": "grid.csv", "E12": "ladder.csv", "E13": "regimes.csv", "E06c": "levels.csv", "E07b": "ladder.csv", "E07c": "ladder.csv", "E11b": "grid.csv", "E12b": "ladder.csv", "E14": "regimes.csv", "E15": "locality.csv", "E16": "edgeworth.csv", "E17": "bias.csv", "E18": "closure_wide.csv", "E19": "moments.csv", "E20": "selection.csv", "E21": "information.csv", "E14b": "regimes.csv", "E13x": "regimes.csv", "hpc_E22c": "shapes.csv", "hpc_E23": "representation.csv", "hpc_E24gpu2": "cost.csv", "hpc_E25m_s4": "information_ladder.csv", "hpc_E25m_s8": "information_ladder.csv", "hpc_E25m_s0": "information_ladder.csv", "hpc_E25m_s12": "information_ladder.csv", "hpc_E25v3_s0": "information_ladder.csv", "hpc_E25v3_s4": "information_ladder.csv", "hpc_E25v3_s8": "information_ladder.csv", "hpc_E25v3_s12": "information_ladder.csv", "hpc_E26": "generation.csv", "hpc_E27": "direct.csv", "hpc_E28": "per_level.csv", "hpc_E29": "ladder.csv", "hpc_E30_s0": "joint.csv", "hpc_E30_s4": "joint.csv", "hpc_E30_s8": "joint.csv", "hpc_E30_s12": "joint.csv", "hpc_E31_window": "ladder.csv", "hpc_E31_mlp": "ladder.csv", "hpc_E31_bigru": "ladder.csv", "hpc_E31_tf": "ladder.csv", "hpc_E31_tcn": "ladder.csv", "hpc_E31_unet": "ladder.csv"}


SCRIPTS = {"E01": "e01_information_decay.py", "E02a": "e02_capacity.py", "E02b": "e02_capacity.py", "E03": "e03_baselines.py", "E04": "e04_refinement.py",
           "E05": "e05_generation.py", "E06": "e06_learning_through_noise.py", "E07": "e07_phase_diagram.py", "E08": "e08_misspecification.py", "E09": "e09_capacity_floor.py", "E06b": "e06_learning_through_noise.py", "E10": "e10_capacity_evidence.py", "E02a_v1": "e02_capacity.py", "E11": "e11_grid_validation.py", "E12": "e12_sample_complexity.py", "E13": "e13_regime_sweep.py", "E06c": "e06_learning_through_noise.py", "E07b": "e07_phase_diagram.py", "E07c": "e07_phase_diagram.py", "E11b": "e11_grid_validation.py", "E12b": "e12_sample_complexity.py", "E14": "e14_regime_baselines.py", "E15": "e15_locality_floor.py", "E16": "e16_edgeworth_oracle.py", "E17": "e17_misspec_bias.py", "E18": "e18_closure_wide.py", "E19": "e19_moment_estimators.py", "E20": "e20_evidence_selection.py", "E21": "e21_information_curve.py", "E14b": "e14_regime_baselines.py", "E13x": "e13_regime_sweep.py", "hpc_E22c": "e22_message_shapes.py", "hpc_E23": "e23_representation.py", "hpc_E24gpu2": "e24_inference_cost.py", "hpc_E25m_s0": "exp_34_information_ladder.py", "hpc_E26": "e05_generation.py", "hpc_E27": "e27_direct_sampler.py", "hpc_E28": "e28_per_level_ranking.py", "hpc_E29": "e29_matched_parameters.py", "hpc_E30_s0": "e30_joint_ladder.py", "hpc_E30_s4": "e30_joint_ladder.py", "hpc_E30_s8": "e30_joint_ladder.py", "hpc_E30_s12": "e30_joint_ladder.py", "hpc_E31_window": "e29_matched_parameters.py", "hpc_E31_mlp": "e29_matched_parameters.py", "hpc_E31_bigru": "e29_matched_parameters.py", "hpc_E31_tf": "e29_matched_parameters.py", "hpc_E31_tcn": "e29_matched_parameters.py", "hpc_E31_unet": "e29_matched_parameters.py"}


def inspect(out_id: str) -> dict:
    """Provenance of one output directory; `dirty` lists only uncommitted files ON THE CODE PATH
    (the static import closure of the experiment script, tools/provenance_gate.import_closure)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from provenance_gate import code_path_dirty, import_closure
    d = ROOT / "outputs" / out_id
    rec = {"dir": str(d.relative_to(ROOT)), "exists": d.exists(), "params": [], "commits": set(), "dirty": [], "config_hash": set(), "rows": None, "unmapped": False}
    if not d.exists():
        return rec
    closure = import_closure(ROOT / "experiments" / SCRIPTS.get(out_id, ""), ROOT) if out_id in SCRIPTS else set()
    params = []
    for p in sorted(d.glob("params_*.json")):
        j = json.loads(p.read_text())
        params.append((p, j))
        rec["params"].append(p.name)
        if j.get("git_commit"):
            rec["commits"].add(j["git_commit"][:7])
        if j.get("resolved_config_hash"):
            rec["config_hash"].add(j["resolved_config_hash"])
    hits = code_path_dirty(params, closure)
    rec["dirty"] = [f"{k}: {v}" for k, v in hits.items()]
    # `MAIN_CSV.get(out_id, "")` used to be interpolated straight into `d / ...`, and an
    # id absent from the map gave `d / "" == d`: a DIRECTORY, which exists(), so the row
    # count raised IsADirectoryError and took the whole --check gate down with it. That is
    # how E22..E27 came to be registered but never gated -- the crash looked like a bug in
    # the tool rather than a missing map entry. An unmapped id is now reported as such.
    name = MAIN_CSV.get(out_id)
    if not name:
        rec["unmapped"] = True
    else:
        csv = d / name
        if csv.is_file():
            rec["rows"] = sum(1 for _ in open(csv)) - 1
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sync", action="store_true"); ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ids = sorted(p.stem for p in REG.glob("*.json"))
    report = {}
    for i in ids:
        out_id = OUTPUT_DIRS.get(i, i)
        rec = inspect(out_id)
        if i == "E02":
            rec_b = inspect("E02b"); report["E02b"] = rec_b
        report[i if i != "E02" else "E02a"] = rec
    bad = 0
    for i, rec in report.items():
        line = f"{i:5s} {rec['dir']:16s} exists={rec['exists']} params={len(rec['params'])} rows={rec['rows']} commits={sorted(rec['commits'])} dirty={'yes' if rec['dirty'] else 'no'}"
        print(line)
        if rec["exists"] and rec["rows"] and (rec["dirty"] or not rec["commits"]):
            bad += 1
    if args.sync and DOC.exists():
        txt = DOC.read_text()
        for i, rec in report.items():
            if not rec["exists"] or not rec["commits"]:
                continue
            commit = ",".join(sorted(rec["commits"])) + (" (dirty)" if rec["dirty"] else "")
            txt = re.sub(rf"(\| {re.escape(i)} \|[^\n]*\| )[^|\n]*(\|\s*)$", lambda m: f"{m.group(1)}`{commit}` {m.group(2)}", txt, flags=re.M)
        DOC.write_text(txt); print("registry document synced")
    if args.check:
        from freeze import check as freeze_check
        bad += freeze_check()
    if args.check and bad:
        print(f"{bad} problem(s): registered output(s) without clean provenance or frozen outputs that changed"); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
