#!/usr/bin/env python
"""Freeze a registered output: pin every file the manuscript may read by SHA-256.

    python tools/freeze.py E02a            # freeze outputs/E02a (registry experiments/registry/E02.json)
    python tools/freeze.py --check         # every frozen output must still match; exit 1 otherwise
    python tools/freeze.py --status        # list what is frozen and what is not

The contract (docs/EXPERIMENTAL_CONTRACT.md). A number enters the paper only from a
FROZEN output: a directory whose CSV and params files are hashed into its registry entry
together with the commit that produced them and the moment they were frozen. Freezing
refuses a directory whose params record a dirty code path or no commit. After freezing,
the files are immutable in the sense that `--check` (run by CI and by
tools/check_manuscript.py) fails on any byte change; a re-run is a new artefact and is
frozen under a new registry entry or a versioned key, never over the old one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "experiments" / "registry"
sys.path.insert(0, str(ROOT / "tools"))
from registry import OUTPUT_DIRS, inspect  # noqa: E402

FROZEN_GLOBS = ("*.csv", "params_*.json", "winners.json", "floor_fits.json")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _registry_file(out_id: str) -> Path:
    """The registry entry that owns an output directory. One entry can own several directories
    (E02.json registers both the capacity part in outputs/E02a and the family part in
    outputs/E02b), so a direct map is tried first and then the id with its trailing part
    letter removed."""
    for rid, d in OUTPUT_DIRS.items():
        if d == out_id and (REG / f"{rid}.json").exists():
            return REG / f"{rid}.json"
    if (REG / f"{out_id}.json").exists():
        return REG / f"{out_id}.json"
    trimmed = out_id.rstrip("abcdefgh")
    if trimmed != out_id and (REG / f"{trimmed}.json").exists():
        return REG / f"{trimmed}.json"
    # A sharded run (E25 splits its 16 seeds across four jobs, each its own directory so
    # that concurrent jobs never share a merged CSV -- see N-27) freezes every shard into
    # the one registry entry: `frozen_outputs` is keyed by directory, so the shards coexist.
    # hpc_E25m_s4 -> E25, hpc_E25v3_s12 -> E25.
    m = re.match(r"^(?:hpc_)?(E\d+)[a-z0-9]*(?:_s\d+)$", out_id)
    if m and (REG / f"{m.group(1)}.json").exists():
        return REG / f"{m.group(1)}.json"
    raise SystemExit(f"no registry entry maps to outputs/{out_id}")


def _files(out_id: str) -> list[Path]:
    d = ROOT / "outputs" / out_id
    files = []
    for g in FROZEN_GLOBS:
        files += sorted(d.glob(g))
    return files


def freeze(out_id: str, force: bool = False) -> int:
    rec = inspect(out_id)
    if not rec["exists"] or not rec["rows"]:
        raise SystemExit(f"outputs/{out_id}: nothing to freeze (exists={rec['exists']}, rows={rec['rows']})")
    if (rec["dirty"] or not rec["commits"]) and not force:
        raise SystemExit(f"outputs/{out_id}: refusing to freeze -- provenance dirty={rec['dirty']} commits={sorted(rec['commits'])} (use --force for a scratch freeze that must never be cited)")
    files = _files(out_id)
    if not files:
        raise SystemExit(f"outputs/{out_id}: no files match {FROZEN_GLOBS}")
    reg = _registry_file(out_id)
    j = json.loads(reg.read_text())
    frozen = j.setdefault("frozen_outputs", {})
    for f in files:
        frozen[str(f.relative_to(ROOT))] = _sha(f)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    j.setdefault("frozen_at", {})[out_id] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    j.setdefault("frozen_commit", {})[out_id] = head
    j.setdefault("frozen_run_commits", {})[out_id] = sorted(rec["commits"])
    if "status" in j and "FROZEN" not in j["status"]:
        j["status"] = f"FROZEN {out_id} {j['frozen_at'][out_id]}; " + j["status"]
    reg.write_text(json.dumps(j, indent=1) + "\n")
    print(f"froze outputs/{out_id}: {len(files)} file(s) -> {reg.relative_to(ROOT)}")
    return 0


def check(verbose: bool = True) -> int:
    bad = 0; n = 0
    for reg in sorted(REG.glob("*.json")):
        j = json.loads(reg.read_text())
        for rel, sha in j.get("frozen_outputs", {}).items():
            n += 1
            p = ROOT / rel
            if not p.exists():
                print(f"MISSING  {rel} (frozen in {reg.name})"); bad += 1
            elif _sha(p) != sha:
                print(f"CHANGED  {rel} (frozen in {reg.name}): a frozen output was modified; re-runs are new artefacts"); bad += 1
    if verbose:
        print(f"frozen files checked: {n}, problems: {bad}")
    return bad


def is_frozen(rel: str) -> bool:
    """True if the root-relative file is pinned in some registry entry AND matches."""
    for reg in REG.glob("*.json"):
        j = json.loads(reg.read_text())
        sha = j.get("frozen_outputs", {}).get(rel)
        if sha is not None:
            p = ROOT / rel
            return p.exists() and _sha(p) == sha
    return False


def status() -> int:
    for reg in sorted(REG.glob("*.json")):
        j = json.loads(reg.read_text())
        fo = j.get("frozen_outputs", {})
        dirs = sorted({rel.split("/")[1] for rel in fo})
        print(f"{reg.stem:5s} frozen files={len(fo):3d} dirs={dirs} status={j.get('status', '')[:70]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("out_id", nargs="?"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--status", action="store_true"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.check:
        return 1 if check() else 0
    if a.status:
        return status()
    if not a.out_id:
        ap.error("give an output id (E02a, E03, ...) or --check / --status")
    return freeze(a.out_id, a.force)


if __name__ == "__main__":
    sys.exit(main())
