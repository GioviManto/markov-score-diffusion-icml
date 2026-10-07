"""Every frozen output is committed to git.

`tools/freeze.py --check` compares frozen files with their hashes on disk, so a frozen file that exists on one
machine but was never committed passes there and fails on every fresh clone. On 6 Oct 2026 outputs/E37/all.csv
(frozen in E37.json since 5 Oct, never committed) looked like a stray file during a clean-up and was moved to the
Trash; it was restored, byte-identical, on 7 Oct. This test turns that state into a failure on the machine that
froze the file, before anything can be lost.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_every_frozen_output_is_tracked():
    tracked = set(subprocess.run(["git", "ls-files", "outputs"], cwd=ROOT, capture_output=True, text=True,
                                 check=True).stdout.split())
    missing = [f"{reg.name}: {path}" for reg in sorted((ROOT / "experiments" / "registry").glob("*.json"))
               for path in json.loads(reg.read_text()).get("frozen_outputs", {}) if path not in tracked]
    assert not missing, "frozen but not committed (git add them):\n" + "\n".join(missing)
