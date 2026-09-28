"""No two SLURM submit cases may write to the same output directory.

N-27: the six E29 ladder jobs were each given one architecture but all wrote to
`{OUT}/E29`. Each job rebuilds the merged CSV from the cells its own config names, so
the last job to finish left a `ladder.csv` holding one architecture out of six. The
per-cell JSONs survived and the ladder was rebuilt, but a truncated merged CSV is not
distinguishable from a complete one by inspection and the freeze machinery would have
frozen it.

The rule this pins is not "E29 is fixed" but the general one: concurrent jobs must not
share a merged output file. Cases that are *designed* to append to one directory in
sequence (E03's screen/confirm/matched/length, E14's screen/confirm) are exempt by name,
because they are chained with --after and never run concurrently.
"""
from __future__ import annotations

import collections
import pathlib
import re

SUBMIT = pathlib.Path(__file__).resolve().parents[1] / "hpc" / "icml_submit.sh"

# Chained by --after, so they share a directory on purpose and never run at the same time.
SEQUENTIAL_FAMILIES = {"e03", "e14", "e14b", "e24c", "e24", "e27"}

# `\s*` before the `)`: bash allows `e30-s0 )` and the first version of this regex did
# not, so three of four E30 cases were invisible to the guard the moment they were added.
CASE = re.compile(r"^\s*([a-z0-9-]+)\s*\)\s*echo (.*?);;\s*$", re.M | re.S)
OUTDIR = re.compile(r"--output-dir \{OUT\}/([A-Za-z0-9_]+)")


def _cases() -> dict[str, str]:
    text = SUBMIT.read_text()
    out = {}
    for name, body in CASE.findall(text):
        m = OUTDIR.search(body)
        if m:
            out[name] = m.group(1)
    return out


def _family(case: str) -> str:
    return case.split("-")[0]


def test_submit_cases_parse():
    cases = _cases()
    assert len(cases) > 20, f"parsed only {len(cases)} cases; the regex has drifted"
    # Every case in the file must parse. Counting `)` lines independently means a regex
    # that silently stops matching some of them fails here instead of quietly covering less.
    declared = len(re.findall(r"^\s*[a-z0-9-]+\s*\)\s*echo ", SUBMIT.read_text(), re.M))
    assert len(cases) == declared, f"parsed {len(cases)} of {declared} declared cases"
    for name in ("e29-window", "e25-s0", "e30-s0", "e30-s4", "e30-s8", "e30-s12"):
        assert name in cases, f"{name} not parsed"


def test_no_two_concurrent_cases_share_an_output_dir():
    cases = _cases()
    by_dir = collections.defaultdict(list)
    for name, d in cases.items():
        by_dir[d].append(name)

    clashes = []
    for d, names in sorted(by_dir.items()):
        if len(names) < 2:
            continue
        families = {_family(n) for n in names}
        if len(families) == 1 and families.pop() in SEQUENTIAL_FAMILIES:
            continue  # chained with --after, by design
        clashes.append(f"{d} <- {sorted(names)}")

    assert not clashes, (
        "submit cases share an output directory and would clobber each other's merged CSV "
        "(N-27):\n  " + "\n  ".join(clashes)
    )
