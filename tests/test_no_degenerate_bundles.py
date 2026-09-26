"""Guard against the degenerate-bundle defect, repo-wide.

THE DEFECT. `rng_for(tag, *keys)` returns a FRESH generator seeded from its arguments.
Called inside a comprehension whose loop variable is not one of those arguments,

    A = np.stack([prior.sample(rng_for("tag", seed, n), L) for _ in range(n)])

it is re-seeded identically on every iteration, so every row of `A` is the same sequence
and the "training set of n chains" is one chain replicated n times. Nothing raises. The
run completes, the CSV looks normal, and every learning curve is flat in n because more
copies of one sequence carry no more information.

WHY A TEST AND NOT A CODE REVIEW. This defect has now occurred three times in this
project. It voided E14's first runs (amendment N-22, 20 September). It was then fixed in
`e02_capacity.py` -- where the broken line was left behind as dead code rather than
deleted -- but never swept for elsewhere. So it was still present in `exp_32` and
`exp_34`, and on 25-26 September it silently voided all four E25 jobs (669323-669326,
~30 CPU-hours) before anyone looked at a number. A reviewer's eye has failed at this
three times; an assertion does not get tired.

WHAT IS ALLOWED. `rng_for` inside a comprehension is fine when the loop variable is one
of its arguments, e.g. `[... rng_for("tag", k) ... for k in range(n)]`, because each
iteration then gets a different seed. That is wasteful but correct, and
`exp_05_reverse_dynamics` does it deliberately.
"""

from __future__ import annotations

import ast
import pathlib

ROOTS = ("experiments", "src", "tools")


def _comprehension_offenders(path: pathlib.Path):
    """Yield (lineno, source) for comprehensions that re-seed rng_for per iteration."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:                       # not our problem here
        return
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            continue
        # every name bound by this comprehension's `for` clauses
        bound: set[str] = set()
        for gen in node.generators:
            for t in ast.walk(gen.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)
        for call in (c for c in ast.walk(node)
                     if isinstance(c, ast.Call)
                     and isinstance(c.func, ast.Name) and c.func.id == "rng_for"):
            # the call is safe iff one of its arguments mentions a bound loop variable
            names = {n.id for a in list(call.args) + [kw.value for kw in call.keywords]
                     for n in ast.walk(a) if isinstance(n, ast.Name)}
            if not (names & bound):
                yield node.lineno, ast.unparse(call)


def test_no_rng_for_reseeded_inside_comprehension():
    offenders = []
    for root in ROOTS:
        for path in sorted(pathlib.Path(root).rglob("*.py")):
            for lineno, src in _comprehension_offenders(path):
                offenders.append(f"{path}:{lineno}: {src}")
    assert not offenders, (
        "rng_for() is re-seeded identically on every iteration of a comprehension, so every "
        "element will be the SAME draw (see this file's docstring; it voided E14 and E25).\n"
        "Hoist the generator out of the comprehension:\n"
        "    rng = rng_for(tag, seed, n)\n"
        "    A = np.stack([prior.sample(rng, L) for _ in range(n)])\n\n"
        + "\n".join(offenders)
    )
