"""E38's memory sampler, exercised without /proc: the peak it reports is the largest RSS seen during the call.

The cluster run reads /proc/self/statm in a background thread; macOS has no statm, so the logic is pinned here with
a fake source that rises during the call and falls back after it.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import e38_memory as e38


class _Fake(e38._PeakSampler):
    def __init__(self, during: float):
        super().__init__(period=1e-4)
        self.level, self.during, self.inside = 100.0, during, threading.Event()

    def now(self) -> float:
        return self.during if self.inside.is_set() else self.level


def test_the_peak_is_the_largest_rss_seen_during_the_call():
    s = _Fake(during=350.0)
    before = s.now()
    with s:
        s.inside.set()
        time.sleep(0.02)          # the "call": RSS sits at 350 MB while it runs
        s.inside.clear()
    assert before == 100.0 and s.peak == 350.0


def test_a_call_that_allocates_nothing_reads_its_own_level():
    s = _Fake(during=100.0)
    with s:
        s.inside.set(); time.sleep(0.005); s.inside.clear()
    assert s.peak == 100.0
