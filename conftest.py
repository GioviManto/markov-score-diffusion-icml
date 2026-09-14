"""Root conftest: make the src-layout package importable without installation, and
put `experiments/` on the path so tests can import `frozen_config` and `common`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT / "experiments", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end fits that take minutes, not seconds")
