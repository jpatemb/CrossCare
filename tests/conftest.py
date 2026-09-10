"""Make the project root importable from tests.

This repo is deliberately not a Python package — `manual_loop.py` does a
flat `from tools import ...` and is run as `python3 manual_loop.py` from the
repo root, relying on Python putting the script's own directory on
sys.path. Adding an `__init__.py` here would break that, so instead the
repo root goes on sys.path once, for the whole tests/ directory.

Offline guards (no API key, no network) live in the repo-root conftest.py,
not here.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
