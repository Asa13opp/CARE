"""Central path definitions for the online_rubric_rl project.

All Python scripts and tests should import from here instead of hardcoding
paths or deriving them with Path.parents[N].
"""
from __future__ import annotations

from pathlib import Path

# ── File-location-based roots ────────────────────────────────────────
# This module lives at:  src/online_rubric_rl/project_paths.py
_MODULE = Path(__file__).resolve()

# The online_rubric_rl_rlver project root (parents[2] from .../online_rubric_rl/project_paths.py)
ONLINE_RUBRIC_ROOT: Path = _MODULE.parents[2]

# The grandparent empathy/ directory (shared with sibling projects)
PROJECT_ROOT: Path = _MODULE.parents[3]

# ── Derived paths ────────────────────────────────────────────────────
SRC_ROOT: Path = ONLINE_RUBRIC_ROOT / "src"
OUTPUT_ROOT: Path = ONLINE_RUBRIC_ROOT / "outputs"
LOG_DIR: Path = ONLINE_RUBRIC_ROOT / "logs"
CONFIG_DIR: Path = ONLINE_RUBRIC_ROOT / "configs"
DATA_DIR: Path = ONLINE_RUBRIC_ROOT / "data"
SCRIPTS_DIR: Path = ONLINE_RUBRIC_ROOT / "scripts"
TESTS_DIR: Path = ONLINE_RUBRIC_ROOT / "tests"
VENDOR_DIR: Path = ONLINE_RUBRIC_ROOT / "vendor"
