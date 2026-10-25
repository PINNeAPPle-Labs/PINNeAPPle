"""Where the app keeps its state, and its limits (environment variables, all optional; see the README).

State lives under ``PINN_ARENA_DATA_DIR`` (default ``/data/pinn_arena`` in the container, ``~/.pinneapple/apps/
pinn_arena`` locally): the SQLite database, uploaded datasets and trained weights.
"""
from __future__ import annotations

import os
from pathlib import Path

from pinneapple_core.service import data_dir

PREFIX = "PINN_ARENA"
NAME = "pinn_arena"


def _int(name: str, default: int) -> int:
    return int(os.environ.get(f"{PREFIX}_{name}") or default)


DATA_DIR: Path = data_dir(NAME, PREFIX)
DB_PATH: Path = DATA_DIR / "arena.sqlite"
DATASET_DIR: Path = DATA_DIR / "datasets"
MODELS_DIR: Path = DATA_DIR / "models"

# limits of one experiment (a public deployment trains on its own CPU)
MAX_EPOCHS = _int("MAX_EPOCHS", 5000)
MAX_ARCHITECTURES = _int("MAX_ARCHITECTURES", 8)
MAX_COLLOCATION = _int("MAX_COLLOCATION", 20000)
MAX_RUNNING = _int("MAX_RUNNING", 2)            # experiments training at once in this server process
FREE_EXPERIMENTS = 100   # default free experiments per caller and month; PINN_ARENA_FREE_EXPERIMENTS (read by appkit)

for _d in (DATASET_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
