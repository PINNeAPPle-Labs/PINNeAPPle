"""Train the natural-convection heat-sink surrogate with monotonicity constraints (zoo model heatsink_natural_mlp).

    python scripts/zoo/train_heatsink_natural.py OUT_DIR        # ~15 min on one CPU core

The recipe is :func:`pinneapple_design.thermal.heatsink.monotone.train_monotone`; this writes its bundle to
OUT_DIR/heatsink_natural.pt, which ``scripts/zoo/build_assets.py`` turns into the zoo model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinneapple_design.thermal.heatsink.monotone import train_monotone  # noqa: E402

if __name__ == "__main__":
    out = Path(sys.argv[1])
    bundle = train_monotone("natural", log=lambda s: print(s, flush=True),
                            artifacts=str(ROOT / "apps" / "heatsink_sizer" / "artifacts"))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out / "heatsink_natural.pt")
