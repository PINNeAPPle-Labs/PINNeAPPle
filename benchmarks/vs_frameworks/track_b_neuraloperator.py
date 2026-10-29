# Track B (operator learning, Darcy 64 x 64), neuraloperator FNO (neuraloperator 2.0.0).
# Model as in neuraloperator/neuraloperator 2.0.0 examples/models/plot_FNO_darcy.py:
# neuralop.models.FNO(n_modes=..., in_channels=1, out_channels=1, hidden_channels=32, projection_channel_ratio=2),
# n_layers=4, everything else the library default: positional_embedding="grid",
# lifting_channel_ratio=2 (channel-MLP lifting), channel MLP in each block with soft-gating skip, linear FNO skip,
# no domain padding, no normalisation. The example's Trainer (AdamW, CosineAnnealingLR, H1 / L2 losses) is replaced by
# the protocol's shared loop so the three FNOs are trained identically.
# Modes: neuraloperator counts n_modes as the number of retained frequencies per dimension, positive and negative
# together, with the last (rfft) dimension keeping n_modes // 2 + 1. n_modes=(24, 22) keeps exactly the 24 x 12 block
# that modes=12 keeps in Li et al. (2021), PINNeAPPle's FNO2d and PhysicsNeMo's FNO (weights (32, 32, 24, 12) per
# layer); n_modes=(12, 12) would keep a 12 x 7 block (a quarter of the spectral weights).
"""neuraloperator runner of track B."""
from __future__ import annotations

from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
from pinneapple_tools.benchmark_suite.crossframework.darcy import run_track_b

N_MODES = (2 * proto.B["modes"], 2 * (proto.B["modes"] - 1))
MODEL = {"class": "neuralop.models.FNO", "n_modes": list(N_MODES), "hidden_channels": proto.B["width"],
         "n_layers": proto.B["layers"], "projection_channel_ratio": 2, "positional_embedding": "grid"}


def build():
    from neuralop.models import FNO
    return FNO(n_modes=N_MODES, in_channels=1, out_channels=1,
               hidden_channels=proto.B["width"], n_layers=proto.B["layers"], projection_channel_ratio=2)


if __name__ == "__main__":
    run_track_b(None, framework="neuraloperator", source="neuraloperator 2.0.0 examples/models/plot_FNO_darcy.py",
                build=build, model_config=MODEL)
