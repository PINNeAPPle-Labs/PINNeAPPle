# Track B (operator learning, Darcy 64 x 64), PhysicsNeMo FNO (nvidia-physicsnemo 2.2.2).
# Model as in NVIDIA/physicsnemo v2.2.2 examples/cfd/darcy_fno/config.yaml + train_fno_darcy.py:
# physicsnemo.models.fno.FNO(in_channels=1, out_channels=1, decoder_layers=1, decoder_layer_size=32, dimension=2,
# latent_channels=32, num_fno_layers=4, num_fno_modes=12, padding=9), coord_features=True (default: 2 grid channels).
# Structure: lifting by a 1x1 conv network, 4 spectral layers, zero padding by 9 cells, MLP decoder (1 hidden layer of
# 32, SiLU). The example's own loop (scalar normalisers, MSE loss, StaticCaptureTraining with AMP / CUDA graphs on GPU,
# exponential decay) is replaced by the protocol's shared loop so the three FNOs are trained identically.
"""PhysicsNeMo runner of track B."""
from __future__ import annotations

from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
from pinneapple_tools.benchmark_suite.crossframework.darcy import run_track_b

MODEL = {"class": "physicsnemo.models.fno.FNO", "latent_channels": proto.B["width"],
         "num_fno_modes": proto.B["modes"], "num_fno_layers": proto.B["layers"], "decoder_layers": 1,
         "decoder_layer_size": 32, "padding": 9, "coord_features": True}


def build():
    from physicsnemo.models.fno import FNO
    return FNO(in_channels=1, out_channels=1, decoder_layers=1, decoder_layer_size=32, dimension=2,
               latent_channels=proto.B["width"], num_fno_layers=proto.B["layers"], num_fno_modes=proto.B["modes"],
               padding=9, coord_features=True)


if __name__ == "__main__":
    run_track_b(None, framework="physicsnemo", source="NVIDIA/physicsnemo v2.2.2 examples/cfd/darcy_fno",
                build=build, model_config=MODEL)
