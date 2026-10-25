# Track B (operator learning, Darcy 64 x 64), PINNeAPPle FNO.
# Model: pinneapple_neural.architectures.neural_operators.fno.FNO2d(1, 1, width=32, modes1=12, modes2=12, layers=4,
# use_grid=True), the class documented in docs/guides/choosing_a_method.md ("Neural operators") and used by
# pp.benchmark's "fno" method (pinneapple_analysis/benchmark/builtin/fields.py::_operator_model).
# Structure: 1x1 conv lifting (3 -> 32 channels: a + 2 grid channels), 4 x (spectral conv + 1x1 conv, GELU after each,
# including the last), 1x1 conv projection 32 -> 1 (no projection MLP), no padding.
# Training: the shared loop of pinneapple_tools.benchmark_suite.crossframework.darcy (protocol track B).
"""PINNeAPPle runner of track B."""
from __future__ import annotations

from pinneapple_tools.benchmark_suite.crossframework import protocol as proto
from pinneapple_tools.benchmark_suite.crossframework.darcy import run_track_b

MODEL = {"class": "pinneapple_neural.architectures.neural_operators.fno.FNO2d", "width": proto.B["width"],
         "modes": proto.B["modes"], "layers": proto.B["layers"], "use_grid": True}


def build():
    from pinneapple_neural.architectures.neural_operators.fno import FNO2d
    return FNO2d(1, 1, width=proto.B["width"], modes1=proto.B["modes"], modes2=proto.B["modes"],
                 layers=proto.B["layers"], use_grid=True)


if __name__ == "__main__":
    run_track_b(None, framework="pinneapple", source="PINNeAPPle FNO2d at the commit under test", build=build,
                model_config=MODEL)
