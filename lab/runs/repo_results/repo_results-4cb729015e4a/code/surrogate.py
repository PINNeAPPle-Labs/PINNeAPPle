"""HeatSink Sizer surrogate. Moved into the library as ``pinneapple_design.thermal.heatsink.surrogate``; this module is that one (same object), so
``from heatsink_sizer.surrogate import ...`` keeps working."""
import sys

import pinneapple_design.thermal.heatsink.surrogate as _m

import os

_m.ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
sys.modules[__name__] = _m
