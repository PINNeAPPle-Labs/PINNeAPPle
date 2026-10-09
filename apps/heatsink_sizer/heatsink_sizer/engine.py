"""HeatSink Sizer engine. Moved into the library as ``pinneapple_design.thermal.heatsink.engine``; this module is that one (same object), so
``from heatsink_sizer.engine import ...`` keeps working."""
import sys

import pinneapple_design.thermal.heatsink.engine as _m

sys.modules[__name__] = _m
