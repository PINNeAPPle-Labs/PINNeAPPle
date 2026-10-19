"""HeatSink Sizer sizer. Moved into the library as ``pinneapple_design.thermal.heatsink.sizer``; this module is that one (same object), so
``from heatsink_sizer.sizer import ...`` keeps working."""
import sys

import pinneapple_design.thermal.heatsink.sizer as _m

sys.modules[__name__] = _m
