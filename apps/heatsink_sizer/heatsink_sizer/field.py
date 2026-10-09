"""HeatSink Sizer field. Moved into the library as ``pinneapple_design.thermal.heatsink.field``; this module is that one (same object), so
``from heatsink_sizer.field import ...`` keeps working."""
import sys

import pinneapple_design.thermal.heatsink.field as _m

sys.modules[__name__] = _m
