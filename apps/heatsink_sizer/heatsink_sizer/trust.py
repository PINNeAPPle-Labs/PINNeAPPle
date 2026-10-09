"""HeatSink Sizer trust. Moved into the library as ``pinneapple_design.thermal.heatsink.trust``; this module is that one (same object), so
``from heatsink_sizer.trust import ...`` keeps working."""
import sys

import pinneapple_design.thermal.heatsink.trust as _m

sys.modules[__name__] = _m
