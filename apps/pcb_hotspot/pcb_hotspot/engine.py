"""PCB Hotspot engine. Moved into the library as ``pinneapple_physics.pcb.engine``; this module is that one (same object), so
``from pcb_hotspot.engine import ...`` keeps working."""
import sys

import pinneapple_physics.pcb.engine as _m

sys.modules[__name__] = _m
