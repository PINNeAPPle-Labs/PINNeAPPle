"""PCB Hotspot solver. Moved into the library as ``pinneapple_physics.pcb.solver``; this module is that one (same object), so
``from pcb_hotspot.solver import ...`` keeps working."""
import sys

import pinneapple_physics.pcb.solver as _m

sys.modules[__name__] = _m
