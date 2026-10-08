"""Inverse Heat Lab engine. Moved into the library as ``pinneapple_analysis.inverse_problems.heat_transfer``; this module is that one (same object), so
``from inverse_heat.engine import ...`` keeps working."""
import sys

import pinneapple_analysis.inverse_problems.heat_transfer as _m

import os

_m.EXAMPLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "use_cases",
                                          "fin_convection_inverse"))
sys.modules[__name__] = _m
