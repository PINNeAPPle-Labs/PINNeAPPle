"""Plate-fin heat-sink sizing (the engine of the HeatSink Sizer app).

    from pinneapple_design.thermal.heatsink import DesignInput, OperatingInput, evaluate_design, SizingRequest, size
    r = evaluate_design(DesignInput(...), OperatingInput(...))        # closed-form network + 3D finite-volume base
    s = size(SizingRequest(...), load_surrogates())                   # screen thousands, verify the best with physics

Modules: ``engine`` (one design, verified), ``field`` (3D finite-volume base plate), ``sizer`` (requirements in,
designs out), ``surrogate`` (MLP screening model, training), ``trust`` (surrogate trust report).
"""
from .engine import H_BAND, DesignInput, OperatingInput, evaluate_design, model_scope, physics_resistance
from .field import solve_base_field
from .sizer import PROCESSES, SizingRequest, size
from .surrogate import COMMON_RANGES, MODE_RANGES, Surrogate, load_surrogates

__all__ = ["DesignInput", "OperatingInput", "evaluate_design", "model_scope", "physics_resistance", "H_BAND",
           "solve_base_field", "SizingRequest", "size", "PROCESSES", "Surrogate", "load_surrogates",
           "COMMON_RANGES", "MODE_RANGES"]
