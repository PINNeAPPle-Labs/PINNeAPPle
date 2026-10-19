"""Flight dynamics: a 6-DOF rigid aircraft driven by vortex-lattice aerodynamics, thrust, gravity and landing-gear
contact, with trim, stability derivatives, a pilot / autopilot, take-off and turbulence-encounter simulations, and
3-D scenes of the flights.

    from pinneapple_simulation import flight
    ac = flight.catalog.build("a320")                       # also light_ga, regional_turboprop, motor_glider
    tr = flight.trim(ac, V=128.0, altitude=3000.0)
    hist = flight.simulate_takeoff(ac)                      # ground roll, rotation at V_R, climb over 35 ft
    m = flight.takeoff_metrics(ac, hist); cf = flight.raymer_takeoff(ac)
    hist, m, *_ = flight.turbulence_encounter(ac, 128.0, 3000.0, flight.turbulence_spec(3000.0, "severe"))
    scene = flight.visuals.flight_scene(ac, hist)           # pp.viz.web_viewer(scene, "viewer/") / Blender

Modules: :mod:`.turbulence` (MIL-F-8785C / MIL-HDBK-1797 Dryden and von Karman, discrete gusts, frozen 3-D field),
:mod:`.rigid_body`, :mod:`.aero`, :mod:`.propulsion`, :mod:`.gear`, :mod:`.mass`, :mod:`.aircraft`, :mod:`.trim`,
:mod:`.control`, :mod:`.simulate`, :mod:`.takeoff`, :mod:`.encounter`, :mod:`.catalog`, :mod:`.visuals`,
:mod:`.figures`, :mod:`.verification`. Guide: ``docs/guides/flight_dynamics.md``.
"""
from . import catalog
from .aircraft import Aircraft, Controls
from .encounter import (
    discrete_gust,
    encounter_metrics,
    gust_sensitivity,
    pratt_check,
    turbulence_encounter,
)
from .simulate import FlightHistory, simulate
from .takeoff import (
    default_schedule,
    raymer_takeoff,
    raymer_variants,
    simulate_takeoff,
    takeoff_metrics,
)
from .trim import ground_equilibrium, modes, stability_derivatives, trim
from .turbulence import (
    DiscreteGust,
    DrydenFilter,
    FrozenField3D,
    FrozenLine,
    ShearWind,
    pratt_gust_load,
    psd,
    turbulence_spec,
)

__all__ = ["catalog", "Aircraft", "Controls", "simulate", "FlightHistory", "trim", "ground_equilibrium",
           "stability_derivatives", "modes", "simulate_takeoff", "takeoff_metrics", "raymer_takeoff", "raymer_variants",
           "default_schedule", "turbulence_encounter", "encounter_metrics", "gust_sensitivity", "discrete_gust",
           "pratt_check", "turbulence_spec", "psd", "FrozenLine", "FrozenField3D", "DrydenFilter", "DiscreteGust",
           "ShearWind", "pratt_gust_load"]


def __getattr__(name):
    if name in ("visuals", "figures", "verification"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
