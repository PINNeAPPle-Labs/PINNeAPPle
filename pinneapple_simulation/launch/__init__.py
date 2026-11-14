"""Ascent to orbit: a launch vehicle from a pad to a near-circular orbit, with the standard atmosphere it drags
against, a gravity-turn and velocity-to-be-gained guidance, the velocity bookkeeping that closes on the Tsiolkovsky
budget, and the checks a result needs before anyone flies it.

    from pinneapple_simulation.launch import Site, Target, simulate_ascent, small_launcher
    hist = simulate_ascent(small_launcher(), Site("Tanegashima", 30.4, 130.97), Target(500.0))
    print(hist.summary())                 # perigee, apogee, eccentricity, max Q, the losses
    print(hist.losses)                    # thrust, gravity, drag and steering [m/s], and the closure

Modules: :mod:`.atmosphere` (US Standard Atmosphere 1976), :mod:`.vehicle` (stages, thrust, impulse, drag),
:mod:`.sizing` (the design closure that sizes the stages from the measured losses and solves the pitch schedule), :mod:`.ascent` (the 3-DOF
simulation and the guidance), :mod:`.verification` (the checks), :mod:`.visuals` (the 3-D flight and the
mission-control display). Guide: ``docs/guides/launch_ascent.md``.
"""
from .ascent import (
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH,
    AscentHistory,
    Earth,
    GuidanceConfig,
    Site,
    Target,
    circular_speed,
    eci_to_geodetic,
    launch_azimuth_deg,
    orbit_elements,
    simulate_ascent,
)
from .vehicle import CD_MACH, LaunchVehicle, Stage, cd_of_mach, small_launcher

__all__ = ["MU_EARTH", "OMEGA_EARTH", "R_EARTH", "AscentHistory", "Earth", "GuidanceConfig", "Site", "Target",
           "circular_speed", "eci_to_geodetic", "launch_azimuth_deg", "orbit_elements", "simulate_ascent",
           "CD_MACH", "LaunchVehicle", "Stage", "cd_of_mach", "small_launcher"]
