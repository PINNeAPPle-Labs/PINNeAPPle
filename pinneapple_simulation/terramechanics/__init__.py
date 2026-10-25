"""Terramechanics for planetary rover wheels: soils with provenance, the Wong-Reece wheel model and its extensions,
deformable terrain with ruts, a single-wheel testbed with published data, and PINN contact surrogates.

    from pinneapple_simulation.terramechanics import WongReeceModel, Wheel, Grousers, get_soil
    wheel = Wheel(radius=0.25, width=0.20, grousers=Grousers(count=24, height=0.025))
    model = WongReeceModel(get_soil("GRC-1"), wheel, g=1.62)
    model.steady_state(load=180.0, slip=[0.1, 0.2, 0.4])           # sinkage, drawbar pull, torque per slip

Modules: :mod:`.soils` (database), :mod:`.wheel`, :mod:`.wong_reece` (stresses and forces: Bekker pressure-sinkage,
Janosi-Hanamoto shear, slip-sinkage, grousers, multi-pass, bulldozing, lateral force), :mod:`.terrain`
(deformable heightmap with ruts and grouser imprints), :mod:`.testbed` (single-wheel validation),
:mod:`.surrogate` (PINN training for the wheel-soil contact), :mod:`.mobility` (screening with the
``bekker_wong_surrogate_2d`` PINN: drawbar pull against slip, constraints R1-R4 checked one by one). See ``docs/core_concepts/lunar_rover_terramechanics.md``.
"""
from .soils import SOILS, Soil, get_soil, list_soils, register_soil
from .wheel import Grousers, Wheel
from .wong_reece import (
    ContactResult,
    WongReeceModel,
    bearing_capacity_factors,
    bulldozing_resistance,
    physics_mismatch,
    slip_angle,
    slip_ratio,
)

__all__ = ["Soil", "SOILS", "get_soil", "list_soils", "register_soil", "Wheel", "Grousers", "WongReeceModel",
           "ContactResult", "bulldozing_resistance", "bearing_capacity_factors", "slip_ratio", "slip_angle",
           "physics_mismatch"]


def __getattr__(name):
    import importlib
    if name in ("testbed", "surrogate", "terrain", "mobility"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
