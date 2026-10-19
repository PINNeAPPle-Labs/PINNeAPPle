"""Metamorphic relations the physics of each zoo model implies (#698), checked without any reference solution by
:func:`check_zoo_relations` (``pinneapple_analysis.validation.metamorphic``).

Each model has groups of relations; each group samples its own box of inputs (a nominal design with some inputs
varied), so the relation is tested inside the model's validity range.

* ``airfoil_cst_rans_mlp``: mirroring the camber line and the angle of attack (``M -> -M``, ``alpha -> -alpha``) flips
  the sign of lift and moment and keeps the drag (geometric symmetry of the incompressible flow); lift grows with the
  angle of attack in the attached range (up to 8 degrees, below the stall of the training set).
* ``heatsink_forced_mlp``: more air speed gives no higher thermal resistance and no lower pressure drop; a better
  conductor gives no higher resistance; above 2 m/s the resistance does not depend on the dissipated power (to 2 %).
  Below 2 m/s it does, in the physics engine too (+6.5 % when the power doubles at 0.6 m/s), so it is not tested there.
"""
from __future__ import annotations

from typing import Any

from pinneapple_analysis.validation.metamorphic import (
    MetamorphicReport,
    box_sampler,
    check_relations,
    relations_from_spec,
)

__all__ = ["ZOO_RELATIONS", "check_zoo_relations"]

_CST_SYM = {"T0": [0.10, 0.24], "T1": [0.08, 0.22], "T2": [0.08, 0.22], "M0": [-0.03, 0.03], "M1": [-0.02, 0.02],
            "M2": [-0.04, 0.04], "alpha_deg": [-2.0, 2.0]}          # the box that stays in range when mirrored
_CST = {"T0": [0.10, 0.24], "T1": [0.08, 0.22], "T2": [0.08, 0.22], "M0": [-0.03, 0.08], "M1": [-0.02, 0.14],
        "M2": [-0.04, 0.12]}
_MIRROR = {"M0": "-M0", "M1": "-M1", "M2": "-M2", "alpha_deg": "-alpha_deg"}
_SINK = {"base_width_mm": 80.0, "base_depth_mm": 80.0, "base_thickness_mm": 6.0, "n_fins": 20.0,
         "fin_thickness_mm": 1.5, "fin_height_mm": 40.0, "k_w_mk": 200.0, "source_width_mm": 40.0,
         "source_depth_mm": 40.0, "t_ambient_c": 25.0, "air_velocity_m_s": 2.0, "power_w": 100.0}
_SINK_VARY = {"air_velocity_m_s": [0.5, 6.0], "k_w_mk": [150.0, 400.0], "power_w": [10.0, 400.0],
              "fin_height_mm": [10.0, 70.0]}
_SINK_FAST = {**_SINK_VARY, "air_velocity_m_s": [2.0, 6.0], "power_w": [10.0, 200.0]}   # doubled power stays in range
_SINK_FIXED = {k: v for k, v in _SINK.items() if k not in _SINK_VARY}

ZOO_RELATIONS: dict[str, list[dict[str, Any]]] = {
    "airfoil_cst_rans_mlp": [
        {"box": _CST_SYM, "relations": [
            # each side of the relation is a prediction with its own error: the tolerance is twice the accepted
            # mean absolute error of the card (Cl 0.01, Cm 0.002; Cd relative)
            {"kind": "symmetric", "name": "camber-mirror-lift", "inputs": _MIRROR, "outputs": {"Cl": "-Cl"},
             "scale": 0.01, "tol": 2.0},
            {"kind": "symmetric", "name": "camber-mirror-moment", "inputs": _MIRROR, "outputs": {"Cm": "-Cm"},
             "scale": 0.002, "tol": 2.0},
            {"kind": "symmetric", "name": "camber-mirror-drag", "inputs": _MIRROR, "outputs": {"Cd": "Cd"},
             "tol": 0.04}]},
        {"box": {**_CST, "alpha_deg": [-2.0, 8.0]}, "relations": [
            {"kind": "monotonic", "name": "lift-grows-with-alpha", "input": "alpha_deg", "output": "Cl",
             "direction": "increasing", "step": 0.05, "tol": 1e-3}]},
    ],
    "heatsink_forced_mlp": [
        {"box": _SINK_VARY, "fixed": _SINK_FIXED, "relations": [
            {"kind": "monotonic", "name": "resistance-falls-with-air-speed", "input": "air_velocity_m_s",
             "output": "r_base_k_w", "direction": "decreasing", "step": 0.05, "tol": 1e-3},
            {"kind": "monotonic", "name": "pressure-drop-grows-with-air-speed", "input": "air_velocity_m_s",
             "output": "pressure_drop_pa", "direction": "increasing", "step": 0.05, "tol": 1e-3},
            {"kind": "monotonic", "name": "resistance-falls-with-conductivity", "input": "k_w_mk",
             "output": "r_base_k_w", "direction": "decreasing", "step": 0.05, "tol": 1e-3}]},
        # Not a law below about 2 m/s: the physics engine's own resistance rises 6.5 % when the power doubles at
        # 0.6 m/s (air properties and outlet temperature change), 0.3-1 % at 2.3 m/s. Tested where it holds, as a
        # relative change of each sample's own resistance.
        {"box": _SINK_FAST, "fixed": _SINK_FIXED, "relations": [
            {"kind": "invariant", "name": "resistance-independent-of-power", "inputs": {"power_w": "2 * power_w"},
             "output": "r_base_k_w", "tol": 0.02, "per_sample": True}]},
    ],
}


def check_zoo_relations(model, name: str | None = None, *, n: int = 64, seed: int = 0, search: str = "random",
                        relations: list[dict[str, Any]] | None = None) -> MetamorphicReport:
    """Run the relations of a zoo model (``model``: a loaded zoo model or anything with the same ``predict``)."""
    import warnings

    name = name or getattr(model, "name", None)
    groups = relations if relations is not None else ZOO_RELATIONS.get(name)
    if groups is None:
        raise KeyError(f"no metamorphic relations declared for zoo model {name!r}")

    def f(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                     # the boxes are inside the validity range
            return model.predict(x)

    results = []
    for gi, g in enumerate(groups):
        sampler = box_sampler(g["box"], g.get("fixed"))
        rep = check_relations(f, relations_from_spec(g["relations"]), sampler, n=n, seed=seed + 101 * gi,
                              search=search)
        results += rep.results
    return MetamorphicReport(results)
