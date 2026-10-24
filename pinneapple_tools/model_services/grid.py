"""Grid-operator services: a trained field model (e.g. a PhysicsNeMo FNO) and PINNeAPPle's reference solver behind
the same ``/v1/infer`` route, each answer checked with the residual-based Darcy checks (no reference needed).

``grid_operator_service(predictor, name, envelope=...)``
    ``inputs = {"a": n x n coefficient}`` -> ``{"u": n x n}``; trust = ``darcy_checks`` physics (certified residual
    bounds, boundary condition, flux balance, maximum principle) + the coefficient against the training envelope.
``darcy_solver_service(n)``
    the finite-volume solver (``operator_benchmarks.darcy_2d``) as a service with the same checks: the fallback a
    client calls when a surrogate's answer is not APPROVED.
"""
from __future__ import annotations

import numpy as np

from .registry import ModelService

__all__ = ["grid_operator_service", "darcy_solver_service", "darcy_trust"]


def darcy_trust(a, u, f: float = 1.0, envelope=None) -> dict:
    from pinneapple_physics import darcy_checks as dc
    rep, checks = dc.darcy_trust_report(np.asarray(a, float), np.asarray(u, float), f, envelope=envelope,
                                        kinds=("physics", "applicability"))
    return {"decision": rep.decisao, "checks": [c.to_dict() for c in checks]}


def grid_operator_service(predictor, name: str, *, envelope=None, f: float = 1.0, card: dict | None = None,
                          task: str = "Darcy-type field operator") -> ModelService:
    def predict(inputs):
        return {"u": predictor.predict(np.asarray(inputs["a"], float))}

    def trust(inputs, out):
        return darcy_trust(inputs["a"], out["u"], f, envelope)

    return ModelService(name, "infer", task, predict, trust, dict(card or {}))


def darcy_solver_service(name: str = "pinneapple-darcy-fv", f: float = 1.0) -> ModelService:
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import darcy_2d

    def predict(inputs):
        return {"u": darcy_2d(np.asarray(inputs["a"], float), f)}

    def trust(inputs, out):
        return darcy_trust(inputs["a"], out["u"], f)

    return ModelService(name, "infer", "finite-volume reference solver (-div(a grad u) = f, u = 0 on the boundary)",
                        predict, trust, {"description": "two-point flux finite volumes, harmonic-mean faces "
                                                        "(pinneapple_simulation.numerical_solvers.operator_benchmarks)"})
