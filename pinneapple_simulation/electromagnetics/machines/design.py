"""Rotor design of the V-magnet IPM machine with the design-study core of :mod:`pinneapple_design.design_optimizer`:
the FEA as a reference evaluator, a Gaussian-process surrogate with active learning, NSGA-II, re-verification of
every Pareto design by FEA, and the FEA calls this saves against a GA that calls the FEA directly.

* :func:`rotor_evaluator` (an ``IPMRotorEvaluator``) -- design vector (magnet width ``W4``, thickness ``H3``, V depth ``H0``, opening ``W0``,
  bridge ``H1`` of :class:`~.geometry.VRotor`) to mean torque and torque ripple at a fixed current (FEA) and magnet
  mass (exact). Designs whose polygons cross or leave the pole (:func:`~.geometry.geometry_errors`) are returned as
  invalid, without a solve.
* :func:`rotor_design_problem` -- maximise torque, minimise magnet mass, ripple below a limit.
* :func:`hybrid_vs_fea_only` -- surrogate-assisted NSGA-II (LHS start, rounds of GP fit, NSGA-II on the GP, FEA on
  points along its front) against NSGA-II on the FEA itself; the hypervolume of the FEA-verified front of each,
  and how many FEA calls the FEA-only GA needs to reach the hybrid's hypervolume.
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from .geometry import ROTOR_VARIABLES, MachineGeometry, geometry_errors, prius_2004
from .pmsm import IPMMachine

__all__ = ["rotor_evaluator", "rotor_design_problem", "hybrid_vs_fea_only", "ROTOR_VARIABLES"]


def rotor_evaluator(**kwargs):
    """An :class:`IPMRotorEvaluator` (keyword arguments of its constructor). The class is built on first use: the
    design core it subclasses imports torch."""
    return _evaluator_class()(**kwargs)


def _evaluator_class():
    from pinneapple_design.design_optimizer.study import Evaluator

    class IPMRotorEvaluator(Evaluator):
        """FEA of the machine with the rotor variables of each design row. Outputs ``torque`` (mean over
        ``n_positions`` angles in 60 electrical degrees, N m), ``ripple`` (peak to peak over mean), ``magnet_mass``
        (kg), ``_valid``; the number of FEA solves is counted in :attr:`n_fea`."""

        fidelity = "reference"

        def __init__(self, base: MachineGeometry | None = None, names=tuple(v[0] for v in ROTOR_VARIABLES),
                     I_peak: float = 250.0, gamma_deg: float = 50.0, n_gap: int = 288, n_positions: int = 6,
                     magnet_density: float = 7500.0, name: str = "ipm-fea"):
            self.base = base or prius_2004()
            self.names = list(names)
            self.I_peak, self.gamma = I_peak, math.radians(gamma_deg)
            self.n_gap, self.n_positions = n_gap, n_positions
            self.rho_m = magnet_density
            self.outputs = ["torque", "ripple", "magnet_mass", "_valid"]
            self.name = name
            self.description = (f"2-D nonlinear FEA, {n_gap} gap nodes, {n_positions} rotor angles, "
                                f"{I_peak:g} A at {gamma_deg:g} deg")
            self.n_fea = 0
            self.seconds = 0.0

        def geometry(self, x) -> MachineGeometry:
            rotor = dataclasses.replace(self.base.rotor, **{n: float(v) for n, v in zip(self.names, x, strict=True)})
            return dataclasses.replace(self.base, rotor=rotor)

        def evaluate(self, X):
            out = {k: np.full(len(X), np.nan) for k in ("torque", "ripple", "magnet_mass")}
            out["_valid"] = np.zeros(len(X), bool)
            for i, x in enumerate(X):
                g = self.geometry(x)
                r = g.rotor
                out["magnet_mass"][i] = 2 * r.poles * r.W4 * r.H3 * g.length * self.rho_m
                if geometry_errors(r.sector(1e-3)):
                    continue
                t0 = time.perf_counter()
                try:
                    m = IPMMachine(g, n_gap=self.n_gap)
                    op = m.operating_point(I_peak=self.I_peak, gamma=self.gamma, n_positions=self.n_positions)
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    continue
                finally:
                    self.n_fea += 1
                    self.seconds += time.perf_counter() - t0
                out["torque"][i], out["ripple"][i] = op["torque_mean"], op["ripple"]
                out["_valid"][i] = True
            return out

    return IPMRotorEvaluator


def _row(problem, ev, i) -> dict:
    row = dict(zip(problem.space.names, map(float, ev.X[i]), strict=True))
    row.update({k: float(ev.Y[k][i]) for k in ("torque", "magnet_mass", "ripple")})
    return row


def rotor_design_problem(evaluator=None, ripple_max: float = 0.30):
    """Maximise ``torque``, minimise ``magnet_mass``, subject to ``ripple <= ripple_max``."""
    from pinneapple_design.design_optimizer.study import (
        Constraint,
        DesignProblem,
        DesignSpace,
        Objective,
        Variable,
    )

    ev = evaluator or rotor_evaluator()
    space = DesignSpace([Variable(n, lo, hi, unit="m") for n, lo, hi in ROTOR_VARIABLES if n in ev.names])
    return DesignProblem("V-IPM rotor (Prius 2004 stator)", space,
                         [Objective("torque", "max", "N m"), Objective("magnet_mass", "min", "kg")],
                         [Constraint("ripple", upper=ripple_max, label="torque ripple (pk-pk / mean)")], ev,
                         description=ev.description)


def hybrid_vs_fea_only(problem, *, n_initial: int = 24, rounds: int = 3, n_verify: int = 8, ga_budget: int = 400,
                       surrogate_budget: int = 1500, seed: int = 0, verifier=None, progress=None) -> dict:
    """Run both studies on ``problem`` (its evaluator is the FEA) and compare them at equal quality.

    Quality is the hypervolume of the feasible FEA-evaluated designs, objectives normalised by the ideal and nadir
    points of all FEA data of both runs. Returns the hybrid's FEA calls, the FEA-only GA's hypervolume history, the
    number of FEA calls at which it first reaches the hybrid's hypervolume (None if not within ``ga_budget``), and
    with ``verifier`` (a finer FEA evaluator) the re-evaluation of every design of the hybrid front."""
    from pinneapple_design.design_optimizer.methods import normalise_front, run
    from pinneapple_design.design_optimizer.study import hypervolume
    from pinneapple_design.design_optimizer.verify import surrogate_assisted, verify_candidates

    fea = problem.evaluator
    n0 = fea.n_fea
    t0 = time.perf_counter()
    hyb = surrogate_assisted(problem, fea, n_initial=n_initial, rounds=rounds, n_verify=n_verify,
                             surrogate_budget=surrogate_budget, seed=seed, kappa=2.0,
                             validity=lambda X: np.array([not geometry_errors(fea.geometry(x).rotor.sector(1e-3))
                                                          for x in X]))
    hyb_calls, hyb_s = fea.n_fea - n0, time.perf_counter() - t0
    if progress:
        progress(f"hybrid done: {hyb_calls} FEA calls")
    n0 = fea.n_fea
    t0 = time.perf_counter()
    ga = run(problem, "nsga2", budget=ga_budget, seed=seed)
    ga_calls, ga_s = fea.n_fea - n0, time.perf_counter() - t0
    if progress:
        progress(f"FEA-only GA done: {ga_calls} FEA calls")

    data_h, data_g = hyb.reference_evals, ga.evaluations
    ok_h = data_h.feasible & ~data_h.relaxed
    ok_g = data_g.feasible & ~data_g.relaxed
    F_all = np.vstack([data_h.F[ok_h], data_g.F[ok_g]])
    ideal, nadir = F_all.min(0), F_all.max(0)
    ref = np.full(F_all.shape[1], 1.1)
    hv_h = hypervolume(normalise_front(data_h.F[ok_h], ideal, nadir), ref)
    curve = []
    Fg = normalise_front(data_g.F, ideal, nadir)
    reach = None
    for k in range(1, len(data_g) + 1):
        hv = hypervolume(Fg[:k][ok_g[:k]], ref) if ok_g[:k].any() else 0.0
        curve.append(hv)
        if reach is None and hv >= hv_h:
            reach = k
    # FEA calls per evaluated design differ (invalid geometries cost no solve): count calls up to ``reach``
    out = {"hybrid": {"fea_calls": hyb_calls, "seconds": hyb_s, "hypervolume": hv_h,
                      "front": [_row(problem, data_h, i) for i in data_h.pareto()],
                      "rounds": hyb.rounds},
           "fea_only": {"fea_calls": ga_calls, "seconds": ga_s, "hypervolume": curve[-1] if curve else 0.0,
                        "hypervolume_curve": curve},
           "ideal": ideal.tolist(), "nadir": nadir.tolist()}
    if reach is not None:
        valid_g = np.asarray(data_g.Y["_valid"], bool)
        out["fea_only"]["calls_to_reach_hybrid"] = int(valid_g[:reach].sum())
        out["fea_call_saving"] = 1.0 - hyb_calls / max(int(valid_g[:reach].sum()), 1)
    else:
        out["fea_only"]["calls_to_reach_hybrid"] = None
        out["fea_call_saving"] = None
    if verifier is not None:
        idx = list(data_h.pareto())
        rep = verify_candidates(problem, data_h, idx, verifier, problem.with_evaluator(verifier))
        out["hybrid"]["finer_fea"] = {"evaluator": verifier.description, "rows": rep.to_dict()}
    return out
