"""Underground gas storage, simulated by the library's own solver: the case that carries the chain through the solver.

The pipe-erosion case (:mod:`.pipe_erosion`) stops where the library has no solver. This one does not: the reference
simulator is :mod:`pinneapple_simulation.ugs` — single-phase real gas, two-point flux, backward Euler, Newton with an
analytic Jacobian — and it runs a 21 x 21 reservoir season in a fraction of a second, so the factory can drive the
whole path here: geometry -> DOE -> solver -> dataset -> quality control.

The case
--------
A depleted gas reservoir turned into a storage site. The design varies the two numbers a geologist actually
delivers — permeability and porosity — and the run is one production season at a fixed bottom-hole pressure floor,
which is what a deliverability test is. The outputs are the two quantities a storage operator signs with:

* ``peak_deliverability`` [m3 standard per day]: the highest withdrawal the formation allows at the floor pressure;
* ``pressure_drop`` [MPa]: the pore-volume-weighted mean pressure lost over the season.

There is no CAD in this case, and the first arrow says so in its own terms: a reservoir is not a solid, its geometry
is the grid. :class:`UGSGridGeometry` builds that grid and measures what a solver reads from it — cells, bulk and pore
volume, the transmissibility range and Peaceman's well index — instead of borrowing a template's solid checks.

What the run reports to the quality control
-------------------------------------------
The simulator already computes exactly what :mod:`..qc` reads, so nothing is inferred: ``converged`` and the reason
when it is not, ``mass_balance_error`` (|GIP_end - GIP_0 - net injected| / GIP_0), the number of cells and a grid check
(positive, finite transmissibilities — a zero one disconnects the reservoir). The grid-convergence index is not
available from a single run, and the check says ``not_run``.

    from pinneapple_autolab.factory.domains import ugs
    run = ugs.run("runs/storage")             # the whole path, about a second
    run.status, run.decision, run.dataset["n_rows"]      # ('done', 'APPROVED', 7)
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pinneapple_analysis.design_study import (
    Base,
    Budget,
    DesignStudy,
    Evaluator,
    Objective,
    Quantity,
    Variable,
)
from pinneapple_autolab.factory.factory import FactoryRun, run_factory
from pinneapple_autolab.factory.interfaces import GeometryArtifact, MaximinDoe
from pinneapple_autolab.simulator import SimOutcome, Simulator

__all__ = ["MD", "REF", "CONDITIONS", "SHAPE", "SPACING", "UGSGridGeometry", "UGSStorage", "spec", "geometry",
           "solver", "doe", "run"]

#: One millidarcy in SI, the unit permeability is reported in: ``1 mD = 9.869233e-16 m2``.
MD = 9.869233e-16

REF = "pinneapple_simulation.ugs:cartesian_grid"

#: The operating conditions of the case, fixed by the study.
CONDITIONS = {"temperature_K": 320.0, "initial_pressure_Pa": 1.2e7, "bhp_floor_Pa": 6.0e6, "season_days": 30.0}

#: The reservoir grid: 21 x 21 cells of 40 x 40 m, one well in the corner cell.
SHAPE, SPACING = (21, 21), (40.0, 40.0)


def _grid(params: Mapping[str, Any], shape=SHAPE, spacing=SPACING):
    from pinneapple_simulation.ugs import CartesianGrid
    k = float(params["permeability_md"]) * MD
    phi = float(params["porosity"])
    return CartesianGrid(shape, spacing, permeability=k, porosity=phi)


class UGSGridGeometry:
    """The geometry arrow of a reservoir case: the grid, built and measured, not a solid.

    ``prepare`` returns the numbers a solver reads from a grid and nothing else: the cell count, the bulk and pore
    volumes, the transmissibility range (``T = A k / d``), Peaceman's well index for the completion, and the two
    checks that say whether the grid may be simulated at all — porosity strictly inside (0, 1), and every
    transmissibility positive and finite, because a zero one silently disconnects the reservoir.
    """

    name = "ugs grid"

    def __init__(self, shape=SHAPE, spacing=SPACING):
        self.shape, self.spacing = tuple(shape), tuple(spacing)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "title": "structured cartesian reservoir grid",
                "physics": ["single_phase_gas_flow"], "quantities": ["peak deliverability", "pressure drop"],
                "parameters": {"permeability_md": {"unit": "mD", "role": "flow capacity"},
                               "porosity": {"unit": "1", "role": "storage capacity"}},
                "ports": {}, "build": "pinneapple_simulation.ugs.CartesianGrid", "references": [],
                "shape": list(self.shape), "spacing_m": list(self.spacing)}

    def prepare(self, params: Mapping[str, float], workdir: str = "") -> GeometryArtifact:
        from pinneapple_analysis.trust.trust_report import CONTRADICTS, SUPPORTS, Check
        from pinneapple_simulation.ugs import Well

        art = GeometryArtifact(ref="ugs_grid", params={str(k): float(v) for k, v in params.items()}, status="built")
        phi = float(params["porosity"])
        ok_phi = 0.0 < phi < 1.0
        art.checks.append(Check("porosity within (0, 1)", SUPPORTS if ok_phi else CONTRADICTS, critical=True,
                                value=phi, limit=1.0, source="pinneapple_simulation.ugs.grid",
                                reason="" if ok_phi else f"a porosity of {phi:g} is not a porous medium"))
        grid = _grid(params, self.shape, self.spacing)
        t = np.asarray(grid.conn_T, float)
        ok_t = bool(t.size and np.all(np.isfinite(t)) and np.all(t > 0))
        art.checks.append(Check("transmissibility positive and finite", SUPPORTS if ok_t else CONTRADICTS,
                                critical=True, value=float(np.min(t)) if t.size else None, limit=0.0,
                                source="pinneapple_simulation.ugs.grid",
                                reason="" if ok_t else "a zero or non-finite transmissibility disconnects the grid"))
        art.metrics = {"cells": int(grid.n_cells), "shape": list(self.shape), "spacing_m": list(self.spacing),
                       "bulk_volume_m3": float(np.sum(grid.volumes)),
                       "pore_volume_m3": float(np.sum(grid.volumes * grid.phi0)),
                       "transmissibility_min": float(np.min(t)) if t.size else None,
                       "transmissibility_max": float(np.max(t)) if t.size else None,
                       "well_index_m3": float(grid.well_index(0, Well("P1", cell=0).r_w)),
                       "ports": {"well": "P1 in cell 0", "boundary": "closed"}}
        if not (ok_phi and ok_t):
            art.status = "refused"
        return art


class UGSStorage(Simulator):
    """The library's underground-storage simulator as the factory's solver: one production season per design.

    ``check`` refuses what the grid or the schedule cannot hold before anything is integrated (a permeability of
    zero, a porosity outside (0, 1), a bottom-hole floor above the initial pressure). ``run`` builds the grid and the
    well from the design, runs the season with ``dt = 1 d`` and returns the two outputs plus the diagnostics the
    quality control reads: the convergence flag with its reason, the measured mass-balance error, the cell count, a
    grid check and the Newton iterations.
    """

    name, fidelity = "pinneapple_simulation.ugs", "reference"
    outputs = ("peak_deliverability", "pressure_drop")

    def __init__(self, *, dt_days: float = 1.0, shape=SHAPE, spacing=SPACING):
        self.dt_days, self.shape, self.spacing = float(dt_days), tuple(shape), tuple(spacing)

    def check(self, params: Mapping[str, Any]) -> list[str]:
        k = float(params.get("permeability_md", 0.0))
        phi = float(params.get("porosity", 0.0))
        floor = float(params.get("bhp_floor_Pa", CONDITIONS["bhp_floor_Pa"]))
        p0 = float(params.get("initial_pressure_Pa", CONDITIONS["initial_pressure_Pa"]))
        bad = []
        if not (k > 0):
            bad.append(f"a permeability of {k:g} mD cannot flow")
        if not 0.0 < phi < 1.0:
            bad.append(f"a porosity of {phi:g} is not a porous medium")
        if not (floor < p0):
            bad.append(f"a bottom-hole floor of {floor / 1e6:g} MPa is not below the initial pressure "
                       f"{p0 / 1e6:g} MPa")
        return bad

    def run(self, params: Mapping[str, Any], workdir: str) -> SimOutcome:
        from pinneapple_simulation.ugs import (
            Control,
            Gas,
            Period,
            Rock,
            Schedule,
            SimConfig,
            UGSSimulator,
            Well,
        )

        t, p0 = float(params.get("temperature_K", CONDITIONS["temperature_K"])), float(
            params.get("initial_pressure_Pa", CONDITIONS["initial_pressure_Pa"]))
        floor = float(params.get("bhp_floor_Pa", CONDITIONS["bhp_floor_Pa"]))
        days = float(params.get("season_days", CONDITIONS["season_days"]))
        grid = _grid(params, self.shape, self.spacing)
        well = Well("P1", cell=0, bhp_min=floor)
        sim = UGSSimulator(grid, Gas(T=t), Rock(cr=1e-9, p_ref=p0), [well], p0=p0)
        res = sim.run(Schedule((Period(days, {"P1": Control("bhp", floor)}, "season"),)),
                      SimConfig(dt_days=self.dt_days))
        outputs = {"peak_deliverability": float(res.deliverability().max()) if res.converged else float("nan"),
                   "pressure_drop": float((res.mean_pressure[0] - res.mean_pressure[-1]) / 1e6)
                   if res.converged else float("nan")}
        tt = np.asarray(grid.conn_T, float)
        meta = {"converged": bool(res.converged), "reason": res.reason,
                "mass_balance_error": float(res.mass_balance_error),
                "mesh": {"cells": int(grid.n_cells),
                         "ok": bool(tt.size and np.all(np.isfinite(tt)) and np.all(tt > 0))},
                "solver": "pinneapple_simulation.ugs.UGSSimulator, backward Euler, dt = "
                          f"{self.dt_days:g} d",
                "steps": int(res.n_steps), "newton_iterations_max": int(max(res.newton_iterations or [0])),
                "grid_convergence": "not studied: one grid per design"}
        return SimOutcome(outputs, ok=bool(res.converged), message=res.reason, cost_s=0.0, meta=meta)


def spec(n_doe: int = 6, seed: int = 0) -> DesignStudy:
    """The storage case as a study spec: permeability and porosity varied, the two operator outputs with units."""
    return DesignStudy(
        kind="surrogate", title="Underground gas storage: deliverability and pressure drop of a season",
        statement="Vary the permeability and the porosity of a storage reservoir to learn the peak deliverability "
                  "and the pressure lost over one production season at the well's pressure floor.",
        base=Base(kind="system", ref=REF,
                  variables=[Variable("permeability_md", 10.0, 1000.0, unit="mD",
                                      description="permeability of the reservoir"),
                             Variable("porosity", 0.12, 0.28, unit="1",
                                      description="porosity of the reservoir")],
                  baseline={"permeability_md": 100.0, "porosity": 0.2},
                  description="21 x 21 cartesian grid of 40 x 40 m cells, one well in cell 0, closed boundaries"),
        evaluator=Evaluator(tool="ugs.storage_season",
                            inputs={"permeability_md": "permeability_md", "porosity": "porosity"},
                            outputs=[Quantity("peak_deliverability", "m3/d", label="peak deliverability"),
                                     Quantity("pressure_drop", "MPa", label="mean pressure lost")],
                            conditions=dict(CONDITIONS), fidelity="reference",
                            description="pinneapple_simulation.ugs: single-phase real gas, backward Euler, 1 d steps"),
        objectives=[Objective("peak_deliverability", "max", label="peak deliverability"),
                    Objective("pressure_drop", "min", label="mean pressure lost")],
        budget=Budget(max_evaluations=1 + n_doe, n_doe=n_doe, n_verify=0, seed=seed),
        metadata={"case": "UGS season", "solver": "pinneapple_simulation.ugs"})


def geometry(**kw: Any) -> UGSGridGeometry:
    """The geometry arrow: the reservoir grid, built and measured."""
    return UGSGridGeometry(**kw)


def solver(**kw: Any) -> UGSStorage:
    """The solver arrow: the library's own underground-storage simulator."""
    return UGSStorage(**kw)


def doe() -> MaximinDoe:
    """The DOE arrow: the space-filling plan of ``pinneapple_autolab.study.campaign.plan_doe``."""
    return MaximinDoe()


def run(out: str, **kw: Any) -> FactoryRun:
    """Run the whole path: grid -> DOE -> season -> dataset -> quality control. About a second for seven designs."""
    s = spec()
    return run_factory(s, solver(), out=out, geometry=geometry(), doe=doe(), **kw)
