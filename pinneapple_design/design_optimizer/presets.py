"""Ready-made design problems built from the library's physics, at several fidelities.

Plate-fin heat sink (:func:`heatsink_problem`)
    Minimise mass and cost of an extruded plate-fin heat sink in ducted forced air without exceeding the source
    temperature, the pressure drop, the fin stress under a shock load, and the extrusion limits (minimum fin gap,
    maximum fin height / gap). Variables: base width, depth and thickness, fin count (integer), fin thickness and
    height. Fidelities of the thermal/hydraulic part:

    * ``"surrogate"``: the zoo model ``heatsink_forced_mlp`` (an MLP trained on 6000 runs of the FVM engine), wrapped
      in torch so the whole problem is differentiable (gradient methods); designs outside its training range are
      invalid;
    * ``"closed_form"``: :func:`pinneapple_physics.closed_form.plate_fin_heatsink.evaluate` (Teertstra forced
      convection, Muzychka-Yovanovich pressure drop, Lee et al. spreading);
    * ``"fvm"``: :func:`pinneapple_design.thermal.heatsink.engine.physics_resistance`, the 3-D finite-volume base
      conduction coupled to the fin correlations (the reference the surrogate was trained on), ~0.3 s per design.

    Mass, cost, stress and the manufacturing quantities are closed-form geometry at every fidelity. The cost model
    is an *assumption* the user sets (:class:`CostModel`), not market data.

CST airfoil (:func:`airfoil_problem`)
    Minimise the section drag coefficient and maximise the spar depth (thickness between 20 and 35 % chord) of a
    2nd-order CST airfoil at Re = 4e6, with a lift constraint, a pitching-moment limit and an aft-thickness
    manufacturing limit. Evaluator: the zoo ensemble ``airfoil_cst_rans_mlp`` (trained on OpenFOAM k-omega SST
    runs), differentiable. No reference solver ships with it: candidates stay "only surrogate" unless verified with
    OpenFOAM (``apps/aero_optimizer/tools/verify.py``).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .study import VALID, Constraint, DesignProblem, DesignSpace, Evaluator, Objective, Variable

__all__ = ["HeatsinkOperating", "HeatsinkLimits", "CostModel", "HeatsinkEvaluator", "heatsink_problem",
           "heatsink_baseline", "AirfoilEvaluator", "airfoil_problem", "airfoil_baseline", "PRESETS",
           "plate_fin_bodies", "airfoil_bodies", "design_file", "HEATSINK_FIDELITIES", "preset_info", "preset_bodies"]

G0 = 9.80665
HEATSINK_FIDELITIES = ("surrogate", "closed_form", "fvm")


@dataclass
class HeatsinkOperating:
    power_w: float = 60.0
    t_ambient_c: float = 35.0
    air_velocity_m_s: float = 2.0
    source_width_mm: float = 30.0
    source_depth_mm: float = 30.0
    tim_k_mm2_w: float = 10.0              # interface resistance R'' (grease ~ 5-20 K mm2/W)
    material: str = "al6063"
    shock_g: float = 50.0                  # lateral design acceleration on the fins (g)


@dataclass
class HeatsinkLimits:
    t_source_max_c: float = 85.0
    pressure_drop_max_pa: float = 30.0
    fin_stress_max_mpa: float = 70.0       # user allowable; e.g. yield of the alloy over a safety factor
    fin_gap_min_mm: float = 1.5            # extrusion (pinneapple_design.thermal.heatsink.sizer.PROCESSES)
    fin_aspect_max: float = 20.0           # fin height / gap, extrusion


@dataclass
class CostModel:
    """Unit cost = material price x mass / material yield + cost per fin + fixed cost (assumed values)."""
    material_per_kg: float = 6.0
    material_yield: float = 0.8
    per_fin: float = 0.10
    fixed: float = 0.5
    currency: str = "EUR"


def _process_limits() -> HeatsinkLimits:
    from pinneapple_design.thermal.heatsink.sizer import PROCESSES
    p = PROCESSES["extruded"]
    return HeatsinkLimits(fin_gap_min_mm=p["gap_min"], fin_aspect_max=p["aspect_max"])


HEATSINK_VARIABLES = [
    Variable("base_width_mm", 40.0, 150.0, unit="mm", description="base width (across the fins)"),
    Variable("base_depth_mm", 40.0, 150.0, unit="mm", description="base depth = flow length"),
    Variable("base_thickness_mm", 2.0, 12.0, unit="mm", description="base plate thickness"),
    Variable("n_fins", 4, 40, kind="integer", unit="-", description="number of fins"),
    Variable("fin_thickness_mm", 1.0, 3.0, unit="mm", description="fin thickness"),
    Variable("fin_height_mm", 10.0, 60.0, unit="mm", description="fin height above the base"),
]
HEATSINK_OUTPUTS = ["mass_g", "cost", "t_source_c", "pressure_drop_pa", "fin_stress_mpa", "fin_gap_mm",
                    "fin_aspect", "r_base_k_w"]


class HeatsinkEvaluator(Evaluator):
    """Geometry (mass, cost, stress, extrusion limits) in closed form plus the thermal/hydraulic model of the chosen
    fidelity. Columns of ``X``: :data:`HEATSINK_VARIABLES`."""

    def __init__(self, fidelity: str = "surrogate", operating: HeatsinkOperating | None = None,
                 cost: CostModel | None = None):
        from pinneapple_physics.closed_form.plate_fin_heatsink import MATERIALS
        if fidelity not in HEATSINK_FIDELITIES:
            raise ValueError(f"fidelity must be one of {HEATSINK_FIDELITIES}")
        self.op = operating or HeatsinkOperating()
        self.costm = cost or CostModel()
        if self.op.material not in MATERIALS:
            raise ValueError(f"unknown material {self.op.material!r}; use one of {sorted(MATERIALS)}")
        self.rho, self.k = MATERIALS[self.op.material]["rho"], MATERIALS[self.op.material]["k"]
        self.kind = fidelity
        self.outputs = list(HEATSINK_OUTPUTS)
        self.fidelity = {"surrogate": "surrogate", "closed_form": "model", "fvm": "reference"}[fidelity]
        self.name = {"surrogate": "zoo:heatsink_forced_mlp (MLP surrogate)",
                     "closed_form": "closed-form resistance network",
                     "fvm": "3-D finite-volume base + fin correlations"}[fidelity]
        self.description = {
            "surrogate": "MLP trained on 6000 FVM-engine runs (pp.zoo heatsink_forced_mlp); differentiable",
            "closed_form": "Teertstra 2000 forced convection, Muzychka & Yovanovich pressure drop, "
                           "Lee et al. 1995 spreading (circular-equivalent)",
            "fvm": "3-D finite-volume conduction in the base coupled to the fin-array correlations "
                   "(the HeatSink Sizer engine); reference for verification"}[fidelity]
        self.differentiable = fidelity == "surrogate"
        self.relaxable = fidelity in ("surrogate", "closed_form")
        self._zoo = None
        if fidelity == "surrogate":
            import pinneapple as pp
            self._zoo = pp.zoo.load("heatsink_forced_mlp")

    # geometry in closed form, NumPy or torch (same expressions)
    def _geometry(self, W, D, tb, N, t, H, xp_max):
        op, cm = self.op, self.costm
        gap = (W - N * t) / xp_max(N - 1, 1.0)
        vol_mm3 = W * D * tb + N * t * H * D
        mass_g = self.rho * vol_mm3 * 1e-6
        cost = cm.material_per_kg * mass_g * 1e-3 / cm.material_yield + cm.per_fin * N + cm.fixed
        # fin as a cantilever under a uniform lateral inertial load: M = rho t D a H^2 / 2, sigma = 6 M / (D t^2)
        stress_mpa = 3.0 * self.rho * op.shock_g * G0 * (H * 1e-3) ** 2 / (t * 1e-3) * 1e-6
        return {"mass_g": mass_g, "cost": cost, "fin_gap_mm": gap, "fin_aspect": H / gap,
                "fin_stress_mpa": stress_mpa}

    def _r_tim(self) -> float:
        return self.op.tim_k_mm2_w / (self.op.source_width_mm * self.op.source_depth_mm)

    def _zoo_inputs(self, X):
        op = self.op
        n = len(X)
        c = lambda v: np.full(n, float(v))  # noqa: E731
        return np.stack([X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], c(self.k), c(op.source_width_mm),
                         c(op.source_depth_mm), c(op.t_ambient_c), c(op.air_velocity_m_s), c(op.power_w)], 1)

    def evaluate(self, X):
        X = np.asarray(X, dtype=float)
        W, D, tb, N, t, H = X.T
        out = self._geometry(W, D, tb, N, t, H, np.maximum)
        op = self.op
        n = len(X)
        valid = (out["fin_gap_mm"] > 0)
        r = np.full(n, np.nan)
        dp = np.full(n, np.nan)
        if self.kind == "surrogate":
            A = self._zoo_inputs(X)
            ok = valid.copy()
            if ok.any():
                with warnings.catch_warnings():       # out-of-range designs are flagged invalid below
                    warnings.simplefilter("ignore")
                    p = self._zoo.predict(**{k: A[ok, i] for i, k in enumerate(self._zoo.input_names)})
                r[ok], dp[ok] = p["r_base_k_w"], p["pressure_drop_pa"]
            valid &= np.asarray(self._zoo.in_range(A), dtype=bool)
        else:
            from pinneapple_design.thermal.heatsink.engine import (
                DesignInput,
                OperatingInput,
                physics_resistance,
            )
            from pinneapple_physics.closed_form.plate_fin_heatsink import evaluate as closed_form
            for i in np.where(valid)[0]:
                d = DesignInput(W[i], D[i], tb[i], int(round(N[i])), t[i], H[i], op.material)
                o = OperatingInput(power_w=op.power_w, t_ambient_c=op.t_ambient_c,
                                   air_velocity_m_s=op.air_velocity_m_s, source_width_mm=op.source_width_mm,
                                   source_depth_mm=op.source_depth_mm, tim_k_mm2_w=op.tim_k_mm2_w)
                try:
                    if self.kind == "fvm":
                        res = physics_resistance(d, o)
                        r[i], dp[i] = res["r_base_k_w"], res["pressure_drop_pa"]
                    else:
                        g = d.geometry()
                        if self.relaxable and abs(N[i] - round(N[i])) > 1e-9:
                            g.n_fins = float(N[i])           # the network accepts a fractional fin count
                        res = closed_form(g, o.operating())
                        r[i] = res["r_total_k_w"] - res["resistances_k_w"]["interface_tim"]
                        dp[i] = res["pressure_drop_pa"]
                except (ValueError, ArithmeticError):
                    valid[i] = False
        out.update(r_base_k_w=r, pressure_drop_pa=dp,
                   t_source_c=op.t_ambient_c + op.power_w * (r + self._r_tim()))
        out[VALID] = valid & np.isfinite(r)
        return out

    def torch(self, X):
        import torch
        if self.kind != "surrogate":
            raise NotImplementedError(f"{self.name} is not differentiable")
        X = torch.as_tensor(X, dtype=torch.float64)
        W, D, tb, N, t, H = X.unbind(1)
        out = self._geometry(W, D, tb, N, t, H, lambda a, b: torch.clamp(a, min=b))
        op, cfg = self.op, self._zoo.config
        gap = torch.clamp(out["fin_gap_mm"], min=1e-3)
        n = X.shape[0]
        c = lambda v: torch.full((n,), float(v), dtype=torch.float64)  # noqa: E731
        feats = torch.stack([W.log(), D.log(), tb.log(), gap.log(), t.log(), H.log(), N.log(), c(math.log(self.k)),
                             c(op.source_width_mm).log() - W.log(), c(op.source_depth_mm).log() - D.log(),
                             c(op.t_ambient_c), c(math.log(op.air_velocity_m_s)), c(math.log(op.power_w))], 1)
        z = (feats - torch.as_tensor(cfg["x_mean"], dtype=torch.float64)) / torch.as_tensor(cfg["x_std"],
                                                                                            dtype=torch.float64)
        y = self._zoo.module(z.float())
        y = (y.y if hasattr(y, "y") else y).double()
        y = y * torch.as_tensor(cfg["y_std"], dtype=torch.float64) + torch.as_tensor(cfg["y_mean"],
                                                                                      dtype=torch.float64)
        r, dp = torch.exp(y[:, 0]), torch.exp(y[:, 1])
        out.update(r_base_k_w=r, pressure_drop_pa=dp, t_source_c=op.t_ambient_c + op.power_w * (r + self._r_tim()))
        with torch.no_grad():
            A = self._zoo_inputs(X.detach().numpy())
            out[VALID] = torch.as_tensor(np.asarray(self._zoo.in_range(A), dtype=bool) &
                                         (out["fin_gap_mm"].detach().numpy() > 0))
        return out


_HEATSINK_OBJECTIVES = {
    "mass_g": lambda cur: Objective("mass_g", "min", "g", "mass"),
    "cost": lambda cur: Objective("cost", "min", cur, f"unit cost ({cur}, assumed model)"),
    "pressure_drop_pa": lambda cur: Objective("pressure_drop_pa", "min", "Pa", "pressure drop"),
    "t_source_c": lambda cur: Objective("t_source_c", "min", "degC", "source temperature"),
    "r_base_k_w": lambda cur: Objective("r_base_k_w", "min", "K/W", "base-to-air resistance"),
}


def heatsink_problem(fidelity: str = "surrogate", operating: HeatsinkOperating | None = None,
                     limits: HeatsinkLimits | None = None, cost: CostModel | None = None,
                     variables: list[Variable] | None = None,
                     objectives: tuple[str, ...] = ("mass_g", "cost")) -> DesignProblem:
    """The plate-fin heat-sink sizing problem (module docstring) at ``fidelity``.

    ``objectives``: outputs to minimise, among ``mass_g``, ``cost``, ``pressure_drop_pa``, ``t_source_c``,
    ``r_base_k_w`` (default mass and cost; mass and pressure drop is the other classic trade-off)."""
    lim = limits or _process_limits()
    ev = HeatsinkEvaluator(fidelity, operating, cost)
    cur = ev.costm.currency
    cons = [
        Constraint("t_source_c", upper=lim.t_source_max_c, unit="degC", kind="physical",
                   label="source temperature"),
        Constraint("pressure_drop_pa", upper=lim.pressure_drop_max_pa, unit="Pa", kind="physical",
                   label="pressure drop"),
        Constraint("fin_stress_mpa", upper=lim.fin_stress_max_mpa, unit="MPa", kind="physical",
                   label=f"fin root stress at {ev.op.shock_g:g} g"),
        Constraint("fin_gap_mm", lower=lim.fin_gap_min_mm, unit="mm", kind="manufacturing", label="fin gap"),
        Constraint("fin_aspect", upper=lim.fin_aspect_max, unit="-", kind="manufacturing",
                   label="fin height / gap"),
    ]
    notes = [f"cost model is an assumption: {asdict(ev.costm)}",
             "ducted forced convection (no bypass), laminar/developing channel flow, no radiation",
             "fin stress: cantilever fin under a uniform lateral inertial load (shock), root bending only"]
    if fidelity == "surrogate":
        notes.append("thermal/hydraulic values from an MLP surrogate: confirm chosen designs with the FVM reference")
    return DesignProblem(
        "heatsink", DesignSpace(variables or HEATSINK_VARIABLES),
        [_HEATSINK_OBJECTIVES[o](cur) for o in objectives],
        cons, ev, description="Plate-fin heat sink, ducted forced air: minimise "
                              + " and ".join(_HEATSINK_OBJECTIVES[o](cur).label for o in objectives)
                              + " within the source temperature, pressure drop, fin stress and extrusion limits.",
        notes=notes)


def heatsink_baseline() -> dict[str, float]:
    """A conventional starting design (100 x 100 mm, 20 fins of 1.5 x 40 mm, 6 mm base) to compare against."""
    return {"base_width_mm": 100.0, "base_depth_mm": 100.0, "base_thickness_mm": 6.0, "n_fins": 20,
            "fin_thickness_mm": 1.5, "fin_height_mm": 40.0}


# --------------------------------------------------------------------------- CST airfoil
AIRFOIL_OUTPUTS = ["Cd", "Cl", "Cm", "LD", "Cd_std", "Cl_std", "thickness", "spar_depth", "aft_thickness"]
_X_SPAR = np.linspace(0.2, 0.35, 16)
_X_ALL = np.linspace(0.0, 1.0, 101)


def _cst_thickness_np(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    from pinneapple_design.aero.geometry import TE_THICKNESS, _bern
    return np.sqrt(x) * (1 - x) * (_bern(2, x) @ (2 * T.T)).T + x * TE_THICKNESS


class AirfoilEvaluator(Evaluator):
    """``airfoil_cst_rans_mlp`` ensemble (mean and spread of Cl, Cd, Cm) plus closed-form section thicknesses."""
    fidelity = "surrogate"
    differentiable = True
    relaxable = True

    def __init__(self):
        import pinneapple as pp
        self._zoo = pp.zoo.load("airfoil_cst_rans_mlp")
        self.outputs = list(AIRFOIL_OUTPUTS)
        self.name = "zoo:airfoil_cst_rans_mlp (MLP ensemble)"
        self.description = ("5-member MLP ensemble trained on OpenFOAM k-omega SST runs at Re 4e6 (pp.zoo "
                            "airfoil_cst_rans_mlp); differentiable")

    def _valid(self, X: np.ndarray) -> np.ndarray:
        from pinneapple_design.aero.geometry import valid
        geo = np.array([valid(x[:6])[0] for x in X])
        return geo & np.asarray(self._zoo.in_range(X[:, :7]), dtype=bool)

    def evaluate(self, X):
        X = np.asarray(X, dtype=float)
        with warnings.catch_warnings():               # out-of-range designs are flagged invalid below
            warnings.simplefilter("ignore")
            p = self._zoo.predict(**{k: X[:, i] for i, k in enumerate(self._zoo.input_names)})
        T = X[:, :3]
        out = {k: np.asarray(p[k], dtype=float) for k in ("Cl", "Cd", "Cm", "LD", "Cl_std", "Cd_std")}
        out["thickness"] = _cst_thickness_np(T, _X_ALL).max(1)
        out["spar_depth"] = _cst_thickness_np(T, _X_SPAR).min(1)
        out["aft_thickness"] = _cst_thickness_np(T, np.array([0.9]))[:, 0]
        out[VALID] = self._valid(X)
        return out

    def torch(self, X):
        import torch

        from pinneapple_design.aero.geometry import BOUNDS, TE_THICKNESS
        X = torch.as_tensor(X, dtype=torch.float64)
        b = torch.as_tensor(BOUNDS, dtype=torch.float64)
        s = (X[:, :6] - b[:, 0]) / (b[:, 1] - b[:, 0]) * 2 - 1
        a = X[:, 6]
        z = torch.cat([s, ((a - 6) / 8)[:, None], (torch.sin(torch.deg2rad(a)) * 4)[:, None]], 1).float()
        c = self._zoo.config
        y = self._zoo.module(z).double() * torch.as_tensor(c["y_std"], dtype=torch.float64) \
            + torch.as_tensor(c["y_mean"], dtype=torch.float64)                    # (K, n, 3)
        cl, cd, cm = y[..., 0], torch.exp(y[..., 1]), y[..., 2]

        def thick(x):
            x = torch.as_tensor(x, dtype=torch.float64)
            B = torch.stack([(1 - x) ** 2, 2 * x * (1 - x), x ** 2], 1)          # 2nd-order Bernstein
            return torch.sqrt(x) * (1 - x) * (2 * X[:, :3] @ B.T) + x * TE_THICKNESS
        out = {"Cl": cl.mean(0), "Cd": cd.mean(0), "Cm": cm.mean(0), "LD": (cl / cd).mean(0),
               "Cl_std": cl.std(0, unbiased=False), "Cd_std": cd.std(0, unbiased=False),
               "thickness": thick(_X_ALL).max(1).values, "spar_depth": thick(_X_SPAR).min(1).values,
               "aft_thickness": thick(np.array([0.9]))[:, 0]}
        with torch.no_grad():
            out[VALID] = torch.as_tensor(self._valid(X.detach().numpy()))
        return out


def airfoil_problem(cl_min: float = 0.6, cm_min: float = -0.12, aft_min: float = 0.012,
                    alpha_range: tuple[float, float] = (0.0, 8.0)) -> DesignProblem:
    """CST airfoil: minimise Cd, maximise spar depth, with Cl >= ``cl_min``, Cm >= ``cm_min`` and thickness at
    90 % chord >= ``aft_min`` (manufacturing: room for the trailing-edge structure)."""
    from pinneapple_design.aero.geometry import BOUNDS, LABELS, PARAMS
    vs = [Variable(p, float(lo), float(hi), unit="-", description=LABELS[p]) for p, (lo, hi) in zip(PARAMS, BOUNDS,
                                                                                                    strict=True)]
    vs.append(Variable("alpha_deg", float(alpha_range[0]), float(alpha_range[1]), unit="deg",
                       description="angle of attack at the design point"))
    return DesignProblem(
        "airfoil", DesignSpace(vs),
        [Objective("Cd", "min", "-", "drag coefficient"),
         Objective("spar_depth", "max", "chord", "spar depth (20-35 % chord)")],
        [Constraint("Cl", lower=cl_min, unit="-", kind="physical", label="lift coefficient"),
         Constraint("Cm", lower=cm_min, unit="-", kind="physical", label="pitching moment"),
         Constraint("aft_thickness", lower=aft_min, unit="chord", kind="manufacturing",
                    label="thickness at 90 % chord")],
        AirfoilEvaluator(),
        description="CST airfoil at Re 4e6: minimise drag and maximise spar depth (lighter spar) at a lift target.",
        notes=["2-D section, fully turbulent RANS training data, Re 4e6, incompressible",
               "no reference solver in this environment: verify chosen sections with OpenFOAM"])


def airfoil_baseline() -> dict[str, float]:
    """NACA 2412 (CST fit) at 4 degrees."""
    from pinneapple_design.aero.geometry import PARAMS, REFERENCE
    return {**{p: float(v) for p, v in zip(PARAMS, REFERENCE["NACA 2412"], strict=True)}, "alpha_deg": 4.0}


PRESETS: dict[str, Any] = {"heatsink": heatsink_problem, "airfoil": airfoil_problem}


def preset_info(name: str) -> dict[str, Any]:
    """What a form needs to set up a preset: its variables, defaults (operating point, limits, cost model), the
    objectives it can minimise, the baseline design and the reference that verifies its candidates (None if none)."""
    if name == "heatsink":
        return {"variables": [asdict(v) for v in HEATSINK_VARIABLES], "operating": asdict(HeatsinkOperating()),
                "limits": asdict(_process_limits()), "cost": asdict(CostModel()),
                "objectives": list(_HEATSINK_OBJECTIVES), "baseline": heatsink_baseline(),
                "reference": "3-D finite-volume heat-sink engine"}
    if name == "airfoil":
        import inspect
        defaults = {k: p.default for k, p in inspect.signature(airfoil_problem).parameters.items()
                    if k in ("cl_min", "cm_min", "aft_min")}
        return {"variables": [asdict(v) for v in airfoil_problem().space.variables], "limits": defaults,
                "objectives": ["Cd", "spar_depth"], "baseline": airfoil_baseline(), "reference": None}
    raise ValueError(f"unknown preset {name!r}: one of {sorted(PRESETS)}")


# --------------------------------------------------------------------------- geometry export
def _box(x0, x1, y0, y1, z0, z1, off):
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]) + off
    return V, F


def plate_fin_bodies(design: dict[str, float]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Closed triangle meshes (metres) of the base plate and each fin: ``[(name, V, F)]``; x across the fins, y along
    the flow, z up."""
    mm = 1e-3
    W, D, tb = design["base_width_mm"] * mm, design["base_depth_mm"] * mm, design["base_thickness_mm"] * mm
    N, t, H = int(round(design["n_fins"])), design["fin_thickness_mm"] * mm, design["fin_height_mm"] * mm
    bodies = [("base", *_box(0, W, 0, D, 0, tb, 0))]
    pitch = (W - t) / max(N - 1, 1)
    for i in range(N):
        bodies.append((f"fin_{i + 1}", *_box(i * pitch, i * pitch + t, 0, D, tb, tb + H, 0)))
    return bodies


def airfoil_bodies(design: dict[str, float], span: float = 0.2, chord: float = 1.0,
                   n: int = 81) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """A closed extruded wing segment (metres) of the CST section, ``chord`` x ``span``, both ends capped."""
    from pinneapple_design.aero.geometry import PARAMS, outline
    o = outline(np.array([design[p] for p in PARAMS]), n)
    x = np.r_[np.array(o["x"])[::-1], np.array(o["x"])[1:]]
    y = np.r_[np.array(o["upper"])[::-1], np.array(o["lower"])[1:]]
    m = len(x)
    P = np.c_[x * chord, y * chord]
    V = np.vstack([np.c_[P, np.zeros(m)], np.c_[P, np.full(m, span)],
                   [P[:, 0].mean(), P[:, 1].mean(), 0.0], [P[:, 0].mean(), P[:, 1].mean(), span]])
    c0, c1 = 2 * m, 2 * m + 1
    F = []
    for i in range(m):
        j = (i + 1) % m
        F += [[i, j, m + j], [i, m + j, m + i], [c0, j, i], [c1, m + i, m + j]]
    return [("wing_segment", V, np.array(F))]


def preset_bodies(preset: str, design: dict[str, float]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The solid bodies of a ``"heatsink"`` (:func:`plate_fin_bodies`) or ``"airfoil"`` (:func:`airfoil_bodies`)
    design, for :func:`design_file`."""
    if preset == "airfoil":
        from pinneapple_design.aero.geometry import PARAMS
        return airfoil_bodies({p: float(design[p]) for p in PARAMS})
    if preset == "heatsink":
        return plate_fin_bodies(design)
    raise ValueError(f"unknown preset {preset!r}: heatsink or airfoil")


def design_file(bodies: list[tuple[str, np.ndarray, np.ndarray]], fmt: str, name: str = "design") -> bytes:
    """``"stl"`` (binary, all bodies) or ``"step"`` (AP214 faceted B-rep, one solid per body) bytes, written by
    :mod:`pinneapple_design.geometry_library.exporters`."""
    from types import SimpleNamespace

    from pinneapple_design.geometry_library.exporters import normalise_format, stl_bytes, to_step
    fmt = normalise_format(fmt)
    if fmt == "stl":
        off = np.cumsum([0] + [len(V) for _, V, _ in bodies[:-1]])
        V = np.vstack([b[1] for b in bodies])
        F = np.vstack([b[2] + o for b, o in zip(bodies, off, strict=True)])
        return stl_bytes(V, F, f"PINNeAPPle design: {name}, metres")
    if fmt == "step":
        geom = SimpleNamespace(name=name, title=name, bodies=[
            SimpleNamespace(name=n, vertices=V, faces=F, closed=True, aero=True) for n, V, F in bodies])
        return to_step(geom)
    raise ValueError("design files are written as stl or step")
