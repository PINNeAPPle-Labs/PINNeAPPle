"""Cheap physical models that turn a geometry into first estimates, the mechanisms behind them and a surface map.

They are meant for the qualitative step before any simulation: which way does each geometric change push the
result, how strongly, and why. Each model evaluates a closed triangulated body (or an ``Assembly`` of parts) face
by face, so a change in one part shows up in the mechanism and in the part it belongs to.

* ``ExternalFlow``: drag and lift of a body in a stream. Windward faces: modified Newtonian pressure
  ``Cp = Cp_stag sin^2(delta)``; leeward faces inclined less than the separation angle recover pressure, steeper ones
  are separated and see the base pressure; friction from the flat-plate skin-friction law on attached faces
  (Hoerner, *Fluid-Dynamic Drag*, 1965, ch. 3 and 6; Anderson, *Hypersonic and High-Temperature Gas Dynamics*, for
  the Newtonian law).
* ``ConvectiveCooling``: heat rejected by a body at uniform temperature: local flat-plate correlations from each
  part's leading edge (laminar 0.332 Re_x^0.5 Pr^(1/3), turbulent 0.0296 Re_x^0.8 Pr^(1/3)), reduced coefficient in
  separated regions, natural convection from the vertical-plate law, optional fin efficiency tanh(mL)/(mL)
  (Incropera & DeWitt, ch. 3, 7 and 9).
* ``Cantilever``: a body clamped at one end with a tip load: non-uniform Euler-Bernoulli beam with section
  properties from a voxelization, tip deflection, bending stress, mass and a Rayleigh estimate of the first
  frequency (Timoshenko & Gere; Rao, *Mechanical Vibrations*).
* ``ScalingModel``: any user formula of the descriptors (``pinneapple_design.qualitative.descriptors``).

The estimates are order-of-magnitude: they rank variants and explain trends; the numbers come from the quantitative
step (``QualitativePreview.quantify``).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .descriptors import Assembly, as_mesh, describe, face_geometry, section_profile

__all__ = ["Cantilever", "ConvectiveCooling", "Evaluation", "ExternalFlow", "QualitativeModel", "ScalingModel",
           "MODELS"]


@dataclass
class Quantity:
    name: str
    unit: str
    better: str                     # "min" or "max": the usual goal, used when an objective names no sense
    pt: str
    en: str


@dataclass
class Evaluation:
    """Result of one model on one geometry."""
    quantities: dict[str, float]
    terms: dict[str, dict[str, float]]           # quantity -> mechanism -> additive contribution
    by_part: dict[str, dict[str, float]]         # quantity -> part -> additive contribution
    face_field: np.ndarray                       # per-face indicator for the surface map
    field_label: tuple[str, str]                 # (pt, en)
    notes: list[tuple[str, str]] = field(default_factory=list)       # regime notes (pt, en)
    caveats: list[tuple] = field(default_factory=list)   # reasons for lower confidence: (pt, en[, weight])
    mesh: tuple[np.ndarray, np.ndarray] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class QualitativeModel:
    name = "model"
    title = ("modelo", "model")
    quantities: dict[str, Quantity] = {}
    mechanisms: dict[str, tuple[str, str]] = {}    # mechanism -> (pt, en) explanation
    defaults: dict[str, Any] = {}
    cmap = "coolwarm"

    def __init__(self, **conditions):
        self.conditions = {**self.defaults, **conditions}

    def evaluate(self, geom: Any) -> Evaluation:  # pragma: no cover - interface
        raise NotImplementedError


def _owner(geom, n_faces) -> tuple[np.ndarray, list[str]]:
    if isinstance(geom, Assembly):
        return geom.face_owner(), geom.names
    return np.zeros(n_faces, int), ["body"]


def sheltered_faces(V: np.ndarray, F: np.ndarray, normals: np.ndarray, centroids: np.ndarray, direction,
                    grid: int = 96, tol: float = 0.05) -> np.ndarray:
    """Windward faces hidden behind other surfaces seen from upstream (in the wake of something in front of them):
    the windward triangles are rasterized on the plane normal to the flow, keeping the most upstream depth per
    cell; a windward face whose cell has a surface more than ``tol`` x body length upstream of it is sheltered."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    nd = normals @ d
    wind = np.nonzero(nd < -0.2)[0]
    out = np.zeros(len(F), bool)
    if len(wind) < 2:
        return out
    e1 = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    P = np.stack([V @ e1, V @ e2], axis=1)
    depth = V @ d
    lo, hi = P.min(0), P.max(0)
    h = (hi - lo).max() / grid + 1e-300
    best = np.full((grid + 1, grid + 1), np.inf)
    for f in wind:
        tri = P[F[f]]
        i0, j0 = np.floor((tri.min(0) - lo) / h).astype(int)
        i1, j1 = np.ceil((tri.max(0) - lo) / h).astype(int)
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        pts = np.stack([lo[0] + (ii + 0.5) * h, lo[1] + (jj + 0.5) * h], -1).reshape(-1, 2)
        a, b, c = tri
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-300:
            continue
        w0 = ((b[1] - c[1]) * (pts[:, 0] - c[0]) + (c[0] - b[0]) * (pts[:, 1] - c[1])) / den
        w1 = ((c[1] - a[1]) * (pts[:, 0] - c[0]) + (a[0] - c[0]) * (pts[:, 1] - c[1])) / den
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (1 - w0 - w1 >= -1e-9)
        dz = depth[F[f]].mean()
        ci, cj = ii.ravel()[inside], jj.ravel()[inside]
        ci, cj = np.clip(ci, 0, grid), np.clip(cj, 0, grid)
        best[ci, cj] = np.minimum(best[ci, cj], dz)
    L = float(np.ptp(depth))
    cpos = np.stack([centroids @ e1, centroids @ e2], 1)
    ij = np.clip(np.floor((cpos[wind] - lo) / h).astype(int), 0, grid)
    out[wind] = centroids[wind] @ d > best[ij[:, 0], ij[:, 1]] + tol * L
    return out


def _group(values: np.ndarray, owner: np.ndarray, names: list[str]) -> dict[str, float]:
    return {n: float(values[owner == i].sum()) for i, n in enumerate(names)}


# ----------------------------------------------------------------------------------------------------- flow
class ExternalFlow(QualitativeModel):
    name = "external_flow"
    title = ("escoamento externo (arrasto e sustentação)", "external flow (drag and lift)")
    quantities = {
        "drag": Quantity("drag", "N", "min", "arrasto", "drag"),
        "drag_coefficient": Quantity("drag_coefficient", "-", "min", "coeficiente de arrasto (área frontal)",
                                     "drag coefficient (frontal area)"),
        "lift": Quantity("lift", "N", "max", "sustentação", "lift"),
        "wake_area": Quantity("wake_area", "m2", "min", "área de esteira separada", "separated wake area"),
    }
    mechanisms = {
        "stagnation": ("pressão nas faces que encaram o escoamento", "pressure on the faces facing the flow"),
        "recovery": ("recuperação de pressão na traseira afilada (escoamento colado)",
                     "pressure recovery on a gently tapered rear (attached flow)"),
        "separation": ("sucção da esteira atrás de faces íngremes ou da base reta (escoamento separado)",
                       "wake suction behind steep faces or a blunt base (separated flow)"),
        "friction": ("atrito na superfície molhada", "skin friction on the wetted surface"),
    }
    defaults = {"U": 10.0, "rho": 1.225, "nu": 1.5e-5, "direction": (1.0, 0.0, 0.0), "lift_axis": (0.0, 0.0, 1.0),
                "Cp_stag": 1.0, "Cp_base": -0.25, "separation_angle": 30.0, "recovery": 0.85}

    def evaluate(self, geom):
        c = self.conditions
        V, F = as_mesh(geom)
        A, n, ctr = face_geometry(V, F)
        owner, names = _owner(geom, len(F))
        d = np.asarray(c["direction"], float)
        d /= np.linalg.norm(d)
        e = np.asarray(c["lift_axis"], float)
        e /= np.linalg.norm(e)
        q = 0.5 * c["rho"] * c["U"] ** 2
        L = float(np.ptp(V @ d))
        Re = c["U"] * L / c["nu"]
        Cf = 1.328 / math.sqrt(Re) if Re < 5e5 else max(0.074 * Re ** -0.2 - 1742 / Re, 0.0)
        nd = n @ d
        s_sep = math.sin(math.radians(c["separation_angle"]))
        shelter = sheltered_faces(V, F, n, ctr, d)               # in the wake of something upstream
        wind = (nd < 0) & ~shelter
        sep = (nd > s_sep) | shelter
        att_lee = (nd >= 0) & ~sep
        Cp = np.where(wind, c["Cp_stag"] * nd ** 2, 0.0)
        Cp = np.where(att_lee, c["recovery"] * c["Cp_stag"] * nd ** 2, Cp)
        Cp = np.where(sep, c["Cp_base"], Cp)
        f_press = -Cp * nd * A * q                                  # pressure force along d per face
        f_fric = np.where(sep, 0.0, Cf * q * A * np.sqrt(np.clip(1 - nd ** 2, 0, 1)))
        lift = float(np.sum(-Cp * (n @ e) * A * q))
        mech = {"stagnation": np.where(wind, f_press, 0.0), "recovery": np.where(att_lee, f_press, 0.0),
                "separation": np.where(sep, f_press, 0.0), "friction": f_fric}
        drag_f = sum(mech.values())
        D = float(drag_f.sum())
        frontal = describe((V, F), d).frontal_area
        wake = float(np.sum(np.where(sep, np.abs(nd) * A, 0.0)))
        notes = [(f"Re = {Re:.2g} ({'laminar' if Re < 5e5 else 'turbulento'} na camada-limite, Cf ≈ {Cf:.4f})",
                  f"Re = {Re:.2g} ({'laminar' if Re < 5e5 else 'turbulent'} boundary layer, Cf ≈ {Cf:.4f})")]
        cav = []
        if 2e5 < Re < 2e6:
            cav.append(("Re na faixa de transição: atrito e ponto de separação mudam com detalhes (rugosidade, "
                        "turbulência do escoamento)",
                        "Re in the transition range: friction and the separation point depend on details "
                        "(roughness, free-stream turbulence)"))
        lee = nd > 0
        near = lee & (np.abs(np.degrees(np.arcsin(np.clip(nd, -1, 1))) - c["separation_angle"]) < 5)
        if lee.any() and A[near].sum() > 0.1 * A[lee].sum():
            cav.append((f"parte da traseira está perto do ângulo de separação ({c['separation_angle']:.0f}°): o "
                        "arrasto pode saltar com pequenas mudanças, e vórtices longitudinais (não modelados) "
                        "costumam aumentá-lo nessa faixa",
                        f"part of the rear is close to the separation angle ({c['separation_angle']:.0f}°): drag "
                        "can jump with small changes, and longitudinal vortices (not modelled) usually raise it in "
                        "this range", 2))
        if shelter.any():
            notes.append((f"{shelter.sum()} faces voltadas ao escoamento estão abrigadas atrás de outras (tratadas "
                          "como esteira)", f"{shelter.sum()} windward faces are sheltered behind others (treated "
                          "as wake)"))
        return Evaluation(
            quantities={"drag": D, "drag_coefficient": D / (q * max(frontal, 1e-300)), "lift": lift, "wake_area": wake},
            terms={"drag": {k: float(v.sum()) for k, v in mech.items()}},
            by_part={"drag": _group(drag_f, owner, names)},
            face_field=Cp, field_label=("Cp estimado (vermelho: estagnação, azul: esteira)",
                                        "estimated Cp (red: stagnation, blue: wake)"),
            notes=notes, caveats=cav, mesh=(V, F), extra={"Re": Re, "q": q, "frontal_area": frontal})


# ----------------------------------------------------------------------------------------------------- heat
class ConvectiveCooling(QualitativeModel):
    name = "convective_cooling"
    title = ("resfriamento por convecção", "convective cooling")
    quantities = {
        "heat_rate": Quantity("heat_rate", "W", "max", "calor dissipado", "heat rejected"),
        "thermal_resistance": Quantity("thermal_resistance", "K/W", "min", "resistência térmica", "thermal resistance"),
        "mean_h": Quantity("mean_h", "W/m2K", "max", "coeficiente médio de convecção", "mean convection coefficient"),
        "cooling_time": Quantity("cooling_time", "s", "min", "constante de tempo de resfriamento", "cooling time constant"),
        "mass": Quantity("mass", "kg", "min", "massa", "mass"),
    }
    mechanisms = {
        "leading_edges": ("bordas de ataque: camada-limite fina, troca intensa", "leading edges: thin boundary layer, strong exchange"),
        "developed": ("superfície a jusante: a camada-limite engrossa e a troca cai", "downstream surface: the boundary layer thickens, exchange drops"),
        "wake": ("regiões separadas (recirculação): troca fraca", "separated regions (recirculation): weak exchange"),
        "fin_losses": ("perda por eficiência de aleta (condução até a ponta)", "fin efficiency loss (conduction to the tip)"),
        "air_heating": ("o ar esquenta ao atravessar o corpo e troca menos (efetividade, NTU)",
                        "the air heats up across the body and exchanges less (effectiveness, NTU)"),
    }
    defaults = {"U": 5.0, "k": 0.026, "nu": 1.6e-5, "Pr": 0.71, "dT": 40.0, "direction": (1.0, 0.0, 0.0),
                "gravity_axis": (0.0, 0.0, -1.0), "beta": 1 / 300, "rho_solid": 2700.0, "c_solid": 900.0,
                "k_solid": 200.0, "separation_angle": 30.0, "wake_factor": 0.5, "base_part": None,
                "rho_fluid": 1.16, "cp_fluid": 1007.0}

    def evaluate(self, geom):
        c = self.conditions
        V, F = as_mesh(geom)
        A, n, ctr = face_geometry(V, F)
        owner, names = _owner(geom, len(F))
        Pr13 = c["Pr"] ** (1 / 3)
        natural = c["U"] <= 0
        if natural:
            g = -np.asarray(c["gravity_axis"], float)
            g /= np.linalg.norm(g)                                  # "up"
            z = ctr @ g
            zmin = np.array([(V[np.unique(F[owner == i])] @ g).min() for i in range(len(names))])[owner]
            x = np.maximum(z - zmin, 1e-3 * np.ptp(V @ g))
            Gr = 9.81 * c["beta"] * c["dT"] * x ** 3 / c["nu"] ** 2
            # local laminar vertical-plate law (integral method): Nu_x = 0.508 Pr^1/2 (0.952 + Pr)^-1/4 Gr_x^1/4
            h = 0.508 * c["Pr"] ** 0.5 * (0.952 + c["Pr"]) ** -0.25 * Gr ** 0.25 * c["k"] / x
            Ra = Gr * c["Pr"]
            sep = np.zeros(len(F), bool)
            Rmax = float(Ra.max())
            notes = [(f"convecção natural, Ra máx ≈ {Rmax:.2g}", f"natural convection, max Ra ≈ {Rmax:.2g}")]
        else:
            d = np.asarray(c["direction"], float)
            d /= np.linalg.norm(d)
            xmin = np.array([(V[np.unique(F[owner == i])] @ d).min() for i in range(len(names))])[owner]
            x = np.maximum(ctr @ d - xmin, 1e-2 * np.ptp(V @ d))
            Rex = c["U"] * x / c["nu"]
            h = np.where(Rex < 5e5, 0.332 * Rex ** 0.5, 0.0296 * Rex ** 0.8) * Pr13 * c["k"] / x
            # faces that meet the flow head-on: stagnation-region law of a blunt face, Nu_D = 0.93 Re_D^0.5 Pr^1/3
            # (Sparrow & Geiger 1985), D the size of the part across the flow
            head = (n @ d) < -0.5
            if head.any():
                Dp = np.array([max(math.sqrt(max(describe(geom.parts[nm] if isinstance(geom, Assembly) else (V, F), d)
                                                 .frontal_area, 1e-12)), 1e-6) for nm in names])[owner]
                h = np.where(head, 0.93 * np.sqrt(c["U"] * Dp / c["nu"]) * Pr13 * c["k"] / Dp, h)
            sep = ((n @ d) > math.sin(math.radians(c["separation_angle"]))) | sheltered_faces(V, F, n, ctr, d)
            # recirculating regions: a fraction of the mean flat-plate coefficient over the whole body length (not
            # of the local attached values, which are high on short leading pieces)
            ReL = c["U"] * float(np.ptp(V @ d)) / c["nu"]
            Lb = float(np.ptp(V @ d))
            h_plate = (0.664 * ReL ** 0.5 if ReL < 5e5 else 0.037 * ReL ** 0.8) * Pr13 * c["k"] / Lb
            h = np.where(sep, c["wake_factor"] * h_plate, h)
            notes = [(f"convecção forçada, Re_L ≈ {ReL:.2g}", f"forced convection, Re_L ≈ {ReL:.2g}")]
        # fin efficiency of the parts attached to the base part
        eta = np.ones(len(F))
        if c["base_part"] is not None and isinstance(geom, Assembly):
            Vb, _ = geom.parts[c["base_part"]]
            lo, hi = Vb.min(0), Vb.max(0)
            for i, name in enumerate(names):
                if name == c["base_part"]:
                    continue
                Vi, Fi = geom.parts[name]
                dist = np.linalg.norm(np.maximum(0, np.maximum(lo - Vi, Vi - hi)), axis=1)
                Lf = float(dist.max())
                Ai = A[owner == i]
                t = 2 * describe((Vi, Fi)).volume / max(Ai.sum(), 1e-300)
                hm = float(np.sum(h[owner == i] * Ai) / max(Ai.sum(), 1e-300))
                mL = math.sqrt(2 * hm / (c["k_solid"] * max(t, 1e-9))) * Lf
                eta[owner == i] = math.tanh(mL) / mL if mL > 1e-9 else 1.0
        q_ideal = h * A * c["dT"]
        q_fin = q_ideal * eta
        # the air crossing the body's envelope heats up: Q = C dT (1 - exp(-hA / C)), C = rho U A_env cp
        # (effectiveness-NTU; forced convection only)
        heating = 1.0
        if not natural:
            Ld = np.ptp(V @ d)
            A_env = float(np.prod(np.ptp(V, axis=0))) / max(Ld, 1e-300)
            C = c["rho_fluid"] * c["U"] * A_env * c["cp_fluid"]
            ntu = float(q_fin.sum()) / c["dT"] / C
            heating = (1 - math.exp(-ntu)) / ntu if ntu > 1e-9 else 1.0
        q_face = q_fin * heating
        Q = float(q_face.sum())
        hA = Q / c["dT"]
        S = float(A.sum())
        desc = describe((V, F))
        mass = c["rho_solid"] * desc.volume
        lead = (~sep) & (x <= np.quantile(x, 0.25))
        mech = {"leading_edges": float(q_ideal[lead].sum()), "developed": float(q_ideal[(~sep) & ~lead].sum()),
                "wake": float(q_ideal[sep].sum()), "fin_losses": float((q_fin - q_ideal).sum()),
                "air_heating": float((q_face - q_fin).sum())}
        cav = []
        Bi = (hA / S) * desc.characteristic_length / c["k_solid"]
        if Bi > 0.1:
            cav.append((f"Biot ≈ {Bi:.2f} > 0,1: o sólido não fica isotérmico, gradientes internos importam",
                        f"Biot ≈ {Bi:.2f} > 0.1: the solid is not isothermal, internal gradients matter"))
        if not natural and heating < 0.7:
            cav.append((f"o ar já sai muito aquecido (efetividade {heating:.0%} do ideal): mais área rende pouco, "
                        "mais vazão rende mais", f"the air leaves already hot ({heating:.0%} of ideal): more area gains "
                        "little, more airflow gains more"))
        if not natural and isinstance(geom, Assembly) and len(names) > 2:
            gap = _min_gap(geom, c["base_part"])
            Ld = float(np.ptp(V @ d))
            delta = 5 * Ld / math.sqrt(max(c["U"] * Ld / c["nu"], 1.0))
            if gap is not None and gap < 2 * delta:
                cav.append((f"espaço entre peças ({gap * 1e3:.1f} mm) menor que duas camadas-limite "
                            f"({2 * delta * 1e3:.1f} mm): elas se juntam e o escoamento vira de canal; sem duto, parte "
                            "do ar desvia por fora e a perda de carga sobe",
                            f"gap between parts ({gap * 1e3:.1f} mm) below two boundary layers ({2 * delta * 1e3:.1f} "
                            "mm): they merge and the flow becomes channel flow; without a duct part of the air "
                            "bypasses and the pressure drop rises", 2 if gap < delta else 1))
        if np.any(eta < 0.7):
            cav.append(("há aletas com eficiência abaixo de 70%: alongá-las rende pouco",
                        "some fins are below 70% efficiency: making them longer gains little"))
        return Evaluation(
            quantities={"heat_rate": Q, "thermal_resistance": 1 / max(hA, 1e-300), "mean_h": hA / S,
                        "cooling_time": mass * c["c_solid"] / max(hA, 1e-300), "mass": mass},
            terms={"heat_rate": mech}, by_part={"heat_rate": _group(q_face, owner, names)},
            face_field=h * eta, field_label=("h local estimado (W/m²K)", "estimated local h (W/m²K)"),
            notes=notes + [(f"Biot ≈ {Bi:.3f}", f"Biot ≈ {Bi:.3f}")], caveats=cav, mesh=(V, F))

    cmap = "inferno"


def _min_gap(geom: Assembly, skip: str | None) -> float | None:
    """Smallest clear distance between the bounding boxes of two parts that face each other (overlap in the
    other two directions), ignoring ``skip`` (the base)."""
    boxes = [(n, V.min(0), V.max(0)) for n, (V, _) in geom.parts.items() if n != skip]
    best = None
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            _, lo1, hi1 = boxes[i]
            _, lo2, hi2 = boxes[j]
            sep = np.maximum(lo2 - hi1, lo1 - hi2)
            if np.sum(sep > 0) == 1:
                g = float(sep.max())
                best = g if best is None else min(best, g)
    return best


# ----------------------------------------------------------------------------------------------------- structure
class Cantilever(QualitativeModel):
    name = "cantilever"
    title = ("viga engastada com carga na ponta", "cantilever with a tip load")
    quantities = {
        "tip_deflection": Quantity("tip_deflection", "m", "min", "deflexão na ponta", "tip deflection"),
        "stiffness": Quantity("stiffness", "N/m", "max", "rigidez", "stiffness"),
        "max_stress": Quantity("max_stress", "Pa", "min", "tensão máxima de flexão", "maximum bending stress"),
        "safety_factor": Quantity("safety_factor", "-", "max", "fator de segurança", "safety factor"),
        "mass": Quantity("mass", "kg", "min", "massa", "mass"),
        "stiffness_to_mass": Quantity("stiffness_to_mass", "N/m/kg", "max", "rigidez por massa", "stiffness per mass"),
        "first_frequency": Quantity("first_frequency", "Hz", "max", "primeira frequência natural", "first natural frequency"),
    }
    mechanisms = {
        "root": ("flexibilidade no terço junto ao engaste (momento fletor maior)",
                 "flexibility in the third next to the clamp (largest bending moment)"),
        "middle": ("flexibilidade no terço do meio", "flexibility in the middle third"),
        "tip": ("flexibilidade no terço da ponta (momento pequeno)", "flexibility in the tip third (small moment)"),
        "mass": ("massa retirada ou acrescentada (o denominador)", "mass removed or added (the denominator)"),
    }
    defaults = {"E": 70e9, "rho": 2700.0, "load": 1000.0, "axis": None, "load_axis": 2, "clamp": "min",
                "yield_stress": 250e6, "n_slices": 40, "n_grid": 40}
    cmap = "viridis"

    def evaluate(self, geom):
        c = self.conditions
        V, F = as_mesh(geom)
        A, n, ctr = face_geometry(V, F)
        ext = np.ptp(V, axis=0)
        ax = int(np.argmax(ext)) if c["axis"] is None else int(c["axis"])
        la = int(c["load_axis"])
        if la == ax:
            raise ValueError("the load must be transverse to the beam axis")
        prof = section_profile((V, F), axis=ax, n_slices=c["n_slices"], n_grid=c["n_grid"])
        _, oa, ob = prof["axes"]
        Isec, cfib, cen = (prof["I_a"], prof["c_a"], prof["centroid_b"]) if la == ob else (prof["I_b"], prof["c_b"], prof["centroid_a"])
        x, dx = prof["x"], prof["dx"]
        lo, hi = V[:, ax].min(), V[:, ax].max()
        Lb = hi - lo
        xi = (x - lo) if c["clamp"] == "min" else (hi - x)
        E, P = c["E"], c["load"]
        gap = Isec <= 0
        Is = np.where(gap, np.max(Isec) * 1e-9 + 1e-30, Isec)
        flex = P * (Lb - xi) ** 2 / (E * Is) * dx                    # contribution of each slice to the tip deflection
        delta = float(flex.sum())
        M = P * (Lb - xi)
        sigma = M * cfib / Is
        smax = float(sigma.max())
        mass = c["rho"] * describe((V, F)).volume
        # Rayleigh quotient with the static tip-load deflection shape
        order = np.argsort(xi)
        curv = (M / (E * Is))[order]
        slope = np.cumsum(curv) * dx
        y = np.cumsum(slope) * dx
        mlin = c["rho"] * prof["A"][order]
        omega2 = float(np.sum(E * Is[order] * curv ** 2) * dx / max(np.sum(mlin * y ** 2) * dx, 1e-300))
        thirds = np.digitize(xi / Lb, [1 / 3, 2 / 3])
        mech = {k: float(flex[thirds == i].sum()) for i, k in enumerate(("root", "middle", "tip"))}
        # bending stress indicator on the surface: M c / I with c the face's distance from the section centroid
        xf = ctr[:, ax]
        k = np.clip(np.searchsorted(x, xf), 0, len(x) - 1)
        xif = (xf - lo) if c["clamp"] == "min" else (hi - xf)
        sface = P * np.clip(Lb - xif, 0, None) * np.abs(ctr[:, la] - cen[k]) / Is[k]
        cav = []
        from scipy import ndimage

        split = [ndimage.label(o)[1] > 1 for o in prof["occupancy"]]
        if any(split):
            cav.append(("há trechos com a seção dividida em pedaços separados (recorte, treliça): a teoria de viga "
                        "supõe que trabalham juntos, mas isso depende de como estão ligados; confira com FEM",
                        "parts of the span have the section split into separate pieces (cut-out, truss): beam "
                        "theory assumes they act together, which depends on how they are tied; check with FEM", 2))
        if gap.any():
            cav.append(("a seção some em algum trecho (peças desconectadas): a viga não transmite carga ali",
                        "the section vanishes somewhere (disconnected parts): the beam carries no load there"))
        if Lb < 3 * max(ext[la], ext[[i for i in range(3) if i not in (ax, la)][0]]):
            cav.append(("viga curta (L < 3 × altura): cisalhamento e efeitos 3D pesam, a teoria de viga subestima a "
                        "deflexão", "short beam (L < 3 × depth): shear and 3-D effects matter, beam theory "
                        "underestimates the deflection"))
        return Evaluation(
            quantities={"tip_deflection": delta, "stiffness": P / delta, "max_stress": smax,
                        "safety_factor": c["yield_stress"] / max(smax, 1e-300), "mass": mass,
                        "stiffness_to_mass": P / delta / mass, "first_frequency": math.sqrt(max(omega2, 0)) / (2 * math.pi)},
            terms={"tip_deflection": mech}, by_part={},
            face_field=sface / 1e6, field_label=("tensão de flexão estimada (MPa)", "estimated bending stress (MPa)"),
            notes=[(f"eixo da viga: {'xyz'[ax]}, carga em {'xyz'[la]}, L = {Lb:.3g} m",
                    f"beam axis: {'xyz'[ax]}, load along {'xyz'[la]}, L = {Lb:.3g} m")],
            caveats=cav, mesh=(V, F), extra={"profile": prof, "xi": xi})


# ----------------------------------------------------------------------------------------------------- custom
class ScalingModel(QualitativeModel):
    """User formulas of the geometric descriptors, e.g. ``ScalingModel({"conduction_resistance": lambda d, c:
    d.length / (c["k"] * d.frontal_area)}, better={"conduction_resistance": "min"}, k=200)``. The mechanisms are the
    descriptors: each one's share of a change comes from swapping it alone (in log space)."""
    name = "scaling"
    title = ("modelo de escala do usuário", "user scaling model")

    def __init__(self, formulas: Mapping[str, Callable], better: Mapping[str, str] | None = None,
                 direction=(1.0, 0.0, 0.0), **conditions):
        super().__init__(**conditions)
        self.formulas = dict(formulas)
        self.direction = direction
        better = dict(better or {})
        self.quantities = {k: Quantity(k, "", better.get(k, "min"), k, k) for k in self.formulas}

    def evaluate(self, geom):
        V, F = as_mesh(geom)
        d = describe((V, F), self.direction)
        vals = {k: float(f(d, self.conditions)) for k, f in self.formulas.items()}
        return Evaluation(quantities=vals, terms={}, by_part={}, face_field=np.zeros(len(F)),
                          field_label=("", ""), mesh=(V, F), extra={"descriptors": d})

    def log_attribution(self, base_geom, var_geom, quantity: str) -> dict[str, float]:
        """Share of ln(q_var / q_base) attributed to each descriptor (one-at-a-time swap from the baseline)."""
        db = describe(base_geom, self.direction)
        dv = describe(var_geom, self.direction)
        f = self.formulas[quantity]
        q0 = f(db, self.conditions)
        out = {}
        for k, v in dv.__dict__.items():
            if not np.isscalar(v) or v == getattr(db, k):
                continue
            trial = describe(base_geom, self.direction)
            setattr(trial, k, v)
            try:
                qk = f(trial, self.conditions)
            except Exception:  # noqa: BLE001 - a formula may not use or accept every descriptor
                continue
            if qk > 0 and q0 > 0 and abs(math.log(qk / q0)) > 1e-12:
                out[k] = math.log(qk / q0)
        return out


MODELS: dict[str, type[QualitativeModel]] = {m.name: m for m in (ExternalFlow, ConvectiveCooling, Cantilever)}
