"""Plate-fin heat-sink designs evaluated several ways at once: the exact solid from NVIDIA USD BRep (volume, area,
mass), PINNeAPPle's closed-form thermal and hydraulic model (``pinneapple_physics.closed_form.plate_fin_heatsink``:
Teertstra forced convection, fin efficiency, spreading resistance, pressure drop, and radiation once an emissivity
and a sky view fraction are given) and, for the questions the network cannot answer -- several chips, a small chip
next to a large one, an off-centre source -- a 3-D finite-volume solve of the base plate with every footprint on it.

Two entry points:

``evaluate_design``
    one design from ``(n_fins, fin_height, fin_t)`` under a :class:`Duty`: the network's hot spot, pressure drop,
    fin efficiency and exact mass. What examples 16 and 18 use.
``evaluate_full``
    one design from a dict of design variables (the three above plus base thickness, material, the film-coefficient
    multiplier and the interface resistance): the same network, the 3-D base solve with all the chips, the per-chip
    hot spots, and the cross-check between the two models. What example 10 uses.

:func:`verify` re-evaluates a design a surrogate proposed with both models before it is accepted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pinneapple_design.thermal.heatsink.field import solve_base_field
from pinneapple_physics.closed_form import plate_fin_heatsink as pfh

from .brep import evaluate_part
from .parts import heat_sink, heat_sink_geometry

__all__ = ["Duty", "Source", "evaluate_design", "evaluate_full", "design_table", "table_rows", "sample_designs",
           "sample_designs_full", "design_matrix", "verify", "FULL_VARS", "FULL_BOUNDS", "checks", "FEATURES"]

FEATURES = ("n_fins", "fin_height", "fin_t")

#: The design variables of the richer study (example 10): the three shape variables, the base thickness, the material
#: (as its conductivity, so a surrogate can interpolate) and the two uncertainty levers an engineer must live with.
FULL_VARS = ("n_fins", "fin_height", "fin_t", "base_t", "k_material", "h_scale", "tim_k_m2_w")
FULL_MATERIALS = tuple(pfh.MATERIALS)
FULL_BOUNDS = {"n_fins": (6.0, 26.0), "fin_height": (0.012, 0.050), "fin_t": (0.0008, 0.0030),
               "base_t": (0.003, 0.010), "h_scale": (0.85, 1.15), "tim_k_m2_w": (5e-6, 2e-5)}


@dataclass(frozen=True)
class Source:
    """A rectangular heat source on the bottom face of the base: centre ``(x, y)`` [m] from the corner, its size [m]
    and its power [W]."""
    x: float
    y: float
    width: float
    depth: float
    power: float


@dataclass
class Duty:
    """The job the heat sink has to do: the sources, the air, the base and the material.

    ``sources`` empty means one centred square source of ``power_w`` over ``source`` (the single-chip case the
    examples 16 and 18 use). ``emissivity`` and ``sky_fraction`` switch radiation on; ``sky_fraction`` is the share
    of the wetted area with an unobstructed view of the room (0 for a fully ducted sink, which is the default and
    what the network assumed before). ``tim_k_m2_w`` is the interface material's ``R''``.
    """
    power_w: float = 60.0
    t_ambient_c: float = 25.0
    air_velocity_m_s: float = 3.0
    source: float = 0.04            # square heat source side [m]
    base_t: float = 0.005
    length: float = 0.1
    width: float = 0.1
    material: str = "al6063"
    root_fillet: float = 0.0        # m, inner fin-root radius of the extrusion (exact solid only)
    sources: tuple[Source, ...] = ()
    emissivity: float = 0.0
    sky_fraction: float = 0.0
    tim_k_m2_w: float = 1.0e-5

    def all_sources(self) -> tuple[Source, ...]:
        if self.sources:
            return tuple(self.sources)
        return (Source(0.5 * self.length, 0.5 * self.width, self.source, self.source, self.power_w),)

    @property
    def total_power_w(self) -> float:
        return float(sum(s.power for s in self.all_sources()))


def _network(n_fins: int, fin_height: float, fin_t: float, duty: Duty, *, base_t: float | None = None,
             material: str | None = None, h_scale: float = 1.0, tim: float | None = None) -> dict:
    """PINNeAPPle's closed-form network for one geometry under ``duty`` (the fins, the spreading, the interface)."""
    g = pfh.HeatSinkGeometry(duty.width, duty.length, duty.base_t if base_t is None else base_t, int(n_fins), fin_t,
                             fin_height, duty.material if material is None else material)
    srcs = duty.all_sources()
    side = math.sqrt(max(s.width * s.depth for s in srcs))          # the network's equivalent footprint: the biggest
    op = pfh.OperatingPoint(duty.total_power_w, duty.t_ambient_c, duty.air_velocity_m_s, side, side,
                            duty.tim_k_m2_w if tim is None else tim)
    return pfh.evaluate(g, op, h_scale=h_scale, emissivity=duty.emissivity, sky_fraction=duty.sky_fraction)


def evaluate_design(n_fins: int, fin_height: float, fin_t: float, duty: Duty | None = None, *, brep: bool = True,
                    chord_tolerances=()) -> dict:
    """One design: closed-form thermal / hydraulic result and (``brep``) the exact solid's volume, area, mass."""
    duty = duty or Duty()
    th = _network(n_fins, fin_height, fin_t, duty)
    cf = heat_sink_geometry(n_fins=int(n_fins), fin_height=fin_height, fin_t=fin_t, base_t=duty.base_t,
                            length=duty.length, width=duty.width, root_fillet=duty.root_fillet)
    out = {"n_fins": int(n_fins), "fin_height": float(fin_height), "fin_t": float(fin_t), "gap": cf["gap"],
           "r_total": th["r_total_k_w"], "t_source": th["t_source_max_c"], "dp": th["pressure_drop_pa"],
           "fin_efficiency": th["fin_efficiency"], "mass_closed_form": th["mass_kg"], "volume_closed_form": cf["volume"],
           "area_closed_form": cf["total_area"], "h_rad": th["h_rad_w_m2k"], "radiation_share": th["radiation_share"],
           "warnings": th["warnings"]}
    if brep:
        rho = pfh.MATERIALS[duty.material]["rho"]
        rep = evaluate_part(heat_sink(n_fins=int(n_fins), fin_height=fin_height, fin_t=fin_t, base_t=duty.base_t,
                                      length=duty.length, width=duty.width, root_fillet=duty.root_fillet),
                            chord_tolerances=list(chord_tolerances),
                            density=rho)
        out.update(volume=rep.volume, area=rep.area, mass=rep.raw["mass"]["mass"], manifold=rep.manifold,
                   faces=rep.raw["faces"], brep_s=rep.seconds, _rep=rep)
    return out


def evaluate_full(design: dict, duty: Duty | None = None, *, brep: bool = False, grid: int = 40) -> dict:
    """One design of the richer study: the network, the 3-D base solve with every chip, and their cross-check.

    The design dict keys are :data:`FULL_VARS` (``material`` may replace ``k_material`` by name). The reported hot
    spot is the *3-D solve's*: the network cannot see a small chip next to a large one, and the solve can. The
    network's value is kept as ``t_source_network`` and the two are compared.
    """
    duty = duty or Duty()
    n_fins = int(design["n_fins"])
    fin_height, fin_t = float(design["fin_height"]), float(design["fin_t"])
    base_t = float(design.get("base_t", duty.base_t))
    material = design.get("material") or _material_for(design.get("k_material"))
    h_scale = float(design.get("h_scale", 1.0))
    tim = float(design.get("tim_k_m2_w", duty.tim_k_m2_w))
    th = _network(n_fins, fin_height, fin_t, duty, base_t=base_t, material=material, h_scale=h_scale, tim=tim)
    srcs = duty.all_sources()
    fvm = solve_base_field(base_width=duty.width, base_depth=duty.length, base_thickness=base_t,
                           k=pfh.MATERIALS[material]["k"], nx=grid, ny=grid,
                           r_fins_k_w=th["resistances_k_w"]["fins_convection"],
                           sources=[{"power_w": s.power, "width": s.width, "depth": s.depth, "x": s.x, "y": s.y}
                                    for s in srcs])
    t_chip = [duty.t_ambient_c + c["theta_max"] + c["power_w"] * tim / (s.width * s.depth)
              for c, s in zip(fvm["sources"], srcs, strict=True)]
    cf = heat_sink_geometry(n_fins=n_fins, fin_height=fin_height, fin_t=fin_t, base_t=base_t, length=duty.length,
                            width=duty.width, root_fillet=duty.root_fillet)
    out = {"n_fins": n_fins, "fin_height": fin_height, "fin_t": fin_t, "base_t": base_t, "material": material,
           "h_scale": h_scale, "tim_k_m2_w": tim, "gap": cf["gap"],
           "t_source_max_c": float(max(t_chip)), "t_chip_c": [float(x) for x in t_chip],
           "rise_k": float(max(t_chip) - duty.t_ambient_c),
           "t_source_network": th["t_source_max_c"], "r_total": th["r_total_k_w"], "dp": th["pressure_drop_pa"],
           "fin_efficiency": th["fin_efficiency"], "mass_closed_form": th["mass_kg"], "mass": th["mass_kg"],
           "volume_closed_form": cf["volume"], "volume": cf["volume"], "area_closed_form": cf["total_area"],
           "h_w_m2k": th["h_w_m2k"], "h_rad": th["h_rad_w_m2k"], "radiation_share": th["radiation_share"],
           "fvm_hotspot_xy_m": fvm["hotspot_xy_m"], "fvm_energy_error": fvm["energy_balance_rel_error"],
           "fvm_grid": fvm["grid"], "warnings": th["warnings"]}
    if brep:
        rho = pfh.MATERIALS[material]["rho"]
        rep = evaluate_part(heat_sink(n_fins=n_fins, fin_height=fin_height, fin_t=fin_t, base_t=base_t,
                                      length=duty.length, width=duty.width, root_fillet=duty.root_fillet),
                            density=rho)
        out.update(volume=rep.volume, area=rep.area, mass=rep.raw["mass"]["mass"], manifold=rep.manifold,
                   faces=rep.raw["faces"], brep_s=rep.seconds)
    return out


def _material_for(k) -> str:
    """The material whose conductivity is nearest ``k`` (so a surrogate can be trained on a continuous variable)."""
    if k is None:
        return "al6063"
    return min(pfh.MATERIALS, key=lambda m: abs(pfh.MATERIALS[m]["k"] - float(k)))


def checks(res: dict, *, t_limit_c: float | None = None, dp_limit_pa: float = 30.0,
           network_band: float = 0.20) -> list[dict]:
    """Engineering checks of one :func:`evaluate_full` result, as ``{name, value, limit, passed, detail, kind}``."""
    out = []

    def add(name, value, limit, detail, passed=None, kind="max"):
        v = float(value)
        out.append({"name": name, "value": v, "limit": float(limit),
                    "passed": bool(v <= limit) if passed is None else bool(passed), "detail": detail, "kind": kind})

    add("3-D base solve closes its own energy balance", res["fvm_energy_error"], 1e-6,
        "heat out of the top face against the chip power, from the finite-volume solve alone")
    if t_limit_c is not None:
        add("hot spot below the limit (3-D solve)", res["t_source_max_c"], t_limit_c,
            "bottom-face maximum over the chips, plus the interface drop")
    add("pressure drop inside the duct budget", res["dp"], dp_limit_pa, "the fan or blower has to make this")
    add("fin efficiency above 50 %", res["fin_efficiency"], 0.5, "extra height buys little below this",
        passed=res["fin_efficiency"] >= 0.5, kind="min")
    add("network and 3-D solve agree", abs(res["t_source_network"] - res["t_source_max_c"])
        / max(res["rise_k"], 1e-9), network_band,
        "relative gap between the closed-form network and the finite-volume base solve: the band is where the two "
        "models of the same physics disagree")
    if res["radiation_share"] > 0:
        add("radiation share reported", res["radiation_share"], 1.0, "share of the surface coefficient that is "
            "radiative; not a limit, only a quantity the answer now carries")
    return out


def design_table(designs, duty: Duty | None = None, *, brep: bool = True) -> list[dict]:
    rows = [evaluate_design(*d, duty or Duty(), brep=brep) for d in designs]
    for r in rows:
        r.pop("_rep", None)
    return rows


def table_rows(designs: list[dict], duty: Duty | None = None, *, brep: bool = False, grid: int = 40) -> list[dict]:
    """``evaluate_full`` over a list of design dicts, with the private keys dropped."""
    rows = []
    for d in designs:
        r = evaluate_full(d, duty or Duty(), brep=brep, grid=grid)
        r.update({k: d[k] for k in FULL_VARS if k in d})
        rows.append(r)
    return rows


def sample_designs(n: int, *, n_range=(6, 20), h_range=(0.015, 0.045), t_range=(0.001, 0.003), min_gap=0.002,
                   width: float = 0.1, seed: int = 0) -> list[tuple[int, float, float]]:
    """Latin-hypercube designs inside the ranges with at least ``min_gap`` between fins."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        u = (rng.permutation(n) + rng.random(n)) / n
        v = (rng.permutation(n) + rng.random(n)) / n
        w = (rng.permutation(n) + rng.random(n)) / n
        for a, b, c in zip(u, v, w, strict=True):
            nf = int(round(n_range[0] + a * (n_range[1] - n_range[0])))
            h = h_range[0] + b * (h_range[1] - h_range[0])
            t = t_range[0] + c * (t_range[1] - t_range[0])
            if (width - nf * t) / (nf - 1) >= min_gap and len(out) < n:
                out.append((nf, h, t))
    return out


def sample_designs_full(n: int, *, seed: int = 0, duty: Duty | None = None, min_gap: float = 0.002) -> list[dict]:
    """Latin-hypercube designs over :data:`FULL_VARS`, with a manufacturable fin gap and the material as an index."""
    duty = duty or Duty()
    keys = [k for k in FULL_VARS if k != "k_material"]
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    while len(out) < n:
        u = (np.column_stack([rng.permutation(n) + rng.random(n) for _ in range(len(keys) + 1)]) + 0.5) / n
        for row in u:
            d = {k: FULL_BOUNDS[k][0] + v * (FULL_BOUNDS[k][1] - FULL_BOUNDS[k][0])
                 for k, v in zip(keys, row[:-1], strict=True)}
            d["n_fins"] = int(round(d["n_fins"]))
            d["material"] = FULL_MATERIALS[min(int(row[-1] * len(FULL_MATERIALS)), len(FULL_MATERIALS) - 1)]
            d["k_material"] = pfh.MATERIALS[d["material"]]["k"]
            if (duty.width - d["n_fins"] * d["fin_t"]) / max(d["n_fins"] - 1, 1) >= min_gap and len(out) < n:
                out.append(d)
    return out


def design_matrix(designs) -> np.ndarray:
    """``(n, 7)`` array of :data:`FULL_VARS`, the material as its conductivity."""
    return np.array([[float(d.get("n_fins", 0)), float(d["fin_height"]), float(d["fin_t"]),
                      float(d.get("base_t", 0.005)),
                      float(d.get("k_material", pfh.MATERIALS[d.get("material", "al6063")]["k"])),
                      float(d.get("h_scale", 1.0)), float(d.get("tim_k_m2_w", 1e-5))] for d in designs])


def verify(n_fins: int, fin_height: float, fin_t: float, duty: Duty | None = None, **kw) -> dict:
    """Closed form + exact solid for a design a surrogate proposed (the evaluation that decides)."""
    return evaluate_design(n_fins, fin_height, fin_t, duty, **kw)


def surrogate_service(net, X_train, *, name: str = "pinneapple-heatsink-surrogate", t_ambient: float = 25.0):
    """A heat-sink surrogate ``(n_fins, fin_height, fin_t) -> (T rise, dp, mass)`` as a PINNeAPPle Model Service.
    Inputs ``{"designs": [[n, H, t], ...]}``; the trust report flags designs outside the training box and outputs
    that break physics (a negative temperature rise, pressure drop or mass)."""
    from pinneapple_tools.model_services.registry import ModelService
    lo, hi = np.min(X_train, 0), np.max(X_train, 0)

    def predict(inputs):
        P = net.predict(np.asarray(inputs["designs"], float))
        return {"t_source": (P[:, 0] + t_ambient).tolist(), "dp": P[:, 1].tolist(), "mass": P[:, 2].tolist()}

    def trust(inputs, out):
        X = np.asarray(inputs["designs"], float)
        inside = np.all((X >= lo) & (X <= hi), 1)
        positive = (np.array(out["t_source"]) > t_ambient) & (np.array(out["dp"]) > 0) & (np.array(out["mass"]) > 0)
        ok = inside & positive
        checks = [{"name": "inside the training box", "passed": bool(inside.all()), "value": float(inside.mean())},
                  {"name": "physical outputs (rise, dp, mass > 0)", "passed": bool(positive.all()),
                   "value": float(positive.mean())}]
        return {"decision": "APPROVED" if ok.all() else "REVIEW", "checks": checks, "per_design": ok.tolist()}

    return ModelService(name, "infer", "heat-sink rating surrogate (PhysicsNeMo FullyConnected)", predict, trust,
                        {"inputs": "designs: [[n_fins, fin_height_m, fin_t_m], ...]", "envelope": [lo.tolist(),
                                                                                              hi.tolist()]})
