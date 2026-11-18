"""A liquid-cooled cold plate: parallel microchannels under a chip set, the conjugate solid/coolant model, and the
hydraulic state that says whether the model is inside its correlations.

Geometry (metres; the plate lies in ``x``-``y``, the coolant flows along ``x``)
    ``n_channels`` slots of width ``channel_width`` and depth ``channel_depth`` are milled into a plate
    ``length`` x ``width`` x ``base_t``. The metal left under a slot is the fin the heat crosses (``base_t -
    channel_depth``); the wall between two slots is ``pitch - channel_width`` of full metal. Chips sit on the top
    face as squares with their own size and power.

Physics
    Per unit plate area, with the thickness integrated out::

        -div(G grad T) + sigma (T - T_f) = q''
        G     = k_metal (base_t - depth) + k_eff depth      [W/K]   (k_eff = k_metal over a wall, k_coolant over a slot)
        sigma = h (1 + 2 depth / width)                     [W/m^2 K]

    ``G`` is the thickness-integrated in-plane conductance: full metal over a wall, the fin plus the water over a
    slot -- the water conducts poorly, so a slot is a band of low ``G``. ``sigma`` is the surface coupling of a slot
    to the coolant over its base and its two side walls (the lid is insulated). ``T_f`` is the mean coolant
    temperature, ``T_f = T_in + dT_f / 2`` with ``dT_f = Q_coolant / (m_dot c_p)`` from the coolant's own energy
    balance -- the unknown the whole problem is closed on. The outer walls of the plate are adiabatic (a guard ring
    of low conductance sits between them and the Dirichlet boundary the finite volumes need, see :func:`fields`), so
    the coolant carries all of the chip power and that balance is itself a check.
    Because the plate problem is linear in ``T - T_in``, the coupling is an affine fixed point and is solved in
    closed form from two solves instead of by iteration.

    Coolant side, per channel: laminar duct flow. ``h`` from the fully developed Nusselt number of a slot with three
    adiabatic walls and one at uniform heat flux (``Nu_Dh = 5.385``, Shah & London 1978) with Hausen's thermal-entry
    correction (``Nu = Nu_fd + 0.0668 Gz / (1 + 0.04 Gz^(2/3))``, Incropera eq. 8.86); water properties from
    Incropera & DeWitt Table A.6, interpolated; pressure drop from the Muzychka-Yovanovich apparent-friction model
    with the same rectangular-duct ``f Re`` fit and the Kays & London contraction / expansion losses that
    ``pinneapple_physics.closed_form.plate_fin_heatsink`` uses.

    The model reports its limits instead of hiding them (:attr:`ChannelState.warnings`): turbulent channels, a
    channel width-to-depth ratio where a slot's Nusselt number no longer holds, a base temperature close to
    saturation, and a developing length outside the correlation.

What the examples use
    :func:`fields` returns the dimensionless ``a``, ``s``, ``f`` of exactly the ``-div(a grad u) + s u = f`` that
    ``pinneapple_physics.darcy_checks`` certifies (and ``operator_benchmarks.darcy_reaction_2d`` solves), so a
    surrogate's temperature field is checked against the conjugate problem rather than against a conduction plate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pinneapple_physics.closed_form import plate_fin_heatsink as pfh
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob

__all__ = ["WATER_K", "NU_SLOT", "RE_LAMINAR_MAX", "water_properties", "Chip", "Coolant", "ColdPlate",
           "ChannelState", "fields", "hydraulics", "reacting_solve", "solve", "checks", "sample_designs",
           "design_matrix", "plate_for", "DESIGN_VARS", "BOUNDS"]

#: Water at 1 atm (Incropera & DeWitt, Table A.6): T [C], k [W/m K], rho [kg/m3], cp [J/kg K], mu [Pa s], Pr [-].
_WATER_T = (20.0, 30.0, 40.0, 50.0, 60.0)
_WATER_K = (0.598, 0.615, 0.631, 0.644, 0.654)
_WATER_RHO = (998.0, 996.0, 993.0, 988.0, 983.0)
_WATER_CP = (4182.0, 4178.0, 4180.0, 4182.0, 4185.0)
_WATER_MU = (1.002e-3, 7.98e-4, 6.53e-4, 5.47e-4, 4.67e-4)
_WATER_PR = (7.01, 5.42, 4.32, 3.57, 2.99)

#: Thermal conductivity of the coolant sitting in a slot [W/m K] (water at 20-60 C, Table A.6): what a channel cell
#: conducts in-plane once the metal is removed.
WATER_K = 0.63

#: Fully developed Nusselt number of a duct with three adiabatic walls and one at uniform heat flux (Shah & London,
#: *Laminar Flow Forced Convection in Ducts*, 1978, Table 2). The cold plate's slots are milled and lidded, so this
#: is the limit the model uses; it is the conservative side up to a width/depth of about 0.5 and optimistic beyond
#: (reported in :attr:`ChannelState.warnings`).
NU_SLOT = 5.385

#: Reynolds number above which a duct's flow is no longer laminar; the correlations above stop applying there.
RE_LAMINAR_MAX = 2300.0


def _interp(x: float, xs, ys) -> float:
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return float(ys[i - 1] + f * (ys[i] - ys[i - 1]))
    return float(ys[-1])


def water_properties(t_c: float) -> dict[str, float]:
    """Water at 1 atm, linearly interpolated in Incropera & DeWitt Table A.6 (clamped to 20-60 C)."""
    return {"k": _interp(t_c, _WATER_T, _WATER_K), "rho": _interp(t_c, _WATER_T, _WATER_RHO),
            "cp": _interp(t_c, _WATER_T, _WATER_CP), "mu": _interp(t_c, _WATER_T, _WATER_MU),
            "pr": _interp(t_c, _WATER_T, _WATER_PR)}


@dataclass(frozen=True)
class Chip:
    """A square heat source on the top face: centre ``(x, y)`` [m] from the plate corner, ``size`` [m], ``power`` [W]."""
    x: float
    y: float
    size: float
    power: float


@dataclass
class Coolant:
    """Water at ``t_inlet_c`` with ``flow_lpm`` litres per minute through all channels together.

    ``h_scale`` multiplies the correlation's heat-transfer coefficient: use it to propagate the 15-25 % spread these
    correlations show (``0.8`` pessimistic, ``1.2`` optimistic)."""
    t_inlet_c: float = 40.0
    flow_lpm: float = 1.0
    h_scale: float = 1.0


@dataclass
class ColdPlate:
    length: float = 0.100
    width: float = 0.100
    base_t: float = 0.003
    n_channels: int = 20
    channel_width: float = 0.0008
    channel_depth: float = 0.0015
    material: str = "cu110"
    chips: tuple[Chip, ...] = (Chip(0.030, 0.050, 0.014, 110.0), Chip(0.072, 0.050, 0.014, 110.0))
    coolant: Coolant = field(default_factory=Coolant)

    @property
    def pitch(self) -> float:
        return self.width / max(self.n_channels, 1)

    @property
    def wall_t(self) -> float:
        return self.pitch - self.channel_width

    @property
    def total_power(self) -> float:
        return float(sum(c.power for c in self.chips))

    def geometry_errors(self) -> list[str]:
        errs = []
        if self.material not in pfh.MATERIALS:
            errs.append(f"unknown material '{self.material}' (use one of {sorted(pfh.MATERIALS)})")
        if self.n_channels < 1:
            errs.append("need at least one channel")
        if min(self.length, self.width, self.base_t, self.channel_width, self.channel_depth) <= 0:
            errs.append("all dimensions must be positive")
        elif self.channel_depth >= self.base_t:
            errs.append("the channel is deeper than the plate (no fin left under the slot)")
        elif self.wall_t <= 0:
            errs.append(f"channels do not fit: {self.n_channels} x {self.channel_width:.4g} m exceeds the width")
        if self.coolant.flow_lpm <= 0:
            errs.append("the flow must be positive")
        if not self.chips:
            errs.append("no heat source")
        for c in self.chips:
            if c.size <= 0 or c.power < 0:
                errs.append("chips need a positive size and a non-negative power")
            elif c.x - c.size / 2 < 0 or c.x + c.size / 2 > self.length or c.y - c.size / 2 < 0 \
                    or c.y + c.size / 2 > self.width:
                errs.append(f"chip at ({c.x:.3g}, {c.y:.3g}) hangs over the plate")
        return errs


@dataclass
class ChannelState:
    """The coolant side of one design: duct state, heat-transfer coefficient, pressure drop and limits."""
    n_channels: int
    width: float
    depth: float
    dh: float
    area: float
    velocity: float
    flow_m3_s: float
    m_dot_cp: float
    re: float
    pr: float
    gz: float
    nu: float
    h: float
    dp: float
    pumping_w: float
    t_out_c: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def hydraulics(cfg: ColdPlate) -> ChannelState:
    """Duct state, ``h`` (with ``h_scale``), pressure drop and pumping power of one design's channels."""
    errs = cfg.geometry_errors()
    if errs:
        raise ValueError("; ".join(errs))
    w, d, L = cfg.channel_width, cfg.channel_depth, cfg.length
    area = w * d
    dh = 4 * area / (2 * (w + d))
    flow = cfg.coolant.flow_lpm * 1e-3 / 60.0
    water = water_properties(cfg.coolant.t_inlet_c)
    velocity = (flow / cfg.n_channels) / area
    re = water["rho"] * velocity * dh / water["mu"]
    pr = water["pr"]
    gz = dh * re * pr / L
    nu = NU_SLOT + 0.0668 * gz / (1.0 + 0.04 * gz ** (2.0 / 3.0))          # Hausen on the Shah & London value
    h = cfg.coolant.h_scale * nu * water["k"] / dh

    asp = min(w, d) / max(w, d)
    f_re = 24 * (1 - 1.3553 * asp + 1.9467 * asp ** 2 - 1.7012 * asp ** 3 + 0.9564 * asp ** 4 - 0.2537 * asp ** 5)
    l_plus = L / (dh * re)
    f_app = math.sqrt((3.44 / math.sqrt(l_plus)) ** 2 + f_re ** 2) / re
    sigma_flow = cfg.n_channels * area / (cfg.width * d)                    # open-area ratio at the manifold
    kc, ke = 0.42 * (1 - sigma_flow ** 2), (1 - sigma_flow ** 2) ** 2
    dp = (4 * f_app * L / dh + kc + ke) * water["rho"] * velocity ** 2 / 2
    m_dot_cp = flow * water["rho"] * water["cp"]

    warn: list[str] = []
    if re > RE_LAMINAR_MAX:
        warn.append(f"channel Reynolds {re:.0f} > {RE_LAMINAR_MAX:.0f}: the flow may be turbulent and the laminar "
                    "correlations are then optimistic on dp and conservative on h")
    if not 0.1 <= gz <= 1000:
        warn.append(f"Graetz number {gz:.3g} outside 0.1-1000: the thermal-entry correction is outside the range it "
                    "was fitted on")
    if w / d > 1.5:
        warn.append(f"channel width/depth {w / d:.2f} > 1.5: Nu = 5.385 is the slot limit (three heated walls give "
                    "more), so h is conservative")
    if w / d < 0.1:
        warn.append(f"channel width/depth {w / d:.2f} < 0.1: the fin under the slot is very wide and thin")
    if h < 200:
        warn.append(f"h = {h:.0f} W/m2 K is low for liquid cooling: check the flow")
    return ChannelState(n_channels=cfg.n_channels, width=w, depth=d, dh=dh, area=area, velocity=velocity,
                        flow_m3_s=flow, m_dot_cp=m_dot_cp, re=re, pr=pr, gz=gz, nu=nu, h=h, dp=dp,
                        pumping_w=dp * flow, warnings=warn)


def _slot_fraction(n: int, length: float, n_channels: int, width: float, channel_width: float) -> np.ndarray:
    """Fraction of each ``y`` cell covered by a slot, ``(n,)``: the slots are narrower than a cell for the
    geometries a cold plate uses, so the property fields are built from the exact overlap rather than from a binary
    mask (which would make ``a`` and ``sigma`` jump with the grid)."""
    dy = width / n
    edges = np.arange(n + 1) * dy
    lo, hi = edges[:-1][:, None], edges[1:][:, None]
    centres = (np.arange(n_channels) + 0.5) * (width / n_channels)
    c_lo, c_hi = (centres - channel_width / 2)[None, :], (centres + channel_width / 2)[None, :]
    overlap = np.clip(np.minimum(hi, c_hi) - np.maximum(lo, c_lo), 0.0, None)
    return overlap.sum(axis=1) / dy


def fields(cfg: ColdPlate, n: int, state: ChannelState | None = None, *, guard: int = 2,
           guard_a: float = 1e-3) -> dict:
    """The dimensionless problem of one design on an ``n x n`` cell-centred grid of the plate.

    Returns ``a`` (in-plane conductance / the solid's), ``s`` (the slot-to-coolant coupling, dimensionless),
    ``f0`` (the chip flux / its peak, before the coolant's own temperature rise is added), the ``channel`` mask,
    ``sigma`` [W/m2 K], ``G`` [W/K], ``q`` [W/m2] and the scales ``G0``, ``q0`` and ``DT`` (kelvin per unit of the
    dimensionless solution), so that ``T = T_in + DT u`` and

        -div(a grad u) + s (u - theta_f) = f0        theta_f = dT_f / (2 DT)

    is the conjugate problem.

    The plate occupies the inner ``n - 2 guard`` cells; the ``guard`` ring around it is a band of ``guard_a`` times
    the solid conductance and no sink, which is how the Dirichlet solver of :func:`reacting_solve` gives the outer
    walls the adiabatic condition a cold plate's housing really has (without it a quarter of the chip power would
    leave through the manifold rim, and the hot spot would be optimistic by several kelvin).
    """
    st = state or hydraulics(cfg)
    L, W = cfg.length, cfg.width
    m = n - 2 * guard                                  # cells across the plate itself
    if m < 6:
        raise ValueError(f"a {n} x {n} grid leaves {m} cells for the plate with a guard of {guard}")
    c = (np.arange(m) + 0.5) / m
    X = c[:, None] * L                                 # x along the channels
    Y = c[None, :] * W                                 # y across them
    frac = np.broadcast_to(_slot_fraction(m, W, cfg.n_channels, W, cfg.channel_width)[None, :], (m, m))
    k_m = pfh.MATERIALS[cfg.material]["k"]
    fin_t = cfg.base_t - cfg.channel_depth
    G0 = k_m * cfg.base_t
    G = k_m * fin_t + (frac * k_m + (1.0 - frac) * WATER_K) * cfg.channel_depth
    sigma = st.h * (1 + 2 * cfg.channel_depth / cfg.channel_width) * frac
    q = np.zeros((m, m))
    dx, dy = L / m, W / m
    for ch in cfg.chips:                       # lumped on the cells of its footprint, so the grid carries exactly its
        inside = (np.abs(X - ch.x) <= ch.size / 2) & (np.abs(Y - ch.y) <= ch.size / 2)   # power, whatever n is
        q = q + np.where(inside, ch.power / (max(int(inside.sum()), 1) * dx * dy), 0.0)
    q0 = float(q.max())

    a = np.full((n, n), guard_a)
    s = np.zeros((n, n))
    f = np.zeros((n, n))
    inner = (slice(guard, guard + m), slice(guard, guard + m))
    Ls = L * n / m                                     # the length the unit square stands for, given the guard ring
    a[inner] = G / G0
    s[inner] = sigma * Ls ** 2 / G0
    f[inner] = q / q0
    full_frac = np.zeros((n, n))
    full_frac[inner] = frac
    return {"a": a, "s": s, "f0": f, "channel": full_frac > 0.5, "frac": full_frac, "sigma": sigma, "G": G, "q": q,
            "G0": G0, "q0": q0, "DT": q0 * Ls ** 2 / G0, "n": n, "guard": guard, "inner": inner, "Ls": Ls,
            "X": X, "Y": Y, "state": st}


def reacting_solve(a: np.ndarray, s: np.ndarray, f: np.ndarray) -> np.ndarray:
    """``-div(a grad u) + s u = f`` on the grid, with one exact reporting of what the solve consumed."""
    return ob.darcy_reaction_2d(a, s, f)


def solve(cfg: ColdPlate, n: int = 64, *, tol: float = 1e-10, max_iter: int = 20) -> dict:
    """The conjugate steady state of one design: the dimensionless field, the coolant state, and the energy balance.

    The coupling is the coolant's own temperature rise, ``dT_f = Q_coolant / (m_dot c_p)``, where ``Q_coolant`` is
    the heat the slots hand over -- itself the sink integral of the solution. Because the plate problem is linear in
    the offset, the fixed point ``theta_f = g(theta_f)`` is affine and solved in closed form from two solves
    (``theta_f = g(0) / (1 - (g(1) - g(0)))``), then confirmed by a third; ``max_iter`` covers the degenerate case
    where the slope is one.
    """
    fl = fields(cfg, n)
    a, s, f0 = fl["a"], fl["s"], fl["f0"]
    G0, DT, st = fl["G0"], fl["DT"], fl["state"]
    h2 = 1.0 / n ** 2

    def image(theta: float) -> tuple[float, np.ndarray, float]:
        u = reacting_solve(a, s, f0 + s * theta)
        q_coolant = G0 * DT * h2 * float(np.sum(s * (u - theta)))
        return q_coolant / st.m_dot_cp / (2 * DT), u, q_coolant

    theta0, _, _ = image(0.0)
    theta1, _, _ = image(1.0)
    slope = theta1 - theta0
    theta_f, iters, converged = theta0, 2, False
    if abs(1.0 - slope) > 1e-12:
        theta_f = theta0 / (1.0 - slope)                # the closed-form fixed point of an affine map
        converged = True
    else:                                               # the slope is one to round-off: fall back to under-relaxation
        while iters < max_iter:
            iters += 1
            new, _, _ = image(theta_f)
            if abs(new - theta_f) <= tol * max(abs(theta_f), 1e-12):
                converged = True
                break
            theta_f = 0.5 * (theta_f + new)
    _, u, q_coolant = image(theta_f)
    out_bnd = float(2 * (np.sum(a[0, :] * u[0, :]) + np.sum(a[-1, :] * u[-1, :])
                         + np.sum(a[:, 0] * u[:, 0]) + np.sum(a[:, -1] * u[:, -1])))
    q_edges = G0 * DT * out_bnd          # out_bnd is already the boundary integral (sum of 2 a u over the faces)
    dt_f = 2 * theta_f * DT
    T = cfg.coolant.t_inlet_c + DT * u
    over_slot = (fl["frac"] > 0) & (fl["frac"] >= fl["frac"].max() * 0.5)
    st.t_out_c = cfg.coolant.t_inlet_c + dt_f
    return {"u": u, "T": T, "fields": fl, "state": st, "theta_f": theta_f, "dt_f": dt_f,
            "t_max_c": float(T.max()), "t_min_c": float(T.min()),
            "t_base_max_c": float(np.where(over_slot, T, -np.inf).max()), "q_coolant": q_coolant,
            "q_edges": q_edges, "q_total": cfg.total_power,
            "energy_closure": abs(q_coolant + q_edges - cfg.total_power) / max(cfg.total_power, 1e-9),
            "n_iter": iters, "converged": converged, "n": n, "peak_chip_flux": fl["q0"]}


def checks(res: dict, *, t_limit_c: float = 85.0, dp_budget_pa: float = 20000.0, t_coolant_limit_c: float = 90.0,
           sat_margin_c: float = 5.0, t_sat_c: float = 100.0) -> list[dict]:
    """The engineering checks of one solved design, as ``{name, value, limit, passed, detail, kind}`` dicts."""
    st = res["state"]
    out = []

    def add(name, value, limit, detail, kind="max", passed=None):
        v = float(value) if value is not None else float("nan")
        out.append({"name": name, "value": v, "limit": float(limit),
                    "passed": bool(v <= limit) if passed is None else bool(passed), "detail": detail, "kind": kind})

    add("hot spot below the limit", res["t_max_c"], t_limit_c,
        f"max T over the plate with h = {st.h:.0f} W/m2 K in the slots (h_scale applied)")
    add("coolant outlet below the fluid limit", st.t_out_c, t_coolant_limit_c,
        "the coolant's own energy balance; water at 1 atm boils at 100 C")
    add("base temperature away from saturation", res["t_base_max_c"], t_sat_c - sat_margin_c,
        f"max metal temperature over a slot; {sat_margin_c:.0f} K below saturation at 1 atm")
    add("coolant Reynolds number laminar", st.re, 2300.0,
        "the Shah & London / Hausen correlations are laminar")
    add("Graetz number inside the fit", st.gz, 1000.0, "thermal-entry correction range")
    add("energy closure of the conjugate solution", res["energy_closure"], 0.02,
        "|heat to the coolant + heat to the manifolds - chip power| / chip power")
    add("pressure drop inside the pump budget", st.dp, dp_budget_pa, f"pumping power {st.pumping_w:.3g} W")
    add("coolant temperature rise", res["dt_f"], 30.0, "m_dot c_p against the chip power")
    return out


# --------------------------------------------------------------------------------------------- design space
#: The design variables the examples sample: channel count, slot width, flow, and the film-coefficient multiplier.
DESIGN_VARS = ("n_channels", "channel_width", "flow_lpm", "h_scale")
BOUNDS = {"n_channels": (10.0, 30.0), "channel_width": (0.0004, 0.0012), "flow_lpm": (0.50, 2.00),
          "h_scale": (0.80, 1.20)}


def design_matrix(designs) -> np.ndarray:
    """``(n, 4)`` array of the design variables, in :data:`DESIGN_VARS` order."""
    return np.array([[float(d[k]) for k in DESIGN_VARS] for d in designs], dtype=np.float64)


def sample_designs(n: int, *, seed: int = 0, min_wall: float = 6e-4, duty: ColdPlate | None = None) -> list[dict]:
    """Latin-hypercube designs inside :data:`BOUNDS` that keep a manufacturable wall (``>= min_wall``) between slots."""
    duty = duty or ColdPlate()
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    while len(out) < n:
        u = (np.column_stack([rng.permutation(n) + rng.random(n) for _ in DESIGN_VARS]) + 0.5) / n
        for row in u:
            d = {k: BOUNDS[k][0] + v * (BOUNDS[k][1] - BOUNDS[k][0]) for k, v in zip(DESIGN_VARS, row, strict=True)}
            d["n_channels"] = int(round(d["n_channels"]))
            d["channel_width"] = round(d["channel_width"], 6)
            d["flow_lpm"] = round(d["flow_lpm"], 4)
            d["h_scale"] = round(d["h_scale"], 4)
            if (duty.width / d["n_channels"] - d["channel_width"]) >= min_wall and len(out) < n:
                out.append(d)
    return out


def plate_for(design: dict, duty: ColdPlate | None = None) -> ColdPlate:
    """The :class:`ColdPlate` of one design, with the duty (plate, chips, coolant inlet) of ``duty``."""
    base = duty or ColdPlate()
    return ColdPlate(length=base.length, width=base.width, base_t=base.base_t,
                     n_channels=int(design["n_channels"]), channel_width=float(design["channel_width"]),
                     channel_depth=base.channel_depth, material=base.material, chips=base.chips,
                     coolant=Coolant(t_inlet_c=base.coolant.t_inlet_c, flow_lpm=float(design["flow_lpm"]),
                                     h_scale=float(design["h_scale"])))
