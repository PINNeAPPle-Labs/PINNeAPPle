"""Maintenance (gunning / repair) scheduling over the forecast. Decision support only.

Port of ``digital-solutions/shared/optimizer.js`` so both sides give the same numbers.

Model (explicit and auditable):
* each zone has a ``deadline`` (x until its critical cell reaches the minimum thickness; 0 = already there);
* an intervention at ``T`` restores ``recovery_frac`` of the usable thickness, capped at the design
  thickness: ``new_deadline = T + min(margin(T) + ext, life_full)``, ``ext = recovery_frac * life_full``;
* several zones can be served in one stop (one setup cost);
* *breach* = x operated with some zone beyond its limit.

Three policies are simulated: run-to-limit (``reactive``), fixed calendar (``fixed``) and ``predictive``
grouped stops (greedy with a search over the grouping window). The recommendation is the cheapest
policy with no breach (or, if none, the one with the least breach).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

from .forecast import ZoneForecast


@dataclass(frozen=True)
class OptimizerConfig:
    horizon: float = 300.0
    heats_per_day: float = 24.0
    recovery_frac: float = 0.45
    setup_hours: float = 6.0
    hours_per_zone: float = 3.0
    cost_per_hour: float = 1.0
    cost_per_zone: float = 4.0
    breach_cost_per_x: float = 50.0
    safety_margin: float = 10.0
    fixed_interval: float = 150.0
    windows: Sequence[float] = (0, 10, 20, 40, 60, 90, 130)


@dataclass
class ZoneState:
    zone: str
    deadline: float
    life_full: float
    ext: float


@dataclass
class Policy:
    id: str
    name: str
    schedule: List[dict]
    stops: int = 0
    zones_served: int = 0
    downtime_hours: float = 0.0
    breach: float = 0.0
    cost: float = 0.0
    window: Optional[float] = None


@dataclass
class OptimizationResult:
    cfg: OptimizerConfig
    zones: List[ZoneState]
    policies: List[Policy]
    recommended: Optional[Policy]
    baseline: Optional[Policy]
    notes: List[str] = field(default_factory=list)


def prepare(zones: Sequence[ZoneForecast], cfg: OptimizerConfig) -> List[ZoneState]:
    out = []
    for z in zones:
        if not math.isfinite(z.min_remaining) or not z.rate > 0:
            continue
        life = z.usable / z.rate
        out.append(ZoneState(z.zone, max(0.0, z.min_remaining), life, cfg.recovery_frac * life))
    return out


def simulate(zs: Sequence[ZoneState], stops: Sequence[dict], cfg: OptimizerConfig) -> dict:
    st = {z.zone: dict(deadline=z.deadline, life_full=z.life_full, ext=z.ext, breach=0.0) for z in zs}
    ordered = sorted(stops, key=lambda s: s["t"])
    for stop in ordered:
        for name in stop["zones"]:
            z = st.get(name)
            if z is None:
                continue
            z["breach"] += max(0.0, stop["t"] - z["deadline"])
            margin = max(z["deadline"] - stop["t"], 0.0)
            z["deadline"] = stop["t"] + min(margin + z["ext"], z["life_full"])
    breach = sum(z["breach"] + max(0.0, cfg.horizon - max(z["deadline"], 0.0)) for z in st.values())
    hours = sum(cfg.setup_hours + cfg.hours_per_zone * len(s["zones"]) for s in ordered)
    served = sum(len(s["zones"]) for s in ordered)
    cost = hours * cfg.cost_per_hour + served * cfg.cost_per_zone + breach * cfg.breach_cost_per_x
    return dict(stops=len(ordered), zones_served=served, downtime_hours=hours, breach=breach, cost=cost)


def fixed_policy(zs, cfg) -> List[dict]:
    out, t = [], cfg.fixed_interval
    while t < cfg.horizon:
        out.append({"t": t, "zones": [z.zone for z in zs]})
        t += cfg.fixed_interval
    return out


def predictive_policy(zs, cfg, window: float) -> List[dict]:
    st = [replace(z) for z in zs]
    stops: List[dict] = []
    for _ in range(500):
        nxt = min(st, key=lambda z: z.deadline)
        t = max(0.0, nxt.deadline - cfg.safety_margin)
        if not math.isfinite(nxt.deadline) or t >= cfg.horizon:
            break
        group = [z for z in st if z.deadline - cfg.safety_margin <= t + window]
        stops.append({"t": round(t, 1), "zones": [z.zone for z in group], "due": {z.zone: z.deadline for z in group}})
        for z in group:
            margin = max(z.deadline - t, 0.0)
            z.deadline = t + min(margin + z.ext, z.life_full)
    return stops


def optimize(forecasts: Dict[str, ZoneForecast] | Sequence[ZoneForecast], cfg: Optional[OptimizerConfig] = None) -> OptimizationResult:
    cfg = cfg or OptimizerConfig()
    zf = list(forecasts.values()) if isinstance(forecasts, dict) else list(forecasts)
    zs = prepare(zf, cfg)
    res = OptimizationResult(cfg, zs, [], None, None)
    if not zs:
        res.notes.append("No zone with a wear trend: nothing to schedule.")
        return res

    def pol(pid, name, stops, window=None):
        return Policy(pid, name, stops, window=window, **{k: v for k, v in simulate(zs, stops, cfg).items()})

    res.policies.append(pol("reactive", "Current: run to limit", []))
    res.policies.append(pol("fixed", f"Fixed calendar ({cfg.fixed_interval:g})", fixed_policy(zs, cfg)))
    best = None
    for w in cfg.windows:
        p = pol("predictive", "Predictive grouped", predictive_policy(zs, cfg, w), window=w)
        if best is None or p.cost < best.cost - 1e-9 or (abs(p.cost - best.cost) < 1e-9 and p.stops < best.stops):
            best = p
    best.name = f"Predictive grouped (window {best.window:g})"
    res.policies.append(best)
    feasible = [p for p in res.policies if p.breach == 0]
    pool, key = (feasible, lambda p: p.cost) if feasible else (res.policies, lambda p: p.breach)
    res.recommended = min(pool, key=key)  # first minimal => reactive wins ties, as in the JS reduce
    res.baseline = res.policies[0]
    res.notes.append("Assumptions are configurable; costs are relative. The recommendation is decision support.")
    return res


def explain(res: OptimizationResult) -> List[dict]:
    rec = res.recommended
    if rec is None:
        return []
    dl = {z.zone: z.deadline for z in res.zones}
    out = []
    for k, s in enumerate(rec.schedule):
        due = s.get("due", dl)
        parts = [f"{n}: " + ("already at the limit - immediate intervention" if due[n] <= 0
                             else f"limit expected in ~{round(due[n])}") for n in s["zones"]]
        tail = (f" Keeps a margin of {res.cfg.safety_margin:g}." if s["t"] > 0 else "")
        out.append({"n": k + 1, "t": s["t"], "zones": s["zones"],
                    "why": "; ".join(parts) + f". The stop groups {len(s['zones'])} zone(s) in one setup." + tail})
    return out
