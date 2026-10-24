"""A mission in plain language -> the rover's lab loop, through a fixed set of tools.

Stages (the Autonomous Physics Lab loop): **mission** (parse the request) -> **plan** (which tools, in which order)
-> **build** (site, rover, beliefs from the orbital map) -> **simulate** (drive the planned route with the multibody
rover on the true terrain) -> **learn** (measurements -> soil belief) -> **optimize** (replan on the new belief) ->
**verify** (trust report against the physics and, on a synthetic site, the known truth) -> **deliver** (report).

The request is turned into a :class:`MissionSpec` by a parser with the ``extract(text, schema) -> dict`` interface of
:class:`pinneapple_design.intent.parse.CallLLM`: pass ``llm=CallLLM(...)`` to use a language model (its answer is
checked against the text: every value needs a quote found in the request), otherwise :class:`RuleParser` does it
deterministically. Either way the agent only calls the tools of :data:`TOOLS`, with arguments it validated.

    from pinneapple_simulation.rover.autolab import demo_site, RoverMission
    m = RoverMission(demo_site(), "Drive the rover to (37, 15) without getting stuck, slip at most 40 %, "
                                  "minimum energy")
    result = m.run(out_dir="out")        # result.report (TrustReport), result.routes, result.log
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SCHEMA = {"goal_xy": "[x, y] in metres", "start_xy": "[x, y] in metres or null", "slip_limit": "fraction 0-1",
          "risk_max": "fraction 0-1", "objective": "energy | distance", "speed_m_s": "m/s",
          "quotes": {"<field>": "exact words of the request the value comes from"}}


@dataclass
class MissionSpec:
    text: str
    goal: tuple
    start: tuple
    slip_limit: float = 0.4
    risk_max: float = 0.05
    objective: str = "energy"
    speed: float = 0.2
    max_replans: int = 2
    parser: str = "rules"
    quotes: dict = field(default_factory=dict)
    defaults_used: list = field(default_factory=list)


_NUM = r"[-+]?\d+(?:[.,]\d+)?"


def _f(s: str) -> float:
    return float(s.replace(",", "."))


class RuleParser:
    """Deterministic parser with the ``extract(text, schema)`` interface of an LLM adapter (English and
    Portuguese keywords)."""

    def extract(self, text: str, schema: dict | None = None) -> dict[str, Any]:
        low = text.lower()
        out: dict[str, Any] = {"quotes": {}}
        pts = list(re.finditer(rf"\(\s*({_NUM})\s*[,;]\s*({_NUM})\s*\)", text))
        goal_kw = re.search(r"\b(to|até|ate|para|goal|alvo|destino|ponto)\b", low)
        start_kw = re.search(r"\b(from|de|partindo|start|início|inicio)\b", low)
        if len(pts) >= 2 and start_kw and start_kw.start() < pts[0].start():
            out["start_xy"] = [_f(pts[0].group(1)), _f(pts[0].group(2))]
            out["quotes"]["start_xy"] = pts[0].group(0)
            pts = pts[1:]
        if pts and (goal_kw or len(pts) == 1):
            out["goal_xy"] = [_f(pts[-1].group(1)), _f(pts[-1].group(2))]
            out["quotes"]["goal_xy"] = pts[-1].group(0)
        m = re.search(rf"(slip|escorregamento|patinagem)[^.\d%]*?({_NUM})\s*(%|percent|por cento)?", low)
        if m:
            v = _f(m.group(2))
            out["slip_limit"] = v / 100 if (m.group(3) or v > 1) else v
            out["quotes"]["slip_limit"] = m.group(0)
        m = re.search(rf"(risk|risco)[^.\d%]*?({_NUM})\s*(%)?", low)
        if m:
            v = _f(m.group(2))
            out["risk_max"] = v / 100 if (m.group(3) or v > 1) else v
            out["quotes"]["risk_max"] = m.group(0)
        m = re.search(rf"({_NUM})\s*(m/s|cm/s)", low)
        if m:
            out["speed_m_s"] = _f(m.group(1)) * (0.01 if m.group(2) == "cm/s" else 1.0)
            out["quotes"]["speed_m_s"] = m.group(0)
        for kw, obj in (("energ", "energy"), ("distân", "distance"), ("distanc", "distance"), ("shortest", "distance"),
                        ("mais curt", "distance")):
            if kw in low:
                out["objective"] = obj
                out["quotes"]["objective"] = kw
                break
        return out


def parse_mission(text: str, site, llm=None) -> MissionSpec:
    """Request -> :class:`MissionSpec`. Values an LLM returns are kept only when their quote is in the text;
    missing values take the site's start / goal or the defaults (listed in ``defaults_used``)."""
    rules = RuleParser().extract(text, SCHEMA)
    got, parser = dict(rules), "rules"
    if llm is not None:
        try:
            ans = dict(llm.extract(text, SCHEMA) or {})
            q = ans.get("quotes", {}) or {}
            kept = {k: v for k, v in ans.items() if k != "quotes" and isinstance(q.get(k), str)
                    and q[k].strip() and q[k].lower() in text.lower()}
            got.update(kept)
            got.setdefault("quotes", {}).update({k: q[k] for k in kept})
            parser = "llm+rules"
        except Exception as e:  # noqa: BLE001 - an unreachable model falls back to the rules
            parser = f"rules (llm failed: {type(e).__name__})"
    spec = MissionSpec(text=text, goal=tuple(site.goal), start=tuple(site.start), parser=parser,
                       quotes=got.get("quotes", {}))
    if "goal_xy" in got:
        spec.goal = tuple(float(v) for v in got["goal_xy"])
    else:
        spec.defaults_used.append("goal = site goal")
    if got.get("start_xy"):
        spec.start = tuple(float(v) for v in got["start_xy"])
    else:
        spec.defaults_used.append("start = site start")
    for key, attr, lo, hi in (("slip_limit", "slip_limit", 0.05, 0.95), ("risk_max", "risk_max", 0.0, 0.5),
                              ("speed_m_s", "speed", 0.02, 0.5)):
        if key in got:
            setattr(spec, attr, float(np.clip(float(got[key]), lo, hi)))
        else:
            spec.defaults_used.append(f"{attr} = {getattr(spec, attr)} (default)")
    if got.get("objective") in ("energy", "distance"):
        spec.objective = got["objective"]
    for p in (spec.start, spec.goal):
        if not (site.x_range[0] <= p[0] <= site.x_range[1] and site.y_range[0] <= p[1] <= site.y_range[1]):
            raise ValueError(f"point {p} is outside the site {site.x_range} x {site.y_range}")
    return spec


# ============================================================================================ tools
TOOLS = {
    "survey_site": "orbital map -> soil units and the prior belief per unit",
    "plan_route": "risk-aware shortest path on the current belief (energy per metre + risk)",
    "drive": "multibody VIPER on the true terrain along a route, stopped by the traction monitor",
    "estimate_soil": "rover measurements -> iterated-EKF update of the soil belief per unit",
    "compare_models": "analytic (prior / learned) vs zoo surrogate vs measured wheel operating points",
    "verify": "trust report: goal, slip limit, physics checks, model agreement, recovery of the known truth",
}


@dataclass
class StepLog:
    stage: str
    tool: str
    summary: dict
    seconds: float


@dataclass
class MissionResult:
    spec: MissionSpec
    routes: list
    runs: list
    belief: Any
    planners: list
    measurements: list
    comparison: dict
    report: Any
    log: list
    retreat: tuple | None = None
    out_dir: str | None = None
    numbers: dict = field(default_factory=dict)


def traction_monitor(slip_limit: float, window_s: float = 2.0, output_dt: float = 0.05, front=None,
                     t_min: float = 2.0):
    """Stop rule for :meth:`VehicleSimulator.run`: the mean slip of the (front) wheels over the last ``window_s``
    seconds above ``slip_limit`` ends the run as "stalled". Samples before ``t_min`` seconds of driving (the start-up
    ramp) are not monitored. ``output_dt`` is not used (kept for compatibility): the window is in time."""
    hist: list = []

    def stop(sample):
        t = float(sample.get("t", t_min))
        if t < t_min:
            return None
        s = np.asarray(sample["slip"], float)
        hist.append((t, float(s[front].mean() if front is not None else s.mean())))
        if hist[-1][0] - hist[0][0] < window_s - 1e-9:
            return None
        m = float(np.mean([v for tt, v in hist if tt > t - window_s + 1e-9]))
        if m > slip_limit:
            return f"stalled: mean slip {m:.2f} over {window_s:g} s > limit {slip_limit:g}"
        return None
    return stop


class RoverMission:
    """The rover lab loop on ``site`` for one request (see the module docstring).

    ``fast=True`` shortens everything for tests (coarse grids, few belief samples, a short drive budget)."""

    def __init__(self, site, text: str, *, llm=None, spec=None, dt: float = 4e-3, terrain_cell: float = 0.03,
                 planner_cell: float = 0.5, n_samples: int = 24, sensors=None, max_time: float = 400.0,
                 fast: bool = False, progress=None):
        from ...vehicle import build_vehicle, viper
        self.site = site
        self.spec = spec or parse_mission(text, site, llm=llm)
        self.vehicle = viper()
        self.model = build_vehicle(self.vehicle)
        self.mass = float(self.model.tree.total_mass())
        self.site_g = 1.62                                  # lunar gravity, as the simulator's default
        self.wheel = self.vehicle.wheel
        self.n_wheels = len(self.vehicle.corners)
        self.front = [i for i, c in enumerate(self.vehicle.corners) if c.front > 0]
        self.dt, self.terrain_cell, self.planner_cell = dt, terrain_cell, planner_cell
        self.n_samples, self.max_time, self.fast = n_samples, max_time, fast
        from .estimation import Sensors
        self.sensors = sensors or Sensors()
        self.progress = progress
        self.log: list[StepLog] = []

    def _step(self, stage, tool, fn, *a, **kw):
        if tool not in TOOLS:
            raise KeyError(f"tool {tool!r} is not an allowed tool")
        t0 = time.perf_counter()
        out, summary = fn(*a, **kw)
        self.log.append(StepLog(stage, tool, summary, time.perf_counter() - t0))
        if self.progress:
            self.progress(stage, tool, summary)
        return out

    # ------------------------------------------------------------------ tools
    def survey_site(self):
        from .estimation import SoilBelief
        bel = SoilBelief([u.orbital for u in self.site.units], self.wheel)
        units = [{"unit": u.name, "orbital_soil": u.orbital.name, "note": u.orbital_note} for u in self.site.units]
        return bel, {"units": units}

    def plan_route(self, belief, start, label):
        from .planner import RiskPlanner
        pl = RiskPlanner(self.site, belief, self.wheel, self.mass, slip_limit=self.spec.slip_limit,
                         n_wheels=self.n_wheels, cell=self.planner_cell, n_samples=self.n_samples,
                         risk_max=self.spec.risk_max, objective=self.spec.objective,
                         half_track=0.5 * self.vehicle.track(), half_width=0.5 * self.vehicle.track() + 0.15)
        route = pl.plan(start, self.spec.goal, label)
        return (route, pl), {"label": label, "length_m": route.length_m, "predicted_energy_J": route.energy_J,
                             "max_risk": route.max_risk, "predicted_max_slip": route.max_slip,
                             "waypoints": np.round(route.waypoints, 2).tolist()}

    def drive(self, route, label):
        from ...vehicle import VehicleSimulator
        ter = self.site.deformable_terrain(cell=self.terrain_cell)
        sim = VehicleSimulator(self.model, ter, self.site.true_soils(), soil_map=self.site.unit_at,
                               wheel_speeds="kinematic")
        stop = traction_monitor(self.spec.slip_limit, front=self.front)
        t0 = time.perf_counter()
        run = sim.run(self.max_time, speed=self.spec.speed, dt=self.dt, output_dt=0.05, route=route.waypoints,
                      stop=stop, record_stresses=False, settle=1.0, ramp=1.0)
        run.meta["wall_s"] = time.perf_counter() - t0
        run.meta["label"] = label
        run.meta["terrain"] = ter
        s = run.summary()
        reached = run.meta["ended"] == "goal"
        return run, {"label": label, "ended": run.meta["ended"], "reached_goal": reached,
                     "distance_m": s["distance_m"], "energy_J": s["energy_J"], "max_slip": s["max_slip"],
                     "max_sinkage_mm": s["max_sinkage_mm"], "sim_time_s": float(run.t[-1]),
                     "wall_s": run.meta["wall_s"]}

    def estimate_soil(self, belief, runs):
        from .estimation import measure
        meas = [measure(r, self.sensors) for r in runs]
        allm = {k: np.concatenate([m[k] for m in meas]) for k in meas[0] if isinstance(meas[0][k], np.ndarray)}
        allm["slip_sd"] = meas[0]["slip_sd"]
        # hold out the last quarter of each unit's rows for a predictive check
        held = {}
        train = np.ones(len(allm["t"]), bool)
        for u in np.unique(allm["unit"]):
            r = np.flatnonzero(allm["unit"] == u)
            r = r[np.argsort(allm["t"][r], kind="stable")]
            k = r[int(0.75 * len(r)):]
            if len(r) >= 8:
                held[int(u)] = k
                train[k] = False
        sub = {k: (v[train] if isinstance(v, np.ndarray) else v) for k, v in allm.items()}
        belief.assimilate(sub)
        checks = {u: belief.predictive_check(u, allm, rows) for u, rows in held.items()}
        summ = {"rows": int(train.sum()), "held_out": {int(u): int(len(r)) for u, r in held.items()},
                "units": {self.site.units[u].name: {"n_obs": belief.units[u].n_obs,
                                                     "mean": dict(zip(_P, np.round(belief.units[u].mean, 4).tolist(),
                                                                      strict=True)),
                                                     "sd": dict(zip(_P, np.round(belief.units[u].sd, 4).tolist(),
                                                                    strict=True))}
                          for u in range(len(self.site.units))},
                "predictive_check": {self.site.units[u].name: c for u, c in checks.items()}}
        return (allm, checks), summ

    def compare_models(self, meas, belief):
        from ...terramechanics.surrogate import ZooContactSurrogate
        from .wheel_models import compare_contact_models
        try:
            import pinneapple as pp
            zoo = ZooContactSurrogate(pp.zoo.load("rover_contact_grc1_viper"))
            sur, inr = zoo.predict, zoo.in_range
        except Exception as e:  # noqa: BLE001 - the comparison still runs without the zoo model
            sur, inr, zoo = None, None, f"unavailable: {type(e).__name__}: {e}"
        out = {}
        for u in np.unique(meas["unit"]):
            r = meas["unit"] == u
            m = {k: meas[k][r] for k in ("load", "slip", "sinkage", "torque")}
            name = self.site.units[int(u)].name
            prior = compare_contact_models(m, self.site.prior_soil(int(u)), self.wheel, surrogate=sur, in_range=inr)
            post = compare_contact_models(m, belief.units[int(u)].soil(), self.wheel)
            out[name] = {"n": prior["n"], "analytic_prior": prior["analytic"], "analytic_learned": post["analytic"],
                         "surrogate_zoo": prior.get("surrogate"), "measured": m}
        brief = {n: {k: ({q: round(v, 3) for q, v in d.items() if not isinstance(v, dict)} if isinstance(d, dict)
                         else d) for k, d in c.items() if k not in ("measured", "n")} for n, c in out.items()}
        out["_surrogate"] = zoo if isinstance(zoo, str) else zoo.meta["name"]
        return out, brief

    # ------------------------------------------------------------------ the loop
    def plan(self) -> list[tuple[str, str]]:
        """The tool sequence the agent will run (replanning repeats plan_route/drive/estimate_soil)."""
        return [("build", "survey_site"), ("plan", "plan_route"), ("simulate", "drive"), ("learn", "estimate_soil"),
                ("learn", "compare_models"), ("optimize", "plan_route"), ("simulate", "drive"),
                ("verify", "verify")]

    def run(self, out_dir: str | None = None) -> MissionResult:
        sp = self.spec
        self.log.append(StepLog("mission", "parse", {k: v for k, v in asdict(sp).items() if k != "text"}, 0.0))
        self.log.append(StepLog("plan", "agent", {"steps": self.plan(), "tools": TOOLS}, 0.0))
        belief = self._step("build", "survey_site", self.survey_site)
        routes, runs, planners, meas_all = [], [], [], []
        start = sp.start
        comparison: dict = {}
        retreat = None
        for attempt in range(sp.max_replans + 1):
            label = "orbital prior" if attempt == 0 else f"learned belief #{attempt}"
            route, pl = self._step("plan" if attempt == 0 else "optimize", "plan_route", self.plan_route, belief,
                                   start, label)
            routes.append(route)
            planners.append(pl)
            run = self._step("simulate", "drive", self.drive, route, label)
            runs.append(run)
            if run.meta["ended"] == "goal":
                break
            meas, _ = self._step("learn", "estimate_soil", self.estimate_soil, belief, runs)
            meas_all.append(meas)
            comparison = self._step("learn", "compare_models", self.compare_models, meas, belief)
            retreat = retreat_point(run, self.site)
            start = retreat
        from .verify import verify_mission
        report = self._step("verify", "verify", verify_mission, self, routes, runs, planners, belief, meas_all,
                            comparison)
        res = MissionResult(sp, routes, runs, belief, planners, meas_all, comparison, report, self.log, retreat,
                            out_dir)
        if out_dir:
            from .report import write_report
            write_report(self, res, out_dir)
        return res


_P = ("log10_k_phi", "n0", "n1", "phi_deg")


def retreat_point(run, site, back: float = 2.0) -> tuple:
    """Where the rover backs out to after a stall: ``back`` metres before the first point where a front wheel
    entered the unit it stalled on, on its own track (the reversal itself is not simulated)."""
    W = run.wheel
    front = [i for i, c in enumerate(run.model.spec.corners) if c.front > 0]
    u_end = int(np.bincount(W["soil_idx"][-1, front]).argmax())
    entered = np.flatnonzero((W["soil_idx"][:, front] == u_end).any(1))
    k = int(entered[0]) if len(entered) else len(run.t) - 1
    d = run.distance()
    j = int(np.searchsorted(d, max(d[k] - back, 0.0)))
    return float(run.q[j, 0]), float(run.q[j, 1])


def to_json(obj):
    """JSON-safe copy of mission outputs (arrays to lists, numbers rounded)."""
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        if isinstance(o, (np.floating, float)):
            return None if not math.isfinite(float(o)) else round(float(o), 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    return json.loads(json.dumps(conv(obj), default=str))


__all__ = ["MissionSpec", "RuleParser", "parse_mission", "RoverMission", "MissionResult", "TOOLS", "SCHEMA",
           "traction_monitor", "retreat_point", "to_json"]
