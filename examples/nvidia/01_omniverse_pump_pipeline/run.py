"""Pump and pipeline digital twin in OpenUSD for Omniverse: valve closure (water hammer) checked by PINNeAPPle.

A fixed-speed pump lifts water through a 2 km, DN600 steel main into a tank; a valve at the end closes. PINNeAPPle
finds the operating point (pump curve = Colebrook friction + valve loss), solves the transient by the method of
characteristics, verifies the solver (Joukowsky head rise, wave period 4L/a, steady state kept), and rates each
closure time against the PN16 pipe class. The twin is written as an OpenUSD stage: the pipe as a mesh with the
pressure keyed over time (the Omniverse timeline plays the surge), pump, valve and three pressure sensors with
their own time series, and the trust decision on the pipeline prim. No PhysicsNeMo: this is Omniverse only.

    python examples/nvidia/01_omniverse_pump_pipeline/run.py            # about 1 min on a CPU
    python examples/nvidia/01_omniverse_pump_pipeline/run.py --smoke
"""
import argparse
import json
import os

import numpy as np

from pinneapple_lab import experiment
from pinneapple_systems.process_components import water_hammer as wh
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure, curves_figure
from pinneapple_tools.nvidia.omniverse import (
    FieldStage,
    kit_render_script,
    render_stage_gif,
    render_stage_png,
    tube,
    validate_stage,
)
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = wh.Pipeline(length=2000.0, diameter=0.6, wave_speed=1150.0, roughness=4.5e-5)
PUMP = wh.PumpCurve(shutoff_head=70.0, k=120.0)
TANK_HEAD = 20.0
VALVE_K = 5.0
PN16 = 16e5                    # Pa, allowable gauge pressure of the PN16 class
CLOSURES = (0.0, 6.0, 20.0, 60.0)


def layout(n):
    """Pipe route in the scene (metres, z up): along x with two bends; arc length = LINE.length."""
    s = np.linspace(0, 1, n)
    pts = np.c_[1400 * s, 300 * np.sin(np.pi * s) ** 2, 10 * s]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return pts * (LINE.length / seg[-1]), seg * (LINE.length / seg[-1])


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("01_omniverse_pump_pipeline")
    architecture_diagram(
        "01 Pump and pipeline twin: PINNeAPPle + Omniverse (OpenUSD)",
        [Block("op", "Operating point\npump curve = Colebrook + valve", "pinneapple", 0, 0),
         Block("moc", "Water hammer, method of\ncharacteristics (MOC)", "pinneapple", 1, 0),
         Block("ver", "Solver verification: Joukowsky,\nperiod 4L/a, steady state", "pinneapple", 1, 1),
         Block("rate", "Pressure class check (PN16)\nper closure time -> decision", "pinneapple", 2, 1),
         Block("usd", "OpenUSD twin: pipe mesh with\np(x, t) time samples, sensors", "omniverse", 2, 0),
         Block("val", "UsdValidation + read-back\nframes (GIF)", "omniverse", 3, 0),
         Block("kit", "Omniverse Kit timeline / RTX,\nlive sensor feed (Nucleus)", "omniverse", 3, 1, ran=False)],
        [("op", "moc", "Q0, H(x)"), ("moc", "ver", "H(x, t)"), ("ver", "rate", "verified"), ("moc", "usd", "p(x, t)"),
         ("rate", "usd", "decision"), ("usd", "val", ".usda"), ("usd", "kit", ".usda")],
        os.path.join(OUT, "architecture.png"),
        subtitle="All physics in PINNeAPPle; the twin is an OpenUSD stage Omniverse plays on its timeline")
    with experiment("examples/nvidia/01_omniverse_pump_pipeline", title="Pump and pipeline water hammer twin "
                    "in OpenUSD", domain="hydraulics", methods=["MOC", "Colebrook"], out_dir=base, register=False,
                    modules=["pinneapple_systems.process_components.water_hammer", "pinneapple_tools.nvidia.omniverse"],
                    params={"length_m": LINE.length, "diameter_m": LINE.diameter, "wave_speed_m_s": LINE.wave_speed,
                            "closures_s": list(CLOSURES)}) as ex:
        n_reaches = 20 if smoke else 80
        t_end = 12.0 if smoke else 90.0
        op = wh.operating_point(LINE, PUMP, valve_k=VALVE_K, tank_head=TANK_HEAD)
        ex.log_metric("flow_m3_s", op.flow, "m^3/s", "pump curve = friction + valve loss (brentq)")
        ex.log_metric("velocity_m_s", op.velocity, "m/s", "Q / A at the operating point")
        checks = []
        with info.step("verification (Joukowsky, period, steady state)"):
            ref = wh.moc_valve_closure(LINE, PUMP, op, closure_time=0.0, t_end=4 * wh.wave_period(LINE.length,
                                       LINE.wave_speed), n_reaches=n_reaches, friction=False)
            dh = ref.head[:, -1].max() - op.valve_head
            jk = wh.joukowsky_head_rise(LINE.wave_speed, op.velocity)
            h = ref.head[:, -1] - op.valve_head
            cross = ref.t[np.where(np.diff(np.sign(h)) != 0)[0]]
            period = 2 * float(np.mean(np.diff(cross)))
            still = wh.moc_valve_closure(LINE, PUMP, op, closure_time=1.0, closure_start=1e9, t_end=20.0,
                                         n_reaches=n_reaches)
            drift = float(np.abs(still.head - still.head[0]).max())
            checks += [{"name": "Joukowsky head rise a V0 / g (instant closure, no friction)", "value": abs(dh - jk) / jk,
                        "limit": 1e-6, "passed": abs(dh - jk) / jk < 1e-6, "detail": f"{dh:.3f} m vs {jk:.3f} m"},
                       {"name": "wave period 4L/a", "value": abs(period - 4 * LINE.length / LINE.wave_speed)
                        / (4 * LINE.length / LINE.wave_speed), "limit": 0.01,
                        "passed": abs(period / (4 * LINE.length / LINE.wave_speed) - 1) < 0.01},
                       {"name": "steady state kept (valve open, 20 s)", "value": drift, "limit": 1e-8,
                        "passed": drift < 1e-8, "detail": "max |H(t) - H(0)| [m]"},
                       {"name": "Courant number = 1", "value": abs(ref.courant - 1), "limit": 1e-12,
                        "passed": abs(ref.courant - 1) < 1e-12}]
        runs = {}
        with info.step("transients (MOC, 4 closure times)"):
            for tc in CLOSURES:
                runs[tc] = wh.moc_valve_closure(LINE, PUMP, op, closure_time=tc, t_end=t_end, n_reaches=n_reaches)
        pmax = {tc: float(r.pressure(LINE).max()) for tc, r in runs.items()}
        pmin = {tc: float(r.pressure(LINE).min()) for tc, r in runs.items()}
        for tc in CLOSURES:
            checks.append({"name": f"max pressure <= PN16, closure {tc:g} s", "value": pmax[tc], "limit": PN16,
                           "passed": pmax[tc] <= PN16, "detail": f"{pmax[tc] / 1e5:.1f} bar"})
            checks.append({"name": f"no column separation, closure {tc:g} s", "value": -pmin[tc] / 1e5,
                           "limit": 0.9, "passed": pmin[tc] > -0.9e5,
                           "detail": f"min gauge {pmin[tc] / 1e5:.2f} bar (vapour at about -0.98 bar)"})
            ex.log_metric(f"max_pressure_bar_closure_{tc:g}s", pmax[tc] / 1e5, "bar", "max over pipe and time, MOC")
        verified = all(c["passed"] for c in checks[:4])
        safe = [tc for tc in CLOSURES if pmax[tc] <= PN16 and pmin[tc] > -0.9e5]
        decision = "APPROVED" if verified and safe else "REJECT"
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        checks_figure(checks, os.path.join(OUT, "checks.png"), title="Solver verification and PN16 rating",
                      decision=f"{decision}; safe closure times: {', '.join(f'{t:g} s' for t in safe) or 'none'}")
        tt = {tc: r.t for tc, r in runs.items()}
        curves_figure([{"title": "Pressure at the valve", "xlabel": "time [s]", "ylabel": "gauge pressure [bar]",
                        "series": [(f"closure {tc:g} s", tt[tc], runs[tc].pressure(LINE)[:, -1] / 1e5)
                                   for tc in CLOSURES], "hlines": [(16.0, "PN16", "r")]},
                       {"title": "Envelope along the pipe", "xlabel": "x [m]", "ylabel": "gauge pressure [bar]",
                        "series": sum([[(f"max, {tc:g} s", r.x, r.pressure(LINE).max(0) / 1e5),
                                        (f"min, {tc:g} s", r.x, r.pressure(LINE).min(0) / 1e5, ":")]
                                       for tc, r in runs.items()], []), "hlines": [(16.0, "PN16", "r")]},
                       {"title": "Verification: instant closure, no friction", "xlabel": "time [s]",
                        "ylabel": "head rise at valve [m]", "series": [("MOC", ref.t, h)],
                        "hlines": [(jk, "Joukowsky a V0 / g", "r")]}],
                      os.path.join(OUT, "transients.png"),
                      suptitle=f"Q0 = {op.flow:.3f} m3/s, V0 = {op.velocity:.2f} m/s, 2L/a = "
                               f"{2 * LINE.length / LINE.wave_speed:.2f} s")
        # the twin: the 20 s closure (first safe one, else the slowest) keyed on the Omniverse timeline
        show = safe[0] if safe else CLOSURES[-1]
        r = runs[show]
        with info.step("OpenUSD twin"):
            route, s = layout(n_reaches + 1)
            V, F = tube(route, radius=12.0, n_around=10)            # drawn 40x thicker than DN600 so it reads
            ring = lambda p: np.repeat(p, 10)                       # noqa: E731
            idx = np.unique(np.linspace(0, len(r.t) - 1, 12 if smoke else 90).astype(int))
            p_bar = r.pressure(LINE) / 1e5
            usd = os.path.join(OUT, "pipeline_twin.usda")
            st = FieldStage(usd, "Pump and pipeline water hammer twin", fps=24)
            pipe = st.add_mesh("Pipeline", V, F, fields={"pressure": ring(p_bar[0]), "head": ring(r.head[0])},
                               units={"pressure": "bar", "head": "m"}, color_by="pressure", cmap="turbo",
                               vrange=(float(p_bar.min()), float(p_bar.max())))
            st.add_time_samples(pipe, "pressure", r.t[idx], [ring(p_bar[i]) for i in idx], cmap="turbo",
                                vrange=(float(p_bar.min()), float(p_bar.max())))
            pv, pf = tube(np.array([route[0] - [60, 0, 0], route[0]]), 45.0, 16)
            st.add_mesh("Pump", pv, pf, color=(0.15, 0.35, 0.8))
            vv, vf = tube(np.array([route[-1], route[-1] + [50, 0, 0]]), 30.0, 4)
            st.add_mesh("Valve", vv, vf, color=(0.85, 0.2, 0.15))
            k = [0, len(s) // 2, len(s) - 1]
            sens = st.add_points("PressureSensors", route[k] + [0, 0, 30], width=25.0,
                                 fields={"pressure": p_bar[0, k]}, units={"pressure": "bar"})
            st.add_time_samples(sens, "pressure", r.t[idx], [p_bar[i, k] for i in idx], color=False)
            st.set_trust(pipe, decision, checks, closure_time_s=show, pressure_class="PN16",
                         solver="pinneapple MOC")
            st.save()
            v = validate_stage(usd, required_primvars={pipe: ["pressure", "head"], sens: ["pressure"]})
            fps = st.stage.GetTimeCodesPerSecond()
            peak = int(np.argmax(p_bar[idx].max(1)))
            render_stage_png(usd, os.path.join(OUT, "usd_preview.png"), time=float(r.t[idx][peak] * fps),
                             elev=60, azim=-95, flat=True, colorbar=("gauge pressure [bar]", float(p_bar.min()),
                                                          float(p_bar.max()), "turbo"),
                             title=f"pipeline_twin.usda read back, t = {r.t[idx][peak]:.1f} s (peak), closure {show:g} s")
            gi = idx[:: max(1, len(idx) // 24)]
            render_stage_gif(usd, os.path.join(OUT, "usd_timeline.gif"), [float(t * fps) for t in r.t[gi]],
                             label=lambda t: f"pipeline_twin.usda, t = {t / fps:5.1f} s", elev=60, azim=-95, flat=True,
                             colorbar=("gauge pressure [bar]", float(p_bar.min()), float(p_bar.max()), "turbo"),
                             size=(6, 4.4), dpi=80)
            with open(os.path.join(OUT, "kit_render.py"), "w") as f:
                f.write(kit_render_script(usd, os.path.join(OUT, "pipeline_rtx.png")))
        ex.check("usd_stage_valid", v["ok"], detail="; ".join(v["problems"]) or "UsdValidation clean")
        info.mark("OpenUSD stage (usd-core)", ran=True, how=f"{v['time_samples']} time samples, validated")
        info.mark("Omniverse Kit timeline / RTX render", ran=False, why="needs Omniverse Kit and an RTX GPU")
        summary = {"decision": decision, "safe_closure_times_s": safe, "operating_point": op.__dict__,
                   "max_pressure_bar": {f"{k:g}": pmax[k] / 1e5 for k in CLOSURES},
                   "min_pressure_bar": {f"{k:g}": pmin[k] / 1e5 for k in CLOSURES}, "checks": checks,
                   "usd": {k: v[k] for k in ("ok", "problems", "meshes", "points", "time_samples")}}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "safe_closure_times_s", "max_pressure_bar")}, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
