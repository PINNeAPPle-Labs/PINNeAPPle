"""One USD scene for Omniverse and the robot sim: the arm is authored once as UsdPhysics, simulated by PINNeAPPle
Robot Sim (NVIDIA Newton) from that same file, and the recorded motion goes back to OpenUSD for the Omniverse
timeline, with the link stress PINNeAPPle computes from the joint-torque sensors keyed on each link.

    python examples/nvidia/12_omniverse_robotsim_arm/run.py           # ~1 min on a CPU

Checks: the stage passes the UsdPhysics parser and validators; Newton loads it with the masses, centres of mass and
inertias PINNeAPPle wrote, and its motion from the USD equals its motion from the URDF of the same arm; the tracked
motion keeps the energy balance; the link stresses stay below yield / 3; the animated stage validates and reads back.
The same stage and torque trajectory drive Isaac Sim through ``IsaacSimBackend`` (script written, not run here).
"""
import argparse
import json
import os

import numpy as np

from pinneapple_lab import experiment
from pinneapple_simulation.multibody import arm as mb
from pinneapple_simulation.robot_sim import IsaacSimBackend, RobotSim, TwoLinkArm, pd_tracking
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure, curves_figure
from pinneapple_tools.nvidia.isaac import author_articulation, validate_physics
from pinneapple_tools.nvidia.omniverse import (
    FieldStage,
    extrude_profile,
    field_colors,
    kit_render_script,
    render_stage_gif,
    render_stage_png,
    validate_stage,
)
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
Q_START, Q_END, DURATION = (-0.6, 1.2), (0.9, -0.4), 0.8
KP, KD = (350.0, 110.0), (5.0, 2.5)


def box_mesh(size, offset):
    """A box as a closed triangle mesh (the visual of a link in its own frame)."""
    a, b, c = (s / 2 for s in size)
    V, F = extrude_profile(np.array([[-b, -c], [b, -c], [b, c], [-b, c]]), 2 * a, n=2, axis=0)
    V[:, 0] -= a
    caps = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]])
    return V + np.asarray(offset, float), np.r_[F, caps]


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("12_omniverse_robotsim_arm")
    architecture_diagram(
        "12 One scene for Omniverse and the robot sim: PINNeAPPle + Omniverse + PINNeAPPle Robot Sim",
        [Block("arm", "TwoLinkArm: masses and\ninertias (6061-T6, payload)", "pinneapple", 0, 0),
         Block("usd", "UsdPhysics stage: bodies,\njoints, drives, scene", "omniverse", 1, 0),
         Block("val", "UsdPhysics parser +\nvalidators", "omniverse", 1, 1),
         Block("newton", "Robot Sim on NVIDIA Newton\nloads the same .usda", "robotsim", 2, 0),
         Block("same", "Same arm from URDF:\nidentical motion?", "pinneapple", 2, 1),
         Block("chk", "Energy balance, link stress\nfrom joint-torque sensors", "pinneapple", 3, 1),
         Block("anim", "Timeline USD: link poses +\nstress colour per frame", "omniverse", 3, 0),
         Block("kit", "Omniverse Kit RTX /\nIsaac Sim replay", "omniverse", 4, 0, ran=False)],
        [("arm", "usd", "spec"), ("usd", "val", ".usda"), ("usd", "newton", ".usda"), ("newton", "same", "q(t)"),
         ("newton", "chk", "tau(t)"), ("newton", "anim", "poses"), ("chk", "anim", "stress"), ("anim", "kit", ".usda")],
        os.path.join(OUT, "architecture.png"),
        subtitle="The file Omniverse opens is the file the simulator ran; PINNeAPPle checks both ends")
    a = TwoLinkArm()
    tree, sec, props = a.tree(), a.section(), a.material_props()
    plan = lambda t: tuple(-x for x in mb.quintic(Q_START, Q_END, DURATION, np.atleast_1d(t)))   # noqa: E731
    feed = lambda t: -mb.fixed_base_inverse_dynamics(tree, *mb.quintic(Q_START, Q_END, DURATION, [t]))[0]  # noqa: E731
    ref = lambda t: tuple(x[0] for x in plan(t)[:2])                                                     # noqa: E731
    q0 = plan(0.0)[0][0]
    with experiment("examples/nvidia/12_omniverse_robotsim_arm", title="Arm: one USD scene for Omniverse and the "
                    "PINNeAPPle robot sim (Newton)", domain="robotics", methods=["Newton Featherstone", "OpenUSD"],
                    out_dir=base, register=False, modules=["pinneapple_simulation.robot_sim",
                                                           "pinneapple_tools.nvidia.isaac",
                                                           "pinneapple_tools.nvidia.omniverse"],
                    params={"payload_kg": a.payload, "kp": KP, "kd": KD}) as ex:
        with info.step("UsdPhysics stage"):
            phys = os.path.join(OUT, "arm_physics.usda")
            st = FieldStage(phys, "Two-link arm (UsdPhysics) for Omniverse, Isaac Sim and the robot sim")
            robot = author_articulation(st, a.spec())
            st.save()
            vp = validate_physics(phys, expect_bodies=3, expect_joints=2)
        dt = 1e-3 if smoke else 2.5e-4
        with info.step("robot sim from the USD and from the URDF"):
            su = RobotSim.from_usd(phys, ground=False)
            sr = RobotSim.from_spec(a.spec(), ground=False)
            mu, mr = su.mass_properties(), sr.mass_properties()
            mp_diff = max(float(np.abs(mu[k] - mr[k]).max()) for k in mu)
            ctrl = pd_tracking(ref, KP, KD, feedforward=feed)
            ru = su.run(DURATION, dt, q0=q0, controller=ctrl, record_every=int(round(1e-3 / dt)) or 1)
            rr = sr.run(DURATION, dt, q0=q0, controller=ctrl, record_every=int(round(1e-3 / dt)) or 1)
            same = float(np.abs(ru.q - rr.q).max())
        track = float(np.abs(ru.q - plan(ru.t)[0]).max())
        E = ru.energy()["total"]
        W = ru.actuator_work()
        eb = float(np.abs((E - E[0]) - W).max() / max(np.abs(W).max(), 1e-12))
        sig, s_checks = mb.link_root_checks(ru.tau, sec, yield_strength=props["yield_strength"], safety=3.0,
                                            names=["shoulder", "elbow"])
        util = sig * 3.0 / props["yield_strength"]
        with info.step("timeline USD (poses + stress)"):
            anim = os.path.join(OUT, "arm_timeline.usda")
            ts = FieldStage(anim, "Arm motion from the robot sim, link stress from PINNeAPPle", fps=60)
            idx = np.unique(np.linspace(0, len(ru.t) - 1, 17 if smoke else 49).astype(int))
            sizes = {L.name: (L.size, L.box_offset) for L in a.spec().links}
            for i, name in enumerate(["base", "link1", "link2"]):
                V, F = box_mesh(*sizes[name])
                cols = None if i == 0 else field_colors(util[idx, i - 1], "RdYlGn_r", (0.0, 1.0))
                ts.add_rigid_animation(name, V, F, ru.t[idx], ru.body_pos[idx, i], ru.body_quat[idx, i],
                                       color=(0.25, 0.25, 0.3), colors_over_time=cols)
            ts.set_trust("/World", "APPROVED" if all(c["passed"] for c in s_checks) else "REJECT", s_checks,
                         simulator="NVIDIA Newton (PINNeAPPle Robot Sim)", physics_stage="arm_physics.usda")
            ts.save()
            va = validate_stage(anim)
            fps = ts.stage.GetTimeCodesPerSecond()
            render_stage_png(phys, os.path.join(OUT, "usd_physics_preview.png"), elev=15, azim=-60,
                             exclude=("/World/Ground",), title="arm_physics.usda (UsdPhysics, authored pose)")
            render_stage_gif(anim, os.path.join(OUT, "usd_timeline.gif"), [float(t * fps) for t in ru.t[idx]],
                             label=lambda t: f"arm_timeline.usda, t = {t / fps:.2f} s", elev=8, azim=-90,
                             colorbar=("link root stress x 3 / yield", 0.0, 1.0, "RdYlGn_r"), size=(6, 4.6), dpi=80)
            peak = int(np.argmax(util[idx].max(1)))
            render_stage_png(anim, os.path.join(OUT, "usd_timeline_peak.png"), time=float(ru.t[idx][peak] * fps),
                             elev=8, azim=-90, colorbar=("link root stress x 3 / yield", 0.0, 1.0, "RdYlGn_r"),
                             title=f"arm_timeline.usda read back at t = {ru.t[idx][peak]:.2f} s (peak stress)")
            with open(os.path.join(OUT, "kit_render.py"), "w") as f:
                f.write(kit_render_script(anim, os.path.join(OUT, "arm_rtx.png")))
            np.savez(os.path.join(OUT, "arm_trajectory.npz"), t=ru.t, q=ru.q, tau=ru.tau)
            isaac = IsaacSimBackend(phys, robot)
            isaac.write_script(os.path.join(OUT, "arm_trajectory.npz"), os.path.join(OUT, "isaac_efforts.npz"),
                               os.path.join(OUT, "isaac_replay.py"))
        checks = [{"name": "UsdPhysics stage valid (parser + validators)", "value": None, "limit": None,
                   "passed": vp["ok"], "detail": "; ".join(vp["problems"]) or str(vp["physics"])},
                  {"name": "Newton reads the USD masses / com / inertia exactly", "value": mp_diff, "limit": 1e-6,
                   "passed": mp_diff <= 1e-6},
                  {"name": "same motion from USD and URDF (rad)", "value": same, "limit": 1e-9, "passed": same <= 1e-9},
                  {"name": "tracking error (rad)", "value": track, "limit": 2e-3, "passed": track <= 2e-3},
                  {"name": "energy balance: actuator work = dE", "value": eb, "limit": 1e-2, "passed": eb <= 1e-2},
                  *s_checks,
                  {"name": "timeline stage valid", "value": None, "limit": None, "passed": va["ok"],
                   "detail": f"{va['time_samples']} time samples" if va["ok"] else "; ".join(va["problems"])}]
        decision = "APPROVED" if all(c["passed"] for c in checks) else "REJECT"
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        ex.log_metric("usd_vs_urdf_motion", same, "rad", "max |q_USD - q_URDF| over the move, Newton")
        ex.log_metric("tracking_error_mrad", track * 1e3, "mrad", "max |q - q_plan|, Newton from the USD")
        info.mark("NVIDIA Newton (robot sim) from USD", ran=True)
        info.mark("Omniverse Kit RTX / Isaac Sim replay", ran=False, why="need Omniverse / Isaac Sim and an RTX GPU")
        curves_figure([{"title": "Newton from USD vs from URDF", "xlabel": "time [s]", "ylabel": "rad",
                        "series": [("shoulder (USD)", ru.t, ru.q[:, 0]), ("elbow (USD)", ru.t, ru.q[:, 1]),
                                   ("URDF", ru.t[::40], rr.q[::40, 0], "ko"), ("URDF ", ru.t[::40], rr.q[::40, 1], "ko")]},
                       {"title": "Energy balance", "xlabel": "time [s]", "ylabel": "J",
                        "series": [("actuator work", ru.t, W), ("dE (kinetic + potential)", ru.t[::40],
                                                                (E - E[0])[::40], "ko")]},
                       {"title": "Link root stress / (yield / 3)", "xlabel": "time [s]", "ylabel": "utilisation",
                        "series": [("upper arm", ru.t, util[:, 0]), ("forearm", ru.t, util[:, 1])],
                        "hlines": [(1.0, "limit", "r")]}],
                      os.path.join(OUT, "robotsim.png"), suptitle="Robot sim run of the USD scene, checked by PINNeAPPle")
        checks_figure(checks, os.path.join(OUT, "checks.png"), title="Scene and motion checks", decision=decision)
        summary = {"decision": decision, "usd_vs_urdf_rad": same, "mass_property_diff": mp_diff,
                   "tracking_error_mrad": track * 1e3, "energy_balance": eb, "peak_util": util.max(0).tolist(),
                   "physics_stage": {k: vp[k] for k in ("ok", "problems", "physics", "total_mass")},
                   "timeline_stage": {k: va[k] for k in ("ok", "problems", "time_samples")}, "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "usd_vs_urdf_rad", "tracking_error_mrad")}, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
