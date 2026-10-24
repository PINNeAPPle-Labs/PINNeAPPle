"""Pick-and-place arm in PINNeAPPle Robot Sim (NVIDIA Newton on the CPU), verified and sized by PINNeAPPle.

A two-link arm (6061-T6 square tubes 50 x 50 x 3 mm, 0.6 m and 0.5 m, a 1.5 kg elbow motor, a 5 kg part in the
gripper) moves the part in 0.8 s. The arm is described once (``TwoLinkArm``) and handed to Newton as URDF and to
PINNeAPPle's multibody tree. PINNeAPPle checks the simulator before using it (energy of a free swing, torque replay
converging to the planned motion as the time step shrinks), a small learning loop (cross-entropy method on Newton
rollouts) tunes the PD gains around the model-based feedforward, and the joint-torque sensors of the final run feed
the structural checks: link-root stress against yield and fatigue, forearm deflection, motor limits.

    python examples/nvidia/03_robotsim_pick_and_place/run.py           # ~2 min on a CPU (Newton kernels compile once)
"""
import argparse
import json
import os

import numpy as np

from pinneapple_lab import experiment
from pinneapple_simulation.multibody import arm as mb
from pinneapple_simulation.numerical_solvers.beam_bvp_fdm import (
    BC_CANTILEVER,
    solve_euler_bernoulli_beam_1d,
)
from pinneapple_simulation.robot_sim import (
    RobotSim,
    TwoLinkArm,
    convergence_order,
    pd_tracking,
    torque_trajectory,
)
from pinneapple_simulation.robot_sim.learn import cem
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure, curves_figure
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
Q_START, Q_END, DURATION = (-0.6, 1.2), (0.9, -0.4), 0.8     # PINNeAPPle tree angles (positive lifts)
SAFETY = 3.0
FATIGUE_6061 = 96.5e6     # Pa, fully reversed, 5e8 cycles (ASM Aerospace Specification Metals, 6061-T6 datasheet)
TIP_LIMIT = 1.0e-3        # m


def plan(t):
    """Planned motion in the simulator's convention (joints about +y: angles and torques change sign)."""
    q, qd, qdd = mb.quintic(Q_START, Q_END, DURATION, np.atleast_1d(t))
    return -q, -qd, -qdd


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("03_robotsim_pick_and_place")
    architecture_diagram(
        "03 Pick-and-place arm: PINNeAPPle + PINNeAPPle Robot Sim (NVIDIA Newton)",
        [Block("arm", "TwoLinkArm: 6061-T6 tubes,\nmotor, payload (masses, inertias)", "pinneapple", 0, 0),
         Block("tree", "Multibody tree: planned\nmotion -> feedforward torques", "pinneapple", 1, 1),
         Block("newton", "Robot Sim on NVIDIA Newton\n(Featherstone, CPU) from URDF", "robotsim", 1, 0),
         Block("ver", "Simulator checks: free-swing\nenergy, replay convergence", "pinneapple", 2, 1),
         Block("cem", "Learning loop (CEM) on Newton\nrollouts: PD gains", "robotsim", 2, 0),
         Block("run", "Final run: joint-torque\nsensors, IMU, energy", "robotsim", 3, 0),
         Block("fea", "Structure: root stress vs yield\nand fatigue, deflection, motors", "pinneapple", 3, 1),
         Block("dec", "Arm decision\n+ trust report", "pinneapple", 4, 1),
         Block("isaac", "Isaac Sim backend\n(same interface)", "isaac", 4, 0, ran=False)],
        [("arm", "newton", "URDF"), ("arm", "tree", "bodies"), ("tree", "ver", "tau(t)"), ("newton", "ver", "runs"),
         ("ver", "cem", "verified"), ("cem", "run", "kp, kd"), ("tree", "cem", "feedforward"), ("run", "fea", "tau(t)"),
         ("fea", "dec", "checks"), ("run", "isaac", "same run()")],
        os.path.join(OUT, "architecture.png"),
        subtitle="PINNeAPPle describes, verifies and sizes the arm; the robot sim (Newton) moves it and measures loads")
    a = TwoLinkArm()
    tree, sec, props = a.tree(), a.section(), a.material_props()
    sim = RobotSim.from_spec(a.spec(), ground=False)
    feed = lambda t: -mb.fixed_base_inverse_dynamics(tree, *mb.quintic(Q_START, Q_END, DURATION, [t]))[0]  # noqa: E731
    ref = lambda t: tuple(x[0] for x in plan(t)[:2])                                                     # noqa: E731
    q0 = plan(0.0)[0][0]
    with experiment("examples/nvidia/03_robotsim_pick_and_place", title="Pick-and-place arm in the PINNeAPPle robot "
                    "sim (NVIDIA Newton), verified and sized", domain="robotics",
                    methods=["Newton Featherstone", "multibody tree", "CEM", "beam FDM"], out_dir=base,
                    register=False, modules=["pinneapple_simulation.robot_sim", "pinneapple_simulation.multibody.arm"],
                    params={"payload_kg": a.payload, "duration_s": DURATION, "section": "50x50x3 6061-T6"}) as ex:
        dts = (2e-3, 1e-3) if smoke else (1e-3, 5e-4, 2.5e-4)
        with info.step("simulator verification (Newton vs PINNeAPPle)"):
            free = RobotSim.from_spec(a.spec(), ground=False)
            e_err, r_err = [], []
            for dt in dts:
                r = free.run(0.5, dt, q0=[0.0, 0.0], record_every=int(round(dts[0] / dt)))
                E = r.energy()["total"]
                e_err.append(float(np.abs(E - E[0]).max() / np.abs(E).max()))
                rr = sim.run(DURATION, dt, q0=q0, controller=torque_trajectory(feed), record_every=int(round(dts[0] / dt)))
                r_err.append(float(np.abs(rr.q - plan(rr.t)[0]).max()))
            p_e, p_r = convergence_order(e_err, dts), convergence_order(r_err, dts)
            r_extrap = 2 * r_err[-1] - r_err[-2]                    # Richardson, first order
        with info.step("learning loop (CEM on Newton rollouts)"):
            def cost(x):
                rr = sim.run(DURATION, 1e-3, q0=q0, controller=pd_tracking(ref, x[:2], x[2:], feedforward=feed))
                e = np.sqrt(np.mean((rr.q - plan(rr.t)[0]) ** 2))
                return float(e + 1e-6 * np.mean(rr.tau ** 2) + 10.0 * rr.saturated / len(rr.t))
            res = cem(cost, [200.0, 100.0, 10.0, 5.0], [100.0, 50.0, 5.0, 2.5], iterations=2 if smoke else 6,
                      population=4 if smoke else 8, lower=[0, 0, 0, 0], seed=0)
            kp, kd = res.best[:2], res.best[2:]
        with info.step("final run (dt = 0.25 ms)"):
            run = sim.run(DURATION, dts[-1], q0=q0, controller=pd_tracking(ref, kp, kd, feedforward=feed),
                          record_every=int(round(1e-3 / dts[-1])) or 1)
        track = float(np.abs(run.q - plan(run.t)[0]).max())
        E = run.energy()["total"]
        W = run.actuator_work()
        eb = float(np.abs((E - E[0]) - W).max() / max(np.abs(W).max(), 1e-12))
        imu = run.imu("link2")["specific_force"]
        peak = np.abs(run.tau).max(0)
        sig, s_checks = mb.link_root_checks(run.tau, sec, yield_strength=props["yield_strength"],
                                            fatigue_strength=FATIGUE_6061, safety=SAFETY, names=["shoulder", "elbow"])
        mpm = props["density"] * sec.A_m2
        acc = float(np.linalg.norm(imu[:, [0, 2]], axis=1).max())          # peak specific force in the arm plane
        beam = solve_euler_bernoulli_beam_1d(L_m=a.l2, E_Pa=props["youngs_modulus"], I_m4=sec.I_m4,
                                             q_N_per_m=mpm * acc, nx=201, bc_type=BC_CANTILEVER, section=sec)
        w_tip = float(np.abs(beam["deflection_m"]).max()) + a.payload * acc * a.l2 ** 3 / (3 * props["youngs_modulus"]
                                                                                        * sec.I_m4)
        checks = [
            {"name": "free swing: energy conserved, first order in dt", "value": p_e, "limit": 0.8,
             "passed": p_e > 0.8 and e_err[-1] < 1e-3, "detail": f"max |dE|/E {e_err[-1]:.2e} at dt {dts[-1]:g} s"},
            {"name": "torque replay converges to the planned motion", "value": p_r, "limit": 0.8,
             "passed": p_r > 0.8 and abs(r_extrap) < 1e-3,
             "detail": f"errors {', '.join(f'{e * 1e3:.2f}' for e in r_err)} mrad; Richardson {r_extrap * 1e3:.3f} mrad"},
            {"name": "closed-loop tracking error (rad)", "value": track, "limit": 2e-3, "passed": track <= 2e-3},
            {"name": "energy balance: actuator work = dE", "value": eb, "limit": 1e-2, "passed": eb <= 1e-2},
            {"name": "no actuator saturation", "value": float(run.saturated), "limit": 0.0, "passed": run.saturated == 0},
            {"name": "shoulder peak torque <= motor (N m)", "value": float(peak[0]), "limit": a.effort_limits[0],
             "passed": peak[0] <= a.effort_limits[0]},
            {"name": "elbow peak torque <= motor (N m)", "value": float(peak[1]), "limit": a.effort_limits[1],
             "passed": peak[1] <= a.effort_limits[1]},
            *s_checks,
            {"name": "forearm tip deflection <= 1 mm", "value": w_tip, "limit": TIP_LIMIT, "passed": w_tip <= TIP_LIMIT,
             "detail": f"peak payload acceleration {acc:.1f} m/s2 (IMU)"}]
        decision = "APPROVED" if all(c["passed"] for c in checks) else "REJECT"
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        ex.log_metric("tracking_error_mrad", track * 1e3, "mrad", "max |q_sim - q_plan| over the move, Newton dt 0.25 ms")
        ex.log_metric("peak_shoulder_torque", peak[0], "N m", "joint-torque sensor, final Newton run")
        ex.log_metric("replay_order", p_r, "", "slope of log(max replay error) vs log(dt)")
        info.mark("NVIDIA Newton (robot sim)", ran=True, how=f"newton {__import__('newton').__version__}, CPU")
        info.mark("Isaac Sim backend", ran=False, why="needs Isaac Sim and an RTX GPU")
        _figures(OUT, run, plan, dts, e_err, r_err, res, sig, props, checks, decision, imu)
        summary = {"decision": decision, "gains": {"kp": kp.tolist(), "kd": kd.tolist()},
                   "tracking_error_mrad": track * 1e3, "peak_torque_Nm": peak.tolist(),
                   "peak_stress_MPa": (sig.max(0) / 1e6).tolist(), "tip_deflection_mm": w_tip * 1e3,
                   "energy_errors": e_err, "replay_errors_rad": r_err, "dts": list(dts), "replay_order": p_r,
                   "energy_order": p_e, "cem": res.history, "cem_evaluations": res.evaluations,
                   "newton_wall_s_final": run.wall_s, "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "tracking_error_mrad", "peak_torque_Nm", "replay_order")},
                     indent=1))
    return summary


def _figures(OUT, run, plan, dts, e_err, r_err, res, sig, props, checks, decision, imu):
    t = run.t
    qp = plan(t)[0]
    curves_figure([{"title": "Joint angles: plan vs Newton", "xlabel": "time [s]", "ylabel": "rad",
                    "series": [("shoulder, sim", t, run.q[:, 0]), ("elbow, sim", t, run.q[:, 1]),
                               ("plan", t[::40], qp[::40, 0], "ko"), ("plan ", t[::40], qp[::40, 1], "ko")]},
                   {"title": "Tracking error", "xlabel": "time [s]", "ylabel": "mrad",
                    "series": [("shoulder", t, 1e3 * (run.q[:, 0] - qp[:, 0])),
                               ("elbow", t, 1e3 * (run.q[:, 1] - qp[:, 1]))]},
                   {"title": "Joint-torque sensors", "xlabel": "time [s]", "ylabel": "N m",
                    "series": [("shoulder", t, run.tau[:, 0]), ("elbow", t, run.tau[:, 1])]}],
                  os.path.join(OUT, "tracking.png"), suptitle="Final Newton run with the learned gains")
    curves_figure([{"title": "Simulator convergence (Newton)", "xlabel": "dt [s]", "ylabel": "error", "logy": True,
                    "series": [("free swing max |dE|/E", dts, e_err, "o-"), ("torque replay max |dq| [rad]", dts, r_err,
                                                                              "s-")]},
                   {"title": "Learning loop (CEM) on Newton rollouts", "xlabel": "iteration", "ylabel": "cost",
                    "logy": True, "series": [("best", [h["iteration"] for h in res.history],
                                              [h["best_cost"] for h in res.history], "o-"),
                                             ("elite mean", [h["iteration"] for h in res.history],
                                              [h["elite_cost"] for h in res.history], "s--")]},
                   {"title": "Link root stress (from torque sensors)", "xlabel": "time [s]", "ylabel": "MPa",
                    "series": [("upper arm", t, sig[:, 0] / 1e6), ("forearm", t, sig[:, 1] / 1e6)],
                    "hlines": [(props["yield_strength"] / 3e6, "yield / 3", "r"),
                               (FATIGUE_6061 / 3e6, "fatigue strength / 3", "m")]}],
                  os.path.join(OUT, "verification.png"), suptitle="PINNeAPPle checks of the simulator and the arm")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm = plt.get_cmap("viridis")
    idx = np.linspace(0, len(t) - 1, 9).astype(int)
    for j, k in enumerate(idx):
        P = run.body_pos[k, :, [0, 2]].T
        R = run._rot(k)
        tip = run.body_pos[k, 2] + R[2] @ np.array([0.5, 0, 0])
        ax.plot([P[1, 0], P[2, 0], tip[0]], [P[1, 1], P[2, 1], tip[2]], "-o", color=cm(j / 8), lw=2, ms=3)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Motion in Newton (strobe, 0.1 s)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "motion.png"), dpi=130)
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="Arm checks", decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
