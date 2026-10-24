"""Learned robot dynamics: PINNeAPPle Robot Sim (NVIDIA Newton) generates the data, a PhysicsNeMo network learns the
arm's inverse dynamics, PINNeAPPle checks the network's physics before it is used as a feedforward controller.

Newton runs the two-link arm through random smooth motions (PD tracking of random multi-sine references) and records
joint angles, rates, accelerations and the delivered torques. ``physicsnemo.models.mlp.FullyConnected`` learns
``tau = f(q, qd, qdd)``. PINNeAPPle then checks what a physically valid inverse-dynamics model must satisfy, without
trusting the training loss: torques against its independent multibody model (rigid-body equations), the gravity
torques at rest, the mass matrix the network implies (``tau(q, 0, qdd) - tau(q, 0, 0)`` must be linear in ``qdd``
with a symmetric positive-definite matrix), and whether a new motion lies inside the training envelope. The network
then drives the pick-and-place motion as feedforward, against PD alone and against the model-based feedforward.

    python examples/nvidia/08_robotsim_physicsnemo_learned_dynamics/run.py      # ~5 min on a CPU
"""
import argparse
import json
import os

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_lab import experiment
from pinneapple_simulation.multibody import arm as mb
from pinneapple_simulation.robot_sim import RobotSim, TwoLinkArm, pd_tracking
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure, curves_figure
from pinneapple_tools.nvidia.physicsnemo import MLPSurrogate, relative_errors
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
Q_START, Q_END, DURATION = (-0.6, 1.2), (0.9, -0.4), 0.8


def features(q, qd, qdd):
    q, qd, qdd = (np.atleast_2d(x) for x in (q, qd, qdd))
    return np.c_[np.sin(q), np.cos(q), np.sin(q[:, :1] + q[:, 1:]), np.cos(q[:, :1] + q[:, 1:]), qd, qdd]


def excitation(sim, rng, t_end, dt):
    """Newton run tracking a random multi-sine reference (q, qd, qdd from the solver's own update)."""
    f = rng.uniform(0.2, 1.5, (3, 2))
    amp = rng.uniform(0.2, 0.9, (3, 2)) / np.arange(1, 4)[:, None]
    ph = rng.uniform(0, 2 * np.pi, (3, 2))
    c = rng.uniform([-0.8, -1.2], [1.0, 1.2])
    ref = lambda t: (c + np.sum(amp * np.sin(2 * np.pi * f * t + ph), 0),                       # noqa: E731
                     np.sum(amp * 2 * np.pi * f * np.cos(2 * np.pi * f * t + ph), 0))
    kp, kd = rng.uniform(150, 400, 2), rng.uniform(3, 10, 2)
    r = sim.run(t_end, dt, q0=ref(0.0)[0], qd0=ref(0.0)[1], controller=pd_tracking(ref, kp, kd))
    qdd = np.diff(r.qd, axis=0) / dt                     # semi-implicit Euler: qd_{k+1} = qd_k + dt qdd_k exactly
    return r.q[:-1], r.qd[:-1], qdd, r.tau[:-1]


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("08_robotsim_physicsnemo_learned_dynamics")
    architecture_diagram(
        "08 Learned arm dynamics: PINNeAPPle + Robot Sim (Newton) + PhysicsNeMo",
        [Block("arm", "TwoLinkArm (masses,\ninertias, payload)", "pinneapple", 0, 0),
         Block("exc", "Newton runs: random smooth\nmotions, torque sensors", "robotsim", 1, 0),
         Block("nn", "physicsnemo FullyConnected\ntau = f(q, qd, qdd)", "physicsnemo", 2, 0),
         Block("tree", "Multibody tree (independent\nrigid-body model)", "pinneapple", 1, 1),
         Block("chk", "Checks: vs rigid body, gravity,\nmass matrix SPD, envelope", "pinneapple", 2, 1),
         Block("ff", "Feedforward in Newton:\npick-and-place tracking", "robotsim", 3, 0),
         Block("dec", "Use the network?\ntrust decision", "pinneapple", 3, 1)],
        [("arm", "exc", "URDF"), ("exc", "nn", "(q, qd, qdd, tau)"), ("arm", "tree", "bodies"), ("nn", "chk", "f"),
         ("tree", "chk", "reference"), ("nn", "ff", "tau_ff(t)"), ("chk", "dec", "checks"), ("ff", "dec", "tracking")],
        os.path.join(OUT, "architecture.png"),
        subtitle="Newton makes the data, PhysicsNeMo learns, PINNeAPPle checks the physics the network must keep")
    a = TwoLinkArm()
    tree = a.tree()
    sim = RobotSim.from_spec(a.spec(), ground=False)
    n_runs, t_run = (3, 0.5) if smoke else (120, 1.0)
    with experiment("examples/nvidia/08_robotsim_physicsnemo_learned_dynamics", title="Learned arm dynamics: Newton "
                    "data, PhysicsNeMo network, PINNeAPPle checks", domain="robotics",
                    methods=["Newton Featherstone", "FullyConnected (PhysicsNeMo)", "multibody tree"], out_dir=base,
                    register=False, modules=["pinneapple_simulation.robot_sim", "pinneapple_tools.nvidia.physicsnemo"],
                    params={"runs": n_runs, "run_s": t_run, "dt": 1e-3}) as ex:
        rng = np.random.default_rng(0)
        with info.step("data (Newton excitation runs)"):
            parts = [excitation(sim, rng, t_run, 1e-3) for _ in range(n_runs)]
            test = [excitation(sim, np.random.default_rng(99 + i), t_run, 1e-3) for i in range(2)]
        Q, QD, QDD, TAU = (np.concatenate([p[i] for p in parts]) for i in range(4))
        X, Y = features(Q, QD, QDD), TAU
        net = MLPSurrogate(X.shape[1], 2, layer_size=128 if not smoke else 32, num_layers=4 if not smoke else 2)
        with info.step("train (PhysicsNeMo FullyConnected)"):
            trun = net.fit(X, Y, ComputeBudget(steps=20 if smoke else 40000), batch_size=512, lr=2e-3)
        info.mark("PhysicsNeMo FullyConnected training", ran=True, how=f"CPU, {trun.steps} steps, {len(X)} samples")
        Qt, QDt, QDDt, TAUt = (np.concatenate([p[i] for p in test]) for i in range(4))
        pred_t = net.predict(features(Qt, QDt, QDDt))
        err_sim = relative_errors(pred_t, TAUt)
        # the independent rigid-body model (tree convention: angles and torques change sign)
        tau_rb = -mb.fixed_base_inverse_dynamics(tree, -Qt, -QDt, -QDDt)
        err_rb = relative_errors(pred_t, tau_rb)
        sim_vs_rb = relative_errors(TAUt, tau_rb)
        # gravity torques at rest over a grid of poses
        g1, g2 = np.meshgrid(np.linspace(-0.8, 1.0, 13), np.linspace(-1.2, 1.2, 13), indexing="ij")
        Qg = np.c_[g1.ravel(), g2.ravel()]
        Z = np.zeros_like(Qg)
        grav_nn = net.predict(features(Qg, Z, Z))
        grav_rb = -mb.fixed_base_inverse_dynamics(tree, -Qg, Z, Z)
        err_g = relative_errors(grav_nn, grav_rb)
        # implied mass matrix: columns tau(q, 0, e_j) - tau(q, 0, 0), symmetric positive definite, linear in qdd
        h = 5.0
        M = np.stack([(net.predict(features(Qg, Z, h * np.eye(2)[j][None].repeat(len(Qg), 0))) - grav_nn) / h
                      for j in range(2)], -1)
        M2 = np.stack([(net.predict(features(Qg, Z, 2 * h * np.eye(2)[j][None].repeat(len(Qg), 0))) - grav_nn) / (2 * h)
                       for j in range(2)], -1)
        asym = np.abs(M[:, 0, 1] - M[:, 1, 0]) / np.abs(M).max((1, 2))
        eig = np.linalg.eigvalsh(0.5 * (M + np.transpose(M, (0, 2, 1))))
        nonlin = np.abs(M2 - M).max((1, 2)) / np.abs(M).max((1, 2))
        M_rb = np.stack([-mb.fixed_base_inverse_dynamics(tree, -Qg, Z, -np.eye(2)[j][None].repeat(len(Qg), 0))
                         - grav_rb for j in range(2)], -1)
        err_M = float(np.linalg.norm(M - M_rb) / np.linalg.norm(M_rb))
        # applicability: the pick-and-place motion and a 3x faster one against the training box
        t = np.arange(0, DURATION + 1e-9, 1e-3)
        qp, qdp, qddp = (-x for x in mb.quintic(Q_START, Q_END, DURATION, t))
        qf, qdf, qddf = (-x for x in mb.quintic(Q_START, Q_END, DURATION / 3, t[t <= DURATION / 3]))
        inside = float(net.in_box(features(qp, qdp, qddp)).mean())
        inside_fast = float(net.in_box(features(qf, qdf, qddf)).mean())
        # use: feedforward in Newton tracking the pick-and-place motion
        plan = lambda tt: tuple(-x for x in mb.quintic(Q_START, Q_END, DURATION, np.atleast_1d(tt)))   # noqa: E731
        ref = lambda tt: tuple(x[0] for x in plan(tt)[:2])                                               # noqa: E731
        ff_nn = lambda tt: net.predict(features(*(x for x in plan(tt))))[0]                             # noqa: E731
        ff_rb = lambda tt: -mb.fixed_base_inverse_dynamics(tree, *mb.quintic(Q_START, Q_END, DURATION, [tt]))[0]  # noqa: E731
        kp, kd = (150.0, 60.0), (4.0, 2.0)
        track = {}
        with info.step("tracking runs (PD, PD + rigid-body ff, PD + PhysicsNeMo ff)"):
            for name, ff in (("PD only", None), ("PD + rigid-body feedforward", ff_rb), ("PD + PhysicsNeMo feedforward", ff_nn)):
                r = sim.run(DURATION, 1e-3, q0=plan(0.0)[0][0], controller=pd_tracking(ref, kp, kd, feedforward=ff))
                track[name] = (r.t, r.q - plan(r.t)[0], float(np.abs(r.q - plan(r.t)[0]).max()))
        checks = [
            {"name": "held-out Newton runs: torque error (worst joint)", "value": float(err_sim.max()), "limit": 0.05,
             "passed": err_sim.max() <= 0.05},
            {"name": "vs independent rigid-body model (worst joint)", "value": float(err_rb.max()), "limit": 0.05,
             "passed": err_rb.max() <= 0.05, "detail": f"Newton itself vs rigid body: {sim_vs_rb.max():.2e}"},
            {"name": "gravity torques at rest vs rigid body", "value": float(err_g.max()), "limit": 0.05,
             "passed": err_g.max() <= 0.05},
            {"name": "implied mass matrix vs rigid body", "value": err_M, "limit": 0.10, "passed": err_M <= 0.10},
            {"name": "implied mass matrix symmetric", "value": float(asym.max()), "limit": 0.10,
             "passed": asym.max() <= 0.10},
            {"name": "implied mass matrix positive definite", "value": float(eig.min()), "limit": None,
             "passed": bool(eig.min() > 0), "detail": f"smallest eigenvalue {eig.min():.3g} kg m2"},
            {"name": "torque linear in acceleration (2x step)", "value": float(nonlin.max()), "limit": 0.10,
             "passed": nonlin.max() <= 0.10},
            {"name": "pick-and-place inside the training envelope", "value": inside, "limit": 1.0,
             "passed": inside >= 1.0},
            {"name": "3x faster motion inside the training envelope", "value": inside_fast, "limit": 1.0,
             "passed": inside_fast >= 1.0, "detail": "a use outside the envelope is flagged, not trusted"}]
        core = checks[:8]
        decision = "APPROVED" if all(c["passed"] for c in core) else ("REVIEW" if checks[0]["passed"] else "REJECT")
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        ex.log_metric("heldout_torque_rel_rms", float(err_sim.max()), "", "worst joint, 2 held-out Newton runs")
        for k, v in track.items():
            ex.log_metric(f"tracking_mrad_{k.replace(' ', '_').replace('+', 'plus')}", v[2] * 1e3, "mrad",
                          "max |q - q_plan| in Newton, dt 1 ms")
        curves_figure([{"title": "Held-out Newton run: shoulder torque", "xlabel": "sample", "ylabel": "N m",
                        "series": [("Newton (sensor)", np.arange(600), TAUt[:600, 0]),
                                   ("PhysicsNeMo", np.arange(600), pred_t[:600, 0], "--"),
                                   ("rigid body", np.arange(0, 600, 25), tau_rb[:600:25, 0], "ko")]},
                       {"title": "Gravity torque at rest, shoulder (q2 = 0)", "xlabel": "q1 [rad]", "ylabel": "N m",
                        "series": [("PhysicsNeMo", g1[:, 6], grav_nn.reshape(13, 13, 2)[:, 6, 0], "o-"),
                                   ("rigid body", g1[:, 6], grav_rb.reshape(13, 13, 2)[:, 6, 0], "k--")]},
                       {"title": "Pick-and-place tracking in Newton", "xlabel": "time [s]", "ylabel": "shoulder error [mrad]",
                        "series": [(f"{k} ({v[2] * 1e3:.1f} mrad)", v[0], 1e3 * v[1][:, 0]) for k, v in track.items()]}],
                      os.path.join(OUT, "dynamics.png"), suptitle=f"PhysicsNeMo inverse dynamics, {len(X)} Newton samples")
        checks_figure(checks, os.path.join(OUT, "checks.png"), title="Learned dynamics checks", decision=decision)
        summary = {"decision": decision, "samples": int(len(X)), "train_steps": trun.steps, "train_wall_s": trun.wall_s,
                   "heldout_rel_rms": err_sim.tolist(), "vs_rigid_body_rel_rms": err_rb.tolist(),
                   "newton_vs_rigid_body": sim_vs_rb.tolist(), "gravity_rel_rms": err_g.tolist(),
                   "mass_matrix_rel": err_M, "mass_matrix_min_eig": float(eig.min()),
                   "tracking_mrad": {k: v[2] * 1e3 for k, v in track.items()}, "envelope_fraction": inside,
                   "envelope_fraction_fast": inside_fast, "parameters": net.n_parameters(), "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "heldout_rel_rms", "mass_matrix_rel", "tracking_mrad")},
                     indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
