"""One-call checks of an arm in the robot sim (used by the Flow Studio node ``lib.robot.arm_check``).

``arm_pick_and_place(payload, duration, ...)`` simulates a :class:`~.arms.TwoLinkArm` rest-to-rest move in Newton
with the multibody-tree feedforward and PD tracking, and returns the PINNeAPPle checks: the torque replay converging
to the planned motion as the step halves, tracking error, energy balance, motor limits and link-root stresses.
"""
from __future__ import annotations

import numpy as np

__all__ = ["arm_pick_and_place"]


def arm_pick_and_place(payload: float = 5.0, duration: float = 0.8, q_start=(-0.6, 1.2), q_end=(0.9, -0.4),
                       dt: float = 5e-4, kp=(350.0, 110.0), kd=(5.0, 2.5), safety: float = 3.0) -> dict:
    """Simulate and check a pick-and-place move (angles from the horizontal, positive lifts). Returns
    ``{"decision", "tracking_error_rad", "peak_torque_Nm", "replay_order", "checks"}``."""
    from pinneapple_simulation.multibody import arm as mb

    from .arms import TwoLinkArm
    from .sim import RobotSim, convergence_order, pd_tracking, torque_trajectory
    a = TwoLinkArm(payload=float(payload))
    tree, sec, props = a.tree(), a.section(), a.material_props()
    sim = RobotSim.from_spec(a.spec(), ground=False)
    plan = lambda t: tuple(-x for x in mb.quintic(q_start, q_end, duration, np.atleast_1d(t)))      # noqa: E731
    feed = lambda t: -mb.fixed_base_inverse_dynamics(tree, *mb.quintic(q_start, q_end, duration, [t]))[0]  # noqa: E731
    q0 = plan(0.0)[0][0]
    errs, dts = [], (2 * dt, dt)
    for h in dts:
        r = sim.run(duration, h, q0=q0, controller=torque_trajectory(feed))
        errs.append(float(np.abs(r.q - plan(r.t)[0]).max()))
    order = convergence_order(errs, dts)
    run = sim.run(duration, dt, q0=q0, controller=pd_tracking(lambda t: tuple(x[0] for x in plan(t)[:2]), kp, kd,
                                                              feedforward=feed))
    track = float(np.abs(run.q - plan(run.t)[0]).max())
    E, W = run.energy()["total"], run.actuator_work()
    eb = float(np.abs((E - E[0]) - W).max() / max(np.abs(W).max(), 1e-12))
    peak = np.abs(run.tau).max(0)
    _, s_checks = mb.link_root_checks(run.tau, sec, yield_strength=props["yield_strength"], safety=safety,
                                      names=["shoulder", "elbow"])
    checks = [{"name": "torque replay converges (order in dt)", "value": order, "limit": 0.8, "passed": order > 0.8},
              {"name": "tracking error (rad)", "value": track, "limit": 2e-3, "passed": track <= 2e-3},
              {"name": "energy balance", "value": eb, "limit": 1e-2, "passed": eb <= 1e-2},
              {"name": "no actuator saturation", "value": float(run.saturated), "limit": 0.0, "passed": run.saturated == 0},
              *[{"name": f"{n} peak torque <= motor", "value": float(p), "limit": float(lim), "passed": bool(p <= lim)}
                for n, p, lim in zip(("shoulder", "elbow"), peak, a.effort_limits, strict=True)],
              *s_checks]
    return {"decision": "APPROVED" if all(c["passed"] for c in checks) else "REJECT", "tracking_error_rad": track,
            "peak_torque_Nm": peak.tolist(), "replay_order": order, "checks": checks}
