"""MuJoCo 4.0 through PINNeAPPle, end to end and without any data file.

Run it with ``python examples/mujoco/quickstart.py`` (needs ``pip install "pinneapple[mujoco]"``). It prints, in order:
a recorded trajectory and its UPD export, the analytic checks, a batch campaign with domain randomisation, and a robot
held at a pose by PINNeAPPle's own multibody tree — two independent engines giving the same answer.
"""
from __future__ import annotations

import numpy as np

from pinneapple_simulation.external_solvers.mujoco import (
    BatchRunner,
    MuJoCoConfig,
    MuJoCoRobotSim,
    MuJoCoRunner,
    mujoco_version,
    summary,
    validate,
)

BALL = """<mujoco><option timestep="0.001" gravity="0 0 -9.81"/><worldbody>
<body name="ball" pos="0 0 1"><freejoint/><geom type="sphere" size="0.05" mass="1"/></body>
</worldbody></mujoco>"""


def trajectory() -> None:
    cfg = MuJoCoConfig(model_xml=BALL, n_steps=300, dt=1e-3)
    traj = MuJoCoRunner(cfg).simulate()
    sample = traj.to_upd()
    print(f"MuJoCo {'.'.join(map(str, mujoco_version()))}")
    print(f"records {traj.t.shape[0]}, qpos {traj.qpos.shape}, z(end) {traj.xpos[-1, 1, 2]:.5f} m "
          f"(closed form {1 - 0.5 * 9.81 * 0.3 ** 2:.5f} m)")
    print(f"energy drift {traj.energy_drift():.2e}, UPD fields {sorted(sample.state)}")


def checks() -> None:
    rows = validate(dt=1e-3)
    for row in rows:
        print(f"  {row['case']:<18} {row['value']:.2e} <= {row['limit']:.0e}  {row['passed']}")
    print(f"decision: {summary(rows)['decision']}")


def batch() -> None:
    runner = BatchRunner(MuJoCoConfig(model_xml=BALL, dt=2e-3), n_envs=64, seed=0)
    runner.set_state("qpos", [0, 0, 1, 1, 0, 0, 0])
    runner.randomize("body", "ball", "mass", 0.5, 2.0)
    rolled = runner.rollout(n_steps=200)
    print(f"batch {rolled['qpos'].shape} on {runner.nthread or 'all'} threads; "
          f"z(end) {rolled['xpos'][:, -1, 1, 2].mean():.4f} m +/- {rolled['xpos'][:, -1, 1, 2].std():.4f} m")


def robot() -> None:
    from pinneapple_simulation.multibody import arm as mb
    from pinneapple_simulation.robot_sim.arms import TwoLinkArm

    arm = TwoLinkArm()
    tree = arm.tree()
    sim = MuJoCoRobotSim.from_spec(arm.spec(), ground=False)
    q0 = np.array([-0.6, 1.2])
    # the tree's joints turn about -y: its angles and torques are the negatives of MuJoCo's
    def feed(t, q, qd):
        return -mb.fixed_base_inverse_dynamics(tree, -np.asarray(q), np.zeros(2), np.zeros(2))[0]

    run = sim.run(0.3, 5e-4, q0=q0, controller=feed)
    print(f"arm held to {np.abs(run.q - q0).max():.1e} rad; energy change "
          f"{np.abs(run.energy()['total'] - run.energy()['total'][0]).max():.1e} J; torque {run.tau[-1].round(3)}")


if __name__ == "__main__":
    trajectory()
    checks()
    batch()
    robot()
