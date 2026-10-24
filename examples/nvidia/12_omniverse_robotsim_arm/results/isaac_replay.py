# Isaac Sim standalone replay, written by pinneapple_tools.nvidia.isaac. Needs Isaac Sim 5.x / 6.x and an RTX GPU:
#   ~/isaacsim/python.sh isaac_efforts_replay.py
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

omni.usd.get_context().open_stage('/home/user/PINNeAPPle/examples/nvidia/12_omniverse_robotsim_arm/results/arm_physics.usda')
for _ in range(10):
    app.update()
world = World(stage_units_in_meters=1.0, physics_dt=0.0016666666666666668, rendering_dt=0.016666666666666666)
robot = SingleArticulation(prim_path='/World/Arm', name="robot")
world.scene.add(robot)
world.reset()
data = np.load('/home/user/PINNeAPPle/examples/nvidia/12_omniverse_robotsim_arm/results/arm_trajectory.npz')
t_ref, q_ref, tau_ref = data["t"], data["q"], data["tau"]
robot.set_joint_positions(q_ref[0])
robot.set_joint_velocities(np.zeros_like(q_ref[0]))
T, Q, E = [], [], []
n = int(round(t_ref[-1] / 0.0016666666666666668))
for k in range(n + 1):
    tk = k * 0.0016666666666666668
    if 'effort' == "effort":
        tau = np.array([np.interp(tk, t_ref, tau_ref[:, j]) for j in range(tau_ref.shape[1])])
        robot.set_joint_efforts(tau)
    else:
        qk = np.array([np.interp(tk, t_ref, q_ref[:, j]) for j in range(q_ref.shape[1])])
        robot.apply_action(ArticulationAction(joint_positions=qk))
    world.step(render=False)
    T.append(tk)
    Q.append(np.asarray(robot.get_joint_positions()))
    E.append(np.asarray(robot.get_measured_joint_efforts()))
np.savez('/home/user/PINNeAPPle/examples/nvidia/12_omniverse_robotsim_arm/results/isaac_efforts.npz', t=np.asarray(T), q=np.asarray(Q), effort=np.asarray(E),
         dof_names=np.asarray(robot.dof_names))
app.close()
