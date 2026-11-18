"""MuJoCo 4.0 integration: loader, runner, trajectory/UPD, batch rollouts, analytic checks, robot backend.

The whole file is skipped when MuJoCo is not installed (``pip install "pinneapple[mujoco]"``). The checks are chosen
so each one has a closed-form answer; none of them needs a reference file.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from pinneapple_simulation.external_solvers.mujoco import (  # noqa: E402
    CASES,
    MUJOCO_MIN_VERSION,
    BatchRunner,
    MuJoCoConfig,
    MuJoCoRobotSim,
    MuJoCoRunner,
    has_batch,
    load_model,
    load_spec,
    model_summary,
    mujoco_version,
    spec_to_mjcf,
    summary,
    validate,
)

BALL = """<mujoco><option timestep="0.001" gravity="0 0 -9.81"/><worldbody>
<body name="ball" pos="0 0 1"><freejoint/><geom type="sphere" size="0.05" mass="1"/></body>
</worldbody></mujoco>"""

KEYFRAMED = """<mujoco><option timestep="0.001"/><worldbody>
<body name="ball" pos="0 0 1"><freejoint/><geom type="sphere" size="0.05" mass="1"/></body>
</worldbody>
<keyframe><key name="high" qpos="0 0 2 1 0 0 0"/></keyframe>
</mujoco>"""


# --------------------------------------------------------------------------- loader
def test_version_and_capabilities():
    version = mujoco_version()
    assert version is not None and tuple(version) >= MUJOCO_MIN_VERSION
    if has_batch():
        import importlib.util

        assert importlib.util.find_spec("mujoco.batch") is not None


def test_config_rejects_bad_inputs():
    with pytest.raises(ValueError):
        MuJoCoConfig()
    with pytest.raises(ValueError):
        MuJoCoConfig(model_xml=BALL, model_path="x.xml")
    with pytest.raises(ValueError):
        MuJoCoConfig(model_xml=BALL, n_substeps=0)
    with pytest.raises(ValueError):
        MuJoCoConfig(model_xml=BALL, dt=-1.0)
    with pytest.raises(FileNotFoundError):
        load_model(MuJoCoConfig(model_path="/definitely/not/here.xml"))


def test_load_model_applies_overrides_and_enables_energy():
    model = load_model(MuJoCoConfig(model_xml=BALL, dt=0.005, gravity=(0, 0, -1.62)))
    assert model.opt.timestep == pytest.approx(0.005)
    assert model.opt.gravity[2] == pytest.approx(-1.62)
    assert int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    info = model_summary(model)
    assert info["mujoco"] == mujoco.__version__ and info["nbody"] == 2


def test_keyframe_reset_by_name_and_index():
    cfg = MuJoCoConfig(model_xml=KEYFRAMED, keyframe="high")
    runner = MuJoCoRunner(cfg)
    runner.reset()
    assert runner.data.qpos[2] == pytest.approx(2.0)
    runner.config.keyframe = 0
    runner.reset()
    assert runner.data.qpos[2] == pytest.approx(2.0)
    runner.config.keyframe = "missing"
    with pytest.raises(KeyError):
        runner.reset()


def test_mjspec_roundtrip():
    spec = load_spec(BALL)
    xml = spec.to_xml() if hasattr(spec, "to_xml") else None
    if xml is None:  # pragma: no cover - very old builds
        pytest.skip("MjSpec.to_xml is not available")
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nq == 7


# --------------------------------------------------------------------------- runner
def test_free_fall_matches_the_closed_form():
    duration, dt = 0.3, 1e-3
    cfg = MuJoCoConfig(model_xml=BALL, n_steps=int(duration / dt), dt=dt)
    traj = MuJoCoRunner(cfg).simulate()
    assert traj.t.shape == (301,)
    assert traj.qpos.shape == (301, 7) and traj.xpos.shape == (301, 2, 3)
    z = traj.xpos[-1, 1, 2]
    vz = traj.cvel[-1, 1, 5]
    assert z == pytest.approx(1 - 0.5 * 9.81 * duration ** 2, rel=5e-3)
    assert vz == pytest.approx(-9.81 * duration, rel=5e-3)
    energy = traj.energy()
    assert set(energy) == {"potential", "kinetic", "total"} and energy["total"].shape == (301,)
    assert traj.energy_drift() < 5e-3


def test_record_every_and_initial_state():
    cfg = MuJoCoConfig(model_xml=BALL, n_steps=100, dt=1e-3)
    traj = MuJoCoRunner(cfg).simulate(record_every=10)
    assert traj.t.shape == (11,) and traj.t[0] == 0.0 and traj.t[-1] == pytest.approx(0.1)
    assert traj.dt == pytest.approx(1e-2)


def test_controller_injection_holds_the_pendulum():
    xml = """<mujoco><option timestep="0.001" gravity="0 0 -9.81"/><worldbody>
    <body name="bob"><joint name="hinge" type="hinge" axis="0 1 0"/>
    <geom type="sphere" size="0.02" mass="1" pos="0 0 -0.5"/></body></worldbody>
    <actuator><motor joint="hinge" ctrllimited="true" ctrlrange="-100 100"/></actuator></mujoco>"""
    cfg = MuJoCoConfig(model_xml=xml, n_steps=500, dt=1e-3)
    runner = MuJoCoRunner(cfg)
    mujoco.mj_forward(runner.model, runner.data)
    runner.data.qpos[0] = np.radians(30.0)
    mujoco.mj_forward(runner.model, runner.data)
    hold = np.array(runner.data.qfrc_bias, dtype=float).copy()

    def ctrl(model, data):
        data.ctrl[:] = hold

    traj = MuJoCoRunner(cfg, controller=ctrl).simulate(q0=[np.radians(30.0)])
    assert np.abs(traj.qpos[:, 0] - np.radians(30.0)).max() < 1e-9


def test_trajectory_dict_compat_and_upd():
    cfg = MuJoCoConfig(model_xml=BALL, n_steps=20, dt=1e-3)
    traj = MuJoCoRunner(cfg).simulate()
    assert traj["qpos"].shape == traj.qpos.shape == (21, 7)
    assert traj.get("time") is traj.t and traj.keys()
    sample = traj.to_upd()
    assert sample.provenance["source"] == "mujoco"
    assert sample.provenance["backend"] == "mj_step"
    assert sample.provenance["dt"] == pytest.approx(1e-3)
    assert set(sample.state) >= {"qpos", "qvel", "xpos", "xquat"}
    assert sample.state["qpos"].shape == (21, 7)
    assert sample.schema["units"]["xpos"] == "m"


def test_trajectory_save_and_body_lookup(tmp_path):
    cfg = MuJoCoConfig(model_xml=BALL, n_steps=10, dt=1e-3)
    traj = MuJoCoRunner(cfg).simulate()
    assert traj.index_body("ball") == 1
    assert traj.body("ball")["pos"].shape == (11, 3)
    path = tmp_path / "traj.npz"
    traj.save(path)
    data = np.load(path, allow_pickle=True)
    assert data["t"].shape == (11,) and list(data["body_names"]) == ["world", "ball"]


# --------------------------------------------------------------------------- batch
@pytest.mark.skipif(not has_batch(), reason="mujoco.batch needs MuJoCo >= 3.17")
def test_batch_rollout_and_domain_randomisation():
    cfg = MuJoCoConfig(model_xml=BALL, dt=2e-3)
    runner = BatchRunner(cfg, n_envs=8, nthread=2, seed=0)
    runner.set_state("qpos", [0, 0, 1, 1, 0, 0, 0])
    runner.randomize("body", "ball", "mass", 0.5, 2.0)
    rolled = runner.rollout(n_steps=50)
    assert rolled["qpos"].shape == (8, 50, 7)
    assert rolled["xpos"].shape == (8, 50, 2, 3)
    masses = np.asarray(runner.batch.body("ball").mass).reshape(-1)
    assert masses.min() >= 0.5 and masses.max() <= 2.0


@pytest.mark.skipif(not has_batch(), reason="mujoco.batch needs MuJoCo >= 3.17")
def test_batch_roll_loop_records_the_initial_state():
    cfg = MuJoCoConfig(model_xml=BALL, dt=2e-3)
    runner = BatchRunner(cfg, n_envs=4, nthread=2, seed=1)
    out = runner.roll_loop(n_steps=20)
    assert out["qpos"].shape == (4, 21, 7)
    assert out["xpos"].shape == (4, 21, 2, 3)


@pytest.mark.skipif(not has_batch(), reason="mujoco.batch needs MuJoCo >= 3.17")
def test_batch_rollout_to_upd():
    cfg = MuJoCoConfig(model_xml=BALL, dt=2e-3)
    samples = BatchRunner(cfg, n_envs=3, nthread=2, seed=2).rollout_to_upd(n_steps=10)
    assert len(samples) == 3
    assert samples[0].provenance["backend"] == "batch"
    assert samples[0].provenance["n_envs"] == 3


# --------------------------------------------------------------------------- validation
def test_analytic_cases_pass():
    rows = validate(dt=1e-3)
    assert [r["case"] for r in rows] == list(CASES)
    assert all(r["passed"] for r in rows), rows
    assert summary(rows)["decision"] == "APPROVED"


def test_validation_rejects_unknown_case():
    with pytest.raises(ValueError):
        validate(cases=["free_fall", "no_such_case"])


# --------------------------------------------------------------------------- robot backend
def test_robot_sim_matches_the_multibody_tree():
    from pinneapple_simulation.multibody import arm as mb
    from pinneapple_simulation.robot_sim.arms import TwoLinkArm

    arm = TwoLinkArm()
    tree = arm.tree()
    sim = MuJoCoRobotSim.from_spec(arm.spec(), ground=False)
    q0 = np.array([-0.6, 1.2])

    # the tree turns about -y, so its torques are the negative of MuJoCo's
    def feed(t, q, qd):
        return -mb.fixed_base_inverse_dynamics(tree, -np.asarray(q), np.zeros(2), np.zeros(2))[0]

    run = sim.run(0.3, 5e-4, q0=q0, controller=feed)
    assert np.abs(run.q - q0).max() < 1e-6
    assert run.backend == "mujoco" and run.body_names[1:] == ["base", "link1", "link2"]
    assert run.joint_names == ["shoulder", "elbow"]
    assert np.allclose(sim.effort_limit, arm.effort_limits)
    energy = run.energy()
    assert np.abs(energy["total"] - energy["total"][0]).max() < 1e-6
    assert np.linalg.norm(run.body_quat, axis=-1) == pytest.approx(1.0)


def test_robot_sim_saturation_and_quaternion_convention():
    from pinneapple_simulation.robot_sim.arms import TwoLinkArm

    sim = MuJoCoRobotSim.from_spec(TwoLinkArm().spec(), ground=False)
    run = sim.run(0.01, 5e-4, q0=[0.0, 0.0], controller=lambda t, q, qd: np.array([1e6, -1e6]))
    assert run.saturated >= 1
    assert np.all(np.abs(run.tau) <= sim.effort_limit + 1e-9)
    # MuJoCo stores (w, x, y, z); RobotRun stores Newton's (x, y, z, w) — at rest the identity is w=1, x=y=z=0
    assert run.body_quat[0, 0] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_robot_sim_from_mjcf_file(tmp_path):
    path = tmp_path / "arm.xml"
    path.write_text("""<mujoco><option timestep="0.001"/><worldbody>
    <body name="b"><joint name="j" type="hinge" axis="0 1 0"/>
    <geom type="capsule" size="0.02 0.2" mass="1" pos="0 0 -0.2"/></body></worldbody>
    <actuator><motor joint="j" ctrllimited="true" ctrlrange="-5 5"/></actuator></mujoco>""")
    sim = MuJoCoRobotSim.from_mjcf(str(path))
    assert sim.joint_names == ["j"] and sim.body_names == ["world", "b"]
    assert sim.effort_limit == pytest.approx([5.0]) and sim.ndof == 1
    run = sim.run(0.02, 1e-3, controller=lambda t, q, qd: [0.0])
    assert run.q.shape == (21, 1) and run.backend == "mujoco"


def test_legacy_dict_adapter_still_works():
    from pinneapple_simulation.external_solvers.mujoco import trajectory_to_upd

    traj = {"time": np.arange(5) * 1e-3, "qpos": np.zeros((5, 2)), "qvel": np.ones((5, 2)),
            "ctrl": np.zeros((5, 1)), "xpos": np.zeros((5, 3, 3)), "sensor": np.zeros((5, 0))}
    sample = trajectory_to_upd(traj)
    assert sample.provenance["source"] == "mujoco"
    assert sample.state["qpos"].shape == (5, 2) and sample.state["xpos"].shape == (5, 9)
    assert sample.domain["coords"]["time"].shape == (5,)


def test_spec_to_mjcf_compiles_with_cylinders():
    from pinneapple_simulation.robot_sim.spec import ArticulationSpec, Joint, Link

    spec = ArticulationSpec("Rod", [
        Link("base", 2.0, (0.1, 0.1, 0.2), (0, 0, 0), com=(0, 0, 0.1), inertia=(0.01, 0.01, 0.01)),
        Link("rod", 1.0, (0.05, 0.4), (0, 0, 0.2), com=(0, 0, 0.2), shape="cylinder")],
        [Joint("swing", "base", "rod", (0, 0, 0.2), "Y", (-90, 90), max_force=5.0)],
        gravity=9.81, ground=False)
    xml = spec_to_mjcf(spec)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.nq == 1 and model.nu == 1
    assert model.opt.timestep > 0
