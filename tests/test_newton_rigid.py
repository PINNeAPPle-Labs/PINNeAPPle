import numpy as np
import pytest

from pinneapple_simulation.external_solvers.newton_physics import RigidScene
from pinneapple_simulation.external_solvers.newton_physics.scene import _rot, _topological


def test_scene_checks_without_newton():
    s = RigidScene().ground(mu=0.3)
    a = s.box("a", (0.1, 0.1, 0.1), pos=(0, 0, 1))
    b = s.sphere("b", 0.05, pos=(0.5, 0, 1))
    s.revolute(-1, a, anchor=(0, 0, 1.2))
    s.revolute(a, b, anchor=(0.25, 0, 1))
    assert s.check() == []
    assert _topological(s.joints) == [0, 1]
    with pytest.raises(ValueError):
        s.sphere("a", 0.1)                                  # duplicate name
    s.revolute(-1, b, anchor=(0, 0, 0))                     # b now has two parents
    assert any("more than one joint" in p for p in s.check())
    bad = RigidScene()
    bad.box("neg", (0.1, -0.1, 0.1))
    assert bad.check()


def test_rotation_matrix_from_quaternion():
    q = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # 90 degrees about z
    assert np.allclose(_rot(q) @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(_rot(q) @ _rot(q).T, np.eye(3))


newton = pytest.importorskip("newton")
from pinneapple_simulation.external_solvers.newton_physics import simulate, validate  # noqa: E402


def test_sliding_box_stops_at_coulomb_distance():
    s = RigidScene().ground(mu=0.4)
    s.box("block", (0.1, 0.1, 0.1), pos=(0, 0, 0.1), vel=(2.0, 0, 0), mu=0.4)
    tr = simulate(s, 0.7, dt=1e-3, record_every=50)
    assert tr.pos[-1, 0, 0] == pytest.approx(2.0 ** 2 / (2 * 0.4 * 9.81), rel=0.03)
    assert abs(tr.vel[-1, 0, 0]) < 1e-3
    sample = tr.to_upd()
    assert sample.state["pos"].shape == (len(tr.t), 3)
    st = tr.to_rigid_body_state()
    assert st.pos.shape == (1, 3) and st.quat.shape == (1, 4)


def test_analytic_checks_pass_with_xpbd():
    rows = validate("xpbd", cases=["free_fall", "pendulum_period", "torque_free_spin"], log=None)
    assert all(r["passed"] for r in rows), rows


def test_pendulum_energy_is_conserved():
    s = RigidScene()
    a = np.radians(10)
    b = s.sphere("bob", 0.05, pos=(np.sin(a), 0, 2 - np.cos(a)))
    s.revolute(-1, b, anchor=(0, 0, 2), axis=(0, 1, 0))
    tr = simulate(s, 2.0, dt=1e-3, record_every=20)
    e = tr.energy()
    swing = e["kinetic"].max()
    assert np.ptp(e["total"]) < 0.05 * swing                # XPBD is slightly dissipative (~2.6 % here)


def test_exact_mass_rescales_inertia():
    s = RigidScene(gravity=0.0)
    s.box("a", (0.1, 0.2, 0.3), mass=2.0)
    tr = simulate(s, 0.01, dt=1e-3, record_every=10)
    assert tr.mass[0] == pytest.approx(2.0)
    ixx = 2.0 * ((0.4) ** 2 + (0.6) ** 2) / 12.0                # m (b^2 + c^2) / 12 with full side lengths
    assert tr.inertia[0, 0, 0] == pytest.approx(ixx, rel=1e-4)


def test_several_welded_trees_build_and_rest():
    """Joints are created tree by tree (Newton needs each articulation's joints contiguous)."""
    from pinneapple_simulation.external_solvers.newton_physics import simulate
    s = RigidScene().ground(mu=0.5)
    a = s.box("a", (0.05, 0.05, 0.05), pos=(0, 0, 0.05))
    b = s.box("b", (0.05, 0.05, 0.05), pos=(1, 0, 0.05))
    c = s.box("c", (0.05, 0.05, 0.05), pos=(0, 0, 0.15))
    d = s.box("d", (0.05, 0.05, 0.05), pos=(1, 0, 0.15))
    s.fixed(a, c)
    s.fixed(b, d)
    tr = simulate(s, duration=0.2, dt=1e-3)
    assert np.abs(tr.pos[-1] - tr.pos[0]).max() < 5e-3


def test_constant_wrench_balances_gravity_and_spins_up():
    from pinneapple_simulation.external_solvers.newton_physics import simulate
    s = RigidScene()
    b = s.box("b", (0.01, 0.01, 0.01), pos=(0, 0, 1), mass=0.1)
    tr = simulate(s, 0.5, wrenches={b: ((0, 0, 0.1 * 9.81), (0, 0, 0))})
    assert abs(tr.pos[-1, 0, 2] - 1.0) < 1e-6
    tr = simulate(s, 0.5, wrenches={b: ((0, 0, 0), (0, 0, 1e-4))})
    inertia = 0.1 * 0.02 ** 2 / 6
    assert tr.omega[-1, 0, 2] == pytest.approx(1e-4 / inertia * 0.5, rel=0.01)
