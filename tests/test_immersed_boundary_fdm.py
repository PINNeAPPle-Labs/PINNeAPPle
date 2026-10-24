"""Immersed boundary FDM, "channel" mode: the face-based projection conserves mass (#627)."""

import numpy as np

import pinneapple_simulation.numerical_solvers.immersed_boundary_fdm as ibm


def _pipe_points(R=0.5, L=3.0, n_ax=150, n_th=64):
    x, th = np.meshgrid(
        np.linspace(0, L, n_ax), np.linspace(0, 2 * np.pi, n_th, endpoint=False), indexing="ij"
    )
    return np.column_stack([x.ravel(), R * np.cos(th).ravel(), R * np.sin(th).ravel()])


def test_channel_flow_rate_equals_the_inlet_flux_at_every_cross_section():
    nx, ny, nz = 40, 16, 16
    U_in = 1.0
    sol = ibm.solve_ibm_internal_flow(
        _pipe_points(),
        (0.0, -0.55, -0.55, 3.0, 0.55, 0.55),
        Re=10.0,
        U_in=U_in,
        nx=nx,
        ny=ny,
        nz=nz,
        n_iter=50,
        bc_mode="channel",
        inlet_hint_point=[0, 0, 0],
    )
    g = sol["grid_fields"]
    fluid = ~g["wall_mask"] & ~g["exterior_mask"]
    dA = (1.1 / ny) * (1.1 / nz)
    # exact value: an incompressible flow carries the inlet flux U_in * A_in through every cross-section
    Q_exact = U_in * fluid[1].sum() * dA
    Q = (g["u"] * fluid).sum(axis=(1, 2)) * dA
    assert sol["bc_info"]["inlet_face"] == "x-"
    assert sol["face_divergence_rms"] < 1e-10
    # the earlier divergence-penalty step lost 60% of the flow by x = 2.5
    assert np.allclose(Q[2:-2], Q_exact, rtol=1e-9)


def test_projection_pins_a_fluid_pocket_that_has_no_outlet():
    fluid = np.zeros((12, 6, 6), bool)
    fluid[:, 1:4, 1:4] = True  # channel from x- to x+
    fluid[3:6, 5, 5] = True  # pocket with no open face to the channel
    fixed = np.zeros_like(fluid)
    fixed[:2] = fluid[:2]
    dirichlet = np.zeros_like(fluid)
    dirichlet[-2:] = fluid[-2:]
    proj = ibm._MACProjection(fluid, fixed, dirichlet, (0.1, 0.1, 0.1))
    assert len(proj.pinned) == 1

    rng = np.random.default_rng(0)
    vel = tuple(rng.normal(size=fluid.shape) * fluid for _ in range(3))
    fixed_vel = (np.where(fixed, 1.0, 0.0), np.zeros(fluid.shape), np.zeros(fluid.shape))
    u, v, w, phi, face_div = proj.project(vel, fixed_vel)
    assert np.abs(face_div).max() < 1e-10
    assert np.isfinite(phi).all() and np.all(u[fixed] == 1.0)
