"""Tests for the 3D LES subgrid closures in ``turbulence_presets``.

Three closures now share one residual assembly (``_LESResiduals3D``):
Smagorinsky, dynamic Smagorinsky and WALE. Because they share the
momentum/continuity/diffusion part, the tests are split the same way:

* the **assembly** (and each closure's ``nu_sgs``) is checked against an
  independently hand-derived SymPy closed form, computed with autograd in
  torch and differentiated by hand in a different tool -- the same
  cross-implementation check ``tests/test_manufactured_solutions.py``
  applies to the RANS closures. WALE's check doubles as the regression
  test for its move onto the shared base class: its numbers must be
  unchanged.
* the **closure-specific** properties are checked on their own: the
  Smagorinsky viscosity must be exactly ``(Cs*Delta)^2 |S|`` and must
  vanish for a solid-body rotation (which is itself an exact solution of
  the incompressible equations, so the whole residual must vanish there),
  and the dynamic coefficient must obey the homogeneity and scaling
  properties that make it a dimensionless Germano coefficient rather than
  a fitted constant.
"""
import numpy as np
import pytest
import sympy as sp
import torch

from pinneapple_physics.pde_environment.turbulence_presets import (
    DynamicSmagorinskyResiduals,
    SmagorinskyResiduals,
    WALEResiduals,
    apply_stencil,
    filter_stencil,
    get_rans_preset,
    van_driest,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _ExactFn(torch.nn.Module):
    """Wraps an analytic field ``(N, 4) -> (N, 4)`` as a model the closures
    can be evaluated on, so derivative errors are the closure's alone."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def _sample_points(seed: int = 0, n: int = 40):
    """Deterministic 4-D points ``(x, y, z, t)`` in [0.5, 1.3]^3 x [0.2, 1.1]."""
    rng = np.random.default_rng(seed)
    return torch.tensor(
        np.column_stack([
            rng.uniform(0.5, 1.3, n),
            rng.uniform(0.5, 1.3, n),
            rng.uniform(0.5, 1.3, n),
            rng.uniform(0.2, 1.1, n),
        ]),
        dtype=torch.float64,
        requires_grad=True,
    )


def _trial_field_consts():
    return dict(
        a1=0.7, a2=0.4, a3=0.3,
        b1=0.5, b2=0.6, b3=0.2,
        c1=0.3, c2=0.5, c3=0.25,
        d1=0.2, d2=0.15,
    )


def _sympy_field():
    """A smooth trial field with non-zero derivatives in every term."""
    x, y, z, t = sp.symbols("x y z t", real=True)
    k = _trial_field_consts()
    u = k["a1"] * sp.sin(x) * sp.cos(y) + k["a2"] * z ** 2 + k["a3"] * t * sp.sin(y)
    v = k["b1"] * x * z + k["b2"] * sp.sin(y) + k["b3"] * t * sp.cos(z)
    w = k["c1"] * y ** 2 + k["c2"] * sp.cos(x) * z + k["c3"] * t * x
    p = k["d1"] * (x + y + z) + k["d2"] * x * y * z
    return x, y, z, t, u, v, w, p


def _torch_field(x):
    k = _trial_field_consts()
    xx, yy, zz, tt = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
    u = k["a1"] * torch.sin(xx) * torch.cos(yy) + k["a2"] * zz ** 2 + k["a3"] * tt * torch.sin(yy)
    v = k["b1"] * xx * zz + k["b2"] * torch.sin(yy) + k["b3"] * tt * torch.cos(zz)
    w = k["c1"] * yy ** 2 + k["c2"] * torch.cos(xx) * zz + k["c3"] * tt * xx
    p = k["d1"] * (xx + yy + zz) + k["d2"] * xx * yy * zz
    return torch.cat([u, v, w, p], dim=1)


def _sympy_les_residuals(nu_eff):
    """Filtered incompressible Navier-Stokes residuals, exactly as
    ``_LESResiduals3D.__call__`` assembles them, with the effective
    viscosity supplied by the caller (over-damped so no clamp is active)."""
    x, y, z, t, u, v, w, p = _sympy_field()

    u_x, u_y, u_z, u_t = sp.diff(u, x), sp.diff(u, y), sp.diff(u, z), sp.diff(u, t)
    v_x, v_y, v_z, v_t = sp.diff(v, x), sp.diff(v, y), sp.diff(v, z), sp.diff(v, t)
    w_x, w_y, w_z, w_t = sp.diff(w, x), sp.diff(w, y), sp.diff(w, z), sp.diff(w, t)

    lap_u = sp.diff(u, x, 2) + sp.diff(u, y, 2) + sp.diff(u, z, 2)
    lap_v = sp.diff(v, x, 2) + sp.diff(v, y, 2) + sp.diff(v, z, 2)
    lap_w = sp.diff(w, x, 2) + sp.diff(w, y, 2) + sp.diff(w, z, 2)

    Sxx, Syy, Szz = u_x, v_y, w_z
    Sxy = sp.Rational(1, 2) * (u_y + v_x)
    Sxz = sp.Rational(1, 2) * (u_z + w_x)
    Syz = sp.Rational(1, 2) * (v_z + w_y)

    nu_x, nu_y, nu_z = sp.diff(nu_eff, x), sp.diff(nu_eff, y), sp.diff(nu_eff, z)
    gS_x = nu_x * Sxx + nu_y * Sxy + nu_z * Sxz
    gS_y = nu_x * Sxy + nu_y * Syy + nu_z * Syz
    gS_z = nu_x * Sxz + nu_y * Syz + nu_z * Szz

    conv_u = u * u_x + v * u_y + w * u_z
    conv_v = u * v_x + v * v_y + w * v_z
    conv_w = u * w_x + v * w_y + w * w_z

    return {
        "momentum_x": u_t + conv_u + sp.diff(p, x) - (nu_eff * lap_u + 2 * gS_x),
        "momentum_y": v_t + conv_v + sp.diff(p, y) - (nu_eff * lap_v + 2 * gS_y),
        "momentum_z": w_t + conv_w + sp.diff(p, z) - (nu_eff * lap_w + 2 * gS_z),
        "continuity": u_x + v_y + w_z,
    }, dict(Sxx=Sxx, Syy=Syy, Szz=Szz, Sxy=Sxy, Sxz=Sxz, Syz=Syz)


# ---------------------------------------------------------------------------
# Smagorinsky: the closed form the model is defined by
# ---------------------------------------------------------------------------

def test_smagorinsky_nu_sgs_is_exactly_cs_delta_squared_times_strain_magnitude():
    """For a velocity field that is *linear* in x, the strain-rate tensor is
    constant and known by hand, so nu_sgs must come out as exactly
    ``(Cs*Delta)^2 * sqrt(2 S:S)`` -- no tolerance on the model's own
    definition (only float64 roundoff)."""
    a, b = 0.3, 0.55

    def field(x):
        xx, yy = x[:, 0:1], x[:, 1:2]
        u = a * xx + b * yy
        v = b * xx - a * yy
        w = torch.zeros_like(xx)
        p = torch.zeros_like(xx)
        return torch.cat([u, v, w, p], dim=1)

    cs, delta = 0.17, 0.1
    x = _sample_points(n=8)
    out = SmagorinskyResiduals(nu=1e-5, cs=cs, delta=delta)(_ExactFn(field), x)

    # S = [[a, b, 0], [b, -a, 0], [0, 0, 0]]  ->  S:S = 2(a^2 + b^2)
    expected = (cs * delta) ** 2 * np.sqrt(2.0 * 2.0 * (a ** 2 + b ** 2))
    got = out["nu_sgs"].detach().numpy().reshape(-1)
    assert np.allclose(got, expected, rtol=0, atol=1e-14), (got, expected)

    # and the field is divergence-free, so continuity is exact
    assert float(out["continuity"].detach().abs().max()) < 1e-12


def test_smagorinsky_solid_body_rotation_is_an_exact_solution_with_no_eddy_viscosity():
    """Solid-body rotation has zero strain, so the Smagorinsky viscosity must
    be exactly zero -- and the rotation field (with the balancing centripetal
    pressure ``p = Omega^2 (x^2 + y^2) / 2``) is an exact solution of the
    incompressible equations. Both facts together test the closure *and* the
    shared assembly: if the diffusion or convection terms were misassembled,
    this residual could not vanish."""
    omega = 0.8

    def field(x):
        xx, yy = x[:, 0:1], x[:, 1:2]
        u = -omega * yy
        v = omega * xx
        w = torch.zeros_like(xx)
        p = 0.5 * omega ** 2 * (xx ** 2 + yy ** 2)
        return torch.cat([u, v, w, p], dim=1)

    x = _sample_points(n=12)
    out = SmagorinskyResiduals(nu=1e-5, cs=0.17, delta=0.05)(_ExactFn(field), x)

    assert float(out["nu_sgs"].detach().abs().max()) < 1e-18
    for name in ("momentum_x", "momentum_y", "momentum_z", "continuity"):
        resid = float(out[name].detach().abs().max())
        assert resid < 1e-10, f"{name} residual of the exact solid-body rotation is {resid:.3e}, expected ~0"


def test_smagorinsky_residual_matches_independent_sympy_closed_form():
    """Full residual (not just nu_sgs) against a hand-derived closed form.

    ``nu_eff = nu + (Cs Delta)^2 |S|`` is carried into the SymPy derivation
    as an explicit function of x, so the ``grad nu_eff`` terms that a
    spatially varying eddy viscosity brings are part of the comparison --
    those are exactly the terms an implementation that forgot to
    differentiate through nu_sgs would drop."""
    cs, delta, nu = 0.17, 0.2, 1e-3
    x, y, z, t = sp.symbols("x y z t", real=True)
    _, S = _sympy_les_residuals(nu)
    magS = sp.sqrt(2 * (S["Sxx"] ** 2 + S["Syy"] ** 2 + S["Szz"] ** 2
                        + 2 * S["Sxy"] ** 2 + 2 * S["Sxz"] ** 2 + 2 * S["Syz"] ** 2))
    nu_eff = nu + (cs * delta) ** 2 * magS

    closed, _ = _sympy_les_residuals(nu_eff)
    pts = _sample_points(n=24).detach().numpy()
    lambdified = {
        name: sp.lambdify((x, y, z, t), expr, "numpy") for name, expr in closed.items()
    }

    out = SmagorinskyResiduals(nu=nu, cs=cs, delta=delta)(_ExactFn(_torch_field), torch.tensor(pts, requires_grad=True))
    for name in closed:
        expected = np.array([float(lambdified[name](*row)) for row in pts])
        got = out[name].detach().numpy().reshape(-1)
        diff = np.abs(got - expected).max()
        assert diff < 1e-9, f"{name}: torch autograd vs. independent sympy closed form, max abs diff={diff:.3e}"


def test_wale_residual_matches_independent_sympy_closed_form():
    """Regression for WALEResiduals after its move onto the shared
    ``_LESResiduals3D`` base: the closed form of the WALE eddy viscosity
    (Nicoud & Ducros) and the full residual must still be reproduced to
    machine precision, term for term."""
    cw, delta, nu = 0.325, 0.2, 1e-3
    x, y, z, t, u, v, w, p = _sympy_field()

    u_x, u_y, u_z = sp.diff(u, x), sp.diff(u, y), sp.diff(u, z)
    v_x, v_y, v_z = sp.diff(v, x), sp.diff(v, y), sp.diff(v, z)
    w_x, w_y, w_z = sp.diff(w, x), sp.diff(w, y), sp.diff(w, z)
    g = sp.Matrix([[u_x, u_y, u_z], [v_x, v_y, v_z], [w_x, w_y, w_z]])
    S = (g + g.T) / 2
    g2 = g * g
    Sd = (g2 + g2.T) / 2 - sp.trace(g2) / 3 * sp.eye(3)
    SS = sum(S[i, j] ** 2 for i in range(3) for j in range(3))
    SdSd = sum(Sd[i, j] ** 2 for i in range(3) for j in range(3))
    nu_sgs = (cw * delta) ** 2 * SdSd ** sp.Rational(3, 2) / (SS ** sp.Rational(5, 2) + SdSd ** sp.Rational(5, 4))
    nu_eff = nu + nu_sgs

    closed, _ = _sympy_les_residuals(nu_eff)
    pts = _sample_points(n=20).detach().numpy()
    lambdified = {
        name: sp.lambdify((x, y, z, t), expr, "numpy") for name, expr in closed.items()
    }
    nu_sgs_fn = sp.lambdify((x, y, z, t), nu_sgs, "numpy")

    out = WALEResiduals(nu=nu, cw=cw, delta=delta)(_ExactFn(_torch_field), torch.tensor(pts, requires_grad=True))
    for name in closed:
        expected = np.array([float(lambdified[name](*row)) for row in pts])
        got = out[name].detach().numpy().reshape(-1)
        diff = np.abs(got - expected).max()
        assert diff < 1e-9, f"{name}: torch autograd vs. independent sympy closed form, max abs diff={diff:.3e}"

    expected_nu = np.array([float(nu_sgs_fn(*row)) for row in pts])
    assert np.abs(out["nu_sgs"].detach().numpy().reshape(-1) - expected_nu).max() < 1e-12


# ---------------------------------------------------------------------------
# van Driest damping
# ---------------------------------------------------------------------------

def test_van_driest_damping_has_the_expected_limits_and_factors_nu_sgs():
    y_plus = torch.tensor([[0.0], [1.0], [25.0], [500.0]])
    d = van_driest(y_plus)
    assert float(d[0]) == pytest.approx(0.0, abs=1e-12)          # D(0) = 0
    assert float(d[-1]) == pytest.approx(1.0, abs=1e-3)          # D -> 1 away from the wall
    assert float(d[1]) < float(d[2])                             # monotone in y+

    def field(x):
        xx, yy = x[:, 0:1], x[:, 1:2]
        u = 0.3 * xx + 0.55 * yy
        v = 0.55 * xx - 0.3 * yy
        return torch.cat([u, v, torch.zeros_like(xx), torch.zeros_like(xx)], dim=1)

    pts = _sample_points(n=6)
    undamped = SmagorinskyResiduals(nu=1e-5, cs=0.17, delta=0.1)(_ExactFn(field), pts)
    damped = SmagorinskyResiduals(
        nu=1e-5, cs=0.17, delta=0.1, damping=lambda x: van_driest(torch.full((x.shape[0], 1), 2.0)),
    )(_ExactFn(field), pts)
    factor = float(van_driest(torch.tensor([[2.0]])))
    assert np.allclose(
        damped["nu_sgs"].detach().numpy(),
        undamped["nu_sgs"].detach().numpy() * factor,
        rtol=0, atol=1e-16,
    )


# ---------------------------------------------------------------------------
# mesh-free test filter
# ---------------------------------------------------------------------------

def _regular_lattice(n: int = 5, h: float = 0.1):
    """``n^3`` points on a regular lattice, plus a dummy time column."""
    lin = np.arange(n) * h
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1).reshape(-1, 3)
    return torch.tensor(np.column_stack([grid, np.zeros(n ** 3)]), dtype=torch.float64)


def test_filter_stencil_reproduces_constants_and_linear_fields():
    """The filter must be a partition of unity (weights sum to one, so a
    constant survives) and, on a regular lattice with a symmetric 7-point
    stencil, reproduce a linear field exactly -- linear reproduction is
    what makes the test filter a *filter* rather than a smoothing kernel.

    Checked on interior points only: at the edge of a finite lattice the
    nearest-7 stencil is one-sided, so it is genuinely not symmetric and
    (correctly) does not reproduce a linear field."""
    x = _regular_lattice()
    idx, w = filter_stencil(x, neighbors=7, kernel="gaussian")
    assert idx.shape == w.shape == (x.shape[0], 7)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-15)
    assert (idx[:, 0] == np.arange(x.shape[0])).all()   # the point itself is always first

    ones = torch.ones((x.shape[0], 1), dtype=torch.float64)
    assert torch.allclose(apply_stencil(ones, idx, w), ones, atol=1e-15)

    pts = x[:, :3]
    linear = (2.0 * pts[:, 0:1] - 3.0 * pts[:, 1:2] + 0.5 * pts[:, 2:3] + 1.0)
    filtered = apply_stencil(linear, idx, w)

    coords = np.unique(pts[:, 0].numpy())
    interior = ((pts[:, 0] > coords[0]) & (pts[:, 0] < coords[-1])
                & (pts[:, 1] > coords[0]) & (pts[:, 1] < coords[-1])
                & (pts[:, 2] > coords[0]) & (pts[:, 2] < coords[-1])).numpy()
    assert interior.sum() > 0
    assert float((filtered - linear)[interior].abs().max()) < 1e-12


def test_filter_stencil_rejects_a_degenerate_stencil():
    x = _regular_lattice(n=3)
    with pytest.raises(ValueError, match="at least 2"):
        filter_stencil(x, neighbors=1)
    with pytest.raises(ValueError, match="kernel"):
        filter_stencil(x, neighbors=7, kernel="triangular")


# ---------------------------------------------------------------------------
# dynamic Smagorinsky: the properties that make C a Germano coefficient
# ---------------------------------------------------------------------------

def _dynamic_on(field, *, clip=False, delta=0.05, points=None, **kwargs):
    closure = DynamicSmagorinskyResiduals(nu=1e-5, delta=delta, clip=clip, **kwargs)
    x = _sample_points(n=60) if points is None else points
    out = closure(_ExactFn(field), x)
    return closure, out


def test_dynamic_coefficient_is_dimensionless_so_velocity_scaling_leaves_it_unchanged():
    """L scales like u^2 and M like u^2, so ``C = -<L:M> / (2 Delta^2 <M:M>)``
    has no dimensions: doubling every velocity must leave C unchanged and
    double ``nu_sgs`` (which carries one velocity). A sign error, a stray
    Delta or a missing factor of two in the contraction shows up here."""
    def field(scale):
        def f(x):
            y = _torch_field(x)
            return torch.cat([scale * y[:, :3], y[:, 3:4]], dim=1)
        return f

    pts = _sample_points(n=60)
    c1, o1 = _dynamic_on(field(1.0), points=pts)
    c2, o2 = _dynamic_on(field(2.0), points=pts)

    C1 = float(c1.last_germano["C"])
    C2 = float(c2.last_germano["C"])
    assert C1 != 0.0 and np.isfinite(C1)
    assert C2 == pytest.approx(C1, rel=1e-9), (C1, C2)

    nu1 = o1["nu_sgs"].detach().numpy().reshape(-1)
    nu2 = o2["nu_sgs"].detach().numpy().reshape(-1)
    assert np.allclose(nu2, 2.0 * nu1, rtol=1e-9, atol=0)


def test_dynamic_viscosity_grows_with_the_filter_width():
    """The test filter's width follows ``Delta`` (``DeltaHat = ratio*Delta``),
    so a wider filter -- a coarser resolution -- must *remove* more of the
    resolved strain and return a larger eddy viscosity. A stencil pinned to
    a fixed set of neighbours, independent of ``Delta``, would return the
    same viscosity at both widths; this is what tells the two apart."""
    pts = _sample_points(n=60)
    _, small = _dynamic_on(_torch_field, delta=0.02, clip=False, points=pts)
    _, large = _dynamic_on(_torch_field, delta=0.20, clip=False, points=pts)
    nu_s = np.abs(small["nu_sgs"].detach().numpy()).mean()
    nu_l = np.abs(large["nu_sgs"].detach().numpy()).mean()
    assert nu_l > nu_s, (nu_s, nu_l)


def test_dynamic_test_filter_width_zero_falls_back_to_the_grid_filter():
    """With ``Delta`` and ``test_ratio`` fixed, the default bandwidth is
    ``test_ratio * Delta`` -- an explicit bandwidth must reproduce it."""
    pts = _sample_points(n=40)
    default, _ = _dynamic_on(_torch_field, delta=0.05, clip=False, points=pts)
    explicit, _ = _dynamic_on(_torch_field, delta=0.05, clip=False, bandwidth=2.0 * 0.05, points=pts)
    assert np.isclose(float(default.last_germano["C"]), float(explicit.last_germano["C"]), rtol=0, atol=0)


def test_dynamic_coefficient_satisfies_the_lilly_least_squares_relation():
    """With clipping off, the reported C must be the minimiser of
    ``sum (L_ij + 2 C Delta^2 M_ij)^2`` -- i.e. the normal equation
    ``<L:M> + 2 C Delta^2 <M:M> = 0`` must hold on the very L and M the
    code exposed. This is the Germano-Lilly algebra itself, checked
    numerically instead of restated."""
    closure, _ = _dynamic_on(_torch_field, clip=False, delta=0.05)
    L = closure.last_germano["L"]
    M = closure.last_germano["M"]
    C = closure.last_germano["C"]

    d2 = 0.05 ** 2
    LM = float((L * M).sum(dim=(1, 2)).mean())
    MM = float((M * M).sum(dim=(1, 2)).mean())
    residual = LM + 2.0 * float(C) * d2 * MM

    scale = max(abs(LM), abs(2.0 * float(C) * d2 * MM), 1e-30)
    assert abs(residual) / scale < 1e-10, (
        f"normal equation residual {residual:.3e} is {abs(residual) / scale:.1%} of its own scale"
    )


def test_dynamic_clipping_keeps_the_viscosity_non_negative_and_finite():
    """A negative SGS viscosity is anti-diffusive and destabilises a
    transient solve; ``clip=True`` must floor it at zero without producing
    NaN/inf, on a field where the unclipped coefficient is negative."""
    pts = _sample_points(n=60)
    unclipped, _ = _dynamic_on(_torch_field, clip=False, points=pts)
    clipped, out = _dynamic_on(_torch_field, clip=True, points=pts)
    assert float(unclipped.last_germano["C"]) < 0.0, "test field should exercise the clipping branch"
    assert float(clipped.last_germano["C"]) == 0.0
    nu = out["nu_sgs"].detach().numpy()
    assert np.isfinite(nu).all() and (nu >= 0.0).all()


def test_dynamic_is_finite_on_a_zero_strain_field():
    """A uniform flow has no strain, so M is identically zero: the
    coefficient is 0/0 and must come back as a finite zero, not NaN."""
    def uniform(x):
        xx = x[:, 0:1]
        return torch.cat([
            0.0 * xx + 1.0,
            0.0 * xx + 2.0,
            0.0 * xx,
            0.0 * xx,
        ], dim=1)

    _, out = _dynamic_on(uniform, points=_sample_points(n=20))
    for name, value in out.items():
        assert torch.isfinite(value).all(), f"{name} is not finite on a zero-strain field"
    assert float(out["nu_sgs"].detach().abs().max()) < 1e-30


def test_dynamic_rejects_a_degenerate_test_filter():
    with pytest.raises(ValueError, match="test_ratio"):
        DynamicSmagorinskyResiduals(test_ratio=1.0)
    with pytest.raises(ValueError, match="averaging"):
        DynamicSmagorinskyResiduals(averaging="local")


def test_dynamic_pass_pointwise_averaging_also_gives_a_finite_positive_viscosity():
    _, out = _dynamic_on(_torch_field, clip=True, averaging="none", points=_sample_points(n=60))
    nu = out["nu_sgs"].detach().numpy()
    assert np.isfinite(nu).all()
    assert (nu >= 0.0).all()
    assert nu.max() > 0.0


def test_dynamic_accepts_a_precomputed_stencil_for_a_fixed_collocation_set():
    """Rebuilding the neighbour search on every residual evaluation is
    wasted work when the training points are fixed; a precomputed stencil
    must give exactly the same closure, not an approximation of it."""
    pts = _sample_points(n=40)
    reference, _ = _dynamic_on(_torch_field, delta=0.05, clip=False, points=pts)
    stencil = filter_stencil(pts, neighbors=8, kernel="gaussian", bandwidth=2.0 * 0.05)
    reused, _ = _dynamic_on(_torch_field, delta=0.05, clip=False, points=pts, stencil=stencil)
    assert float(reference.last_germano["C"]) == float(reused.last_germano["C"])


def test_dynamic_closure_trains_through_the_filter():
    """The mesh-free filter averages *values already computed*, so gradients
    must reach the network across neighbour links -- if the stencil were
    built on a detached copy of the values (an easy mistake) the loss would
    still fall to zero but no gradient would flow."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4),
    ).double()
    x = _sample_points(n=30)
    closure = DynamicSmagorinskyResiduals(nu=1e-5, delta=0.05, clip=False)
    out = closure(net, x)
    loss = sum(v.pow(2).mean() for k, v in out.items() if k != "nu_sgs")
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0.0 for g in grads)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def test_get_rans_preset_resolves_the_les_closures():
    assert isinstance(get_rans_preset("smagorinsky", delta=0.1), SmagorinskyResiduals)
    assert isinstance(get_rans_preset("dynamic-smagorinsky", delta=0.1), DynamicSmagorinskyResiduals)
    assert isinstance(get_rans_preset("dsmag", delta=0.1), DynamicSmagorinskyResiduals)
    assert isinstance(get_rans_preset("wale", delta=0.1), WALEResiduals)
