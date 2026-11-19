"""2-D plane elasticity (plane stress / plane strain) in the Q1 finite-element solver.

``FEMSolver.solve_from_spec`` used to send every kind that was not axisymmetric to the Poisson
assembly, so a ``linear_elasticity`` ProblemSpec at dim=2 was solved as a scalar Laplace problem.
These tests pin the two-field (ux, uy) path with analytic answers only:

* uniaxial tension in plane stress -- ``u = (s*x/E, -nu*s*y/E)`` is a linear field, so a Q1 mesh
  reproduces it nodally exactly (the patch test of Zienkiewicz, Taylor & Zhu, *The Finite Element
  Method*, 7th ed. (2013), ch. 6);
* the same prescribed axial strain in plane strain -- the lateral contraction it needs is
  ``-nu/(1-nu) * s/E`` instead of ``-nu * s/E`` (Timoshenko & Goodier, *Theory of Elasticity*,
  the plane stress / plane strain constitutive relations);
* the strain energy of a prescribed uniform strain ``eps0`` in x, ``0.5 * D[0,0] * eps0^2 * V``,
  whose plane-strain / plane-stress ratio is ``(1-2*nu)/(1-nu)^2``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_physics.pde_environment.conditions import DirichletBC
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_simulation.numerical_solvers.fem import (
    FEMSolver,
    _assemble_plane_elasticity,
    _make_structured_mesh,
)

E_STEEL = 210e9
NU_STEEL = 0.3
SIGMA = 1.0e6                            # prescribed axial stress [Pa]
BOUNDS = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
FIELDS = ("ux", "uy")


def _spec(kind: str, params: dict, conditions=(), *, dim: int = 2, fields=FIELDS,
          bounds=BOUNDS) -> ProblemSpec:
    coords = ("x", "y") if dim == 2 else ("x", "y", "z")
    return ProblemSpec(
        name="plane_elasticity",
        dim=dim,
        coords=coords,
        fields=fields,
        pde=PDETermSpec(kind, fields, coords, params),
        conditions=tuple(conditions),
        domain_bounds=dict(bounds),
    )


def _const(fn):
    """A builder-style Dirichlet value_fn (numpy in, (N, 1) out) from a scalar or a callable."""
    if callable(fn):
        return lambda X, ctx: np.asarray(fn(np.asarray(X)), dtype=np.float64).reshape(-1, 1)
    return lambda X, ctx: np.full((X.shape[0], 1), float(fn))


def _uniaxial_bcs(sx: float = SIGMA, E: float = E_STEEL):
    """Hold ux on the left/right edges (0 and the exact axial value), uy on the bottom edge.

    With the top and the lateral contraction free, ``u = (sx*x/E, -c*sx*y/E)`` is the exact
    solution for both models -- c = nu in plane stress, c = nu/(1-nu) in plane strain -- so the
    interior is a genuine solve and not a boundary-value copy.
    """
    return (
        DirichletBC("left_ux", ("ux",), "tag", {"tag": "left"}, _const(0.0)),
        DirichletBC("right_ux", ("ux",), "tag", {"tag": "right"}, _const(sx / E)),
        DirichletBC("bottom_uy", ("uy",), "tag", {"tag": "bottom"}, _const(0.0)),
    )


def _expected(nodes: torch.Tensor, c: float, sx: float = SIGMA, E: float = E_STEEL):
    """Exact displacement field of the uniaxial state: (ux, uy) at every node."""
    ex = sx * nodes[:, 0].double() / E
    ey = -c * sx * nodes[:, 1].double() / E
    return ex, ey


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """max|a - b| / max|b|."""
    return float((a - b).abs().max() / b.abs().max())


# ---------------------------------------------------------------------------
# 1. Patch test: a Q1 element reproduces the linear uniaxial field exactly
# ---------------------------------------------------------------------------

def test_plane_stress_uniaxial_tension_patch_test():
    """Plane stress, ux = s*x/E on the lateral edges, uy = 0 at the bottom: the exact field
    u = (s*x/E, -nu*s*y/E) is linear, so every interior node must reproduce it.

    Measured on nx=ny=8 (float64 assembly, direct solve): relative error 3.6e-8 for ux and
    4.3e-8 for uy, i.e. round-off, not discretisation.
    """
    out = FEMSolver(nx=8, ny=8, solver="direct")(
        spec=_spec("linear_elasticity", {"E": E_STEEL, "nu": NU_STEEL}, _uniaxial_bcs()))
    nodes = out.extras["nodes"]
    assert out.extras["method"] == "fem_plane_stress"
    ex, ey = _expected(nodes, NU_STEEL)
    assert _rel(out.extras["ux"].reshape(-1), ex) < 1e-6
    assert _rel(out.extras["uy"].reshape(-1), ey) < 1e-6
    # the prescribed edges carry exactly their Dirichlet value
    assert float(out.extras["ux"][-1, 0]) == pytest.approx(SIGMA / E_STEEL, rel=1e-12)
    assert float(out.extras["ux"][0, 0]) == 0.0
    assert float(out.extras["uy"][:, 0].abs().max()) == 0.0
    # ... and the free top edge really contracted (this is not an all-zero field)
    assert float(out.extras["uy"][:, -1].min()) < 0.0


def test_patch_test_holds_on_a_nonunit_box_and_with_callable_boundary_values():
    """The same field on the box [0,2]x[0,3] with the value prescribed by a callable rather than a
    scalar: the patch test does not depend on the mesh being the unit square."""
    bounds = {"x": (0.0, 2.0), "y": (0.0, 3.0)}
    bcs = (
        DirichletBC("left_ux", ("ux",), "tag", {"tag": "left"}, _const(0.0)),
        DirichletBC("right_ux", ("ux",), "tag", {"tag": "right"},
                    _const(lambda P: SIGMA * P[:, 0] / E_STEEL)),
        DirichletBC("bottom_uy", ("uy",), "tag", {"tag": "bottom"},
                    _const(lambda P: -NU_STEEL * SIGMA * P[:, 1] / E_STEEL)),
    )
    out = FEMSolver(nx=6, ny=9, solver="direct")(
        spec=_spec("plane_stress", {"E": E_STEEL, "nu": NU_STEEL}, bcs, bounds=bounds))
    nodes = out.extras["nodes"]
    ex, ey = _expected(nodes, NU_STEEL)
    assert _rel(out.extras["ux"].reshape(-1), ex) < 1e-6
    assert _rel(out.extras["uy"].reshape(-1), ey) < 1e-6


# ---------------------------------------------------------------------------
# 2. Plane strain and plane stress answer differently, by the analytic factor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,model,contraction", [
    ("plane_stress", "plane_stress", NU_STEEL),
    ("plane_strain", "plane_strain", NU_STEEL / (1.0 - NU_STEEL)),
    # the kind names the presets declare (presets/structural.py)
    ("linear_elasticity_plane_stress", "plane_stress", NU_STEEL),
    ("linear_elasticity_plane_strain", "plane_strain", NU_STEEL / (1.0 - NU_STEEL)),
])
def test_lateral_contraction_is_the_analytic_one_for_each_model(kind, model, contraction):
    """Same prescribed axial strain, same BCs; only the constitutive matrix differs.

    Plane stress relaxes sigma_yy = 0 through eps_yy = -nu*eps_xx, plane strain keeps eps_zz = 0
    and needs eps_yy = -nu/(1-nu)*eps_xx, so the free-edge contraction differs by 1/(1-nu) = 1.4286.
    Measured: -1.42857145e-6 (plane stress) and -2.04081632e-6 (plane strain) against the analytic
    -1.42857143e-6 and -2.04081633e-6.
    """
    out = FEMSolver(nx=8, ny=8, solver="direct")(
        spec=_spec(kind, {"E": E_STEEL, "nu": NU_STEEL}, _uniaxial_bcs()))
    nodes = out.extras["nodes"]
    assert out.extras["method"] == f"fem_{model}"
    assert out.extras["model"] == model
    _, ey = _expected(nodes, contraction)
    assert _rel(out.extras["uy"].reshape(-1), ey) < 1e-6     # exact contraction, not a ratio
    # the axial response is the same statically determinate one for both models
    ex, _ = _expected(nodes, contraction)
    assert _rel(out.extras["ux"].reshape(-1), ex) < 1e-6


def test_plane_strain_is_stiffer_by_the_analytic_energy_ratio():
    """Strain energy of the prescribed uniform strain u = (eps0*x, 0) is 0.5*D[0,0]*eps0^2*V*t.

    D_ps[0,0] = E/(1-nu^2) and D_pse[0,0] = E*(1-nu)/((1+nu)*(1-2nu)), so the plane-stress energy
    is smaller by (1-2nu)/(1-nu)^2 = 0.81633 at nu = 0.3. Measured: 0.81632649 against 0.81632653
    (the ratio of two sum-of-products, so a few ulp off).

    Tolerances are 1e-6 relative, not machine epsilon: the shared Q1 shape-function helpers
    (``_q1_shape``/``_q1_grad``) build their tables in float32, so every integrand here inherits a
    ~1e-8 relative floor even though this assembly works in float64.
    """
    nodes, elems = _make_structured_mesh(8, 8, BOUNDS)
    eps0 = 1e-4
    u = torch.zeros(2 * nodes.shape[0], dtype=torch.float64)
    u[0::2] = eps0 * nodes[:, 0].double()                  # ux = eps0*x, uy = 0
    energies = {}
    for model in ("plane_stress", "plane_strain"):
        K, f = _assemble_plane_elasticity(nodes, elems, E_STEEL, NU_STEEL, model=model)
        assert f.abs().max() == 0.0                        # no body force given
        assert torch.equal(K, K.T)                         # symmetric stiffness
        energies[model] = float(0.5 * u @ K @ u)
        # rows sum to zero: a constant displacement carries no strain, so no nodal force
        assert float(K.sum(dim=1).abs().max()) < 1e-10 * float(K.abs().max())

    D_ps = E_STEEL / (1.0 - NU_STEEL ** 2)
    D_pse = E_STEEL * (1.0 - NU_STEEL) / ((1.0 + NU_STEEL) * (1.0 - 2.0 * NU_STEEL))
    assert energies["plane_stress"] == pytest.approx(0.5 * D_ps * eps0 ** 2, rel=1e-6)
    assert energies["plane_strain"] == pytest.approx(0.5 * D_pse * eps0 ** 2, rel=1e-6)
    ratio = energies["plane_stress"] / energies["plane_strain"]
    assert ratio == pytest.approx((1.0 - 2.0 * NU_STEEL) / (1.0 - NU_STEEL) ** 2, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Loads and the parameter paths
# ---------------------------------------------------------------------------

def test_body_force_is_integrated_per_unit_volume_times_thickness():
    """sum(f) must be b * volume * thickness, on both components: the load is
    integral(N_a * b * t * detJ) and the Q1 partition of unity sums to the area."""
    nodes, elems = _make_structured_mesh(4, 6, {"x": (0.0, 2.0), "y": (0.0, 3.0)})
    bx, by, t = 2.5e5, -1.0e5, 0.01
    K, f = _assemble_plane_elasticity(nodes, elems, E_STEEL, NU_STEEL,
                                      thickness=t, body_force=(bx, by))
    volume = 2.0 * 3.0
    assert float(f[0::2].sum()) == pytest.approx(bx * volume * t, rel=1e-6)
    assert float(f[1::2].sum()) == pytest.approx(by * volume * t, rel=1e-6)
    assert float(f.sum()) == pytest.approx((bx + by) * volume * t, rel=1e-6)


def test_body_force_drives_a_solve_and_plane_stress_is_the_more_compliant_model():
    """ux = 0 on the left edge, uy = 0 at the bottom, uniform axial body force: both models solve
    the same load and plane stress (the smaller D[0,0]) deflects more. Measured ratio of
    max|ux|, plane stress / plane strain: 1.0799 -- below the statically determinate bar's
    D_pse[0,0]/D_ps[0,0] = 1.225, because this square body is laterally restrained by the
    uy = 0 edge rather than free."""
    bcs = (
        DirichletBC("left_ux", ("ux",), "tag", {"tag": "left"}, _const(0.0)),
        DirichletBC("bottom_uy", ("uy",), "tag", {"tag": "bottom"}, _const(0.0)),
    )
    params = {"E": E_STEEL, "nu": NU_STEEL, "body_force": (1.0e6, 0.0)}
    out = {m: FEMSolver(nx=16, ny=16, solver="direct")(
        spec=_spec("linear_elasticity", {**params, "model": m}, bcs))
        for m in ("plane_stress", "plane_strain")}
    ux_ps = float(out["plane_stress"].extras["ux"].abs().max())
    ux_pse = float(out["plane_strain"].extras["ux"].abs().max())
    assert ux_ps > 0.0 and ux_pse > 0.0
    assert ux_ps > ux_pse                                       # plane stress is the softer model
    assert 1.0 < ux_ps / ux_pse < 1.25                          # bounded by the determinate bar
    diff = (out["plane_stress"].extras["ux"] - out["plane_strain"].extras["ux"]).abs().max()
    assert float(diff) > 1e-3 * ux_ps                           # the two solves really differ
    # equilibrium: the load reached the right end at all (nothing pinned the far edge)
    assert float(out["plane_stress"].extras["ux"][-1, :].mean()) > 0.0


def test_lame_only_parameters_are_converted_to_e_and_nu():
    """params with only lambda/mu: E = mu*(3*lam+2*mu)/(lam+mu), nu = lam/(2*(lam+mu))."""
    lam = E_STEEL * NU_STEEL / ((1 + NU_STEEL) * (1 - 2 * NU_STEEL))
    mu = E_STEEL / (2 * (1 + NU_STEEL))
    out = FEMSolver(nx=4, ny=4, solver="direct")(
        spec=_spec("linear_elasticity", {"lambda": lam, "mu": mu}, _uniaxial_bcs()))
    assert out.extras["E"] == pytest.approx(E_STEEL, rel=1e-12)
    assert out.extras["nu"] == pytest.approx(NU_STEEL, rel=1e-12)
    ex, ey = _expected(out.extras["nodes"], NU_STEEL)
    assert _rel(out.extras["ux"].reshape(-1), ex) < 1e-6


def test_defaults_apply_when_no_material_is_given():
    """The elasticity capability's own defaults (E=200e9, nu=0.3) apply only when nothing is given."""
    out = FEMSolver(nx=4, ny=4, solver="direct")(
        spec=_spec("elasticity", {}, (DirichletBC("hold", ("ux", "uy"), "all", None,
                                                  _const(0.0)),)))
    assert out.extras["E"] == 200e9
    assert out.extras["nu"] == 0.3


def test_preset_style_two_field_condition_clamps_both_components():
    """A condition naming both fields with a (N, 2) value_fn -- how presets/structural.py
    describes its "fixed" boundary -- holds ux and uy, and a single unnamed field holds both."""
    clamped = FEMSolver(nx=4, ny=4, solver="direct")(
        spec=_spec("plane_stress_2d", {"E": E_STEEL, "nu": NU_STEEL},
                   (DirichletBC("fixed", FIELDS, "all", None,
                                lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float64)),)))
    assert clamped.extras["ux"].abs().max() == 0.0
    assert clamped.extras["uy"].abs().max() == 0.0

    unnamed = FEMSolver(nx=4, ny=4, solver="direct")(
        spec=_spec("plane_strain_2d", {"E": E_STEEL, "nu": NU_STEEL},
                   (DirichletBC("hold", ("u",), "all", None, _const(0.0)),)))
    assert unnamed.extras["ux"].abs().max() == 0.0
    assert unnamed.extras["uy"].abs().max() == 0.0


def test_unknown_model_is_refused():
    with pytest.raises(ValueError):
        FEMSolver(nx=2, ny=2, solver="direct")(
            spec=_spec("linear_elasticity",
                       {"E": E_STEEL, "nu": NU_STEEL, "model": "plane_curved"}, ()))


# ---------------------------------------------------------------------------
# 4. The paths that already worked are unchanged
# ---------------------------------------------------------------------------

def _quadratic(p):
    p = torch.as_tensor(np.asarray(p))
    return p[:, 0] ** 2 + p[:, 1] ** 2            # -lap u = -4


def test_poisson_kind_still_dispatches_to_poisson():
    """kind "poisson" keeps the scalar assembly: extras carry no "method" key and the field is
    still the one-scalar result, reproducing the quadratic nodally exactly (as in
    tests/test_solver_exact_solutions_batch4.py)."""
    bc = DirichletBC("wall", ("u",), "callable",
                     lambda X, ctx: ((X[:, 0] < 1e-6) | (X[:, 0] > 1 - 1e-6)
                                     | (X[:, 1] < 1e-6) | (X[:, 1] > 1 - 1e-6)),
                     lambda X, ctx: np.asarray(_quadratic(X), dtype=np.float64).reshape(-1, 1))
    spec = _spec("poisson", {"source": -4.0}, (bc,), fields=("u",))
    out = FEMSolver(nx=8, ny=8, solver="direct")(spec=spec)
    assert "method" not in out.extras
    assert out.extras["kind"] == "poisson"
    assert out.result.shape == (9, 9)
    exact = _quadratic(out.extras["nodes"])
    assert float((out.result.reshape(-1) - exact).abs().max()) < 1e-5


def test_three_dimensional_elasticity_still_falls_through():
    """The mesh here is 2-D: a dim=3 linear_elasticity spec must NOT be solved as plane stress
    (kind "linear_elasticity" is shared with the 3-D presets, e.g. linear_elasticity_3d),
    so it keeps taking the previous scalar path until a 3-D elasticity solver handles it."""
    bc = DirichletBC("wall", ("u",), "callable",
                     lambda X, ctx: ((X[:, 0] < 1e-6) | (X[:, 0] > 1 - 1e-6)
                                     | (X[:, 1] < 1e-6) | (X[:, 1] > 1 - 1e-6)),
                     lambda X, ctx: np.zeros((X.shape[0], 1)))
    spec = _spec("linear_elasticity", {"E": E_STEEL, "nu": NU_STEEL}, (bc,), dim=3, fields=("u",))
    out = FEMSolver(nx=4, ny=4, solver="direct")(spec=spec)
    assert "method" not in out.extras
    assert out.result.shape == (5, 5)
