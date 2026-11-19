"""Axisymmetric linear elasticity on the Q1 finite-element solver.

``FEMSolver.solve_from_spec`` built its ``axisymmetric_linear_elasticity`` system with 2N unknowns
(``[ur0, uz0, ur1, uz1, ...]``) and then reshaped them as the scalar N-node grid, so the very first
call of a spec with this kind ended in ``RuntimeError: shape '[5, 5]' is invalid for input of size
50``; the Dirichlet conditions were applied to *node* indices used as *DOF* indices, which on an
interleaved two-field system clamps half of the wrong unknowns. Both are fixed (the post-processing
and the condition dispatch; the assembly was already the correct r-weighted weak form). These tests
pin the contract and the physics:

* **Lamé's thick-walled cylinder** under internal pressure, in plane strain, which is the state a
  body of revolution with its two ends held at ``u_z = 0`` is in (Timoshenko & Goodier, *Theory of
  Elasticity*, 3rd ed., ch. 4 and 9). With ``a`` the bore, ``b`` the outer radius, ``p`` the
  internal pressure, ``C = p*a^2/(b^2 - a^2)``,

      sigma_rr = C (1 - b^2/r^2),   sigma_tt = C (1 + b^2/r^2),   sigma_zz = nu (sigma_rr + sigma_tt)
      eps_tt   = (1/E)[sigma_tt - nu (sigma_rr + sigma_zz)] = (C/E)(1+nu)[(1-2nu) + b^2/r^2]
      u_r      = r eps_tt = (C/E)(1+nu)[(1-2nu) r + b^2/r]

  which gives ``sigma_rr(a) = -p``, ``sigma_rr(b) = 0`` and the classical hoop peak
  ``sigma_tt(a) = C (1 + b^2/a^2) = p (a^2 + b^2)/(b^2 - a^2)``. ``_lame_ur`` is checked against
  the library's own axisymmetric operator (``pinneapple_physics.solid.axisymmetric``) before
  anything is compared to it, so the reference is not taken on trust.

* a **manufactured** axisymmetric field, ``u_r = a1 r + a3 r z``, ``u_z = b0 + b2 z``, whose
  divergence is the constant ``(0, 2 (lambda + mu) a3)`` and so is an exact solution under the
  uniform body force ``b = (0, -2 (lambda + mu) a3)``. It is bilinear, hence in the Q1 space, and
  its load is integrated exactly by 2x2 Gauss on rectangles, so the nodal answer must reproduce it
  to round-off -- the hoop term, the ``r`` weight, the shear coupling and the load all enter.

* the scalar Poisson path and the 2-D plane-stress path answering exactly as they did before.

Measured errors are quoted in each docstring. They carry a ~1e-8 relative floor from the shared Q1
shape-function helpers, whose tables are built in float32 (``_q1_shape``/``_q1_grad``), even though
both elasticity assemblies work in float64.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pinneapple_physics.pde_environment.conditions import DirichletBC
from pinneapple_physics.pde_environment.presets import get_preset
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_physics.solid.axisymmetric import (
    equilibrium_residual_rz,
    lame_parameters,
    stresses_rz,
)
from pinneapple_simulation.numerical_solvers.fem import (
    FEMSolver,
    _assemble_axisym_elasticity,
    _axisym_components,
    _make_structured_mesh,
    _plane_components,
)
from pinneapple_simulation.numerical_solvers.registry import SolverRegistry

E_STEEL = 210e9
NU_STEEL = 0.3
A_IN = 0.05                      # bore radius [m]
B_OUT = 0.10                     # outer radius [m]
HEIGHT = 0.20                    # axial length of the slice [m]
P_IN = 20e6                      # internal pressure [Pa]
BOUNDS = {"x": (A_IN, B_OUT), "y": (0.0, HEIGHT)}
COORDS = ("x", "y")              # this solver's grid: x IS the radius r, y the axial z
FIELDS = ("u_r", "u_z")

LAM, MU = lame_parameters(E_STEEL, NU_STEEL)
C_LAME = P_IN * A_IN ** 2 / (B_OUT ** 2 - A_IN ** 2)


# ---------------------------------------------------------------------------
# Lamé reference
# ---------------------------------------------------------------------------

def _lame_ur(r):
    """Radial displacement of the plane-strain Lamé cylinder, ``(C/E)(1+nu)[(1-2nu) r + b^2/r]``."""
    r = np.asarray(r, dtype=np.float64)
    return (C_LAME / E_STEEL) * (1.0 + NU_STEEL) * ((1.0 - 2.0 * NU_STEEL) * r + B_OUT ** 2 / r)


def _lame_sigma_rr(r):
    """``C (1 - b^2/r^2)``: ``-p`` at the bore, 0 at the outer wall."""
    r = np.asarray(r, dtype=np.float64)
    return C_LAME * (1.0 - B_OUT ** 2 / r ** 2)


def _lame_sigma_tt(r):
    """``C (1 + b^2/r^2)``, the hoop stress; its peak is at the bore."""
    r = np.asarray(r, dtype=np.float64)
    return C_LAME * (1.0 + B_OUT ** 2 / r ** 2)


# ---------------------------------------------------------------------------
# Specs and conditions
# ---------------------------------------------------------------------------

def _spec(params: dict, conditions=(), *, bounds=None, fields=FIELDS,
          coords=COORDS, kind="axisymmetric_linear_elasticity") -> ProblemSpec:
    return ProblemSpec(
        name="axisymmetric_elasticity",
        dim=2,
        coords=tuple(coords),
        fields=tuple(fields),
        pde=PDETermSpec(kind, tuple(fields), tuple(coords), params),
        conditions=tuple(conditions),
        domain_bounds=dict(BOUNDS if bounds is None else bounds),
    )


def _const(fn):
    """A builder-style Dirichlet value_fn (numpy in, (N, 1) out) from a scalar or a callable."""
    if callable(fn):
        return lambda X, ctx: np.asarray(fn(np.asarray(X)), dtype=np.float64).reshape(-1, 1)
    return lambda X, ctx: np.full((X.shape[0], 1), float(fn), dtype=np.float64)


def _lame_conditions() -> tuple:
    """``u_r`` = the exact Lamé value at the bore and at the outer wall, ``u_z = 0`` at both ends.

    These are the boundary conditions this solver can express (Dirichlet only -- it has no
    traction condition, so "internal pressure" cannot be applied as ``sigma_rr = -p`` directly).
    They are consistent with the exact Lamé state, which has ``u_z = 0`` and an ``u_r`` that varies
    with ``r`` alone, so the interior is a genuine solve and not a copy of the boundary: a wrong
    stiffness would leave the interior off the analytic profile. See
    ``test_the_lame_field_this_file_compares_against_is_in_equilibrium`` for the verification that
    these data do reproduce ``sigma_rr = -p`` / ``0``.
    """
    return (
        DirichletBC("bore_ur", ("u_r",), "tag", {"tag": "left"}, _const(lambda X: _lame_ur(X[:, 0]))),
        DirichletBC("wall_ur", ("u_r",), "tag", {"tag": "right"}, _const(lambda X: _lame_ur(X[:, 0]))),
        DirichletBC("base_uz", ("u_z",), "tag", {"tag": "bottom"}, _const(0.0)),
        DirichletBC("top_uz", ("u_z",), "tag", {"tag": "top"}, _const(0.0)),
    )


def _lame_spec(params=None) -> ProblemSpec:
    return _spec({"E": E_STEEL, "nu": NU_STEEL} if params is None else params, _lame_conditions())


def _rel(a, b) -> float:
    """max|a - b| / max|b|."""
    a, b = torch.as_tensor(np.asarray(a)), torch.as_tensor(np.asarray(b))
    return float((a - b).abs().max() / b.abs().max())


def _radii(out) -> np.ndarray:
    """The radius of each row of the returned grids (x is the radial coordinate)."""
    nx = out.extras["ux"].shape[0] - 1
    return np.linspace(A_IN, B_OUT, nx + 1) if nx else np.asarray([A_IN])


# ---------------------------------------------------------------------------
# 1. The branch answers, with two grids, and says which method it is
# ---------------------------------------------------------------------------

def test_the_axisymmetric_branch_returns_two_nodal_grids():
    """The reported failure: one DirichletBC over the whole boundary of an axisymmetric spec.

    ``SolverRegistry.build("fem", nx=4, ny=4).solve(spec)`` raised ``RuntimeError: shape '[5, 5]'
    is invalid for input of size 50`` -- 2N unknowns reshaped as the N-node scalar grid. The result
    is now the (nx+1, ny+1) radial grid, both displacement grids are in extras, and the method says
    which formulation answered.
    """
    spec = _spec(
        {"E": 210e9, "nu": 0.3},
        (DirichletBC("held", ("u_r", "u_z"), "tag", {"tag": "boundary"},
                     lambda X, ctx: np.zeros((X.shape[0], 2), dtype=np.float64)),),
    )
    out = SolverRegistry.build("fem", nx=4, ny=4).solve(spec)
    assert out.extras["method"] == "fem_axisymmetric"
    assert out.extras["formulation"] == "axisymmetric_rz"
    assert out.result.shape == (5, 5)                       # the radial displacement grid
    assert tuple(out.extras["ux"].shape) == (5, 5)
    assert tuple(out.extras["uy"].shape) == (5, 5)
    assert torch.equal(out.extras["ur"], out.extras["ux"])
    assert torch.equal(out.extras["uz"], out.extras["uy"])
    assert out.extras["u"].numel() == 2 * 25                # the raw interleaved (2N,) vector
    assert torch.equal(out.result, out.extras["ux"])
    assert float(out.extras["ux"].abs().max()) == 0.0       # a held boundary, an unloaded body
    assert float(out.extras["uy"].abs().max()) == 0.0
    assert out.extras["E"] == 210e9 and out.extras["nu"] == 0.3


def test_the_linear_elasticity_axisymmetric_alias_takes_the_same_branch():
    """``linear_elasticity_axisymmetric`` was dispatched to the axisymmetric assembly too."""
    out = FEMSolver(nx=3, ny=3, solver="direct")(
        spec=_spec({"E": E_STEEL, "nu": NU_STEEL}, _lame_conditions(), kind="linear_elasticity_axisymmetric"))
    assert out.extras["method"] == "fem_axisymmetric"
    assert out.extras["kind"] == "linear_elasticity_axisymmetric"


# ---------------------------------------------------------------------------
# 2. Which component a condition holds (the mirror of the post-processing bug)
# ---------------------------------------------------------------------------

def test_condition_field_names_pick_the_axisymmetric_component():
    """``("u_z",)`` -- the presets' own name for the axial field -- must hold ONLY ``u_z``.

    On the interleaved ``[u_r, u_z]`` system a condition that clamped both components would hold
    the radial displacement wherever only the axial one is meant, which is a much stiffer boundary
    (a fixed bore rather than a sliding base). ``("u",)`` keeps the scalar convention of holding
    both, and the plane table is untouched by the radial/axial names.
    """
    cond = lambda names: SimpleNamespace(fields=names)      # noqa: E731 - a stand-in condition
    assert _axisym_components(cond(("u_z",))) == ((1, 0),)
    assert _axisym_components(cond(("uz",))) == ((1, 0),)
    assert _axisym_components(cond(("u_r",))) == ((0, 0),)
    assert _axisym_components(cond(("ur",))) == ((0, 0),)
    assert _axisym_components(cond(("u_r", "u_z"))) == ((0, 0), (1, 1))
    assert _axisym_components(cond(("u",))) == ((0, 0), (1, 0))      # one unnamed value: both
    assert _plane_components(cond(("ux", "uy"))) == ((0, 0), (1, 1))
    assert _plane_components(cond(("u_z",))) == ((0, 0), (1, 0))     # unchanged for the plane path


def test_an_axial_only_condition_leaves_the_radial_field_alone():
    """A callable selector plus a ``("u_z",)`` condition on a mid-height line of nodes: the axial
    displacement there is held at zero exactly, and the radial field keeps its Lamé value on that
    line (had the condition clamped ``u_r`` too, the whole cylinder would be split and the radial
    displacement on that line would be zero)."""
    middle = lambda X, ctx: np.abs(X[:, 1] - 0.5 * HEIGHT) < 1e-12      # noqa: E731
    conditions = _lame_conditions() + (
        DirichletBC("mid_uz", ("u_z",), "callable", middle, _const(0.0)),
    )
    out = FEMSolver(nx=8, ny=4, solver="direct")(spec=_spec({"E": E_STEEL, "nu": NU_STEEL}, conditions))
    j = int(np.argmin(np.abs(np.linspace(0.0, HEIGHT, 5) - 0.5 * HEIGHT)))
    assert float(out.extras["uy"][:, j].abs().max()) < 1e-15           # the axial field, held
    assert _rel(out.extras["ux"][1:-1, j].numpy(), _lame_ur(_radii(out)[1:-1])) < 1e-3


# ---------------------------------------------------------------------------
# 3. A spec that keys its domain ("r", "z") is meshed on those metres
# ---------------------------------------------------------------------------

def test_a_domain_keyed_r_and_z_is_meshed_on_those_metres():
    """``axisymmetric_linear_elasticity_2d`` declares ``domain_bounds={"r": ..., "z": ...}``.

    The solver's grid is (x, y) = (r, z); without the alias the preset spec would have been meshed
    on the default unit square -- for a body of revolution, silently including the axis and the
    wrong radii. Here the preset's own spec is solved: the nodes must span its radii, its callable
    ``fixed_base_uz`` selector must find the base, and only ``u_z`` may be held there.
    """
    preset = get_preset("axisymmetric_linear_elasticity_2d", fix_z_bottom=True)
    assert preset.domain_bounds == {"r": (10.0, 50.0), "z": (0.0, 100.0)}
    out = FEMSolver(nx=4, ny=4, solver="direct")(spec=preset)
    nodes = out.extras["nodes"]
    assert float(nodes[:, 0].min()) == pytest.approx(10.0) and float(nodes[:, 0].max()) == pytest.approx(50.0)
    assert float(nodes[:, 1].min()) == pytest.approx(0.0) and float(nodes[:, 1].max()) == pytest.approx(100.0)
    assert float(out.extras["uy"][:, 0].abs().max()) == 0.0      # u_z held at the base
    assert out.extras["method"] == "fem_axisymmetric"


# ---------------------------------------------------------------------------
# 4. Lamé, in equilibrium, before it is used as a reference
# ---------------------------------------------------------------------------

def test_the_lame_field_this_file_compares_against_is_in_equilibrium():
    """The reference of the next two tests, checked with the library's own axisymmetric operator.

    ``pinneapple_physics.solid.axisymmetric`` differentiates an (r, z) displacement field by
    autograd (its own tests compare it with the same closed forms). For the field used here it must
    report ``div(sigma) = 0`` in the interior and ``sigma_rr = -p`` at the bore, ``0`` at the outer
    wall -- i.e. the Dirichlet data below are exactly the plane-strain Lamé cylinder under internal
    pressure ``p`` and no external pressure.
    """
    x = torch.tensor([[0.07, 0.11], [0.09, 0.02], [0.06, 0.19]], dtype=torch.float64, requires_grad=True)
    r = x[:, 0:1]
    ur = (C_LAME / E_STEEL) * (1.0 + NU_STEEL) * ((1.0 - 2.0 * NU_STEEL) * r + B_OUT ** 2 / r)
    u = torch.cat([ur, torch.zeros_like(r)], dim=1)
    residual = equilibrium_residual_rz(u, x, E_STEEL, NU_STEEL).detach()
    assert residual.shape == (3, 2)
    assert float(residual.abs().max()) < 1e-9 * C_LAME / A_IN      # [Pa/m]

    walls = torch.tensor([[A_IN, 0.0], [B_OUT, 0.0]], dtype=torch.float64, requires_grad=True)
    rw = walls[:, 0:1]
    uw = torch.cat([(C_LAME / E_STEEL) * (1.0 + NU_STEEL) * ((1.0 - 2.0 * NU_STEEL) * rw + B_OUT ** 2 / rw),
                    torch.zeros_like(rw)], dim=1)
    s_rr = stresses_rz(uw, walls, E_STEEL, NU_STEEL)["s_rr"].detach().ravel()
    assert float(s_rr[0]) == pytest.approx(-P_IN, rel=1e-12)       # internal pressure
    assert float(s_rr[1]) == pytest.approx(0.0, abs=1e-6)          # zero external pressure


# ---------------------------------------------------------------------------
# 5. Lamé, against the finite elements
# ---------------------------------------------------------------------------

def test_lame_thick_cylinder_converges_to_the_analytic_displacement_in_second_order():
    """Nodal ``u_r`` of the solved cylinder against ``(C/E)(1+nu)[(1-2nu) r + b^2/r]``.

    ``A r + B/r`` is not in the Q1 space, so this is a discretisation error, not a round-off one:
    it must fall at second order under radial refinement. Measured relative error on the interior
    nodes (ny = 2, direct solve): 1.159e-4 at nx=8, 2.811e-5 at 16, 6.866e-6 at 32, 1.706e-6 at 64
    -- ratios 4.12, 4.09, 4.02. The axial displacement stays at round-off (max|u_z| ~ 1e-20 m),
    which is the plane-strain state the analytic solution assumes.
    """
    errors = {}
    for nx in (16, 32):
        out = FEMSolver(nx=nx, ny=2, solver="direct")(spec=_lame_spec())
        ur = out.extras["ux"][1:-1, 1].numpy()                 # interior radii, mid-height row
        errors[nx] = _rel(ur, _lame_ur(_radii(out)[1:-1]))
        assert float(out.extras["uy"].abs().max()) < 1e-15      # u_z = 0: plane strain
        assert float(out.extras["ux"][-1].min()) == pytest.approx(_lame_ur(B_OUT), rel=1e-12)
    assert errors[32] < 2e-5
    assert errors[16] / errors[32] > 3.0                        # second order, not first


def test_lame_wall_stresses_match_the_analytic_hoop_and_radial_stress():
    """The stresses recovered from the finite-element displacement at the mid-wall radius.

    ``sigma_rr = (lam+2mu) e_rr + lam e_tt`` and ``sigma_tt = lam e_rr + (lam+2mu) e_tt`` with
    ``e_zz = 0`` (plane strain), ``e_tt = u_r/r`` and ``e_rr`` a central difference of the nodal
    ``u_r``. At r = (a+b)/2 = 0.075 m the analytic values are sigma_rr = -8.8889 MPa and sigma_tt =
    18.8889 MPa; measured at nx=64, ny=2: -8.8850 MPa (0.045 %) and 18.8880 MPa (0.005 %). The
    hoop stress is the larger one there by a wide margin, and the radial stress is compressive.
    """
    nx = 64
    out = FEMSolver(nx=nx, ny=2, solver="direct")(spec=_lame_spec())
    ur = out.extras["ux"][:, 1].double()
    radii = torch.linspace(A_IN, B_OUT, nx + 1, dtype=torch.float64)
    h = float(radii[1] - radii[0])
    e_tt = ur / radii
    e_rr = torch.zeros_like(ur)
    e_rr[1:-1] = (ur[2:] - ur[:-2]) / (2.0 * h)
    e_rr[0] = (ur[1] - ur[0]) / h
    e_rr[-1] = (ur[-1] - ur[-2]) / h
    s_rr = (LAM + 2.0 * MU) * e_rr + LAM * e_tt
    s_tt = LAM * e_rr + (LAM + 2.0 * MU) * e_tt

    mid = nx // 2
    r_mid = float(radii[mid])
    assert float(s_rr[mid]) == pytest.approx(float(_lame_sigma_rr(r_mid)), rel=1e-3)
    assert float(s_tt[mid]) == pytest.approx(float(_lame_sigma_tt(r_mid)), rel=1e-3)
    assert float(s_tt[mid]) > float(s_rr[mid])                  # hoop dominates in a pressured ring
    assert float(s_rr[0]) < 0.0                                 # compression at the bore side
    assert abs(float(s_rr[0])) < P_IN                           # the wall carries at most p
    # the classical hoop peak: sigma_tt(a) = p (a^2 + b^2)/(b^2 - a^2) = 5/3 p for a = b/2
    assert float(_lame_sigma_tt(A_IN)) == pytest.approx(P_IN * (A_IN ** 2 + B_OUT ** 2)
                                                       / (B_OUT ** 2 - A_IN ** 2), rel=1e-12)


# ---------------------------------------------------------------------------
# 6. A manufactured solution: every term of the assembly, to round-off
# ---------------------------------------------------------------------------

#: u_r = A1 r + A3 r z, u_z = B0 + B2 z -- bilinear, so it is in the Q1 space
MA_1, MA_3, MB_0, MB_2 = 1.0e-3, 2.0e-3, 1.0e-4, -5.0e-4
#: the uniform body force that makes it an exact axisymmetric solution
MA_BODY = (0.0, -2.0 * (LAM + MU) * MA_3)


def _manufactured_spec(body_force=MA_BODY) -> ProblemSpec:
    def exact(X, ctx):
        r, z = X[:, 0], X[:, 1]
        return np.stack([MA_1 * r + MA_3 * r * z, MB_0 + MB_2 * z], axis=1)

    return _spec({"E": E_STEEL, "nu": NU_STEEL, "body_force": body_force},
                 (DirichletBC("exact", ("u_r", "u_z"), "tag", {"tag": "boundary"}, exact),))


def test_the_manufactured_body_force_is_the_divergence_of_its_stresses():
    """``div(sigma) = (0, 2 (lam+mu) a3)`` for that field, so the load is ``b = -div(sigma)``.

    Checked with the library's own equilibrium operator, over the (r, z) domain the test meshes.
    """
    x = torch.tensor([[0.06, 0.03], [0.09, 0.17], [0.07, 0.10]], dtype=torch.float64, requires_grad=True)
    r, z = x[:, 0:1], x[:, 1:2]
    u = torch.cat([MA_1 * r + MA_3 * r * z, MB_0 + MB_2 * z], dim=1)
    residual = equilibrium_residual_rz(u, x, E_STEEL, NU_STEEL).detach()
    assert float(residual[:, 0].abs().max()) < 1e-9 * E_STEEL * MA_3      # no radial load
    assert float(residual[:, 1].min()) == pytest.approx(residual[:, 1].max(), rel=1e-15)
    assert float(residual[:, 1].mean()) == pytest.approx(2.0 * (LAM + MU) * MA_3, rel=1e-10)
    assert tuple(MA_BODY) == pytest.approx((0.0, float(-residual[:, 1].mean())), rel=1e-10, abs=1e-9)


def test_the_manufactured_axisymmetric_field_is_reproduced_to_round_off():
    """The exact field prescribed on all four sides under its own body force comes back untouched.

    Being bilinear it lies in the Q1 space, and on rectangles 2x2 Gauss integrates both the
    stiffness (degree 3 in x, 2 in y) and this load (degree 3 in x, 2 in y) exactly, so the only
    error left is round-off. Measured relative error over every node, nx = ny = 4: 4.5e-8 for u_r
    and 5.8e-8 for u_z -- the float32 floor of the shared shape-function tables, not a
    discretisation error (it does not fall when the mesh is refined: 4.9e-8 at n=3, 7.7e-8 at n=5,
    7.2e-8 at n=8).

    What this adds over the Lamé test: the strain state here has ``e_rr = e_tt != e_zz`` and a
    non-zero ``g_rz = a3 r``, with a non-uniform ``u_r`` that the hoop term scales by ``1/r``, and
    a load that only the ``r``-weighted load vector reproduces. Perturbing the hoop coefficient by
    1 % moves this error to 4.6e-4, and dropping the ``r`` weight from the stiffness moves the
    Lamé error from 6.9e-6 to 1.1e-2, so neither test passes on a self-consistent but wrong
    assembly.
    """
    for n in (3, 4):
        out = FEMSolver(nx=n, ny=n, solver="direct")(spec=_manufactured_spec())
        nodes = out.extras["nodes"]
        r = nodes[:, 0].double().numpy()
        z = nodes[:, 1].double().numpy()
        assert _rel(out.extras["ux"].reshape(-1).numpy(), MA_1 * r + MA_3 * r * z) < 1e-6
        assert _rel(out.extras["uy"].reshape(-1).numpy(), MB_0 + MB_2 * z) < 1e-6
        K = out.extras["K"]
        assert float((K - K.T).abs().max() / K.abs().max()) < 1e-15          # symmetric stiffness


def test_the_stiffness_of_a_revolution_has_one_rigid_mode_and_a_hoop_energy():
    """What distinguishes this element from the plane one, on the same mesh: a uniform AXIAL
    translation strains nothing (it is the only rigid-body mode of a body of revolution), while a
    uniform RADIAL displacement does not -- it stretches every hoop fibre, ``e_tt = u_r / r``, and
    carries a large energy. Measured, as a fraction of ``max|K| = 1.35e11``: -7.4e-21 for the axial
    field against 1.4e-5 for the radial one."""
    nodes, elems = _make_structured_mesh(4, 4, BOUNDS)
    K, f = _assemble_axisym_elasticity(nodes, elems, E_STEEL, NU_STEEL)
    assert f.abs().max() == 0.0                                  # no body force given
    scale = float(K.abs().max())
    axial, radial = torch.zeros(2 * nodes.shape[0], dtype=torch.float64), torch.zeros_like(f)
    axial[1::2] = 0.01
    radial[0::2] = 0.01
    e_axial = float(0.5 * axial @ K @ axial)
    e_radial = float(0.5 * radial @ K @ radial)
    assert abs(e_axial) < 1e-15 * scale                          # a rigid axial translation
    assert e_radial > 1e-6 * scale                               # the hoop strain it costs


def test_a_radially_confined_slice_carries_the_constrained_modulus():
    """A displacement-controlled slice: ``u_r = 0`` at both walls, ``u_z = eps z`` at the top.

    Nothing is free, so the exact state (``e_zz = eps``, ``e_rr = e_tt = 0``) is in the Q1 space and
    must come back to round-off; the axial force it needs is the uniaxial-strain (constrained)
    modulus ``M = lambda + 2 mu = E(1-nu)/((1+nu)(1-2nu)) = 282.692 GPa``. Measured on nx=4, ny=8:
    max|u_r| = 1.0e-21 m, ``u_z`` within 5.0e-8 of ``eps z``, and the sum of the top nodal forces
    106009.62 against ``M eps (b^2 - a^2)/2 = 106009.62`` (7e-8 relative). The r-weighted load and
    the axial block of ``C`` are what this number measures.
    """
    eps = 1.0e-4
    conditions = (
        DirichletBC("base_uz", ("u_z",), "tag", {"tag": "bottom"}, _const(0.0)),
        DirichletBC("top_uz", ("u_z",), "tag", {"tag": "top"}, _const(lambda X: eps * X[:, 1])),
        DirichletBC("bore_ur", ("u_r",), "tag", {"tag": "left"}, _const(0.0)),
        DirichletBC("wall_ur", ("u_r",), "tag", {"tag": "right"}, _const(0.0)),
    )
    out = FEMSolver(nx=4, ny=8, solver="direct")(spec=_spec({"E": E_STEEL, "nu": NU_STEEL}, conditions))
    z = np.linspace(0.0, HEIGHT, 9)
    assert float(out.extras["ux"].abs().max()) < 1e-15                 # clamped radially at both walls
    assert _rel(out.extras["uy"][0].double().numpy(), eps * z) < 1e-6

    nodes, elems = _make_structured_mesh(4, 8, BOUNDS)
    K, f = _assemble_axisym_elasticity(nodes, elems, E_STEEL, NU_STEEL)
    u = torch.zeros(2 * nodes.shape[0], dtype=torch.float64)
    u[1::2] = eps * nodes[:, 1].double()
    axial = (K @ u)[1::2]
    top = torch.where((nodes[:, 1] - HEIGHT).abs() < 1e-12)[0]
    constrained = LAM + 2.0 * MU                                      # E(1-nu)/((1+nu)(1-2nu))
    assert float(axial[top].sum()) == pytest.approx(constrained * eps * (B_OUT ** 2 - A_IN ** 2) / 2.0,
                                                    rel=1e-6)


# ---------------------------------------------------------------------------
# 7. The paths that already worked answer exactly as before
# ---------------------------------------------------------------------------

def _quadratic(p):
    p = torch.as_tensor(np.asarray(p))
    return p[:, 0] ** 2 + p[:, 1] ** 2            # -lap u = -4


def test_poisson_kind_still_takes_the_scalar_assembly():
    """kind "poisson" keeps the one-scalar path: a (nx+1, ny+1) result, no "method" key, and the
    quadratic reproduced nodally (as in tests/test_fem_plane_stress.py)."""
    bc = DirichletBC("wall", ("u",), "callable",
                     lambda X, ctx: ((X[:, 0] < 1e-6) | (X[:, 0] > 1 - 1e-6)
                                     | (X[:, 1] < 1e-6) | (X[:, 1] > 1 - 1e-6)),
                     _const(lambda X: _quadratic(X)))
    out = FEMSolver(nx=8, ny=8, solver="direct")(
        spec=_spec({"source": -4.0}, (bc,), fields=("u",), kind="poisson",
                   bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)}))
    assert "method" not in out.extras
    assert out.result.shape == (9, 9)
    assert float((out.result.reshape(-1) - _quadratic(out.extras["nodes"])).abs().max()) < 1e-5


def test_plane_stress_still_answers_the_uniaxial_patch_test():
    """2-D plane stress is untouched: the exact linear field ``u = (s x/E, -nu s y/E)`` with ``ux``
    prescribed on the lateral edges and ``uy = 0`` at the bottom (tests/test_fem_plane_stress.py)."""
    sigma = 1.0e6
    out = FEMSolver(nx=8, ny=8, solver="direct")(
        spec=_spec({"E": E_STEEL, "nu": NU_STEEL},
                   (DirichletBC("left_ux", ("ux",), "tag", {"tag": "left"}, _const(0.0)),
                    DirichletBC("right_ux", ("ux",), "tag", {"tag": "right"}, _const(sigma / E_STEEL)),
                    DirichletBC("bottom_uy", ("uy",), "tag", {"tag": "bottom"}, _const(0.0))),
                   bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)}, fields=("ux", "uy"),
                   kind="linear_elasticity"))
    nodes = out.extras["nodes"]
    assert out.extras["method"] == "fem_plane_stress"
    assert out.extras["model"] == "plane_stress"
    assert "ur" not in out.extras and "uz" not in out.extras
    ex = sigma * nodes[:, 0].double() / E_STEEL
    ey = -NU_STEEL * sigma * nodes[:, 1].double() / E_STEEL
    assert _rel(out.extras["ux"].reshape(-1).double().numpy(), ex.numpy()) < 1e-6
    assert _rel(out.extras["uy"].reshape(-1).double().numpy(), ey.numpy()) < 1e-6
