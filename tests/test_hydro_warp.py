"""NVIDIA Warp kernels behind the accretion solver, against the PyTorch reference (issue #410).

The reference is :mod:`pinneapple_physics.blackhole.hydro`; every test here holds the Warp path to it -- the
dispatch, agreement to round-off, the budget, and the derivative of the accreted mass with respect to alpha -- and
the two that cannot run without the package skip with the install line instead of passing quietly. ``warp-lang`` is
optional: nothing in this file imports it at module level.
"""
import numpy as np
import pytest
import torch

from pinneapple_physics.blackhole import AccretionFlow, RIAFConfig, hydro_warp, torus_state
from pinneapple_physics.blackhole.hydro import accreted_mass

#: relative tolerance the Warp path is held to. Measured on 32 x 16 / 64 x 32 grids over 10 - 20 steps: 3e-14 for
#: the worst primitive (the reference's own round-off), 2e-15 for the boundary budgets. The bound is the numba
#: backend's 1e-10, i.e. the same claim about a second implementation of one scheme.
ROUND_OFF = 1e-10


@pytest.fixture(autouse=True)
def _threads():
    torch.set_num_threads(1)


def _warp():
    """Skip, naming the install line, when NVIDIA Warp is not importable here."""
    status = hydro_warp.warp_status()
    pytest.importorskip("warp", reason=f"NVIDIA Warp is not installed (missing: {status['missing']}); "
                                      f"install it with: {status['install']}")
    return status


# --------------------------------------------------------------------------- dispatch
def test_the_reference_backend_is_spelled_either_way():
    assert RIAFConfig(backend="reference").backend == "torch"
    assert RIAFConfig().backend == "torch"
    assert RIAFConfig(backend="warp").backend == "warp"
    with pytest.raises(ValueError, match="unknown backend"):
        RIAFConfig(backend="jax")


def test_warp_status_always_answers_with_the_install_line():
    status = hydro_warp.warp_status()
    assert status["backend"] == "warp"
    assert set(status) >= {"available", "missing", "install", "device", "version", "reason"}
    assert "warp-lang" in status["install"]
    if status["available"]:
        assert status["missing"] is None
        assert status["device"] is not None
    else:
        assert status["missing"] == "warp-lang"
        assert status["device"] is None
        assert status["reason"]


def test_asking_for_warp_without_the_package_raises_with_the_install_line(monkeypatch):
    """The failure a machine without the extra sees, without needing a machine without the extra."""
    monkeypatch.setattr(hydro_warp, "_WARP", None)
    monkeypatch.setitem(__import__("sys").modules, "warp", None)      # makes `import warp` raise ImportError
    assert hydro_warp.warp_available() is False
    status = hydro_warp.warp_status()
    assert status["available"] is False and status["missing"] == "warp-lang"
    assert "pip install warp-lang" in status["install"]
    assert "pip install warp-lang" in status["reason"]
    with pytest.raises(ImportError, match="warp-lang"):
        hydro_warp.require_warp()
    with pytest.raises(ImportError, match="warp-lang"):
        AccretionFlow(RIAFConfig(nr=8, ntheta=4, backend="warp"))      # fails at construction, not at the first step


def test_the_warp_backend_is_selected_by_the_config():
    """``backend="warp"`` routes ``step`` through the kernels; ``backend="torch"`` never builds a stepper."""
    status = _warp()
    flow = AccretionFlow(RIAFConfig(nr=8, ntheta=4, backend="warp"))
    flow.set_primitives(torus_state(flow))
    flow.step()
    assert isinstance(flow._warp(), hydro_warp.WarpStepper)
    assert flow._warp().device == status["device"]
    ref = AccretionFlow(RIAFConfig(nr=8, ntheta=4, backend="reference"))
    assert "warp" not in str(type(getattr(ref, "_wp_backend", None)))
    assert getattr(ref, "_wp_backend", None) is None


# --------------------------------------------------------------------------- equivalence with the reference
@pytest.mark.parametrize("kwargs", [
    {"nr": 32, "ntheta": 16},                                        # SS viscosity, the default set-up
    {"nr": 24, "ntheta": 12, "viscosity": "ST", "alpha": 0.2},       # the other nu prescription
    {"nr": 24, "ntheta": 12, "viscosity": "none", "t_cap": None},    # inviscid, no temperature ceiling
    {"nr": 24, "ntheta": 12, "outer_bc": "fixed"},                   # held outer ghosts
])
def test_warp_matches_the_torch_reference(kwargs):
    _warp()
    steps = 10
    out = {}
    for backend in ("torch", "warp"):
        flow = AccretionFlow(RIAFConfig(backend=backend, **kwargs))
        flow.set_primitives(torus_state(flow))
        for _ in range(steps):
            flow.step()
        out[backend] = (flow.primitives(), flow.t, dict(flow.boundary), flow.mdot_in, flow.capped)
    ref, war = out["torch"], out["warp"]
    assert ref[4] == war[4]                                          # the same cells were capped, if any
    assert abs(ref[1] - war[1]) <= 1e-12 * abs(ref[1])               # both picked the same dt at every step
    assert abs(ref[3] - war[3]) <= ROUND_OFF * abs(ref[3])           # the accretion rate at the last step
    for q, name in enumerate(("rho", "vr", "vtheta", "vphi", "p")):
        span = np.abs(ref[0][q]).max()
        assert np.max(np.abs(ref[0][q] - war[0][q])) <= ROUND_OFF * (span + 1e-30), name
    for k in ("mass_in", "mass_out", "angmom_in", "angmom_out"):      # what budget() closes on
        assert abs(ref[2][k] - war[2][k]) <= ROUND_OFF * (abs(ref[2][k]) + 1e-30), k


def test_a_longer_run_on_a_larger_grid_still_matches():
    _warp()
    out = {}
    for backend in ("torch", "warp"):
        flow = AccretionFlow(RIAFConfig(nr=64, ntheta=32, backend=backend))
        flow.set_primitives(torus_state(flow))
        for _ in range(25):
            flow.step()
        out[backend] = flow.primitives()
    span = np.abs(out["torch"]).max(axis=(1, 2))
    err = np.abs(out["torch"] - out["warp"]).max(axis=(1, 2)) / (span + 1e-30)
    assert err.max() <= ROUND_OFF


def test_bc_functional_is_the_same_boundary_rule_as_the_in_place_one():
    """`_bc` is a copy of `bc_functional`'s result, so the two cannot drift apart (the autograd path needs the
    out-of-place one, every stepper calls the in-place one)."""
    for kwargs in ({}, {"outer_bc": "fixed"}):
        flow = AccretionFlow(RIAFConfig(nr=16, ntheta=8, **kwargs))
        flow.set_primitives(torus_state(flow))
        flow.step()
        W = flow.W.clone()
        W[1, :, 2:-2] = -0.5                                         # something for the no-inflow clamp to bite on
        a = flow.bc_functional(W)
        b = W.clone()
        flow._bc(b)
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- gradients
def _reference_gradient(alpha, steps, dt, **kwargs):
    """d(accreted mass)/d(alpha) through torch autograd, from the reference path."""
    a = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
    flow = AccretionFlow(RIAFConfig(alpha=a, **kwargs))
    flow.set_primitives(torus_state(flow))
    total = accreted_mass(flow, steps, dt=dt)
    total.backward()
    return float(total.detach()), float(a.grad)


def _finite_difference(alpha, steps, dt, h=1e-4, **kwargs):
    out = []
    for sign in (+1, -1):
        flow = AccretionFlow(RIAFConfig(alpha=alpha + sign * h, **kwargs))
        flow.set_primitives(torus_state(flow))
        out.append(float(accreted_mass(flow, steps, dt=dt)))
    return (out[0] - out[1]) / (2 * h)


SMALL = dict(nr=24, ntheta=12)
STEPS, DT = 8, 0.05


def test_the_reference_gradient_of_the_accreted_mass_is_right():
    """Autograd through the reference's step against a central finite difference of the same quantity.

    Runs everywhere: this is the claim that the observable and the differentiable path are the physics, and it is
    the yardstick the Warp tape is measured against in the next test.
    """
    mass, grad = _reference_gradient(0.1, STEPS, DT, **SMALL)
    assert mass > 0                                                  # gas reaches the hole in these steps
    fd = _finite_difference(0.1, STEPS, DT, **SMALL)
    assert abs(grad - fd) <= 1e-4 * abs(fd), (grad, fd)


def test_the_warp_tape_gradient_matches_the_reference_and_a_finite_difference():
    _warp()
    _, grad = _reference_gradient(0.1, STEPS, DT, **SMALL)
    fd = _finite_difference(0.1, STEPS, DT, **SMALL)
    flow = AccretionFlow(RIAFConfig(nr=SMALL["nr"], ntheta=SMALL["ntheta"], alpha=0.1, backend="warp"))
    flow.set_primitives(torus_state(flow))
    before = flow.W.clone()
    res = hydro_warp.tape_accreted_mass(flow, STEPS, dt=DT)
    assert res["steps"] == STEPS and res["dt"] == DT
    assert torch.equal(flow.W, before)                               # the flow is an input: its state is left alone
    assert abs(res["dmass_dalpha"] - grad) <= 1e-6 * abs(grad), (res["dmass_dalpha"], grad)
    assert abs(res["dmass_dalpha"] - fd) <= 1e-4 * abs(fd), (res["dmass_dalpha"], fd)


def test_the_tape_observable_is_the_mass_the_reference_reports():
    _warp()
    flow = AccretionFlow(RIAFConfig(nr=16, ntheta=8, alpha=0.1, backend="warp"))
    flow.set_primitives(torus_state(flow))
    tape_mass = hydro_warp.tape_accreted_mass(flow, 6, dt=0.05)["mass"]
    ref = AccretionFlow(RIAFConfig(nr=16, ntheta=8, alpha=0.1))
    ref.set_primitives(torus_state(ref))
    assert abs(tape_mass - float(accreted_mass(ref, 6, dt=0.05))) <= ROUND_OFF * abs(tape_mass)


# --------------------------------------------------------------------------- the reference did not move
def test_the_reference_results_are_unchanged_by_the_backend_split():
    """Numbers from a 32 x 16 run at alpha = 0.1, pinned before the Warp backend was added (bit-identical to the
    code before this refactor; the tolerance is for a different BLAS or a different torch build, not for a change
    of scheme)."""
    flow = AccretionFlow(RIAFConfig(nr=32, ntheta=16, alpha=0.1, viscosity="SS"))
    flow.set_primitives(torus_state(flow))
    for _ in range(20):
        flow.step()
    prim = flow.primitives()
    tot = flow.totals()
    expected = {
        "t": 8.35822939307928,
        "mass": 70077.37011124422,
        "angmom": 347958.24990804977,
        "mass_in": 0.0502135697779456,
        "angmom_in": 4.441648378002993e-06,
        "mdot": 0.007017045927932356,
        "rho_max": 0.9370178696535097,
        "rho_sum": 19.99930836939543,
        "vr_max": 0.7111139412820011,
        "vphi_max": 0.4461610781758132,
        "p_max": 0.0038891847327570863,
    }
    got = {"t": flow.t, "mass": tot["mass"], "angmom": tot["angmom"], "mass_in": flow.boundary["mass_in"],
           "angmom_in": flow.boundary["angmom_in"], "mdot": flow.mdot_in,
           "rho_max": float(prim[0].max()), "rho_sum": float(prim[0].sum()),
           "vr_max": float(np.abs(prim[1]).max()), "vphi_max": float(np.abs(prim[3]).max()),
           "p_max": float(prim[4].max())}
    assert flow.capped == 0
    for k, v in expected.items():
        assert abs(got[k] - v) <= 1e-12 * abs(v), (k, got[k], v)


# --------------------------------------------------------------------------- the seam for the other solvers
def test_the_seam_declares_what_is_ported_and_what_is_not():
    assert hydro_warp.WARP_PORTS["blackhole.hydro"]["ported"] is True
    for solver in ("compressible_fv", "shallow_water_fv"):
        entry = hydro_warp.solver_status(solver)
        assert entry["ported"] is False
        assert entry["reason"] and entry["reference"]          # a port nobody has written, with a reason
        with pytest.raises(NotImplementedError, match="not written yet"):
            hydro_warp.not_ported(solver)
    with pytest.raises(KeyError, match="unknown solver"):
        hydro_warp.solver_status("no_such_solver")


# --------------------------------------------------------------------------- the measurement harness
def test_the_measurement_harness_tabulates_a_row_per_backend():
    rows = hydro_warp.bench(grids=((16, 8),), steps=3, warmup=1, backends=("torch", "warp"))
    assert [r["backend"] for r in rows] == ["torch", "warp"]
    assert rows[0]["skipped"] is None and rows[0]["ms_per_step"] > 0 and rows[0]["rel_error"] == 0.0
    if hydro_warp.warp_available() and rows[1]["skipped"] is None:
        assert rows[1]["rel_error"] <= ROUND_OFF                # the speed-up is the same physics
        assert rows[1]["t_end"] == pytest.approx(rows[0]["t_end"], rel=1e-12)
    else:
        assert rows[1]["skipped"]                              # not measured, and it says why
    table = hydro_warp.format_table(rows)
    assert table.startswith("| backend |") and len(table.strip().splitlines()) == len(rows) + 2
