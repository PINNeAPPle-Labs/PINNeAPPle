"""Maxwell physics checks (``pinneapple_physics.maxwell_checks``), ``pinneapple check`` with ``physics = "maxwell"``,
and the electromagnetic kinds and presets of the PDE compiler.

A check is useful only if it passes exact solutions and fails wrong ones, so every test has both.
"""
from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
import torch

from pinneapple_physics import maxwell_checks as mc

C = 299_792_458.0


def tm_mode(n=41, w_factor=1.0, periods=0.7):
    """TM12 standing wave of the unit PEC square, time domain, SI (eps0, mu0); w_factor != 1: not a solution."""
    kx, ky = math.pi, 2 * math.pi
    w = C * math.hypot(kx, ky)
    x = y = np.linspace(0, 1, n)
    t = np.linspace(0, periods * 2 * math.pi / w, n)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    wf = w_factor * w
    f = {"Ez": np.sin(kx * X) * np.sin(ky * Y) * np.cos(wf * T),
         "Bx": -ky * np.sin(kx * X) * np.cos(ky * Y) * np.sin(wf * T) / wf,
         "By": kx * np.cos(kx * X) * np.sin(ky * Y) * np.sin(wf * T) / wf}
    return {"x": x, "y": y, "t": t}, f


def by_key(checks):
    return {c.key: c for c in checks}


# ------------------------------------------------------------------------------------------------ residuals
def test_exact_cavity_mode_passes_and_residuals_converge_at_second_order():
    """maxwell_checks( on the exact solution (TM12 cavity mode): passes, residual O(h^2)."""
    vals = []
    for n in (21, 41, 81):
        grid, f = tm_mode(n)
        ch = by_key(mc.maxwell_checks(grid, f, pec_faces=("x_min", "x_max", "y_min", "y_max")))
        assert all(c.passed is not False for c in ch.values()), [c.to_dict() for c in ch.values()]
        assert ch["gauss_electric"].passed is None              # div D = 0 trivially: reported, not passed
        vals.append([ch[k].value for k in ("gauss_magnetic", "faraday", "ampere_maxwell")])
        assert "discretisation error" in ch["faraday"].detail
    vals = np.array(vals)
    order = np.log2(vals[:-1] / vals[1:])
    assert np.all(order > 1.8), order
    assert by_key(mc.maxwell_checks(*tm_mode(41)))["poynting_balance"].value < 1e-10


def test_wrong_frequency_is_rejected_by_ampere_and_energy():
    """maxwell_checks( reject a field that is not the exact solution (wrong frequency)."""
    grid, f = tm_mode(41, w_factor=1.2)
    ch = by_key(mc.maxwell_checks(grid, f))
    assert ch["faraday"].passed                                 # B was built from E with Faraday's law
    assert not ch["ampere_maxwell"].passed and not ch["poynting_balance"].passed
    assert "does not shrink with h" in ch["ampere_maxwell"].detail
    rep, _ = mc.maxwell_trust_report(grid, f)
    assert rep.decisao == "REJECT"
    good, _ = mc.maxwell_trust_report(*tm_mode(41))
    assert good.decisao == "APPROVED"


def test_divergence_of_B_catches_a_magnetic_monopole_and_gauss_E_a_charge():
    """mc.gauss_magnetic( on an analytic solenoidal field and a monopole; Gauss's law with an exact charge."""
    x = y = z = np.linspace(-1, 1, 33)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid = {"x": x, "y": y, "z": z}
    # solenoidal: B = curl(A) with A = (0, 0, exp(-r^2))
    g = np.exp(-(X ** 2 + Y ** 2 + Z ** 2))
    ok = mc.gauss_magnetic(grid, {"Bx": -2 * Y * g, "By": 2 * X * g, "Bz": 0 * g})
    bad = mc.gauss_magnetic(grid, {"Bx": X * g, "By": Y * g, "Bz": Z * g})
    assert ok.passed and ok.value < 1e-2 and not bad.passed
    # Gaussian charge and its exact field E = Q / (4 pi eps r^2) erf-profile: use E = -grad(phi), phi Gaussian
    rho = mc.EPS0 * (6 - 4 * (X ** 2 + Y ** 2 + Z ** 2)) * g     # -eps lap(g) for phi = g
    E = {"Ex": 2 * X * g, "Ey": 2 * Y * g, "Ez": 2 * Z * g, "rho": rho}
    assert mc.gauss_electric(grid, E).passed
    assert not mc.gauss_electric(grid, {**E, "rho": 2 * rho}).passed


def test_phasor_plane_wave_in_a_conductor_balances_dissipation():
    """maxwell_checks( on the exact solution of a plane wave in a conductor (phasors)."""
    sigma, f0 = 1.0, 1e8
    w = 2 * math.pi * f0
    k = np.sqrt(w * w * mc.MU0 * mc.EPS0 - 1j * w * mc.MU0 * sigma)
    k = -k if k.imag > 0 else k
    x = np.linspace(0, 0.5, 201)
    y = np.linspace(0, 0.1, 5)
    X, _ = np.meshgrid(x, y, indexing="ij")
    Ey = np.exp(-1j * k * X)
    Hz = k / (w * mc.MU0) * Ey                                   # curl E = -j w mu H
    grid = {"x": x, "y": y}
    ch = by_key(mc.maxwell_checks(grid, {"Ey": Ey, "Hz": Hz}, sigma=sigma, omega=w))
    assert ch["faraday"].passed and ch["ampere_maxwell"].passed and ch["poynting_balance"].passed
    assert ch["poynting_balance"].value < 1e-3
    lossless = by_key(mc.maxwell_checks(grid, {"Ey": Ey, "Hz": Hz}, sigma=0.0, omega=w))
    assert not lossless["ampere_maxwell"].passed and not lossless["poynting_balance"].passed


def test_continuity_of_a_moving_charge_packet():
    """Continuity with the exact current of a translating charge packet."""
    v = 2.0
    x = np.linspace(-3, 3, 121)
    t = np.linspace(0, 1, 41)
    X, T = np.meshgrid(x, t, indexing="ij")
    rho = np.exp(-(X - v * T) ** 2)
    assert mc.continuity({"x": x, "t": t}, {"rho": rho, "Jx": v * rho}).passed
    assert not mc.continuity({"x": x, "t": t}, {"rho": rho, "Jx": 0.5 * v * rho}).passed


def test_tangential_continuity_at_a_dielectric_interface_from_fresnel():
    """Tangential E from the closed-form Fresnel coefficients is continuous."""
    from pinneapple_simulation.electromagnetics.waves import fresnel

    n1, n2, th = 1.0, 1.5, 0.5
    rs, _, ts, _ = fresnel(n1, n2, th)
    pts = np.linspace(0, 1, 20)
    k1x = 2 * math.pi * n1 * math.sin(th)
    phase = np.exp(-1j * k1x * pts)                               # phase matching along the interface
    E1 = np.zeros((20, 3), complex)
    E2 = np.zeros((20, 3), complex)
    E1[:, 1] = (1 + rs) * phase                                   # s polarisation: E along y
    E2[:, 1] = ts * phase
    assert mc.tangential_continuity(E1, E2, [0, 0, 1]).passed
    E2[:, 1] = 1.05 * ts * phase
    assert not mc.tangential_continuity(E1, E2, [0, 0, 1]).passed


def test_network_reciprocity_and_passivity():
    """mc.passivity( and reciprocity of networks whose S-matrices are known in closed form."""
    # matched 3 dB attenuator and a lossless 90-degree hybrid
    a = 10 ** (-3 / 20)
    att = np.array([[0, a], [a, 0]], complex)
    hyb = -np.array([[0, 1j, 1, 0], [1j, 0, 0, 1], [1, 0, 0, 1j], [0, 1, 1j, 0]]) / math.sqrt(2)
    assert mc.reciprocity(att).passed and mc.passivity(att).passed and not mc.passivity(att, lossless=True).passed
    assert mc.reciprocity(hyb).passed and mc.passivity(hyb, lossless=True).passed
    amp = np.array([[0.1, 0.0], [3.0, 0.1]], complex)             # an amplifier: active and non-reciprocal
    assert not mc.passivity(amp).passed and not mc.reciprocity(amp).passed
    sweep = np.stack([att, 1.42 * att])                           # 1.42 a = 1.005: the second frequency is active
    assert not mc.passivity(sweep).passed and "index 1" in mc.passivity(sweep).detail


# ------------------------------------------------------------------------------------------------ pinneapple check
@pytest.fixture(scope="module")
def check_cases(tmp_path_factory):
    d = tmp_path_factory.mktemp("maxwell_check")
    mc.save_maxwell_npz(d / "good.npz", *tm_mode(41))
    mc.save_maxwell_npz(d / "bad.npz", *tm_mode(41, w_factor=1.2))
    x = np.linspace(0, 0.5, 101)
    y = np.linspace(0, 0.1, 5)
    X, _ = np.meshgrid(x, y, indexing="ij")
    w = 2 * math.pi * 1e8
    k = w / C
    E = np.exp(-1j * k * X)
    mc.save_maxwell_npz(d / "phasor.npz", {"x": x, "y": y}, {"Ey": E, "Hz": E / (C * mc.MU0)})
    (d / "pinneapple-check.toml").write_text(
        mc.check_config_entry("good", "good.npz") + "\n" + mc.check_config_entry("bad", "bad.npz") + "\n"
        + mc.check_config_entry("phasor", "phasor.npz", parameters={"omega": w}))
    return d


def test_pinneapple_check_maxwell(check_cases):
    from pinneapple_check import run

    rep = run(config=check_cases / "pinneapple-check.toml")
    bad = {v.rule for v in rep.violations if v.target == "bad"}
    assert bad == {"PC301", "PC103"}
    assert not [v for v in rep.violations if v.target in ("good", "phasor")]
    ran = {o.rule for o in rep.outcomes if o.target == "good"}
    assert {"PC102", "PC301", "PC103"} <= ran
    assert {o.rule for o in rep.outcomes if o.target == "phasor"} >= {"PC301", "PC103"}


def test_pinneapple_check_maxwell_cli_without_torch(check_cases):
    code = ("import sys; from pinneapple_check.cli import check_main; "
            "rc = check_main(['--config', sys.argv[1], 'good.npz']); "
            "assert 'torch' not in sys.modules; sys.exit(rc)")
    ok = subprocess.run([sys.executable, "-c", code, str(check_cases / "pinneapple-check.toml")], cwd=check_cases)
    assert ok.returncode == 0
    bad = subprocess.run([sys.executable, "-m", "pinneapple_check", "--config",
                          str(check_cases / "pinneapple-check.toml")], cwd=check_cases, capture_output=True, text=True)
    assert bad.returncode == 1 and "Ampere-Maxwell" in bad.stdout


# ------------------------------------------------------------------------------------------------ compiler kinds
class Exact(torch.nn.Module):
    """A 'network' that returns a closed-form field (torch, differentiable)."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, X):
        # the zero term keeps every derivative in the autograd graph (a linear field has a constant gradient)
        return self.fn(X) + 0 * self.p + 0 * (X ** 2).sum(1, keepdim=True)


def pde_loss(spec, fn, n=256, seed=0, ctx=None):
    from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

    rng = np.random.default_rng(seed)
    b = spec.domain_bounds
    X = np.column_stack([rng.uniform(*b[c], n) for c in spec.coords])
    loss = compile_problem(spec)(Exact(fn), None, {"x_col": torch.as_tensor(X, dtype=torch.float64),
                                                   "ctx": ctx or {}})
    return float(loss["pde"])


def spec_of(kind, coords, fields, params, bounds):
    from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec

    return ProblemSpec(name=kind, dim=len([c for c in coords if c != "t"]), coords=coords, fields=fields,
                       pde=PDETermSpec(kind=kind, fields=fields, coords=coords, params=params),
                       conditions=(), domain_bounds=bounds)


def tm_torch(w_factor=1.0):
    kx, ky = math.pi, math.pi
    w = math.hypot(kx, ky) * w_factor

    def fn(X):
        x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        Ez = torch.sin(kx * x) * torch.sin(ky * y) * torch.cos(w * t)
        Hx = -ky / w * torch.sin(kx * x) * torch.cos(ky * y) * torch.sin(w * t)
        Hy = kx / w * torch.cos(kx * x) * torch.sin(ky * y) * torch.sin(w * t)
        return torch.cat([Ez, Hx, Hy], 1)
    return fn


def test_maxwell_tm_kinds_zero_on_the_exact_mode():
    """"maxwell_tm" and maxwell_tm_integral: the exact solution (cavity mode) has zero residual."""
    for kind, params in (("maxwell_tm", {}), ("maxwell_tm_integral", {"h": 0.05})):
        spec = spec_of(kind, ("x", "y", "t"), ("Ez", "Hx", "Hy"), params,
                       {"x": (0, 1), "y": (0, 1), "t": (0, 1)})
        exact = pde_loss(spec, tm_torch())
        wrong = pde_loss(spec, tm_torch(1.3))
        assert exact < 1e-12 and wrong > 1e-2


def test_maxwell_3d_plane_wave_with_gauss_terms():
    """maxwell_3d: the exact solution (plane wave) has zero residual, a wrong one does not."""
    k = 2.0

    def fn(X, kk=k):
        x, t = X[:, 0:1], X[:, 3:4]
        Ey = torch.cos(kk * x - k * t)
        z = 0 * Ey
        return torch.cat([z, Ey, z, z, z, Ey], 1)             # Hz = Ey / eta, eta = 1

    spec = spec_of("maxwell_3d", ("x", "y", "z", "t"), ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), {"gauss": True},
                   {"x": (0, 1), "y": (0, 1), "z": (0, 1), "t": (0, 1)})
    assert pde_loss(spec, fn) < 1e-12
    assert pde_loss(spec, lambda X: fn(X, 1.5 * k)) > 1e-2


def test_eddy_current_and_frequency_kinds():
    """eddy_current_az and maxwell_tm_frequency: exact solution (skin effect, lossy plane wave) residual zero."""
    delta = 0.2
    sig = 2.0 / delta ** 2

    def skin(X, d=delta):
        x = X[:, 0:1]
        e = torch.exp(-x / d)
        return torch.cat([e * torch.cos(x / d), -e * torch.sin(x / d)], 1)   # exp(-(1+j) x / delta)

    spec = spec_of("eddy_current_az", ("x",), ("Ar", "Ai"), {"omega": 1.0, "sigma": sig, "nu": 1.0}, {"x": (0, 1)})
    assert pde_loss(spec, skin) < 1e-12 and pde_loss(spec, lambda X: skin(X, 0.3)) > 1e-2
    # plane wave in a lossy dielectric: exp(-j k0 n x), n = sqrt(eps_r - j eps_i)
    eps = complex(2.5, -0.4)
    n = np.sqrt(eps)

    def wave(X):
        x = X[:, 0:1]
        a = torch.exp(float(n.imag) * x)                     # Im(n) < 0: decays along +x
        return torch.cat([a * torch.cos(float(n.real) * x), -a * torch.sin(float(n.real) * x)], 1)

    spec = spec_of("maxwell_tm_frequency", ("x", "y"), ("Er", "Ei"), {"k0": 1.0, "eps_r": 2.5, "eps_i": -0.4},
                   {"x": (0, 2), "y": (0, 1)})
    assert pde_loss(spec, wave) < 1e-12
    assert pde_loss(spec, wave, ctx={"eps_fn": lambda X, c: np.full(len(X), 2.0 + 0j)}) > 1e-2


def test_integral_gauss_sees_the_dielectric_interface_the_differential_form_misses():
    """"electrostatics" vs electrostatics_integral against the exact solution of the layered capacitor.

    phi = y is harmonic on both sides of eps1 | eps2, so the differential residual is zero, but D.n jumps at the
    interface; the integral form (flux through each control square) flags it, and is zero for the true field."""
    from pinneapple_physics.closed_form.electromagnetics import two_dielectric_capacitor

    D = 1.0 / (0.5 / 1.0 + 0.5 / 4.0)

    def true(X):
        y = X[:, 1:2]
        return torch.where(y <= 0.5, D * y, D * 0.5 + D * (y - 0.5) / 4.0)

    eps_fn = lambda X, ctx: np.where(X[:, 1] <= 0.5, 1.0, 4.0)      # noqa: E731
    assert np.allclose(true(torch.tensor([[0.0, 0.8]])).item(), two_dielectric_capacitor(0.8, 1, 1, 0.5, 1, 4))
    common = {"eps_fn": eps_fn}
    diff = spec_of("electrostatics", ("x", "y"), ("phi",), common, {"x": (0.1, 0.9), "y": (0.1, 0.9)})
    integ = spec_of("electrostatics_integral", ("x", "y"), ("phi",), {**common, "h": 0.02},
                    {"x": (0.1, 0.9), "y": (0.1, 0.9)})
    linear = lambda X: X[:, 1:2]                                    # noqa: E731
    assert pde_loss(diff, linear, n=2000) < 1e-20 and pde_loss(diff, true, n=2000) < 1e-20
    assert pde_loss(integ, true, n=2000) < 1e-20
    assert pde_loss(integ, linear, n=2000) > 1.0


# ------------------------------------------------------------------------------------------------ presets
EM_PRESETS = ["em_two_dielectric_capacitor", "em_two_dielectric_capacitor_differential", "em_wire_az",
              "em_tm_cavity", "em_tm_cavity_integral", "em_skin_effect", "em_cylinder_scattering"]


@pytest.mark.parametrize("name", EM_PRESETS)
def test_presets_have_an_analytic_solution_that_satisfies_their_conditions(name):
    import pinneapple as pp

    prob = pp.PhysicalProblem.from_preset(name)
    sol = pp.solve(prob, "analytic")
    rng = np.random.default_rng(1)
    b = prob.domain_bounds
    X = np.column_stack([rng.uniform(*b[c], 400) for c in prob.coords])
    assert np.all(np.isfinite(sol.predict(X)))
    for c in prob.conditions:
        if c.kind != "dirichlet":
            continue
        Xb = X.copy()
        side = rng.integers(0, 2, len(X))
        k = [i for i, cc in enumerate(prob.coords) if cc != "t"][0]
        Xb[:, k] = np.where(side == 0, b[prob.coords[k]][0], b[prob.coords[k]][1])
        m = c.mask(Xb, {})
        want = np.asarray(c.value_fn(Xb[m], {}))
        idx = [prob.fields.index(f) for f in c.fields]
        np.testing.assert_allclose(sol.predict(Xb[m])[:, idx], want.reshape(m.sum(), -1), atol=1e-10)


def test_wire_preset_analytic_solves_its_pde():
    """The em_wire_az analytic solution satisfies magnetostatics_az (finite differences)."""
    import pinneapple as pp

    prob = pp.PhysicalProblem.from_preset("em_wire_az", a=0.3, J=2.0)
    f = pp.solve(prob, "analytic").predict
    h = 1e-4
    for x, y in ((0.1, 0.05), (0.6, -0.4)):
        P = np.array([[x, y], [x + h, y], [x - h, y], [x, y + h], [x, y - h]])
        u = f(P)[:, 0]
        lap = (u[1:].sum() - 4 * u[0]) / h ** 2
        J = 2.0 if math.hypot(x, y) < 0.3 else 0.0
        assert lap + J == pytest.approx(0.0, abs=1e-4)
