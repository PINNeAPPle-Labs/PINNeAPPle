"""Rollout validation (#701). References: the pseudo-spectral 2-D vorticity solver (`multiscale.ns2d`) and the
Kuramoto-Sivashinsky ETDRK4 integrator (cross-validated against Exponax), each run with a halved time step as the
"model"; an FNO stepper trained on Kuramoto-Sivashinsky; the reference integrator with injected high-wavenumber
dissipation."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from pinneapple_analysis.validation.rollout import (
    divergence_time,
    invariant_drift,
    long_run_statistics,
    rollout,
    validate_rollout,
)
from pinneapple_simulation.numerical_solvers.kuramoto_sivashinsky import KuramotoSivashinsky1D

KS, KS_FINE = KuramotoSivashinsky1D(dt=0.25), KuramotoSivashinsky1D(dt=0.125)
K = np.abs(np.fft.fftfreq(128) * 128)


def attractor(seed: int, solver=KS) -> np.ndarray:
    return solver.step(solver.initial_state(seed), int(100 / solver.dt))


def damped(u: np.ndarray) -> np.ndarray:
    """The reference step with 30 % of the energy above mode 20 removed every step (numerical dissipation)."""
    vh = np.fft.fft(KS.step(u, 4))
    vh[K >= 20] *= 0.7
    return np.fft.ifft(vh).real


@pytest.fixture(scope="module")
def ks_reference():
    u0 = attractor(0)
    return u0, KS.trajectory(u0, 300, 4)                                # frames 1 time unit apart


def test_reference_integrators_pass(ks_reference):
    u0, ref = ks_reference
    alt = KS_FINE.trajectory(u0, 300, 8)                                # the same system, half the time step
    rep = validate_rollout(alt, 1, ref, dt=1.0, conserved=["mass"], chaotic=True)
    assert all(c["status"] != "fail" for c in rep.checks), rep.checks
    th = rep.evidence["divergence"]["trust_horizon"]
    assert th is not None and th > 40                                   # chaos: finite, set by the Lyapunov time
    long_ref = KS.trajectory(attractor(1), 1500, 4)
    long_alt = KS_FINE.trajectory(attractor(2, KS_FINE), 1500, 8)       # another initial state entirely
    st = validate_rollout(long_alt, 1, long_ref=long_ref, dt=1.0)
    assert st["rollout.statistics"]["status"] == "pass"
    from pinneapple_physics.multiscale.ns2d import VorticityNS2D, random_vorticity
    w0 = random_vorticity(np.random.default_rng(0), 64, k0=4.0)
    t = np.arange(0, 3.01, 0.1)
    nref = VorticityNS2D(64, 1e-3).simulate(w0, t, dt=1e-2)
    nalt = VorticityNS2D(64, 1e-3).simulate(w0, t, dt=5e-3)
    rep = validate_rollout(nalt, 2, nref, dt=0.1, conserved=["mass"])
    assert all(c["status"] == "pass" for c in rep.checks), rep.checks
    assert rep.evidence["divergence"]["trust_horizon"] is None


def test_injected_dissipation_is_flagged_by_the_spectrum_first(ks_reference):
    u0, ref = ks_reference
    pred = rollout(damped, u0, 300)
    dv = divergence_time(pred, ref, 1, dt=1.0)
    assert dv["times"]["spectrum"] is not None and dv["times"]["error"] is not None
    assert dv["times"]["spectrum"] < dv["times"]["error"]              # before the pointwise error crosses 20 %
    rep = validate_rollout(pred, 1, ref, dt=1.0, chaotic=True)
    assert rep["rollout.spectrum"]["status"] == "fail"
    long = rollout(damped, attractor(3), 1500)
    st = validate_rollout(long, 1, long_ref=KS.trajectory(attractor(1), 1500, 4), dt=1.0)
    assert st["rollout.statistics"]["status"] == "fail"                 # a different attractor in the long run


def test_fno_rollout_has_a_finite_horizon_consistent_with_its_error(ks_reference):
    torch = pytest.importorskip("torch")
    from pinneapple_analysis.benchmark.builtin.fields import _operator_model
    from pinneapple_neural import train_model
    X, Y = [], []
    for s in range(10):
        tr = KS.trajectory(attractor(10 + s), 150, 4)
        X.append(tr[:-1])
        Y.append(tr[1:])
    X, Y = np.concatenate(X), np.concatenate(Y)
    sc = float(X.std())
    Xt = torch.tensor(X / sc, dtype=torch.float32)
    Yt = torch.tensor((Y - X) / sc, dtype=torch.float32)
    torch.manual_seed(0)
    m = _operator_model("fno", {"width": 32, "modes": 24}, (128,))
    batches = [(Xt[i:i + 32], Yt[i:i + 32]) for i in range(0, len(Xt), 32)]
    import tempfile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_model(m, lambda mm, yh, b: ((yh - b["y"]) ** 2).mean(), batches, batches[:4], epochs=25, lr=2e-3,
                    device="cpu", save_best=False, log_dir=tempfile.mkdtemp())
    m.eval()

    def step(u):
        with torch.no_grad():
            return u + m(torch.tensor(u[None] / sc, dtype=torch.float32)).numpy()[0] * sc
    u0, ref = ks_reference
    pred = rollout(step, u0, 300)
    rep = validate_rollout(pred, 1, ref, dt=1.0, chaotic=True, conserved=["mass"])
    dv = rep.evidence["divergence"]
    th = dv["trust_horizon"]
    assert th is not None and 1 <= th < 57                              # finite, shorter than the integrator's
    assert dv["times"]["error"] == th                                   # the pointwise error sets the horizon
    k = int(th)
    assert dv["error"][k - 1] <= 0.2 < dv["error"][k]
    assert dv["error"][1] < 0.05                                        # accurate for the first step
    assert rep["rollout.invariant.mass"]["status"] in ("pass", "fail")  # reported either way


def test_invariants_and_statistics_closed_forms():
    t = np.arange(50)
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    u = np.stack([np.sin(x) * np.exp(-0.01 * tt) + 1.0 for tt in t])
    d = invariant_drift(u, 1, dt=1.0)
    assert d["mass"]["max_drift"] < 1e-12                               # sin has zero mean: mass is exactly 1
    assert d["energy"]["rate"] < 0
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((400, 64)), rng.standard_normal((400, 64))
    st = long_run_statistics(a, b, 1)
    assert st["w1"] < 3 * st["w1_floor"] + 0.02
    blown = np.concatenate([u[:10], np.full((5, 64), np.nan)])
    assert validate_rollout(blown, 1)["rollout.finite"]["status"] == "fail"


def test_pp_validate_runs_rollout_checks(ks_reference):
    import pinneapple as pp
    u0, ref = ks_reference
    rep = pp.validate(lambda u: KS_FINE.step(u, 8),
                      {"rollout": {"ndim": 1, "dt": 1.0, "conserved": ["mass"], "chaotic": True}},
                      data={"u0": u0, "reference": ref}, level="quick")
    assert rep.decision == "warn" and rep["rollout.horizon"].status == "warn"   # a finite horizon asks for review
    assert rep["rollout.invariant.mass"].status == "pass"
    bad = pp.validate(damped, {"rollout": {"ndim": 1, "dt": 1.0, "chaotic": True}},
                      data={"u0": u0, "reference": ref}, level="quick")
    assert bad["rollout.spectrum"].status == "fail"
    from pinneapple_analysis.validation.report_artifact import check_report
    assert check_report(rep) == [] and check_report(bad) == []


def test_state_vector_rollout_with_a_custom_invariant_named_like_a_builtin():
    # a harmonic oscillator (q, p): energy q^2 + p^2 is the user's "energy"; the field energy of the built-in must
    # not be used in its place, and a state vector (ndim = 0) has no spectrum
    th = 0.1 * np.arange(101)
    exact = np.stack([np.cos(th), -np.sin(th)], 1)
    euler = [np.array([1.0, 0.0])]
    for _ in range(100):
        q, p = euler[-1]
        euler.append(np.array([q + 0.1 * p, p - 0.1 * q]))               # explicit Euler gains energy
    energy = {"energy": lambda tr, ndim: (np.asarray(tr) ** 2).sum(-1)}
    ok = validate_rollout(exact, 0, exact, conserved=["energy"], invariants=energy, dt=0.1)
    bad = validate_rollout(np.stack(euler), 0, exact, conserved=["energy"], invariants=energy, dt=0.1)
    assert ok["rollout.invariant.energy"]["status"] == "pass" and ok["rollout.spectrum"]["status"] == "not-applicable"
    assert bad["rollout.invariant.energy"]["status"] == "fail"
    growth = 1.01 ** 100                                                # |z|^2 grows by (1 + dt^2) per step
    assert bad["rollout.invariant.energy"]["value"] == pytest.approx((growth - 1) / growth, rel=1e-6)   # / its max


def test_pp_validate_passes_burn_in_to_the_long_run_statistics():
    from pinneapple_analysis.validation.validate import validate
    rng = np.random.default_rng(0)
    ref = np.cumsum(rng.standard_normal((400, 32)), 0) * 0.01           # a drifting signal: the window matters
    data = {"u0": ref[0], "long_u0": ref[0], "long_reference": ref}
    w1 = [validate(lambda u: u + 0.01, {"rollout": {"ndim": 1, "dt": 1.0, "burn_in": b}}, data=data)
          ["rollout.statistics"].value for b in (0, 200)]
    assert w1[0] != pytest.approx(w1[1])
