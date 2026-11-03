"""The reduced-order-modeling workflow (#1208): POD-Galerkin, DMD and operator inference against the FOM."""
import numpy as np
import pytest

from pinneapple_analysis.reduced_order import make_fom, reduced_order_workflow
from pinneapple_analysis.reduced_order.workflow import write_snapshots

SMALL = dict(n=128, dt=2e-3, t_end=1.0, save_every=5, n_params=5, repeats=1)


def test_fom_conserves_mass_and_matches_the_linear_closed_form():
    fom = make_fom("advection_diffusion", n=256, dt=1e-3, t_end=0.5, save_every=50)
    U = fom.solve(0.1)
    m = fom.mass(U)
    assert np.max(np.abs(m - m[0])) / m[0] < 1e-12
    # u0 = 1 + 0.5 sin x + 0.25 cos 2x advected at c = 1 and damped by exp(-mu k^2 t)
    t = fom.times[-1]
    exact = 1 + 0.5 * np.exp(-0.1 * t) * np.sin(fom.x - t) + 0.25 * np.exp(-0.4 * t) * np.cos(2 * (fom.x - t))
    assert np.max(np.abs(U[-1] - exact)) < 2e-3


def test_burgers_galerkin_is_accurate_stable_conservative_and_faster():
    r = reduced_order_workflow(fom="burgers", **SMALL)
    g = r["methods"]["pod_galerkin"]
    assert g["passed"], r["checks"]
    assert g["interpolation_error"] < 0.02 and g["extrapolation_error"] < 0.05
    assert g["invariant_drift"] < 1e-10
    assert g["growth_factor"] <= 1.01
    assert g["speedup"] > 1
    assert r["basis"]["energy_captured"] >= 0.99999
    assert r["recommended"] == "pod_galerkin"
    # DMD of a nonlinear flow has a growing mode, and the learned quadratic model is unstable about the uniform
    # state it should decay to: both caught by the stability check, not by the error over the tested horizon
    assert r["methods"]["dmd"]["max_eigenvalue"] > 1
    assert r["methods"]["opinf"]["max_eigenvalue"] > 0
    assert r["methods"]["opinf"]["interpolation_error"] < 0.02
    codes = {c["code"] for c in r["checks"]}
    assert {"opinf.extrapolation", "dmd.stability", "pod_galerkin.speedup"} <= codes


def test_advection_diffusion_all_three_roms():
    r = reduced_order_workflow(fom="advection_diffusion", **SMALL)
    for k in ("pod_galerkin", "opinf"):
        assert r["methods"][k]["interpolation_error"] < 0.02, (k, r["summary"])


def test_snapshot_file_skips_the_intrusive_rom_and_the_speedup(tmp_path):
    fom = make_fom("advection_diffusion", n=128, dt=2e-3, t_end=1.0, save_every=5)
    mus = np.linspace(0.05, 0.15, 5)
    path = write_snapshots(str(tmp_path / "s.npz"), [fom.solve(m) for m in mus], fom.times, mus)
    r = reduced_order_workflow(snapshots_file=path, repeats=1)
    assert "skipped" in r["methods"]["pod_galerkin"]
    assert any(c["code"] == "opinf.speedup" and c["status"] == "skip" for c in r["checks"])
    assert np.isfinite(r["methods"]["opinf"]["interpolation_error"])


def test_bad_inputs_are_refused(tmp_path):
    with pytest.raises(ValueError):
        reduced_order_workflow(fom="nope")
    np.savez(tmp_path / "bad.npz", snapshots=np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="times"):
        reduced_order_workflow(snapshots_file=str(tmp_path / "bad.npz"))
