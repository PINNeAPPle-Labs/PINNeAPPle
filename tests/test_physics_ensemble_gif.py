"""Ensemble of five families of physics models (FNO, GNN, DeepONet, PINN, CNN) and its animated GIF."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PIL = pytest.importorskip("PIL.Image")

from pinneapple_physics.advection_diffusion_1d import AdvectionDiffusion1D  # noqa: E402
from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable  # noqa: E402
from pinneapple_physics.ensemble_viz import animate_ensemble  # noqa: E402

AD = AdvectionDiffusion1D()
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "physics_ensemble" / "five_model_families_gif.py"


def calibrated(nu):
    return from_callable(f"nu={nu}", lambda c: AD.exact({**c, "nu": nu}))


def test_gif_shows_every_case_and_follows_the_regime():
    rng = np.random.default_rng(0)
    cases = [AD.random_case(rng, nu) for nu in [0.01] * 6 + [0.2] * 6]
    refs = [AD.exact(c) for c in cases]
    ens = PhysicsEnsemble([calibrated(0.01), calibrated(0.2)], mode="select")
    with pytest.raises(ValueError, match="keep_predictions"):
        animate_ensemble(ens.run(cases, refs), refs)
    run = PhysicsEnsemble([calibrated(0.01), calibrated(0.2)], mode="select").run(cases, refs, keep_predictions=True)
    assert run.active[:6] == ["nu=0.01"] * 6 and run.active[-3:] == ["nu=0.2"] * 3
    assert set(run.predictions[0]["expert_predictions"]) == {"nu=0.01", "nu=0.2"}
    np.testing.assert_allclose(run.predictions[0]["prediction"], refs[0], atol=1e-12)


def test_gif_frames(tmp_path):
    rng = np.random.default_rng(1)
    cases = [AD.random_case(rng, nu) for nu in [0.01] * 3 + [0.2] * 3]
    refs = [AD.exact(c) for c in cases]
    run = PhysicsEnsemble([calibrated(0.01), calibrated(0.2)], mode="combine").run(cases, refs, keep_predictions=True)
    out = animate_ensemble(run, refs, tmp_path / "st.gif", x=AD.x, times=AD.t, frames_per_case=2, dpi=30,
                           regimes=[(0, "low"), (3, "high")])
    with PIL.open(out) as im:
        assert im.n_frames == 6 * 2
    # 1-D fields (one frame per case), e.g. a steady profile
    run1 = PhysicsEnsemble({"a": lambda c: AD.exact(c)[-1], "b": lambda c: 0 * AD.x}).run(
        cases, [r[-1] for r in refs], keep_predictions=True)
    out1 = animate_ensemble(run1, [r[-1] for r in refs], tmp_path / "line.gif", dpi=30)
    with PIL.open(out1) as im:
        assert im.n_frames == 6


def test_five_model_families_each_chosen_in_its_regime(tmp_path):
    spec = importlib.util.spec_from_file_location("five_families", EXAMPLE)
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    torch.manual_seed(0)
    run, s = ex.main(out=tmp_path / "ens.gif", scale=0.3, cases_per_regime=8, frames_per_case=1, dpi=30)
    assert (tmp_path / "ens.gif").stat().st_size > 0
    assert run.names == ["FNO", "GNN", "DeepONet", "PINN", "CNN"]
    # every family is in-distribution in one regime only: the ensemble beats every single model by far
    assert s["ensemble"] < 0.6 * s["best_single_in_hindsight"]["error"]
    # and after a few cases of each regime it uses that regime's model (allowing the weakest, the GNN, to lose)
    chosen_late = {run.active[8 * k + 7] for k in range(5)}
    assert len(chosen_late & {"FNO", "DeepONet", "PINN", "CNN"}) == 4
