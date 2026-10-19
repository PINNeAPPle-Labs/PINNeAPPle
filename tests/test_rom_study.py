"""ROM study (pinneapple_neural.architectures.rom.study): held-out errors, the projection floor, the baselines, the
equal-budget neural surrogate and the portable ROM file."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.snapshots import snapshots_from_arrays
from pinneapple_neural.architectures.rom import ParametricPOD
from pinneapple_neural.architectures.rom.study import (
    ReducedModel,
    fit_reduced_model,
    rom_study,
    split_snapshots,
)

X1 = np.linspace(0, 2 * np.pi, 300)


def _wave(n=100, dt=0.1):
    t = np.arange(n) * dt                     # two travelling waves: an exact rank-4 linear system
    U = np.array([1 + np.sin(X1 - 2 * s) + 0.3 * np.sin(2 * X1 + 3 * s) for s in t])
    return snapshots_from_arrays({"u": U}, times=t)


def _bump(n=50, seed=0):
    P = np.random.default_rng(seed).uniform(0, 1, (n, 2))
    X = np.array([np.exp(-((X1 - 3 - p[0]) ** 2) / (0.3 + p[1])) for p in P])
    return snapshots_from_arrays({"u": X}, params=P, param_names=["shift", "width"])


def test_split_holds_out_the_last_times_and_never_trains_on_them():
    s = _wave()
    sp = split_snapshots(s)
    assert sp.how == "forecast" and set(sp.train).isdisjoint(sp.test) and sp.test.min() > sp.train.max()
    il = split_snapshots(s, how="interleaved")
    assert set(il.train).isdisjoint(il.test) and il.test.min() > 0 and il.test.max() < s.n - 1


def test_dmd_and_opinf_forecast_a_linear_wave_and_time_interpolation_cannot():
    r = rom_study(_wave(), ranks=[2, 4, 8], methods=["persistence", "pod_rbf", "dmd", "opinf"], timing_repeats=2)
    err = {(x["method"], x["rank"]): x["error_mean"] for x in r.rows}
    assert err[("persistence", None)] > 0.5                          # the waves move: the last snapshot is wrong
    assert err[("opinf", 4)] < 1e-3          # rank 4 holds the two waves (OpInf works on centred coefficients)
    assert err[("dmd", 8)] < 1e-3            # uncentred DMD: the mean is one more mode
    assert err[("dmd", 4)] > 0.1 and err[("opinf", 2)] > 0.1
    assert min(v for (m, _), v in err.items() if m == "pod_rbf") > 0.5   # extrapolating in time fails
    floor = {p["rank"]: p["error_mean"] for p in r.projection}
    assert floor[4] < 1e-8
    assert r.best()["method"] in ("dmd", "opinf")
    d = r.to_dict(tolerance=0.05)
    assert d["best"]["error_mean"] <= 0.05 and d["pareto"] and len(d["singular_values"]) == len(r.split.train)


def test_parametric_rom_beats_the_nearest_design_and_respects_the_projection_floor():
    r = rom_study(_bump(), ranks=[4, 8], methods=["nearest", "pod_rbf", "pod_gpr"], timing_repeats=2)
    err = {(x["method"], x["rank"]): x["error_mean"] for x in r.rows}
    assert err[("pod_rbf", 8)] * 3 < err[("nearest", None)]
    floor = {p["rank"]: p["error_mean"] for p in r.projection}
    for (_m, k), e in err.items():
        if k is not None:
            assert e >= floor[k] - 1e-12                              # no regression beats the best approximation


def test_neural_surrogate_trains_within_its_budget():
    s = _bump()
    sp = split_snapshots(s)
    m = fit_reduced_model(s, sp.train, "neural", budget_s=1.0)
    b = m.config["budget"]
    assert b["stopped_by"] == "wall_s" and b["used_s"] == pytest.approx(1.0, abs=0.3) and b["steps"] > 10
    e = np.linalg.norm(m.predict(params=s.params[sp.test]) - s.X[sp.test], axis=1) / np.linalg.norm(s.X[sp.test], axis=1)
    assert np.isfinite(e).all()


@pytest.mark.parametrize("method,data", [("pod_rbf", "p"), ("pod_gpr", "p"), ("nearest", "p"), ("neural", "p"),
                                         ("dmd", "t"), ("opinf", "t"), ("neural", "t"), ("pod_rbf", "t")])
def test_exported_rom_predicts_exactly_what_the_fitted_one_does(method, data):
    s = _bump() if data == "p" else _wave()
    sp = split_snapshots(s)
    m = fit_reduced_model(s, sp.train, method, 4, budget_s=0.3)
    m2 = ReducedModel.from_bytes(m.to_bytes())
    q = {"params": s.params[sp.test]} if data == "p" else {"times": s.times[sp.test]}
    np.testing.assert_allclose(m2.predict(**q), m.predict(**q), rtol=1e-10, atol=1e-12)
    assert m2.manifest()["format"] == "pinneapple-rom/1"
    inside = m2.in_range(**q)
    assert inside.dtype == bool and (data == "p" or not inside.any())      # forecast times are outside the window


def test_parametric_pod_round_trip_with_amplitude_split():
    s = _bump()
    P, X = s.params, s.X * (1 + 10 * s.params[:, :1])
    rom = ParametricPOD(r=6, energy=None, regressor="gpr", normalize=True).fit(P, X)
    back = ParametricPOD.from_arrays(rom.to_arrays(), rom.config())
    np.testing.assert_allclose(back.predict(P[:5]), rom.predict(P[:5]), rtol=1e-9, atol=1e-12)


def test_time_steppers_refuse_irregular_time_steps():
    s = _wave()
    s.times = np.r_[s.times[:50], s.times[50:] + 0.037]
    r = rom_study(s, ranks=[4], methods=["dmd", "opinf"], timing_repeats=1)
    assert not r.rows and {x["method"] for x in r.skipped} == {"dmd", "opinf"}


def test_fields_in_different_units_are_scaled_and_log_parameters_round_trip():
    s0 = _bump()
    s = snapshots_from_arrays({"u": s0.X * 1e3, "v": s0.X[:, ::-1] * 1e-3}, params=s0.params + 0.5,
                              param_names=["shift", "width"])
    r = rom_study(s, ranks=[8], methods=["pod_rbf"], log_params=["width"], timing_repeats=1)
    row = r.rows[0]
    assert "scaled" in r.metric and set(row["field_errors"]) == {"u", "v"}
    assert row["field_errors"]["v"] < 0.05                         # the 1e-3 field is not drowned by the 1e3 one
    m = r.models[("pod_rbf", 8)]
    m2 = ReducedModel.from_bytes(m.to_bytes())
    np.testing.assert_allclose(m2.predict(params=s.params[:4]), m.predict(params=s.params[:4]), rtol=1e-10)
    assert m.config["validity"]["lo"][1] == pytest.approx(s.params[r.split.train, 1].min())   # raw, not log


@pytest.mark.parametrize("method,data", [("pod_rbf", "p"), ("nearest", "p"), ("neural", "p"), ("dmd", "t"),
                                         ("pod_rbf", "t")])
def test_onnx_graph_matches_the_rom(method, data):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    s = _bump() if data == "p" else _wave()
    sp = split_snapshots(s)
    m = fit_reduced_model(s, sp.train, method, 6, budget_s=0.3, log_params=["width"] if data == "p" else ())
    sess = ort.InferenceSession(m.to_onnx(), providers=["CPUExecutionProvider"])
    q = s.params[sp.test] if data == "p" else s.times[sp.test][:, None]
    ref = m.predict(params=q) if data == "p" else m.predict(times=q[:, 0])
    got = sess.run(None, {"x": q.astype(np.float32)})[0]
    assert np.abs(got - ref).max() < 1e-4 * np.abs(ref).max()
    spec = m.onnx_spec()
    assert len(spec["outputs"]) == s.n_values and set(spec["validity_bounds"]) == set(spec["inputs"])


def test_operator_inference_has_no_onnx_graph():
    s = _wave()
    m = fit_reduced_model(s, split_snapshots(s).train, "opinf", 4)
    with pytest.raises(NotImplementedError, match="npz"):
        m.to_onnx()


def test_reconstruction_view_of_a_held_out_design_on_its_body():
    """The beam example of the ROM Lab: FEM and ROM fields on each design's own mesh, the deformed shape in metres,
    the clamp and load sets as markers, and error maps whose numbers are the arrays' own."""
    import os

    from pinneapple_data.snapshots import load_snapshots
    from pinneapple_neural.architectures.rom.views import reconstruction_view
    p = os.path.join(os.path.dirname(__file__), "..", "apps", "rom_lab", "examples", "cantilever_designs.npz")
    s = load_snapshots([("c.npz", open(p, "rb").read())])
    sp = split_snapshots(s, how="random", test_fraction=0.2, seed=0)
    m = fit_reduced_model(s, sp.train, "pod_rbf", 8, log_params=["H", "W"])
    i = int(sp.test[0])
    fv, info = reconstruction_view(s, m, i, reported={"study mean": 0.1})
    names = [f["name"] for f in fv.meta["fields"]]
    assert names == ["displacement:full", "displacement:rom", "displacement:error",
                     "von_mises:full", "von_mises:rom", "von_mises:error"]
    np.testing.assert_allclose(fv.meta["bounds"]["max"][1:], s.snapshot_points[i].max(0)[1:], rtol=1e-5)
    u = s.field_values("displacement")[i] * 1e-3
    assert fv.meta["deform"]["FEM"]["max"] == pytest.approx(np.linalg.norm(u, axis=1).max(), rel=1e-6)
    assert fv.meta["deform"]["FEM"]["unit"] == "m"
    kinds = {mk["name"]: mk["kind"] for mk in fv.meta["markers"]}
    assert kinds["clamp"] == "bc" and kinds["loaded tip"] == "load" and "hotspot" in kinds.values()
    vm_t, vm_p = s.field_values("von_mises")[i], m.predict(params=s.params[i:i + 1])[0]
    vm_p = s.field_values("von_mises", vm_p[None])[0]
    e = info["errors"]["von_mises"]
    assert e["rel_l2"] == pytest.approx(np.linalg.norm(vm_p - vm_t) / np.linalg.norm(vm_t))
    assert e["max"] == pytest.approx(np.abs(vm_p - vm_t).max())
    assert {r["name"] for r in e["regions"]} == {"clamp zone", "loaded tip zone", "rest of the body"}
    assert sum(r["share_error"] for r in e["regions"]) == pytest.approx(1.0)
    assert "study mean: 10.00 %" in e["summary"][0]
    with pytest.raises(ValueError, match="no mesh"):
        reconstruction_view(_bump(), fit_reduced_model(_bump(), range(40), "pod_rbf", 4), 45)
