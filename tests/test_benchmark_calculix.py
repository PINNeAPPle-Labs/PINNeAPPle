"""``calculix_beam`` and ``calculix_bracket``: the structural benchmark with CalculiX as the reference solver (#68).

The campaign's data comes from a real ``ccx``, so the tests that solve skip without the solver (native binary or the
Docker image), as the rest of the repository does; the design space, the meshes and the closed form never skip, because
they are what the campaign would be wrong about if they were wrong.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis import benchmark as B
from pinneapple_analysis.benchmark.builtin import calculix as C
from pinneapple_simulation.external_solvers.calculix.runner import ccx_backend

needs_ccx = pytest.mark.skipif(ccx_backend() is None, reason="CalculiX not available (no ccx, no Docker)")
HEX_TETS = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))


def _volumes(X: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Volume of each hexahedron, from the six tetrahedra that share its 0-6 diagonal (positive when the element is
    not inverted)."""
    P = X[E]
    out = np.zeros(len(E))
    for a, b, c, d in HEX_TETS:
        out += np.abs(np.einsum("ij,ij->i", np.cross(P[:, b] - P[:, a], P[:, c] - P[:, a]), P[:, d] - P[:, a])) / 6
    return out


def test_the_two_problems_and_the_two_methods_are_registered():
    assert {"calculix_beam", "calculix_bracket"} <= set(B.problems())
    assert {"mlp_calculix", "pod_mlp_calculix"} <= set(B.methods())
    for pid in ("calculix_beam", "calculix_bracket"):
        prob = B.get_problem(pid)
        for mid in ("mlp_calculix", "pod_mlp_calculix"):
            meth = B.get_method(mid)
            assert meth.supports(prob), f"{mid} does not support {pid}"
            assert isinstance(meth.space(prob), dict) and meth.space(prob), f"{mid} has no grid on {pid}"
        assert prob.columns() and prob.main_metric() in {(c.set, c.metric) for c in prob.columns()}


def test_the_design_box_and_the_ood_designs_that_leave_it():
    for part in ("beam", "bracket"):
        lo = np.array([b[1] for b in C.BOUNDS[part]])
        hi = np.array([b[2] for b in C.BOUNDS[part]])
        rng = np.random.default_rng(0)
        U = C._lhs(64, rng)
        D = np.array([10.0 ** np.asarray(C._features(part, u)) for u in U])
        assert ((D >= lo * (1 - 1e-9)) & (D <= hi * (1 + 1e-9))).all(), "a Latin-hypercube point left the box"
        outside = [C._outside(part, np.log10(np.asarray(o))) for o in C.OOD[part]]
        assert all(outside), f"an OOD design of {part} sits inside the training box"


def test_a_design_round_trips_through_its_four_logarithms():
    for part in ("beam", "bracket"):
        for design in (C.REFERENCE[part], *C.OOD[part]):
            d = C._design(part, np.log10(np.asarray(design)))
            back = [d["L"], d["L"] / d["H"], d["W"] / d["H"], d["P"]] if part == "beam" else \
                [d["t"], d["b"] / d["t"], d["width"] / d["t"], d["P"]]
            assert np.allclose(back, design, rtol=1e-12), f"{part}: {back} != {design}"


def test_the_beam_mesh_fills_its_box_and_keeps_one_topology():
    vol = []
    for L in (0.6, 1.2, 2.0):
        d = {"L": L, "H": L / 12, "W": L / 12, "P": 1000.0}
        m = C._model("beam", d)
        v = _volumes(m.nodes, m.elements)
        assert v.min() > 0, "an inverted element"
        vol.append(v.sum())
        assert v.sum() == pytest.approx(L * (L / 12) * (L / 12), rel=1e-9)
        assert (len(m.nodes), len(m.elements)) == (336, 180)
    assert len({len(C._model("beam", {"L": L, "H": L / 12, "W": L / 12, "P": 1.0}).nodes)
                for L in (0.6, 2.0)}) == 1


def test_the_bracket_mesh_is_one_conforming_body():
    """Two blocks merged on the plane z = 1: no duplicated node left, no inverted element, and the volume is the L's
    (base a x w x t plus the upright (b - t) x w x t) for every design, which is what tells a merge gone wrong from a
    merge that worked."""
    for bt, wt in ((4.0, 4.0), (6.0, 6.0), (10.0, 10.0)):
        d = C._design("bracket", np.log10(np.asarray([8e-3, bt, wt, 1000.0])))
        m = C._model("bracket", d)
        key = np.round(m.nodes / 1e-9).astype(np.int64)
        assert len(np.unique(key, axis=0)) == len(m.nodes), "duplicated nodes: the two blocks were not merged"
        v = _volumes(m.nodes, m.elements)
        assert v.min() > 0, "an inverted element"
        expected = d["a"] * d["width"] * d["t"] + (d["b"] - d["t"]) * d["width"] * d["t"]
        assert v.sum() == pytest.approx(expected, rel=1e-9)
        assert (len(m.nodes), len(m.elements)) == (405, 208), "the topology has to be the same for every sample"


def test_the_closed_form_is_the_beam_theory_it_says():
    L, W, H, P = 1.2, 0.1, 0.1, 1000.0
    ref = C._analytic("beam", {"L": L, "W": W, "H": H, "P": P})
    E, nu = C.E_STEEL, C.NU_STEEL
    G = E / (2 * (1 + nu))
    inertia = W * H ** 3 / 12
    assert ref["w"] == pytest.approx(P * L ** 3 / (3 * E * inertia) + P * L / (5 / 6 * G * W * H))
    assert ref["sigma"] == pytest.approx(P * L / 2 * (H / 2) / inertia)
    assert ref["sigma"] == pytest.approx(3.6e6, rel=1e-3)          # 3.6 MPa: a sanity anchor in MPa
    b = {"t": 8e-3, "b": 48e-3, "width": 48e-3, "a": 80e-3, "P": 1000.0}
    rb = C._analytic("bracket", b)
    assert rb["w"] is None, "the bracket has no closed-form deflection and the row has to say so"
    assert rb["sigma"] == pytest.approx(1000 * (48e-3 - 8e-3) / 2 * 4e-3 / (48e-3 * 8e-3 ** 3 / 12))


@needs_ccx
@pytest.mark.parametrize("pid", ["calculix_beam", "calculix_bracket"])
def test_the_solver_is_verified_against_the_closed_form(pid):
    checks = B.get_problem(pid).verify_solver()
    assert checks and all(c.passed for c in checks), [(c.name, c.value, c.detail) for c in checks if not c.passed]
    assert any("vs Timoshenko" in c.name for c in checks) == (pid == "calculix_beam")


@needs_ccx
def test_a_smoke_campaign_prepares_and_passes_every_check(tmp_path):
    man = B.prepare(problems=["calculix_beam"], out_dir=tmp_path / "b", smoke=True, n_workers=2, log=lambda *a: None)
    man = man["calculix_beam"]
    assert man["status"] == "ok" and not man["blocking"]
    v = json.loads((tmp_path / "b" / "validation" / "calculix_beam.json").read_text())
    names = [c["name"] for c in v["checks"]]
    for name in ("every sample solved (ccx returned a result)", "one topology for every sample",
                 "every run is in equilibrium (reactions balance the load)", "OOD designs outside the training box"):
        assert name in names, name
    assert not [c for c in v["checks"] + v["validation"] if c["passed"] is False]
    path = next((tmp_path / "b" / "prepared" / "calculix_beam" / "runs").glob("*.json"))
    rec = json.loads(path.read_text())
    assert rec["status"] == "ok" and rec["w"] > 0 and rec["sigma"] > 0 and rec["eq"] < 1e-4
    z = np.load(path.with_suffix(".npz"))
    assert z["u"].shape == (len(z["x"]), 3) and z["vm"].shape == (len(z["x"]),)
    assert np.abs(z["u"]).max() < 1.0 and 0 < z["vm"].max() < 0.01, "the fields are dimensionless (u / L, vm / E)"
    assert (tmp_path / "b" / "prepared" / "calculix_beam" / "geometry" / "reference.inp").exists()


@needs_ccx
def test_a_smoke_run_trains_both_methods_on_both_parts(tmp_path):
    res = B.run(problems=["calculix_beam", "calculix_bracket"], methods=["mlp_calculix", "pod_mlp_calculix"],
                out_dir=tmp_path / "b", smoke=True, budget={"wall_s": 2.0}, seeds=1, n_trials=1, n_workers=2,
                log=lambda *a: None)
    assert not res.blocked, res.blocked
    df = res.table()
    assert set(df["method"]) == {"mlp_calculix", "pod_mlp_calculix"} and len(df) == 4
    assert "INVALID" not in set(df["status"]), df[["problem", "method", "status"]]
    # each method reports on the columns it produces: the MLP the two integrals, the POD + MLP the fields as well (and
    # the integrals read off the predicted field). Whether any of them converges in a 2 s smoke budget is not the test.
    integ, field = "rel. err, held-out test (n = 2)", "rel. L2, held-out test (n = 2)"
    for _, r in df.iterrows():
        if r["method"] == "mlp_calculix":
            assert np.isfinite(r[f"Displacement {integ}"]) and np.isfinite(r[f"Peak stress {integ}"])
            assert np.isnan(r[f"Displacement {field}"]) and np.isnan(r[f"von Mises {field}"])
        else:
            for label in (f"Displacement {integ}", f"Peak stress {integ}", f"Displacement {field}", f"von Mises {field}"):
                assert np.isfinite(r[label]), label
    md = (tmp_path / "b" / "table.md").read_text()
    assert "Beam theory (Timoshenko, M c / I; no data)" in md
    for pid in ("calculix_beam", "calculix_bracket"):
        row = json.loads((tmp_path / "b" / "prepared" / pid / "analytic_reference.json").read_text())
        for block in row["seeds"][0].values():
            assert "sigma" in block and 0 <= np.mean(block["sigma"]["values"]) < 1
        assert ("w" in row["seeds"][0].get("test", {})) == (pid == "calculix_beam")


def test_a_field_vector_reshapes_back_into_nodes_and_channels(tmp_path):
    """The trainer's layout: a channel-major vector reshaped to (samples, nodes, 4) comes back interleaved, and every
    field metric would then be measured on a scrambled field while the training loss still fell to zero."""
    n = 5
    u = np.arange(3 * n, dtype=np.float32).reshape(n, 3) / 100
    vm = (np.arange(n, dtype=np.float32) + 1) / 1e5
    path = tmp_path / "s.npz"
    np.savez_compressed(path, x=np.zeros((n, 3)), u=u, vm=vm)
    F = C._fields({"npz": str(path)})
    assert F.shape == (4 * n,)
    back = F.reshape(1, -1, 4)[0]
    assert np.array_equal(back[:, :3], u) and np.array_equal(back[:, 3], vm)
    mean, std = C._channel_stats(np.stack([F, 2 * F]))          # one mean and one std per channel, tiled per node
    per_node = mean.reshape(-1, 4)
    assert per_node.shape == (n, 4) and np.allclose(per_node[:, 0], 1.5 * u[:, 0].mean())
    assert np.allclose(per_node[:, 3], 1.5 * vm.mean()) and np.all(std.reshape(-1, 4) > 0)
