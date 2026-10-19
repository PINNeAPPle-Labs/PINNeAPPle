"""pp.validate (#696): the same call on a zoo model, an FNO trained in plain PyTorch and the same FNO as an ONNX file;
decisions equal to running the underlying checks by hand; an out-of-range query fails without ground truth."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinneapple_analysis.validation import schema  # noqa: E402
from pinneapple_analysis.validation.validate import ValidationCheck, validate  # noqa: E402
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob  # noqa: E402

N = 32


def darcy(n, seed, tau=3.0, high=12.0, low=3.0):
    a = ob.threshold_coefficient(ob.periodic_grf(n, (N, N), alpha=2.0, tau=tau, seed=seed), high=high, low=low)
    return a, np.stack([ob.darcy_2d(x, 1.0) for x in a])


class UserFNO(torch.nn.Module):
    """A user's model: an FNO with its normalisation inside, trained below with a plain PyTorch loop."""

    def __init__(self, am, asd, us):
        super().__init__()
        from pinneapple_neural.architectures.neural_operators.fno import FNO2d
        self.net = FNO2d(1, 1, width=16, modes1=8, modes2=8, layers=4)
        self.am, self.asd, self.us = am, asd, us

    def forward(self, a):
        return self.net(((a - self.am) / self.asd).unsqueeze(1)).y.squeeze(1) * self.us


@pytest.fixture(scope="module")
def darcy_case(tmp_path_factory):
    A, U = darcy(500, 1)
    torch.manual_seed(0)
    m = UserFNO(float(A[:400].mean()), float(A[:400].std()), float(U[:400].std()))
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    X, Y = torch.tensor(A[:400], dtype=torch.float32), torch.tensor(U[:400], dtype=torch.float32)
    s2 = float(U.std()) ** 2
    for _ in range(30):
        for i in range(0, 400, 20):
            loss = ((m(X[i:i + 20]) - Y[i:i + 20]) ** 2).mean() / s2
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    from pinneapple_neural.architectures.neural_operators.fno_export import export_onnx
    path = tmp_path_factory.mktemp("onnx") / "fno.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diff = export_onnx(m, X[:4], path)
    data = {"train": {"a": A[:400], "u": U[:400]}, "test": {"a": A[400:], "u": U[400:]}}
    return m, str(path), data, diff


def _statuses(rep):
    return {c.id: c.status for c in rep.checks}


def test_same_call_on_pytorch_and_onnx_gives_the_same_report(darcy_case):
    m, path, data, diff = darcy_case
    assert diff < 1e-5
    q = data["test"]["a"][:8]
    rt = validate(m, "darcy_2d", data=data, query=q, level="standard")
    ro = validate(path, "darcy_2d", data=data, query=q, level="standard")
    assert _statuses(rt) == _statuses(ro)
    assert rt.decision == ro.decision
    assert rt["reference.rel_l2"].value == pytest.approx(ro["reference.rel_l2"].value, rel=1e-4)
    assert rt["applicability.envelope"].status == "pass"
    assert rt["reference.rel_l2"].status == "pass" and rt["reference.rel_l2"].value < 0.1
    assert rt["residual.energy_bound"].status == "pass"
    assert rt["uncertainty.residual_bound"].status == "pass"
    for f in schema.FACTORS:
        assert f in rt.credibility()


def test_decisions_match_the_underlying_checks_run_by_hand(darcy_case):
    m, _, data, _ = darcy_case
    from pinneapple_analysis.uncertainty.residual_bound import darcy_problem, energy_error_bound
    from pinneapple_data.audit import audit_dataset
    q = data["test"]["a"][:8]
    rep = validate(m, "darcy_2d", data=data, query=q, level="standard")
    with torch.no_grad():
        p = m(torch.tensor(q, dtype=torch.float32)).numpy()
    by_hand = max(energy_error_bound(darcy_problem(a), u)["relative"] for a, u in zip(q, p, strict=True))
    assert rep["residual.energy_bound"].value == pytest.approx(by_hand, rel=1e-5)
    with torch.no_grad():
        pt = m(torch.tensor(data["test"]["a"], dtype=torch.float32)).numpy()
    e = np.linalg.norm((pt - data["test"]["u"]).reshape(100, -1), axis=1) / \
        np.linalg.norm(data["test"]["u"].reshape(100, -1), axis=1)
    assert rep["reference.rel_l2"].value == pytest.approx(e.mean(), rel=1e-4)
    card = audit_dataset(data, physics="darcy", max_physics_samples=16)
    assert rep["data.leakage"].status == card["leakage"].status
    assert rep["data.physics.solve"].status == card["physics.solve"].status


def test_out_of_range_query_fails_without_ground_truth(darcy_case):
    m, path, data, _ = darcy_case
    shifted, _ = darcy(4, 7701, high=20.0, low=1.0)               # the PhysicsNeMo contrast shift
    for model in (m, path):
        rep = validate(model, "darcy_2d", data={"train": data["train"]}, query=shifted, level="quick")
        assert rep.decision == "fail"
        assert rep.verdict.startswith("fail: out of validity range")
        assert rep["applicability.envelope"].status == "fail"
        assert rep["reference.rel_l2"].status == "not-applicable"   # no ground truth was used
        assert rep["residual.energy_bound"].status == "fail"        # the residual says so too


def test_zoo_model_report_matches_the_zoo_gate():
    import pinneapple as pp
    m = pp.zoo.load("heatsink_forced_mlp")
    ok = {"base_width_mm": [80.0], "base_depth_mm": [80.0], "base_thickness_mm": [6.0], "n_fins": [20.0],
          "fin_thickness_mm": [1.5], "fin_height_mm": [40.0], "k_w_mk": [200.0], "source_width_mm": [40.0],
          "source_depth_mm": [40.0], "t_ambient_c": [25.0], "air_velocity_m_s": [2.0], "power_w": [100.0]}
    rep = validate(m, query=ok, level="standard")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate = pp.zoo.validate(m, inputs=ok)
    assert rep.to_trust_report().decisao == gate.decisao == "APPROVED"
    assert rep["applicability.range"].status == "pass"
    assert rep.get("metamorphic.monotonicity.resistance-falls-with-air-speed").status == "pass"
    data_checks = [c for c in rep.checks if c.id.startswith("data.")]   # the stored audit card of its dataset
    assert data_checks and all(c.status in ("pass", "not-applicable") for c in data_checks)
    bad = {**ok, "air_velocity_m_s": [20.0]}                        # validity range 0.5 - 6 m/s
    rep = validate("heatsink_forced_mlp", query=bad, level="quick")
    assert rep.verdict.startswith("fail: out of validity range")


def test_outputs_json_html_sarif_and_trust_card(darcy_case):
    m, _, data, _ = darcy_case
    rep = validate(m, "darcy_2d", data=data, query=data["test"]["a"][:4], level="thorough")
    d = json.loads(rep.to_json())
    assert d["schema"] == "validation-report/v1" and d["decision"] in ("pass", "warn", "fail")
    assert {c["status"] for c in d["checks"]} <= set(schema.STATUSES)
    assert "<table>" in rep.to_html()
    sarif = json.loads(rep.render("sarif"))
    assert sarif["runs"][0]["tool"]["driver"]["name"].lower().startswith("pinneapple")
    assert "testsuite" in rep.render("junit")
    assert rep.get("robustness.input_noise").status in ("pass", "fail")
    assert rep.get("robustness.fault_injection").status == "not-applicable"


def test_schema_refuses_unknown_ids_and_inherits_families():
    with pytest.raises(KeyError):
        ValidationCheck("made.up", "pass", "")
    with pytest.raises(ValueError):
        ValidationCheck("reference.rel_l2", "ok", "")
    assert schema.spec_of("metamorphic.symmetry.reflect-x").rule == "PC303"
    assert schema.spec_of("data.physics.solve").factor == "input_pedigree"


@pytest.mark.parametrize("H,W,modes", [(16, 16, 4), (15, 17, 7), (24, 24, 13)])
def test_fno_onnx_export_is_exact(tmp_path, H, W, modes):
    from pinneapple_neural.architectures.neural_operators.fno import FNO2d
    from pinneapple_neural.architectures.neural_operators.fno_export import export_onnx
    torch.manual_seed(0)
    net = FNO2d(1, 2, width=8, modes1=modes, modes2=modes, layers=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diff = export_onnx(net, torch.randn(3, 1, H, W), tmp_path / "f.onnx")
    assert diff < 1e-5


def test_pp_validate_is_callable_and_keeps_the_old_namespace():
    import pinneapple as pp
    from pinneapple_analysis.validation import PhysicsValidator
    assert pp.validate.PhysicsValidator is PhysicsValidator          # pp.validate.<name> as before
    rep = pp.validate(lambda x: {"y": 2 * np.asarray(x["x"])}, {"valid_range": {"x": (0, 1)}},
                      query={"x": [0.5, 2.0]}, level="quick")
    assert rep.verdict.startswith("fail: out of validity range")


def test_exact_solver_passes_with_a_source_symmetric_in_one_direction_only():
    # f varies along the second axis: symmetric under x-reflection (first axis) but not under transposition, so only
    # the reflection may be tested; the exact solver must not be rejected for a symmetry the problem does not have
    from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob
    f = np.tile(np.linspace(0.5, 2.0, 16), (16, 1))
    a = ob.threshold_coefficient(ob.periodic_grf(8, (16, 16), alpha=2.0, tau=3.0, seed=3))
    solver = lambda x: np.stack([ob.darcy_2d(y, f) for y in x])        # noqa: E731
    rep = validate(solver, {"pde": "darcy", "f": f}, query=a, level="standard")
    ids = [c.id for c in rep.checks]
    assert "metamorphic.symmetry.reflect-x" in ids and "metamorphic.symmetry.transpose" not in ids
    assert rep["metamorphic.symmetry.reflect-x"].status == "pass" and rep.decision != "fail", str(rep)
