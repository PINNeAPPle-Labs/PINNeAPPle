"""pp.benchmark: the pipeline, the cache and its guard, every correctness gate on a deliberate fault, the advisor."""
import json

import numpy as np
import pytest
import torch

from pinneapple_analysis import benchmark as B
from pinneapple_analysis.benchmark.builtin.standard import Burgers
from pinneapple_analysis.cost import ComputeBudget


def _arrays(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 32)) + shift
    k = np.fft.rfftfreq(32, 1 / 32)
    y = np.fft.irfft(np.fft.rfft(x, axis=1) * np.exp(-0.05 * k ** 2), n=32, axis=1)   # a smoothing operator
    return x.astype(np.float32), y.astype(np.float32)


def _problem(leak=False, **kw):
    tr = _arrays(48, 0)
    te = (tr[0][:8].copy(), tr[1][:8].copy()) if leak else _arrays(8, 2)
    return B.ArrayProblem("smooth", train=tr, val=_arrays(8, 1), test=te, ood=_arrays(4, 3, shift=3.0),
                          data_cost_s=1.0, **kw)


FAST = dict(budget={"wall_s": 1.0}, seeds=2, n_trials=1, n_workers=1, log=lambda *a: None)


def test_run_report_resume_and_config_guard(tmp_path):
    out = tmp_path / "b"
    res = B.run(problems=[_problem()], methods=["fno", "deeponet"], out_dir=out, **FAST)
    md = (out / "table.md").read_text()
    for part in ("Rel. L2, test, all (n = 8)", "## Diagnostics and suggestions", "## Validation", "## Trust card",
                 "## Checks", "| Checks |"):
        assert part in md
    assert (out / "summary.json").exists() and (out / "table.csv").exists() and (out / "config.yaml").exists()
    assert (out / "prepared" / "smooth" / "manifest.json").exists() and (out / "hardware.json").exists()
    for f in ("summary_table.png", "error_vs_budget.png", "pareto_error_vs_inference.png"):
        assert (out / "figures" / f).exists()
    df = res.table()
    assert set(df["method"]) == {"fno", "deeponet"}
    n_runs = len(list((out / "runs").glob("*.json")))
    logs = []
    B.run(problems=[_problem()], methods=["fno", "deeponet"], out_dir=out, **{**FAST, "log": logs.append})
    assert len(list((out / "runs").glob("*.json"))) == n_runs                     # resumed from the cache
    assert any("0 to run" in str(m) for m in logs)
    with pytest.raises(B.ConfigMismatch):
        B.run(problems=[_problem()], methods=["fno", "deeponet"], out_dir=out, **{**FAST, "seeds": 3})
    B.run(problems=[_problem()], methods=["fno"], out_dir=out, force=True, stages=(), **FAST)


def test_leaked_split_blocks_the_run(tmp_path):
    with pytest.raises(B.PreparationError):
        B.run(problems=[_problem(leak=True)], methods=["fno"], out_dir=tmp_path / "b", **FAST)
    v = json.loads((tmp_path / "b" / "validation" / "smooth.json").read_text())
    leak = [c for c in v["checks"] if "shared between splits" in c["name"]][0]
    assert leak["passed"] is False


def test_fno_modes_above_nyquist_are_caught_and_the_pair_is_invalid(tmp_path):
    res = B.run(problems=[_problem()], methods=["fno"], out_dir=tmp_path / "b",
                method_options={"fno": {"modes": [40]}}, stages=(), **FAST)
    c = [c for c in res.arch if c.name.startswith("retained Fourier modes")][0]
    assert c.passed is False and c.value == 40 and c.threshold == 17
    assert "smooth/fno" in res.blocked
    assert any(d.topic == "method" and "Nyquist" in d.finding for d in res.diagnostics)


@B.register_method("lazy")
class Lazy(B.Method):
    """A method that does not learn: constant loss, zero prediction."""
    title = "Lazy"

    def supports(self, problem):
        return isinstance(problem, B.FieldOperatorProblem)

    def space(self, problem):
        return {"lr": [1e-3]}

    def train(self, problem, prepared, cfg, seed, budget, final=True, **_):
        from pinneapple_analysis.benchmark.builtin.fields import FieldData
        from pinneapple_analysis.cost import train_within_budget
        run = train_within_budget(lambda p: 1.0, budget, flops_per_step=1.0)
        D = FieldData(prepared)
        out = {"train": run.to_dict(), "infer_batch1": {"median_s": 1e-6}}
        for split in (("val", "test", "ood") if final else ("val",)):
            y = D.raw[f"{split}_y"]
            out[split] = problem.scores(np.zeros_like(y), y)
        return out

    def architecture_checks(self, problem, prepared, cfg):
        return [B.Check("forward shape", "architecture", f"{problem.id}/lazy", True)]


def test_a_model_that_does_not_learn_is_labelled(tmp_path):
    res = B.run(problems=[_problem()], methods=["lazy"], out_dir=tmp_path / "b", **FAST)
    row = res._build()["tables"]["smooth"]["rows"]["lazy"]
    assert row["status"] == "did not converge" and row["did_not_converge"]
    learned = [c for c in res.checks if c.name.startswith("learned") and c.subject == "smooth/lazy"][0]
    assert learned.passed is False
    assert "did not converge" in (tmp_path / "b" / "table.md").read_text()


class BrokenSign(Burgers):
    """Burgers with a sign error in the PINN residual: + nu u_xx instead of - nu u_xx."""
    id = "burgers_broken"

    def pinn_setup(self, x_in, cfg):
        model, spec, sample, evaluate, exact = super().pinn_setup(x_in, cfg)
        spec.pde_residuals = [spec.pde_residuals[0].replace("- nu*", "+ nu*")]
        return model, spec, sample, evaluate, exact


def test_pinn_residual_gate_catches_a_sign_error(tmp_path):
    good = B.prepare(problems=["burgers"], out_dir=tmp_path / "b", smoke=True, log=lambda *a: None)["burgers"]
    assert good["status"] == "ok"
    pin = B.get_method("pinn")
    cfg = {"width": 16, "depth": 2, "lr": 1e-3, "n_colloc": 64}
    prep = tmp_path / "b" / "prepared" / "burgers"
    ok = [c for c in pin.architecture_checks(Burgers().configure({}, True).bind(prep), prep, cfg)
          if c.name.startswith("PDE residual")][0]
    bad = [c for c in pin.architecture_checks(BrokenSign().configure({}, True).bind(prep), prep, cfg)
           if c.name.startswith("PDE residual")][0]
    assert ok.passed is True and bad.passed is False
    fd = [c for c in pin.architecture_checks(Burgers().configure({}, True).bind(prep), prep, cfg)
          if c.name.startswith("autograd")][0]
    assert fd.passed is True


def test_skip_checks_banner_and_advisor_edits(tmp_path):
    res = B.run(problems=[_problem()], methods=["deeponet"], out_dir=tmp_path / "b", skip_checks=True,
                **{**FAST, "seeds": 1})
    md = (tmp_path / "b" / "table.md").read_text()
    assert "CHECKS SKIPPED" in md and "checks skipped" in md
    seeds = [d for d in res.diagnostics if d.edit.get("seeds")]
    assert seeds and seeds[0].edit["seeds"] == [1, 3]
    new = res.apply_suggestions(tmp_path / "suggested.yaml")
    assert new.seeds == 3 and (tmp_path / "suggested.yaml").exists()


def test_cli_prepare_run_report(tmp_path):
    from pinneapple_analysis.benchmark.cli import main
    cfg = tmp_path / "bench.yaml"
    cfg.write_text("problems: [burgers]\nmethods: [fno]\nbudget: {wall_s: 1}\nseeds: 1\nn_trials: 1\n")
    out = str(tmp_path / "x")
    assert main(["benchmark", "prepare", "--config", str(cfg), "--out", out, "--smoke"]) == 0
    assert main(["benchmark", "run", "--config", str(cfg), "--out", out, "--smoke", "--no-prepare"]) == 0
    assert main(["benchmark", "report", out]) == 0
    assert (tmp_path / "x" / "table.md").exists()


def test_config_round_trip_and_budget(tmp_path):
    c = B.BenchmarkConfig(problems=["burgers"], budget=ComputeBudget(wall_s=10), budget_levels=[0.5, 1], seeds=2)
    c.to_yaml(tmp_path / "c.yaml")
    d = B.BenchmarkConfig.from_yaml(tmp_path / "c.yaml")
    assert d.result_hash() == c.result_hash() and d.budget_for("burgers").wall_s == 10
    with pytest.raises(ValueError):
        B.BenchmarkConfig(budget_levels=[0.5])
    with pytest.raises(ValueError):
        B.BenchmarkConfig.from_dict({"nope": 1})
    _ = torch


def test_tampered_artifacts_are_refused_and_own_artifacts_can_be_adopted(tmp_path):
    out = tmp_path / "b"
    B.prepare(problems=[_problem()], out_dir=out, log=lambda *a: None)
    data = out / "prepared" / "smooth" / "data.npz"
    z = dict(np.load(data))
    z["test_y"] = z["test_y"] + 1.0
    np.savez_compressed(data, **z)                                   # changed after validation
    with pytest.raises(B.PreparationError):
        B.run(problems=[_problem()], methods=["fno"], out_dir=out, prepare_missing=False, stages=(), **FAST)
    other = tmp_path / "c" / "prepared" / "smooth"
    other.mkdir(parents=True)
    tr, va, te = _arrays(48, 0), _arrays(8, 1), _arrays(8, 2)
    np.savez_compressed(other / "data.npz", train_x=tr[0], train_y=tr[1], val_x=va[0], val_y=va[1], test_x=te[0],
                        test_y=te[1], ood_x=_arrays(4, 3, 3.0)[0], ood_y=_arrays(4, 3, 3.0)[1])
    (other / "cost.json").write_text(json.dumps({"train_set_core_s": 5.0}))
    man = B.prepare(problems=[_problem()], out_dir=tmp_path / "c", adopt=True, log=lambda *a: None)["smooth"]
    assert man["adopted"] and man["status"] == "ok" and set(man["hashes"]) == {"data.npz", "cost.json"}


def test_overfit_gate_separates_a_broken_setup_from_a_slow_one():
    import torch

    from pinneapple_analysis.benchmark.builtin.fields import _overfit_check
    w = torch.zeros(1, requires_grad=True)
    frozen = _overfit_check("x/frozen", lambda: (w.detach() - 1.0) ** 2 + 0 * w, [w], steps=50)
    assert frozen.passed is False and frozen.severity == "error"          # gradient cut: cannot learn
    v = torch.zeros(1, requires_grad=True)
    slow = _overfit_check("x/slow", lambda: (v - 1.0) ** 2, [v], steps=200, lr=2e-3)
    assert slow.passed is False and slow.severity == "warning"             # learns, but not to 10 % in 200 steps
    u = torch.zeros(1, requires_grad=True)
    assert _overfit_check("x/ok", lambda: (u - 1.0) ** 2, [u], steps=300, lr=5e-2).passed


def test_equal_flops_arm_is_reported(tmp_path):
    out = tmp_path / "b"
    B.run(problems=[_problem()], methods=["fno", "deeponet"], out_dir=out, equal_flops=True, **FAST)
    assert len(list((out / "runs").glob("flops__smooth__*.json"))) == 4
    md = (out / "table.md").read_text()
    assert "### Equal FLOPs" in md and "| Stopped by |" in md
    assert "equal_flops" in (out / "table.csv").read_text()


def test_budget_study_and_ablations(tmp_path):
    out = tmp_path / "b"
    kw = dict(budget_study={"smooth": {"fno": [2.0]}}, ablations={"smooth": {"fno": {"low_lr": {"lr": 1e-5}}}})
    res = B.run(problems=[_problem()], methods=["fno"], out_dir=out, **FAST, **kw)
    assert (out / "runs" / "study__smooth__fno__B2.json").exists()
    assert (out / "runs" / "diag__smooth__fno__low_lr.json").exists()
    md = (out / "table.md").read_text()
    assert "### Budget study: FNO" in md and "### Setup audit: FNO" in md and "lr = 1e-05" in md
    assert (out / "figures" / "budget_study.png").exists()
    st = res._build()["studies"]["smooth"]
    assert [p["budget_s"] for p in st["study"]["fno"]["all"]] == [1.0, 2.0]       # the full run, then the study run
    assert st["ablation"]["fno"]["baseline"] and st["ablation"]["fno"]["variants"]["low_lr"]["metrics"]
    # studies only add jobs: adding to them reuses the out_dir, changing an existing entry is refused
    more = dict(budget_study={"smooth": {"fno": [2.0, 3.0]}},
                ablations={"smooth": {"fno": {"low_lr": {"lr": 1e-5}, "high_lr": {"lr": 1e-2}}}})
    B.run(problems=[_problem()], methods=["fno"], out_dir=out, stages=(), **FAST, **more)
    with pytest.raises(B.ConfigMismatch):
        B.run(problems=[_problem()], methods=["fno"], out_dir=out, stages=(), **FAST,
              budget_study={"smooth": {"fno": [3.0]}}, ablations=more["ablations"])
    with pytest.raises(B.ConfigMismatch):
        B.run(problems=[_problem()], methods=["fno"], out_dir=out, stages=(), **FAST, budget_study=more["budget_study"],
              ablations={"smooth": {"fno": {"low_lr": {"lr": 1e-4}}}})


def test_yplus_check_uses_the_reynolds_number_of_the_reference_length():
    from pinneapple_analysis.benchmark.validation import yplus_check
    c = yplus_check("airfoil", 4e-6, 1.0, 1.0, 1 / 4e6)          # the aero app's O-grid: first cell 4e-6 chords
    assert c.passed and 0.3 < c.value < 1.0 and "nan" not in c.detail
    assert yplus_check("airfoil", 4e-5, 1.0, 1.0, 1 / 4e6).passed is False   # 10x coarser: y+ ~ 6


def test_airfoil_convergence_is_judged_on_what_each_method_predicts():
    prob = B.get_problem("airfoil")
    assert prob.main_metric_for("mlp") == ("test4", "cd")              # integrals only
    for m in ("pinn", "pinn_rans", "mgn", "mlp_fields"):                # potential flow: Cd = 0 by construction
        assert prob.main_metric_for(m) == ("test4", "u")


def test_equal_flops_arm_with_a_per_instance_pinn(tmp_path):
    """The equal-FLOPs table finds the PINN's run by its instance (the arm trains the first "final" instance)."""
    res = B.run(problems=["burgers"], methods=["pinn", "fno"], out_dir=tmp_path / "b", smoke=True, equal_flops=True,
                budget={"wall_s": 2.0}, seeds=1, n_trials=1, n_workers=1, log=lambda *a: None)
    ef = res._build()["tables"]["burgers"]["equal_flops"]
    assert ef and ef["rows"]["pinn"]["common"] and ef["rows"]["fno"]["common"]


def test_budget_study_curve_uses_each_methods_own_metric():
    from pinneapple_analysis.benchmark.report import study_metric_key

    class Prob:
        def main_metric_for(self, mid):
            return ("test4", "cd") if mid == "mlp" else ("test4", "u")

    pts = [{"budget_s": 1.0, "metrics": {"test4/cd": 1.0, "test4/u": 0.5}}]
    assert study_metric_key(Prob(), "pinn", pts) == "test4/u"      # a field model is judged on its field
    assert study_metric_key(Prob(), "mlp", pts) == "test4/cd"
    assert study_metric_key(object(), "pinn", pts) == "test4/cd"   # no metric declared: the first column
