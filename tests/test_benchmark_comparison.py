"""The standard field comparison of the benchmark report (#63): the same figure in every case that has a planar
field, and a declared reason in every case that has not.

The figure itself (reference, surrogate and the error, one shared colour scale, the L2 in the number) is
``pp.viz.compare_solutions``, tested in ``tests/test_compare_solutions.py``. What is tested here is the benchmark's
half of the rule: the case produces it, the report points at it and quotes the same L2 the function computed, a case
with no 2-D field says why instead of getting a picture, and the surrogate in the picture is the case's best.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis import benchmark as B

FAST = dict(budget={"wall_s": 1.0}, seeds=1, n_trials=1, n_workers=1, log=lambda *a: None)

#: the cases whose field is not planar, with the reason each one gives
DECLARED_WITHOUT_FIELD = {"burgers", "calculix_beam", "calculix_bracket"}


def _arrays_2d(n, seed):
    """A smoothing operator on a 24 x 24 grid: an input field and its low-passed output."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 24, 24)).astype(np.float32)
    k = np.fft.fftfreq(24)
    low = np.exp(-0.05 * (k[:, None] ** 2 + k[None, :] ** 2) * 24).astype(np.complex64)
    y = np.real(np.fft.ifft2(np.fft.fft2(x, axes=(1, 2)) * low[None], axes=(1, 2))).astype(np.float32)
    return x, y


def _problem_2d(id_="smooth2d"):
    return B.ArrayProblem(id_, train=_arrays_2d(24, 0), val=_arrays_2d(8, 1), test=_arrays_2d(6, 2),
                          ood=_arrays_2d(4, 3), data_cost_s=1.0)


#: the FNO's default 16 modes are above the Nyquist limit of a 24-point grid; asking for 6 keeps the pair valid
FNO_2D = {"fno": {"modes": 6}}


def test_every_case_declares_whether_it_has_a_planar_comparison_field():
    """The rule over the whole registry: a case either has the field (no declaration, and its trainers store it) or
    says in words why it has not. The three that cannot are named here, so a case silently dropping out of the figure
    -- or silently getting one it cannot support -- fails this test."""
    without = {pid for pid in B.problems() if getattr(B.get_problem(pid), "comparison_absent", "")}
    assert without == DECLARED_WITHOUT_FIELD
    for pid in without:
        reason = B.get_problem(pid).comparison_absent
        assert len(reason) > 40 and reason.strip() == reason, f"{pid}: the reason is a sentence, not a placeholder"
    for pid in set(B.problems()) - without:
        assert not getattr(B.get_problem(pid), "comparison_absent", ""), f"{pid} declares no field but has the reason"


def test_a_case_of_the_registry_gets_the_same_figure(tmp_path):
    """``darcy`` is a real 2-D case (a solved coefficient field, a 2-D pressure head), at a 16 x 16 grid and the same
    1 s budget: the figure is the case's, drawn by the same code path, with the case's own label on the colour bar."""
    out = tmp_path / "b"
    res = B.run(problems=["darcy"], methods=["fno"], out_dir=out, smoke=True, method_options=FNO_2D,
                problem_options={"darcy": {"n_solve": 16, "n_out": 16}}, **FAST)
    assert (out / "figures" / "comparison_darcy.png").exists()
    entry = json.loads((out / "summary.json").read_text())["comparison"]["darcy"]
    assert entry["method"] == "fno" and entry["metrics"]["n"] == 16 * 16
    with np.load(out / "runs" / entry["npz"]) as z:
        assert str(z["quantity"]) == "u" and str(z["unit"]) == ""
        assert len(np.unique(z["x"])) == 16 and len(np.unique(z["y"])) == 16, "a product grid, not a cloud"
    assert "| Darcy 2-D |" in (out / "table.md").read_text()
    assert res._build()["tables"]["darcy"]["rows"]["fno"]["status"].startswith("pass")


def test_a_two_dimensional_case_gets_the_standard_figure_and_the_number_behind_it(tmp_path):
    out = tmp_path / "b"
    B.run(problems=[_problem_2d()], methods=["fno"], out_dir=out, method_options=FNO_2D, **FAST)
    figure = out / "figures" / "comparison_smooth2d.png"
    assert figure.exists() and figure.stat().st_size > 10_000
    summary = json.loads((out / "summary.json").read_text())
    entry = summary["comparison"]["smooth2d"]
    assert entry["figure"] == "comparison_smooth2d.png" and entry["reason"] == "" and entry["method"] == "fno"
    assert "comparison_smooth2d.png" in summary["figures"]
    # the quoted L2 is the stored field's, measured with the one definition the house uses
    from pinneapple_tools.visualization.fields import field_error_metrics
    with np.load(out / "runs" / entry["npz"]) as z:
        again = field_error_metrics(z["reference"], z["prediction"])
    assert entry["metrics"]["l2_relative"] == pytest.approx(again["l2_relative"], rel=1e-6)
    assert entry["metrics"]["n"] == 24 * 24
    md = (out / "table.md").read_text()
    assert "## Field comparison (1 of 1 case(s))" in md
    assert "`figures/comparison_smooth2d.png`" in md and f"{100 * again['l2_relative']:.2f} %" in md
    assert "no comparison figure" not in md


def test_a_one_dimensional_case_is_declared_instead_of_drawn(tmp_path):
    """``burgers`` is a real case of the registry, its FNO row is a valid one, and its field is a profile: the report
    prints the case's reason and draws nothing."""
    out = tmp_path / "b"
    res = B.run(problems=["burgers"], methods=["fno"], out_dir=out, smoke=True, method_options=FNO_2D,
                problem_options={"burgers": {"n_solve": 64, "n_out": 32}}, **FAST)
    assert res._build()["tables"]["burgers"]["rows"]["fno"]["status"].startswith("pass")
    assert (out / "runs" / "final__burgers__fno__L1.000__s0.json").exists(), "the run itself is a valid one"
    md = (out / "table.md").read_text()
    assert "## Field comparison (0 of 1 case(s))" in md
    assert "Cases with no comparison figure, and why:" in md
    assert B.get_problem("burgers").comparison_absent in md
    assert not (out / "figures" / "comparison_burgers.png").exists()
    entry = json.loads((out / "summary.json").read_text())["comparison"]["burgers"]
    assert entry["figure"] is None and entry["metrics"] is None
    assert "1-D profile" in entry["reason"]


def test_the_case_with_the_lowest_field_error_is_the_one_drawn(tmp_path):
    """Two field methods, one that reproduces the reference and one that predicts zeros: the figure is of the first
    -- the case's best on its own column -- and it is that method's stored field the report read. The two stubs make
    the ranking exact, so the test does not depend on how far a real model got in one second."""
    from pinneapple_analysis.benchmark.builtin.fields import FieldData
    from pinneapple_analysis.benchmark.comparison import grid_coordinates, save_comparison
    from pinneapple_analysis.cost import train_within_budget

    @B.register_method("mirror")
    class Mirror(B.Method):
        """Predicts the reference exactly: the best a field method can do on any column."""
        title = "Mirror (the reference itself)"

        def supports(self, problem):
            return isinstance(problem, B.FieldOperatorProblem)

        def space(self, problem):
            return {"lr": [1e-3]}

        def train(self, problem, prepared, cfg, seed, budget, final=True, comparison_path=None, **_):
            run = train_within_budget(lambda p: 1.0, budget, flops_per_step=1.0)
            D = FieldData(prepared)
            out = {"train": run.to_dict(), "infer_batch1": {"median_s": 1e-6}}
            for split in (("val", "test", "ood") if final else ("val",)):
                y = D.raw[f"{split}_y"]
                out[split] = problem.scores(y, y)
            if final:
                y = D.raw["test_y"][0]
                X, Y = grid_coordinates(y.shape)
                save_comparison(comparison_path, y, y, x=X, y=Y, quantity="field", instance="test0")
            return out

        def architecture_checks(self, problem, prepared, cfg):
            return [B.Check("forward shape", "architecture", f"{problem.id}/mirror", True)]

    @B.register_method("zeros")
    class Zeros(B.Method):
        """Predicts a field of zeros: a relative L2 of exactly 1, which loses to anything that has learned."""
        title = "Zeros"

        def supports(self, problem):
            return isinstance(problem, B.FieldOperatorProblem)

        def space(self, problem):
            return {"lr": [1e-3]}

        def train(self, problem, prepared, cfg, seed, budget, final=True, **_):
            run = train_within_budget(lambda p: 1.0, budget, flops_per_step=1.0)
            D = FieldData(prepared)
            out = {"train": run.to_dict(), "infer_batch1": {"median_s": 1e-6}}
            for split in (("val", "test", "ood") if final else ("val",)):
                y = D.raw[f"{split}_y"]
                out[split] = problem.scores(np.zeros_like(y), y)
            return out

        def architecture_checks(self, problem, prepared, cfg):
            return [B.Check("forward shape", "architecture", f"{problem.id}/zeros", True)]

    out = tmp_path / "b"
    B.run(problems=[_problem_2d("smooth2d_best")], methods=["mirror", "zeros"], out_dir=out, **FAST)
    summary = json.loads((out / "summary.json").read_text())
    entry = summary["comparison"]["smooth2d_best"]
    assert entry["method"] == "mirror" and entry["method_title"] == "Mirror (the reference itself)"
    assert entry["npz"] == "final__smooth2d_best__mirror__L1.000__s0.npz"
    assert entry["metrics"]["l2"] == 0.0 and entry["metrics"]["l2_relative"] == 0.0
    assert summary["tables"]["smooth2d_best"]["rows"]["zeros"]["cells"], "the zeros model was measured too"
    assert (out / "figures" / "comparison_smooth2d_best.png").exists()


def test_a_case_without_a_stored_field_says_so_instead_of_failing(tmp_path):
    """A run whose sidecar is missing -- one made before the trainers stored the field, or a best method that could
    not store it -- still reports: the entry names the case and the missing sidecar, and no figure is invented."""
    from pinneapple_analysis.benchmark import comparison
    out = tmp_path / "b"
    res = B.run(problems=[_problem_2d("smooth2d_old")], methods=["fno"], out_dir=out, method_options=FNO_2D, **FAST)
    (out / "runs" / "final__smooth2d_old__fno__L1.000__s0.npz").unlink()          # the sidecar this figure needs
    entries = comparison.select(res.plan, res._build()["tables"])
    assert entries["smooth2d_old"]["figure"] is None and entries["smooth2d_old"]["metrics"] is None
    assert "no comparison field was stored" in entries["smooth2d_old"]["reason"]
    assert entries["smooth2d_old"]["method"] == "fno", "the case is still named, with the reason it has no picture"
