"""Scope of use (pinneapple_veriphysics.scope): which uses a result serves and why, in the Decision Record and the
Evidence Report -- beyond one approve / reject verdict."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_veriphysics.decision import DecisionRecord
from pinneapple_veriphysics.evidence_report import render_evidence_report_pdf
from pinneapple_veriphysics.scope import ScopeOfUse, UseScope, scope_from_batch, scope_from_verdicts


def _record(**kw):
    base = dict(run_id="r1", problem_description="2-D Darcy", recommended_strategy="FNO surrogate",
                reasoning="trained with PhysicsNeMo", trust_score=None, trust_coverage=0.0, trustworthy=None)
    return DecisionRecord(**{**base, **kw})


def _darcy_verdicts(wall_factor=1.0):
    from pinneapple_physics import darcy_checks as dc
    from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob
    a = ob.threshold_coefficient(ob.periodic_grf(3, (48, 48), alpha=2.0, tau=3.0, seed=2))
    u = np.stack([ob.darcy_2d(x) for x in a])
    u[:, :2, :] *= wall_factor                        # wrong next to one wall only
    env = dc.CoefficientEnvelope.fit(a, q=0.0)
    return [dc.use_verdicts(a[i], u[i], envelope=env) for i in range(len(a))]


def test_use_scope_lines_and_evidence_kinds():
    u = UseScope("interior field", True, "error ~1.2 % <= 5 %", "values away from the walls", "estimated")
    assert u.status == "USABLE" and "[estimated]" in u.line() and "needs: values" in u.line()
    assert UseScope("x", None).status == "not assessed"
    with pytest.raises(ValueError):
        UseScope("x", True, evidence="vibes")


def test_scope_from_darcy_use_verdicts_says_where_and_why():
    v = _darcy_verdicts(wall_factor=1.5)[0]
    scope = scope_from_verdicts("FNO, sample 0", v, evidence={"inputs": "input check"}, notes=["estimate validated"])
    assert scope.usable == ["interior field"] and scope.not_usable == ["walls and fluxes"]
    assert scope.summary() == "use for: interior field; not for: walls and fluxes; preconditions met"
    walls = next(u for u in scope.uses if u.use == "walls and fluxes")
    assert ">" in walls.reason and walls.limits == {"wall": 0.05, "flux": 0.02} and walls.evidence == "estimated"
    inputs = next(u for u in scope.uses if u.use == "inputs")
    assert inputs.evidence == "input check" and inputs.precondition and inputs.status == "MET"
    assert ScopeOfUse.from_dict(scope.to_dict()) == scope
    text = scope.render()
    assert "[NOT USABLE] walls and fluxes" in text and "note: estimate validated" in text


def test_scope_from_batch_uses_the_share_of_samples():
    good = _darcy_verdicts()
    scope = scope_from_batch("solver", good)
    assert all(u.ok and u.fraction_ok == 1.0 for u in scope.uses)
    mixed = good[:1] + _darcy_verdicts(wall_factor=1.5)[:2]          # walls OK in 1 of 3 samples
    s = scope_from_batch("mixed", mixed, min_fraction=0.9)
    walls = next(u for u in s.uses if u.use == "walls and fluxes")
    assert walls.ok is False and walls.fraction_ok == pytest.approx(1 / 3) and "median" in walls.reason
    assert "(33 % of samples)" in walls.line()
    with pytest.raises(ValueError):
        scope_from_batch("empty", [])
    out = scope_from_verdicts("x", {"inputs": {"ok": False, "reason": "max a 20 > 12"}}, evidence="input check")
    assert out.summary() == "precondition not met: inputs" and out.uses[0].status == "NOT MET"


def test_decision_record_renders_the_scope_and_is_unchanged_without_it():
    plain = _record()
    assert "Scope of use" not in plain.render()
    scope = scope_from_verdicts("FNO", _darcy_verdicts(1.5)[0], evidence={"inputs": "input check"})
    rec = _record(scope_of_use=scope.to_dict())
    text = rec.render()
    assert "Scope of use -- FNO" in text and "not for: walls and fluxes" in text
    assert text.index("Scope of use") < text.index("Per-check confidence breakdown")
    assert plain.render() == _record(scope_of_use=None).render()


def test_evidence_report_has_the_scope_section_and_figure(tmp_path):
    pytest.importorskip("reportlab")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(np.random.default_rng(0).random((16, 16)))
    png = tmp_path / "map.png"
    fig.savefig(png)
    plt.close(fig)
    scope = scope_from_verdicts("FNO", _darcy_verdicts(1.5)[0])
    without = render_evidence_report_pdf(_record())
    text_only = render_evidence_report_pdf(_record(scope_of_use=scope.to_dict()))
    scope.figure = str(png)
    with_fig = render_evidence_report_pdf(_record(scope_of_use=scope.to_dict()))
    assert all(p.startswith(b"%PDF-") for p in (without, text_only, with_fig))
    assert len(without) < len(text_only) < len(with_fig)
    assert b"/Subtype /Image" in with_fig and b"/Subtype /Image" not in text_only
    pypdf = pytest.importorskip("pypdf")
    import io
    pages = "".join(pg.extract_text() for pg in pypdf.PdfReader(io.BytesIO(text_only)).pages)
    assert "Where this result can be used" in pages and "not for: walls and fluxes" in pages
    scope.figure = str(tmp_path / "missing.png")
    assert render_evidence_report_pdf(_record(scope_of_use=scope.to_dict())).startswith(b"%PDF-")
