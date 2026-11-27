"""The weather, ensemble and CFD-surrogate records are the same ``Trust`` JSON (#345).

The issue is done when "the weather, ensemble and one CFD surrogate example all emit the same ``Trust`` JSON, and the
docs show it". "The same" is what this file holds, three ways:

* the three documents validate against **one** versioned JSON Schema (:data:`TRUST_SCHEMA`) and not three schemas;
* they agree **key for key** — the recursion below walks the three dictionaries and reports every path at which a key
  exists in one and not in another, with the two documented exceptions (the case's own inputs, and the lists that have
  one entry per expert);
* they come out of the **same code path** — every residual carries the ensemble adapter's source, and every provenance
  carries the ensemble adapter's fingerprint method, which is the signature of ``from_ensemble`` and not of three
  producers that happen to agree.

The numbers are asserted too, and the numbers are measurements: the weather record's residual level comes from
``residual_baseline`` on the initialisations the scored case was held out of, the CFD one from the median residual of
the POD surrogate over the Reynolds numbers it was fitted for.
"""
from __future__ import annotations

import copy
import json

import pytest

from pinneapple_analysis.trust.producers import (
    KOVASZNAY_FIT,
    cfd_surrogate_trust,
    kovasznay_case,
    navier_stokes_residual,
    three_records,
)
from pinneapple_analysis.trust.record import TRUST_SCHEMA, Trust


@pytest.fixture(scope="module")
def records() -> dict[str, Trust]:
    """The three records of the issue, produced once for the whole file (the weather testbed is the slow one)."""
    return three_records()


@pytest.fixture(scope="module")
def documents(records) -> dict[str, dict]:
    return {kind: json.loads(record.to_json()) for kind, record in records.items()}


# ---------------------------------------------------------------------------------------- the same shape
def test_the_three_documents_validate_against_one_schema(documents):
    """The "done when": three JSONs, one schema, no problems."""
    for kind, document in documents.items():
        problems = Trust.validate_document(document)
        assert problems == [], f"{kind}: {problems}"


def _paths(value, prefix=""):
    """Every key path of a document, with the two forms of a ``TrustValue`` folded into one.

    A field that was measured carries ``value``/``unit``; a field that was not carries ``reason`` instead. That is the
    object's rule, and both are the same field of the record — so the walk treats them as ``<value>`` and the schema's
    ``oneOf`` is what keeps them honest. Lists become ``[<element>]``: the number of residuals is the number of
    experts, and the number of gates follows from it.
    """
    out: set[str] = set()
    if isinstance(value, dict):
        if "status" in value:
            return {f"{prefix}/<value>"}
        for k, v in value.items():
            out |= _paths(v, f"{prefix}/{k}")
        out.add(prefix)
    elif isinstance(value, list):
        out.add(f"{prefix}[{len(value)}]")
        for v in value:
            out |= _paths(v, f"{prefix}[*]")
    else:
        out.add(prefix)
    return out


def test_the_three_documents_agree_key_for_key(documents):
    """Every key of one is a key of the other two; nothing is added and nothing is missing."""
    paths = {kind: _paths(doc) for kind, doc in documents.items()}

    def keys_only(s: set[str]) -> set[str]:
        # the case's own inputs are the case's, and the list lengths are one per expert: neither is the schema
        return {p for p in s if "/provenance/inputs" not in p and "[" not in p}

    weather, ensemble, surrogate = (keys_only(paths[k]) for k in ("weather", "ensemble", "surrogate"))
    assert weather == ensemble, sorted(weather ^ ensemble)
    assert weather == surrogate, sorted(weather ^ surrogate)


def _experts_of(document: dict) -> list[str]:
    """The expert names the provenance of a record names: ``PhysicsEnsemble(pod-3, pod-6)`` -> ``["pod-3", "pod-6"]``."""
    model = document["provenance"]["model"]
    inside = model[model.index("(") + 1:model.rindex(")")] if "(" in model else ""
    return [name.strip() for name in inside.split(",") if name.strip()]


def test_the_lists_are_one_entry_per_expert_in_every_document(documents):
    """The only lists that differ between the three are the per-expert ones, and they follow the models exactly."""
    for kind, document in documents.items():
        experts = _experts_of(document)
        assert experts, kind
        assert [r["expert"] for r in document["residuals"]] == experts, kind
        assert [g["name"] for g in document["gates"] if g["name"].startswith("residual:")] == \
               [f"residual:{r['expert']}/{r['law']}" for r in document["residuals"]], kind


def test_the_three_come_out_of_the_same_adapter(records):
    """``from_ensemble`` leaves its own signature on every record: the residual source and the fingerprint method."""
    for kind, record in records.items():
        assert record.residuals, f"{kind}: no residual"
        for residual in record.residuals:
            assert residual.value.source == ("pinneapple_physics.ensemble.PhysicsEnsemble.predict "
                                             "(residual_fn on this query)"), (kind, residual.name)
        assert "of the expert names, kinds and declared costs" in record.provenance.model_fingerprint_method
        assert record.provenance.model.startswith("PhysicsEnsemble(")


def test_the_schema_refuses_a_document_that_lost_a_field(documents):
    """The schema is what makes the three comparable: drop a field from one and it stops being a record."""
    broken = copy.deepcopy(documents["surrogate"])
    broken.pop("provenance")
    assert any("provenance" in p for p in Trust.validate_document(broken))
    wrong_badge = copy.deepcopy(documents["ensemble"])
    wrong_badge["badge"] = "excellent"
    assert Trust.validate_document(wrong_badge)
    measured_with_reason = copy.deepcopy(documents["weather"])
    measured_with_reason["in_domain"]["reason"] = "both at once"
    assert Trust.validate_document(measured_with_reason)


# ---------------------------------------------------------------------------------------- the numbers
def test_the_weather_record_reads_what_weather_evaluate_measured(documents):
    """The forecast record: a horizon from ``weather.evaluate.horizons``, a residual level from a held-out case."""
    d = documents["weather"]
    assert d["kind"] == "weather"
    assert d["in_domain"]["value"] == 1.0
    by_expert = {r["expert"]: r["value"] for r in d["residuals"]}
    # the two semi-Lagrangian schemes are within 30 % of their own in-domain level on the hardest case
    for expert in ("semi-lagrangian-linear", "semi-lagrangian-cubic"):
        value, level = by_expert[expert]["value"], by_expert[expert]["reference"]
        assert 0.0 < level < 0.05, (expert, level)
        assert 0.95 < value / level < 1.5, (expert, value, level)
    # persistence's residual is 1 by construction: the field it holds explains none of the change
    assert by_expert["persistence"]["value"] == 1.0
    # the forecast never loses skill inside the five days scored, so the horizon says so instead of inventing one
    assert d["horizon"]["no_skill"]["status"] == "not_measured"
    assert "beyond them" in d["horizon"]["no_skill"]["reason"]
    assert d["horizon"]["requested"]["value"] == 48.0
    assert d["uncertainty"]["coverage"]["status"] == "measured"
    assert d["uncertainty"]["coverage"]["value"] > 0.9


def test_the_ensemble_record_reads_the_in_domain_level_of_each_expert(documents):
    """The combined model: the graph expert is exact at ``u = 1``, where its in-domain median says it is not."""
    d = documents["ensemble"]
    assert d["kind"] == "ensemble"
    assert d["in_domain"]["value"] == 1.0
    by_expert = {r["expert"]: r["value"] for r in d["residuals"]}
    assert by_expert["fno"]["value"] == 0.0
    assert by_expert["gnn"]["value"] == 0.0
    assert 0.0 < by_expert["gnn"]["reference"] < 0.02
    assert d["badge"] == "fading"                       # only because nothing here measures a horizon
    assert "horizon unmeasured" in d["badge_reason"]


def test_the_cfd_surrogate_reads_the_momentum_residual_of_its_own_field(documents):
    """The CFD record: the momentum law on the field the surrogate emits, against the level it has in its own range."""
    d = documents["surrogate"]
    assert d["kind"] == "surrogate"
    assert d["provenance"]["model"] == "PhysicsEnsemble(pod-3, pod-6)"
    assert d["in_domain"]["value"] == 1.0
    assert d["provenance"]["inputs"]["re"] == 70.0
    by_expert = {r["expert"]: r["value"] for r in d["residuals"]}
    for expert in ("pod-3", "pod-6"):
        value, level = by_expert[expert]["value"], by_expert[expert]["reference"]
        assert 0.01 < level < 0.02, (expert, level)      # the discretisation floor of the reference solution
        assert 1.5 < value / level < 2.0, (expert, value, level)
    # the two truncations disagree by ~1e-6 over the fitted parameters and by the same at an interpolated one, so the
    # interval cannot cover a 1.9 % error: the coverage gate fails honestly instead of being narrowed to pass
    assert d["uncertainty"]["coverage"]["value"] < 0.5
    assert d["uncertainty"]["level"] == 0.9
    assert "calibration fail" in d["badge_reason"]


def test_the_cfd_surrogate_outside_its_envelope_is_no_skill():
    """Re = 200 is outside the Reynolds numbers it was fitted for, and the record says no skill rather than fading."""
    record = cfd_surrogate_trust(200.0)
    document = json.loads(record.to_json())
    assert Trust.validate_document(document) == []
    assert record.badge() == "no_skill"
    assert document["in_domain"]["value"] == 0.0
    residual = max(r.value.value for r in record.residuals)
    level = max(r.value.reference for r in record.residuals)
    assert residual / level > 10.0                        # the momentum law is violated an order of magnitude harder
    assert "domain no_skill" in record.badge_reason()


# ---------------------------------------------------------------------------------------- the pieces
def test_the_navier_stokes_residual_of_the_exact_solution_is_the_discretisation_floor():
    """The residual below is what the grid's own finite differences leave on the exact solution: measured, not declared."""
    axes, exact = kovasznay_case(50.0)
    residual = navier_stokes_residual(axes, exact, 50.0)
    assert 0.005 < residual["momentum"] < 0.02, residual
    assert 0.003 < residual["mass"] < 0.02, residual
    assert residual["momentum_scale"] > 1.0


def test_the_surrogate_is_fitted_on_the_reynolds_numbers_it_declares():
    """The in-domain level is the median over exactly these cases, so they are part of the contract."""
    assert KOVASZNAY_FIT == (20.0, 40.0, 60.0, 80.0, 100.0, 120.0)
    for reynolds in KOVASZNAY_FIT:
        axes, fields = kovasznay_case(reynolds)
        assert fields["U"].shape == (48, 48, 2)
        assert fields["p"].shape == (48, 48)
        assert navier_stokes_residual(axes, fields, reynolds)["momentum"] < 0.05


# ---------------------------------------------------------------------------------------- pp.trust kinds
def test_pp_trust_reads_a_pinn_at_points_it_never_trained_on():
    """The second model kind the issue names: a PINN, whose residual is re-evaluated and never read off its loss."""
    import pinneapple as pp

    solution = pp.solve("burgers_1d", method="pinn", epochs=40, hidden_dim=8, n_layers=2, n_collocation=64)
    record = pp.trust(solution, None, n_points=64)
    assert record.kind == "pinn"
    assert Trust.validate_document(record.to_json()) == []
    residual = record.residuals[0]
    assert residual.law == "pde" and residual.value.is_measured
    assert residual.value.value > 0.0
    # no in-domain level was measured for it, so the gate is unmeasured and the badge is fading — never a silent pass
    assert residual.value.reference is None
    assert any(g.name.startswith("residual:") and g.status == "unmeasured" for g in record.gates())
    assert record.provenance.model_fingerprint and len(record.provenance.model_fingerprint) == 64


def test_pp_trust_refuses_a_grid_operator_by_name():
    """A grid operator predicts a field, not a value at a point: refused with the reason, not scored as if it were."""
    import pinneapple as pp
    from pinneapple_core.errors import NotSupported
    from pinneapple_neural.architectures.neural_operators.fno import FourierNeuralOperator
    from pinneapple_neural.physics_model import as_physics_model

    problem = pp.PhysicalProblem.from_preset("burgers_1d", nu=0.01)
    operator = as_physics_model(FourierNeuralOperator(in_channels=2, out_channels=1, width=8, modes=4, layers=1))
    with pytest.raises(NotSupported, match="cannot be sampled at points"):
        pp.trust(operator, None, problem, n_points=16)


def test_the_producers_name_the_three_kinds_of_the_issue(records):
    assert set(records) == {"weather", "ensemble", "surrogate"}
    assert records["weather"].kind == "weather"
    assert records["ensemble"].kind == "ensemble"
    assert records["surrogate"].kind == "surrogate"
    assert json.loads(records["surrogate"].to_json())["schema"] == TRUST_SCHEMA["properties"]["schema"]["const"]
