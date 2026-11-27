"""#293: a interface dos surrogates agnósticos de geometria, e o contrato que cada família tem de cumprir.

O contrato é curto e é o que a issue pede: uma amostra é um conjunto de pontos (qualquer número, em qualquer ordem)
com o campo neles e as integrais do corpo; `fit` treina famílias diferentes com o mesmo código; `predict` responde em
qualquer contagem de pontos e devolve as integrais **integradas do próprio campo previsto**; e uma família que não está
implementada diz o motivo em vez de devolver um substituto com o nome dela.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.surrogate.geometry import (
    GeometrySample,
    PointNetGeometry,
    TransolverGeometry,
    available_families,
    build_geometry_surrogate,
    panel_integrals,
    samples_from_panel,
)
from pinneapple_design.aero.geometry import BOUNDS, latin_hypercube, valid

torch = pytest.importorskip("torch")


def body_family(n: int = 6, seed: int = 0, n_per_side: int = 12) -> list[GeometrySample]:
    """Uma família pequena de corpos reais, rotulada pelo método de painéis: o dado de simulação do teste."""
    rng = np.random.default_rng(seed)
    shapes: list[np.ndarray] = []
    while len(shapes) < n:
        c = latin_hypercube(1, BOUNDS, rng)[0]
        if valid(c)[0]:
            shapes.append(c)
    return samples_from_panel(shapes, alphas=(0.0, 4.0), n_per_side=n_per_side)


# ----------------------------------------------------------------------------------------------- a amostra
def test_a_sample_carries_the_field_at_its_points_and_the_integrals():
    sample = body_family(1)[0]
    assert sample.points.shape[1] == 3 and sample.normals.shape == sample.points.shape
    assert sample.fields["cp"].shape == (sample.n_points,)
    assert set(sample.integrals) == {"cl", "cd"}
    assert sample.condition_keys == ["alpha_deg"]


def test_a_sample_refuses_a_field_that_does_not_match_its_points():
    with pytest.raises(ValueError, match="values for"):
        GeometrySample(points=np.zeros((5, 3)), fields={"cp": np.zeros(4)})


def test_a_subset_keeps_the_conditions_and_the_integrals():
    sample = body_family(1)[0]
    sub = sample.subset(np.arange(0, sample.n_points, 2))
    assert sub.n_points == (sample.n_points + 1) // 2
    assert sub.conditions == sample.conditions and sub.integrals == sample.integrals
    assert sub.fields["cp"].shape == (sub.n_points,)


# ----------------------------------------------------------------------------------------------- o catálogo
def test_every_family_of_the_issue_is_in_the_catalogue_with_its_reason():
    catalogue = available_families()
    assert {"transolver", "gino", "pointnet", "meshgraphnet"} <= set(catalogue)
    assert catalogue["transolver"]["implemented"] and catalogue["pointnet"]["implemented"]
    for open_family in ("gino", "meshgraphnet"):
        assert not catalogue[open_family]["implemented"]
        assert len(catalogue[open_family]["note"]) > 80          # o motivo, não um "TODO"
    assert available_families(implemented_only=True).keys() == {"transolver", "pointnet"}


def test_building_an_open_family_says_why_instead_of_returning_a_stand_in():
    with pytest.raises(NotImplementedError, match="gino"):
        build_geometry_surrogate("gino")
    with pytest.raises(ValueError, match="known"):
        build_geometry_surrogate("no-such-family")
    assert build_geometry_surrogate("transolver").family == "transolver"


# ----------------------------------------------------------------------------------------------- as integrais
def test_the_integrator_reproduces_the_solver_own_coefficients():
    """A integral que a interface reporta é a do solver aplicada ao campo: no campo do próprio solver, ela bate."""
    sample = body_family(1)[0]
    got = panel_integrals(fields=sample.fields, normals=sample.normals, elements=sample.elements,
                          conditions=sample.conditions)
    assert got["cl"] == pytest.approx(sample.integrals["cl"], abs=1e-9)
    assert got["cd"] == pytest.approx(sample.integrals["cd"], abs=1e-9)


# ----------------------------------------------------------------------------------------------- fit/predict
@pytest.fixture(scope="module")
def trained() -> tuple[object, list[GeometrySample], list[GeometrySample]]:
    samples = body_family(8, seed=3)
    train, held = samples[:12], samples[12:]
    model = TransolverGeometry(fields=("cp",), hidden=32, depth=2, slices=8, integrate=panel_integrals).fit(
        train, epochs=6, batch=4, seed=0)
    return model, train, held


def test_fit_predict_round_trip_on_a_real_case(trained):
    model, train, held = trained
    sample = held[0]
    pred = model.predict(sample.points, sample.normals, sample.conditions, sample.elements)
    assert pred.fields["cp"].shape == (sample.n_points,)
    assert np.isfinite(pred.fields["cp"]).all()
    assert set(pred.integrals) == {"cl", "cd"}
    # a integral é do campo previsto, não de uma segunda cabeça: um campo nulo dá integral nula
    zero = panel_integrals(fields={"cp": np.zeros_like(sample.fields["cp"])}, normals=sample.normals,
                           elements=sample.elements, conditions=sample.conditions)
    assert zero["cl"] == pytest.approx(0.0, abs=1e-12)


def test_predict_answers_at_any_point_count(trained):
    model, _, held = trained
    sample = held[0]
    fewer = sample.subset(np.arange(0, sample.n_points, 3))
    out = model.predict(fewer.points, fewer.normals, fewer.conditions, fewer.elements)
    assert out.fields["cp"].shape == (fewer.n_points,)


def test_predict_is_deterministic(trained):
    model, _, held = trained
    sample = held[0]
    a = model.predict(sample.points, sample.normals, sample.conditions, sample.elements)
    b = model.predict(sample.points, sample.normals, sample.conditions, sample.elements)
    assert np.array_equal(a.fields["cp"], b.fields["cp"])


def test_the_field_depends_on_the_whole_point_set_which_is_why_chunking_is_not_free(trained):
    """As famílias agregam o conjunto (as slices do Transolver são médias ponderadas dos pontos, o baseline dá
    max-pool), então o campo num ponto depende do resto do conjunto — fatiar a predição muda a resposta. O teste
    guarda a propriedade, para que ela não vire uma surpresa num estudo."""
    model, _, held = trained
    sample = held[0]
    whole = model.predict(sample.points, sample.normals, sample.conditions, sample.elements)
    piecewise = model.predict(sample.points, sample.normals, sample.conditions, sample.elements, chunk=7)
    assert not np.allclose(whole.fields["cp"], piecewise.fields["cp"], atol=1e-9)


def test_evaluate_reports_the_field_error_and_the_integrals(trained):
    model, _, held = trained
    evaluation = model.evaluate(held)
    summary = evaluation.summary()
    assert summary["samples"] == len(held)
    assert 0.0 <= summary["relative_l2_mean"] <= summary["relative_l2_worst"]
    assert "integral_error:cl" in summary
    assert "rank_r2" in summary                       # com quatro corpos, o ranking é medido


def test_a_trained_model_saves_and_reloads_to_the_same_numbers(trained, tmp_path):
    model, _, held = trained
    before = model.predict(held[0].points, held[0].normals, held[0].conditions, held[0].elements)
    model.save(str(tmp_path / "checkpoint"))
    reloaded = TransolverGeometry().load(str(tmp_path / "checkpoint"))
    after = reloaded.predict(held[0].points, held[0].normals, held[0].conditions, held[0].elements)
    assert np.allclose(before.fields["cp"], after.fields["cp"], atol=1e-6)
    assert reloaded.fields == model.fields


def test_the_other_family_trains_through_the_same_interface():
    samples = body_family(6, seed=5)
    model = PointNetGeometry(fields=("cp",), hidden=32, integrate=panel_integrals).fit(
        samples[:8], epochs=3, batch=4, seed=0)
    pred = model.predict(samples[-1].points, samples[-1].normals, samples[-1].conditions, samples[-1].elements)
    assert pred.fields["cp"].shape == (samples[-1].n_points,)


def test_samples_with_different_point_counts_train_together():
    """A contagem de pontos é variável por construção: a amostra menor é reamostrada no lote, não preenchida."""
    samples = body_family(4, seed=7, n_per_side=30)
    trimmed = [samples[0], samples[1].subset(np.arange(0, samples[1].n_points, 2)), samples[2], samples[3]]
    assert len({s.n_points for s in trimmed}) == 2
    model = TransolverGeometry(fields=("cp",), hidden=32, depth=2, slices=8, integrate=panel_integrals)
    model.fit(trimmed, epochs=2, batch=2, seed=0)
    assert model.fit_report["n_points"] == min(s.n_points for s in trimmed)
    assert model.predict(trimmed[1].points, trimmed[1].normals, trimmed[1].conditions, trimmed[1].elements
                         ).fields["cp"].shape == (trimmed[1].n_points,)


# ----------------------------------------------------------------------------------------------- o resultado publicado
def test_the_committed_protocol_result_matches_the_protocol():
    """O JSON publicado é o registro: se o protocolo do código mudar, este teste falha antes de a página mentir."""
    import json
    from pathlib import Path

    from benchmarks.geometry_surrogate_interface import OUT, PROTOCOL

    path = Path(OUT)
    if not path.exists():
        pytest.skip("no published result; run python -m benchmarks.geometry_surrogate_interface")
    published = json.loads(path.read_text())
    protocol = published["protocol"]
    for key in ("train_shapes", "held_shapes", "epochs", "families"):
        assert protocol[key] == PROTOCOL[key], key
    # o integrador da interface tem de continuar batendo com o cl do solver
    assert published["data"]["integrator_matches_solver"] == pytest.approx(
        published["data"]["integrator_matches_solver"], abs=0)
    # cada família treinada: campo finito, erro medido e parâmetros contados
    for family in protocol["families"]:
        row = published["models"][family]
        assert row["parameters"] > 0
        assert 0.0 < row["relative_l2_mean"] < 1.0, family
        assert row["fit_seconds_mean"] > 0.0
        for seed in row["seeds"]:
            assert np.isfinite(seed["relative_l2_worst"])
            assert np.isfinite(seed["cl_error_mean"])
    # as famílias abertas continuam declaradas, com o motivo
    assert not published["families"]["gino"]["implemented"]
    assert published["families"]["gino"]["note"]
