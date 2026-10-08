"""Relatório de confiança: decisão APPROVED/REVIEW/REJECT e conversão do TrustScore."""
import pytest

from pinneapple_analysis.trust.trust_gate import TrustScore
from pinneapple_analysis.trust.trust_report import (
    APPROVED, CONTRADICTS, NOT_RUN, REJECT, REVIEW, SUPPORTS,
    Check, TrustReport, checagem_booleana, checagem_erro_relativo, from_trust_score,
)


def _tudo_ok():
    return [
        checagem_erro_relativo("acuracia_referencia", 0.01, 0.02, critica=True),
        checagem_booleana("conservacao_massa", True, critica=True),
        checagem_booleana("fora_da_distribuicao", True),
        checagem_erro_relativo("calibracao_incerteza", 0.05, 0.10),
        checagem_erro_relativo("dado_experimental", 0.03, 0.05),
    ]


def test_tudo_passou_aprova():
    r = TrustReport(_tudo_ok(), model="m")
    assert r.decisao == APPROVED
    assert r.score == pytest.approx(1.0) and r.cobertura == 1.0 and r.motivos == []


def test_falha_critica_rejeita():
    checks = _tudo_ok()
    checks[0] = checagem_erro_relativo("acuracia_referencia", 0.30, 0.02, critica=True)
    r = TrustReport(checks)
    assert r.decisao == REJECT
    assert any("acuracia_referencia" in m and "crítica" in m for m in r.motivos)


def test_falha_nao_critica_pede_revisao():
    checks = _tudo_ok()
    checks[3] = checagem_erro_relativo("calibracao_incerteza", 0.5, 0.10)
    assert TrustReport(checks).decisao == REVIEW


def test_pouca_cobertura_pede_revisao_mesmo_sem_falha():
    checks = _tudo_ok()[:2] + [checagem_booleana("x", None, motivo_nao_executada="sem dado") for _ in range(3)]
    r = TrustReport(checks, min_coverage=0.6)
    assert r.cobertura == pytest.approx(2 / 5)
    assert r.decisao == REVIEW
    assert any("cobertura" in m for m in r.motivos)


def test_not_run_nao_entra_no_score():
    checks = [checagem_booleana("a", True), checagem_booleana("b", None, motivo_nao_executada="x")]
    r = TrustReport(checks, min_coverage=0.5)
    assert r.score == pytest.approx(1.0)
    assert r.decisao == APPROVED


def test_score_ponderado():
    checks = [Check("a", SUPPORTS, weight=3.0), Check("b", CONTRADICTS, weight=1.0)]
    r = TrustReport(checks, min_score=0.5)
    assert r.score == pytest.approx(0.75)
    assert r.decisao == REVIEW  # há uma contradição não crítica


def test_status_invalido_falha():
    with pytest.raises(ValueError):
        Check("x", "talvez")


def test_from_trust_score_converte_subscores():
    ts = TrustScore(combined=0.8, ood_score=0.9, residual_score=0.2, ensemble_score=None)
    checks = {c.name: c for c in from_trust_score(ts)}
    assert checks["ood"].status == SUPPORTS and checks["ood"].critical
    assert checks["residuo_pde"].status == CONTRADICTS and checks["residuo_pde"].critical
    assert checks["ensemble_incerteza"].status == NOT_RUN and checks["ensemble_incerteza"].reason


def test_relatorio_a_partir_do_trust_score_rejeita_residuo_ruim():
    ts = TrustScore(combined=0.5, ood_score=0.9, residual_score=0.1)
    r = TrustReport(from_trust_score(ts))
    assert r.decisao == REJECT


def test_markdown_e_dict_trazem_decisao_e_checagens():
    r = TrustReport(_tudo_ok(), model="surrogate-x")
    md = r.to_markdown()
    assert "surrogate-x" in md and "APPROVED" in md and "acuracia_referencia" in md
    d = r.to_dict()
    assert d["decisao"] == APPROVED and len(d["checagens"]) == 5
