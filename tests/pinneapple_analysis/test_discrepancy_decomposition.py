"""Decomposição da discrepância: as quatro parcelas recuperadas num caso sintético conhecido (#212).

O aceite do #212 é "recupera componentes injetados conhecidos num caso sintético, com os intervalos declarados". O caso
sintético aqui não é uma parábola escolhida para dar certo: o modelo de física é um **solver de verdade**, diferenças
centrais para ``-u'' = pi² sin(pi x)`` com ``u(0) = u(1) = 0``, cuja solução discreta é exata e conhecida:

    u_h(x) = sin(pi x) * (pi² / lambda_h),   lambda_h = 2 (1 - cos(pi h)) / h²

ou seja o erro de discretização de cada ponto é ``sin(pi x) (pi²/lambda_h - 1)``, que vale
``sin(pi x) pi² h²/12`` no limite assintótico — um número escrito à mão, não medido pelo mesmo código que o estima. A
ordem observada tem de sair 2.

Sobre ele se injetam as outras duas parcelas, e cada uma é comparada com o valor que se injetou: erro de modelo
``delta(x) = 0.05 sin(2 pi x)`` (a física resolve uma equação que não tem esse termo) e ruído gaussiano com
``sigma = 0.02`` em ``k = 6`` réplicas por ponto.

Os três números que o teste cobra: as duas **identidades** fecham no erro de máquina (elas são contas, não ajustes), e
cada intervalo **declarado** contém o valor que se injetou. A calibração dos intervalos é medida, não afirmada: 300
sementes, e a taxa de cobertura tem de ficar perto dos 95% que os intervalos prometem.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.trust.trust_report import Check, TrustReport
from pinneapple_analysis.verification.discrepancy_decomposition import (
    COMPONENTS,
    DiscrepancyDecomposition,
    decompose_discrepancy,
)

# ------------------------------------------------------------------------------------------- o caso sintético
X = np.linspace(0.1, 0.9, 9)
H = (0.2, 0.1, 0.05)
TRUTH = np.sin(np.pi * X)
MODEL_FORM_ERROR = 0.05 * np.sin(2.0 * np.pi * X)
SIGMA = 0.02
REPLICATES = 6


def fd_solution(x, h):
    """Solução discreta exata de ``-u'' = pi² sin(pi x)`` por diferenças centrais (autofunção do operador discreto)."""
    lam = (2.0 / h ** 2) * (1.0 - np.cos(np.pi * h))
    return np.sin(np.pi * x) * (np.pi ** 2 / lam)


def observations(seed: int, *, noise: float = SIGMA, model_error: np.ndarray | None = None,
                 replicates: int = REPLICATES) -> np.ndarray:
    model_error = MODEL_FORM_ERROR if model_error is None else model_error
    rng = np.random.default_rng(seed)
    return (TRUTH + model_error)[:, None] + noise * rng.standard_normal((X.size, replicates))


def decomposed(seed: int = 11) -> DiscrepancyDecomposition:
    return decompose_discrepancy(fd_solution, X, observations(seed), H)


# ------------------------------------------------------------------------------------------- identidades
def test_the_variance_identity_closes_in_the_round_off_of_the_machine():
    """MSE(modelo fino, réplicas) = viés² + ruído² — a decomposição da ANOVA, sem tolerância escolhida a dedo."""
    d = decomposed()
    assert d.closure < 1e-12
    obs = observations(11)
    fine = fd_solution(X, H[-1])
    mse = float(np.mean((obs - fine[:, None]) ** 2))
    assert d.total_mse == pytest.approx(mse, rel=0.0, abs=0.0)


def test_the_bias_identity_is_exact_point_by_point():
    """viés = numérico + modelo, ponto a ponto: os dois termos saem do mesmo ``u_extrapolado``, então se cancelam."""
    d = decomposed()
    assert d.bias_closure < 1e-12
    for p in d.points:
        assert p.bias == pytest.approx(p.numerical + p.model_form, abs=1e-15)
        assert p.bias == pytest.approx(p.observed_mean - p.finest_model, abs=1e-15)


# ------------------------------------------------------------------------------------------- recuperação
def test_the_injected_components_are_recovered_within_the_stated_intervals():
    """As três parcelas de verdade, cada uma contra o número que se injetou, com o intervalo declarado."""
    d = decomposed()
    injected = {
        "numerical": float(np.sqrt(np.mean((TRUTH - fd_solution(X, H[-1])) ** 2))),
        "model_form": float(np.sqrt(np.mean(MODEL_FORM_ERROR ** 2))),
        "noise": SIGMA,
    }
    assert d.covers(injected) == {"numerical": True, "model_form": True, "noise": True}
    # e o ponto estimado não é só "está dentro do intervalo": ele está perto do injetado
    assert d.component("numerical").value == pytest.approx(injected["numerical"], rel=0.10)
    assert d.component("model_form").value == pytest.approx(injected["model_form"], rel=0.20)
    assert d.component("noise").value == pytest.approx(injected["noise"], rel=0.25)


def test_the_numerical_component_is_the_discretization_error_of_the_finest_grid():
    """A parcela numérica, ponto a ponto, é ``u_extrapolado - u_fino`` — e o erro exato da malha fina é conhecido."""
    d = decomposed()
    u_fine = fd_solution(X, H[-1])
    exact_error = TRUTH - u_fine                                   # o que falta para a malha fina chegar na verdade
    estimated = np.array([p.numerical for p in d.points])
    # o mesmo sinal e o mesmo tamanho: a estimativa reproduz o erro de discretização, não o ruído
    assert np.all(np.sign(estimated) == np.sign(exact_error))
    assert np.sqrt(np.mean(estimated ** 2)) == pytest.approx(np.sqrt(np.mean(exact_error ** 2)), rel=0.10)
    # e a ordem observada do esquema é a ordem do esquema
    for p in d.points:
        assert p.observed_order == pytest.approx(2.0, abs=0.05)
    assert d.fraction_asymptotic == 1.0


def test_the_model_component_is_the_injected_bias_not_the_noise():
    """Sem erro de modelo, a parcela de modelo cai para o tamanho da incerteza da média — e não herda o ruído."""
    d = decompose_discrepancy(fd_solution, X, observations(3, model_error=np.zeros_like(X)), H)
    # o erro de modelo injetado é zero; a parcela estimada tem de caber dentro do próprio intervalo
    assert d.component("model_form").covers(0.0)
    assert d.component("model_form").value < 0.01          # ruído/sqrt(k) vezes t é da ordem de 0.015
    # e o viés que sobra é o numérico: o solver é o mesmo, então ele continua lá
    assert d.component("numerical").value == pytest.approx(
        float(np.sqrt(np.mean((TRUTH - fd_solution(X, H[-1])) ** 2))), rel=0.10)


def test_intervals_are_calibrated_the_coverage_over_many_seeds_is_the_promised_one():
    """A taxa de cobertura medida com 300 sementes: o que os intervalos de 95% prometem, eles entregam."""
    injected = {
        "numerical": float(np.sqrt(np.mean((TRUTH - fd_solution(X, H[-1])) ** 2))),
        "model_form": float(np.sqrt(np.mean(MODEL_FORM_ERROR ** 2))),
        "noise": SIGMA,
    }
    hits = dict.fromkeys(injected, 0)
    seeds = range(300)
    for seed in seeds:
        covers = decomposed(seed).covers(injected)
        for name, hit in covers.items():
            hits[name] += int(hit)
    for name in injected:
        rate = hits[name] / len(seeds)
        # 300 sementes para um intervalo nominal de 95%: duas caudas, folga de 3 pontos percentuais
        assert 0.90 <= rate <= 1.0, f"{name}: cobertura medida {rate:.1%} contra os 95% declarados"


# ------------------------------------------------------------------------------------------- refusals
def test_it_refuses_a_single_replicate_because_the_noise_would_be_a_guess():
    with pytest.raises(ValueError, match="at least 2 replicates"):
        decompose_discrepancy(fd_solution, X, observations(1, replicates=1), H)


def test_it_refuses_two_refinements_because_richardson_needs_three():
    with pytest.raises(ValueError, match="three refinements"):
        decompose_discrepancy(fd_solution, X, observations(1), (0.1, 0.05))


def test_it_refuses_refinements_out_of_order():
    with pytest.raises(ValueError, match="coarse to fine"):
        decompose_discrepancy(fd_solution, X, observations(1), (0.05, 0.1, 0.2))


def test_it_refuses_a_model_that_does_not_give_one_value_per_point():
    with pytest.raises(ValueError, match="one quantity of interest per point"):
        decompose_discrepancy(lambda x, h: np.sin(x)[:5], X, observations(1), H)


# ------------------------------------------------------------------------------------------- reaproveitamento
def test_it_reuses_solves_paid_elsewhere():
    """A sequência de resultados já calculados dá o mesmo número que re-rodar o solver."""
    precomputed = [fd_solution(X, h) for h in H]
    a = decompose_discrepancy(precomputed, X, observations(11), H)
    b = decomposed()
    assert a.components["numerical"].value == pytest.approx(b.component("numerical").value, rel=0.0, abs=0.0)
    assert a.closure == pytest.approx(b.closure, rel=1e-12)


def test_the_checks_enter_the_trust_report():
    """As duas checagens que são de fato checagens entram no relatório de confiança e o aprovam."""
    checks = decomposed().checks()
    assert [c.name for c in checks] == ["discrepancy.closure", "discrepancy.asymptotic"]
    assert all(isinstance(c, Check) and c.status == "supports" for c in checks)
    report = TrustReport(checks=checks, model="fd-1d")  # o relatório decide com o que recebeu
    assert report.decisao == "APPROVED"
    assert report.cobertura == 1.0


def test_a_broken_identity_is_reported_as_a_contradiction():
    """Se a conta da variância não fecha, a checagem reprova em vez de passar em silêncio."""
    d = decomposed()
    d.closure = 1e-3
    status = {c.name: c.status for c in d.checks()}
    assert status["discrepancy.closure"] == "contradicts"


def test_the_report_prints_the_three_numbers_and_the_identities():
    text = decomposed().report()
    for label in ("ruído de medida", "erro de modelo", "erro numérico", "identidade da variância",
                  "identidade do viés"):
        assert label in text
    payload = decomposed().to_dict()
    assert set(COMPONENTS).issubset(payload["components"])
    assert payload["components"]["noise"]["low"] <= payload["components"]["noise"]["value"] <= \
        payload["components"]["noise"]["high"]
