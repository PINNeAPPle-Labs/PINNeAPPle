"""O relatório de decisão a partir de um ensemble: distribuição, risco por restrição, quem dirige o risco e a
recomendação com o domínio de validade (#219, roadmap S38).

O aceite da issue é uma frase concreta -- *para um problema de benchmark, o relatório dá a probabilidade de violação
da restrição e nomeia as combinações de entrada que a causam, com o status de confiança do S8* --, e é isso que os
testes medem, em dois níveis:

* **a aritmética do relatório**, sobre um caso com probabilidade escrita à mão: ``T = 4 P - 0.4 h`` com ``P`` e ``h``
  uniformes numa caixa dá uma probabilidade de violação que se integra no papel, e é ela que o ensemble de 20 000
  realizações tem de reproduzir. Assim uma fórmula errada (probabilidade trocada pela de *não* violar, quantil no
  lugar do máximo, correlação no lugar da margem) aparece como número diferente e não como texto plausível;
* **o caso real**, a placa do benchmark ``inverse_heat``: a quantidade de interesse é a temperatura máxima da placa
  resolvida por volumes finitos, as entradas incertas são a potência dissipada e o coeficiente de convecção -- que é
  exatamente o envelope que o caso declara (#58) --, e as restrições são o limite de segurança (85 degC) e o de
  derating (60 degC), escritos como ``pp.Constraint`` (#221) e checados pelo motor do #221. A malha grossa é o modelo
  do ensemble e a fina é a referência do S8, e a diferença entre as duas é o que o relatório põe na confiança.

Medido nesta máquina (2026-11-25), 200 realizações, semente 0: ``T_max`` média 73.9 degC, desvio 21.9, p05 45.6,
p95 111.9; ``P(T_max <= 85)`` = 0.115 e ``P(T_max <= 60)`` = 0.325. O primeiro driver é a potência (correlação
+0.90 com a folga), o segundo a convecção (-0.75), e a faixa recomendada é ``P <= 20`` com ``h`` no piso.
"""
from __future__ import annotations

import pytest

import pinneapple as pp
from pinneapple_analysis.benchmark.validity import ValidityDomain
from pinneapple_analysis.decision import DecisionReport, decision_report
from pinneapple_analysis.inverse_problems.heat_transfer import PLATE
from pinneapple_analysis.trust.trust_report import SUPPORTS, TrustReport
from pinneapple_simulation.numerical_solvers.heat_fv_reference import plate_2d

BOX = ValidityDomain("caixa", {"P": (2.0, 20.0), "h": (5.0, 100.0)}, source="declarada pelo gerador do teste")

T50 = pp.Constraint("T_max <= 50 degC", quantity="T_max", op="<=", value=50.0, unit="degC", family="safety")
T65 = pp.Constraint("T_max <= 65 degC", quantity="T_max", op="<=", value=65.0, unit="degC", family="operational")


# --------------------------------------------------------------------------- a aritmética, com a resposta no papel
def _analytic(n, rng):
    """``T = 4 P - 0.4 h`` com ``P ~ U[2, 20]`` e ``h ~ U[5, 100]``."""
    return {"P": rng.uniform(2.0, 20.0, n), "h": rng.uniform(5.0, 100.0, n)}


def _analytic_eval(inputs):
    return {"T_max": 4.0 * inputs["P"] - 0.4 * inputs["h"]}


def _exact_violation(limit: float) -> float:
    """``P(4P - 0.4h > limit)`` integrada no papel.

    A condição é ``h < 10P - 2.5 * limit``; com ``h`` uniforme em [5, 100] o integrando em ``P`` é
    ``clamp(10P - 2.5 limit, 5, 100) - 5`` e a integral fica ``[5P^2 - (2.5 limit + 5) P]`` entre o ponto onde a
    reta cruza ``h = 5`` e ``P = 20``, dividida por ``18 * 95``.
    """
    p_lo = (5.0 + 2.5 * limit) / 10.0                     # onde h = 5, isto é, a violação começa
    p_lo = max(2.0, p_lo)
    p_hi = min(20.0, (100.0 + 2.5 * limit) / 10.0)        # onde h = 100 e a violação acaba
    if p_hi <= p_lo:
        return 0.0

    def antiderivative(p: float) -> float:
        c = 2.5 * limit + 5.0
        return 5.0 * p * p - c * p

    return (antiderivative(p_hi) - antiderivative(p_lo)) / (18.0 * 95.0)


def test_a_probabilidade_de_violacao_e_a_do_ensemble_e_nao_uma_formula_a_parte():
    """``T = 4P - 0.4h`` viola 50 degC com probabilidade ``0.14327``, integrada no papel dentro do teste."""
    assert _exact_violation(50.0) == pytest.approx(0.14327, abs=1e-5)
    assert _exact_violation(65.0) == pytest.approx(0.03088, abs=1e-5)
    report = decision_report("analítico", "T_max", BOX, _analytic, _analytic_eval, [T50], n=20_000, seed=1,
                             trust_check=False)
    assert report.probabilities[T50.name] == pytest.approx(_exact_violation(50.0), abs=0.01)
    assert report.distribution["mean"] == pytest.approx(4.0 * 11.0 - 0.4 * 52.5, abs=0.3)
    assert report.distribution["min"] <= report.distribution["p05"] <= report.distribution["median"]
    assert report.distribution["median"] <= report.distribution["p95"] <= report.distribution["max"]
    assert report.drivers[0].worst_margin > 0


def test_o_relatorio_nomeia_as_entradas_que_dirigem_o_risco():
    """A potência empurra para a violação e a convecção puxa para fora dela: as duas têm de aparecer, com o lado."""
    report = decision_report("analítico", "T_max", BOX, _analytic, _analytic_eval, [T50], n=2_000, seed=2,
                             trust_check=False)
    by_var = {d.variable: d for d in report.inputs}
    assert set(by_var) == {"P", "h"}
    assert by_var["P"].correlation > 0.3 and by_var["P"].safe_side == "<"        # acima do limiar a folga acaba
    assert by_var["h"].correlation < -0.3 and by_var["h"].safe_side == ">"      # abaixo do limiar também
    assert by_var["P"].violation_rate_risky > by_var["P"].violation_rate_safe
    assert by_var["P"].violation_rate_safe == 0.0                               # o lado seguro não tem violação
    assert "P < " in report.recommendation["text"] and "h > " in report.recommendation["text"]
    # e a combinação que concentra o risco é a potência alta com a convecção baixa
    worst = report.combinations[0]
    assert worst.variables == {"P": ">", "h": "<="}
    assert worst.violation_rate > report.probabilities[T50.name]
    assert sum(c.n for c in report.combinations) == report.n


def test_os_piores_casos_sao_os_de_maior_violacao():
    report = decision_report("analítico", "T_max", BOX, _analytic, _analytic_eval, [T50], n=500, seed=3,
                             top_k=5, trust_check=False)
    margins = [m.worst_margin for m in report.drivers]
    assert margins == sorted(margins, reverse=True) and len(margins) == 5
    assert margins[0] > 0


def test_sem_violacao_o_relatorio_diz_que_o_envelope_inteiro_e_operavel():
    loose = pp.Constraint("T_max <= 1000 degC", quantity="T_max", op="<=", value=1000.0, family="safety")
    report = decision_report("analítico", "T_max", BOX, _analytic, _analytic_eval, [loose], n=500, seed=4,
                             trust_check=False)
    assert report.probabilities[loose.name] == 0.0
    assert report.violated == {}
    assert "envelope inteiro é operável" in report.recommendation["text"]
    assert report.inputs == []                     # sem violação não há quem separe


def test_uma_restricao_branda_aparece_separada_da_dura():
    soft = pp.Constraint("derating", quantity="T_max", op="<=", value=50.0, family="operational", hard=False)
    report = decision_report("analítico", "T_max", BOX, _analytic, _analytic_eval, [soft], n=500, seed=5,
                             trust_check=False)
    assert soft.name not in report.probabilities and soft.name in report.soft_probabilities
    assert report.probabilities == {} and report.violated == {}


def test_uma_realizacao_fora_do_dominio_e_contada_e_dita():
    narrow = ValidityDomain("estreito", {"P": (2.0, 10.0), "h": (5.0, 100.0)}, source="teste")
    report = decision_report("analítico", "T_max", narrow, _analytic, _analytic_eval, [T50], n=400, seed=6,
                             trust_check=False)
    assert any("fora do domínio de validade" in n for n in report.notes)


def test_o_relatorio_e_serializavel_e_imprimivel():
    report = decision_report("analítico", "T_max", BOX, _analytic, _analytic_eval, [T50], n=200, seed=7,
                             trust_check=False)
    d = report.to_dict()
    assert d["probabilities"][T50.name] == report.probabilities[T50.name]
    assert d["domain"]["name"] == "caixa" and "recommendation" in d and "member" not in d["recommendation"]
    md = report.to_markdown()
    assert "## Probabilidade de violar cada restrição" in md and "## O que dirige o risco" in md
    assert "**Validade:**" in md and "safety" in md


# --------------------------------------------------------------------------- o caso real: a placa do inverse_heat
_HEAT = dict(lx=PLATE["lx"], ly=PLATE["ly"], t=PLATE["t"], k=PLATE["k"], t_air=PLATE["t_air"], device=PLATE["device"])
PLATE_DOMAIN = ValidityDomain("placa inverse_heat", {"P": (2.0, 20.0), "h": (5.0, 100.0)},
                              source="a faixa em que o caso inverse_heat gera os seus pares de treino")

#: as duas restrições do caso: o limite de segurança do componente e o derating de operação
PLATE_SAFETY = pp.Constraint("T_max <= 85 degC", quantity="T_max", op="<=", value=85.0, unit="degC",
                             family="safety")
PLATE_OPERATING = pp.Constraint("T_max <= 60 degC", quantity="T_max", op="<=", value=60.0, unit="degC",
                                family="operational")


def _plate(n, rng):
    return {"P": rng.uniform(2.0, 20.0, n), "h": rng.uniform(5.0, 100.0, n)}


def _plate_field(inputs, nx, ny):
    """Uma realização do modelo de volumes finitos da placa; a temperatura máxima é a quantidade de interesse."""
    _x, _y, T = plate_2d(float(inputs["h"]), power=float(inputs["P"]), nx=nx, ny=ny, **_HEAT)
    return {"T_max": float(T.max())}


def test_o_caso_real_da_placa_da_a_probabilidade_de_violar_cada_restricao():
    """Malha 40x24 no ensemble, 200x120 na referência: 200 realizações sobre o envelope que o caso declara."""
    report = decision_report("placa inverse_heat", "T_max", PLATE_DOMAIN, _plate,
                             lambda inp: _plate_field(inp, 40, 24), [PLATE_SAFETY, PLATE_OPERATING],
                             reference=lambda inp: _plate_field(inp, 200, 120), n=200, seed=0)
    assert isinstance(report, DecisionReport) and isinstance(report.trust, TrustReport)

    p85, p60 = report.probabilities[PLATE_SAFETY.name], report.probabilities[PLATE_OPERATING.name]
    assert p85 == pytest.approx(0.180, abs=0.03)          # medido: 0.180
    assert p60 == pytest.approx(0.435, abs=0.03)          # medido: 0.435
    assert p60 > p85 > 0.0                                 # o derating é mais apertado que a segurança

    assert report.distribution["mean"] == pytest.approx(66.87, abs=2.0)   # medido: 66.87 (desvio 36.14)
    assert report.distribution["max"] > 200.0              # há realizações muito acima do limite de segurança
    assert report.distribution["p95"] > report.distribution["median"]

    # o caso que dirige o risco: potência alta com convecção baixa
    worst = report.drivers[0]
    assert worst.qoi["T_max"] == pytest.approx(report.distribution["max"], abs=1e-6)
    assert worst.inputs["P"] > 12.0 and worst.inputs["h"] < 25.0
    by_var = {d.variable: d for d in report.inputs}
    assert by_var["P"].correlation > 0.5 and by_var["P"].safe_side == "<"
    assert by_var["h"].correlation < -0.5 and by_var["h"].safe_side == ">"
    assert all(d.violation_rate_safe == 0.0 for d in report.inputs)     # o lado seguro não tem violação

    # a combinação que concentra o risco: carga alta com convecção baixa, medido célula a célula
    hottest = report.combinations[0]
    assert hottest.variables == {"P": ">", "h": "<="} and hottest.violation_rate > 0.9
    assert sum(c.n for c in report.combinations if c.constraint == PLATE_OPERATING.name) == report.n

    # a confiança é sobre a evidência: a malha grossa contra a fina
    acc = next(c for c in report.trust.checks if c.name == "accuracy against the reference")
    assert acc.status == SUPPORTS and acc.value < 0.01
    assert report.verdict == "não opera no envelope inteiro"
    assert "P(T_max <= 85 degC violada)" in report.summary()
    assert "**Validade:**" in report.to_markdown()


def test_sem_referencia_a_confianca_diz_que_a_acuracia_nao_foi_checada():
    """A ausência de referência é ``not_run`` crítico com o motivo, e a página diz isso -- não vira um passe."""
    report = decision_report("analítico", "T_max", BOX, _analytic, _analytic_eval, [T50], n=200, seed=8)
    acc = next(c for c in report.trust.checks if c.name == "accuracy against the reference")
    assert acc.status == "not_run" and acc.critical and "reference" in acc.reason
    assert "accuracy against the reference: not_run" in report.to_markdown()
    # a cobertura cai porque a checagem existe e não rodou: o S8 conta a ausência, mesmo quando não reprova
    assert report.trust.cobertura < 1.0


def test_uma_amostra_pequena_nao_resolve_probabilidade_pequena_e_o_relatorio_diz_isso():
    report = decision_report("analítico", "T_max", BOX, _analytic, _analytic_eval, [T50], n=20, seed=9,
                             min_members=100)
    sample = next(c for c in report.trust.checks if c.name == "sample size")
    assert sample.status != SUPPORTS and sample.value == 20.0
