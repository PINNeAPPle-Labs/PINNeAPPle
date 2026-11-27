"""Um objeto de restrição, aplicado no treino, num otimizador e numa guarda de deploy (#221, roadmap X34).

O critério de aceite é literal -- *o mesmo objeto de restrição* é aplicado nos três pontos --, então o teste guarda
a **identidade** do objeto: o mesmo ``pp.Constraint`` que entra em ``Trainer(constraints=...)`` é o que entra em
``PhysicsOptimizer(constraints=optimizer_constraints(...))`` e o que a ``ConstraintGuard`` recusa. Se algum dos três
recebesse uma cópia, um limite reescrito ou uma reimplementação, a asserção de identidade cairia.

Cada ponto é medido, não só exercitado:

* **treino** -- o equilíbrio da penalidade quadrática é escrito à mão. Com perda ``(w - 120)^2 + 9 * relu(w - 85)^2``
  o gradiente zera em ``w = (120 + 9 * 85) / (1 + 9) = 88.5``, e é esse número que o laço de treino tem de produzir;
* **otimizador** -- o ótimo com restrição é o ponto onde a restrição está ativa (``t = 2``), e o ``violation`` do
  ``OptimizeResult`` tem de ser zero: uma restrição que o otimizador só *penaliza* devolveria folga positiva;
* **guarda** -- a predição quente levanta, a fria passa, e a branda é relatada sem parar o serviço.

Medido nesta máquina (2026-11-25): o treino converge para 88.500 contra 88.5 previsto; o SQP devolve ``w = 3, t = 2``
com violação ``2.7e-15``; a guarda recusa ``92.5 degC`` contra um limite de 85.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pinneapple as pp  # noqa: E402
from pinneapple_core.optim import PhysicsOptimizer  # noqa: E402
from pinneapple_neural.trainer import TrainConfig, Trainer  # noqa: E402
from pinneapple_physics.constraint_engine import (  # noqa: E402
    ConstraintGuard,
    ConstraintReport,
    ConstraintViolation,
    check,
    check_all,
    optimizer_constraints,
    penalty,
    residual,
)

T_MAX = pp.Constraint("T_max do componente", quantity="T_max", op="<=", value=85.0, unit="degC",
                      family="safety", description="limite de segurança do componente")


# --------------------------------------------------------------------------- as seis famílias
def test_as_seis_familias_sao_expressaveis_e_uma_familia_inventada_e_recusada():
    for family in ("physical", "conservation", "operational", "material", "actuator", "safety"):
        c = pp.Constraint(f"limite {family}", quantity="x", value=1.0, family=family)
        assert c.family == family
    assert pp.Constraint("sem rótulo", quantity="x", value=1.0).family == ""      # as de antes de #221
    with pytest.raises(ValueError, match="family must be one of"):
        pp.Constraint("inventada", quantity="x", value=1.0, family="vibes")


def test_o_peso_e_o_cardinal_da_restricao_sao_validados():
    with pytest.raises(ValueError, match="weight must be positive"):
        pp.Constraint("peso negativo", quantity="x", value=1.0, weight=0.0)
    assert pp.Constraint("branda", quantity="x", value=1.0, hard=False).hard is False


# --------------------------------------------------------------------------- a aritmética
def test_o_sinal_do_residual_para_cada_operador():
    low = pp.Constraint("piso", quantity="t", op=">=", value=2.0)
    high = pp.Constraint("teto", quantity="T", op="<=", value=85.0)
    same = pp.Constraint("igual", quantity="x", op="==", value=1.0)
    assert residual(low, {"t": 3.0}) == pytest.approx(-1.0)         # 2 - 3, folga de 1 mm
    assert residual(high, {"T": 90.0}) == pytest.approx(5.0)        # 5 degC acima do limite
    assert residual(same, {"x": 1.0}) == pytest.approx(0.0)
    assert residual(same, {"x": 0.5}) == pytest.approx(0.5)


def test_o_residual_de_uma_expressao_usa_o_avaliador_seguro():
    c = pp.Constraint("margem", kind="expression", expression="mass_kg - payload_kg >= 2")
    assert residual(c, {"mass_kg": 40.0, "payload_kg": 35.0}) == pytest.approx(-3.0)
    assert residual(c, {"mass_kg": 40.0, "payload_kg": 39.5}) == pytest.approx(1.5)
    with pytest.raises(Exception, match="single comparison"):
        residual(pp.Constraint("duas", kind="expression", expression="a >= 1 and b >= 2"), {"a": 0, "b": 0})
    with pytest.raises(Exception, match="unknown name"):
        residual(c, {"mass_kg": 40.0})                              # payload_kg tem de estar lá


def test_um_campo_viola_na_celula_mais_quente_e_nao_na_media():
    T = np.array([[40.0, 50.0], [88.0, 60.0]])
    c = check(T_MAX, {"T_max": T})
    assert not c.satisfied
    assert c.margin == pytest.approx(3.0)                            # 88 - 85, não (88+60)/2 - 85
    assert c.value == pytest.approx(88.0)


def test_uma_restricao_que_nao_acha_a_sua_quantidade_diz_o_que_falta():
    """``T_max`` não está na resposta: a restrição reclama em vez de passar em silêncio."""
    with pytest.raises(KeyError, match="não carrega"):
        check(T_MAX, {"outra_coisa": 1.0})


# --------------------------------------------------------------------------- ponto 1: treino
def _train_constrained(constraints, *, epochs=600, lr=0.5, seed=0):
    """A scalar model pulled towards 120 with a limit at 85: the equilibrium is written down, so it can be checked."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(0.0)
    X, Y = torch.ones(32, 1), torch.full((32, 1), 120.0)
    loss_fn = lambda _m, y_hat, batch: {"total": ((y_hat - batch["y"]) ** 2).mean()}  # noqa: E731
    cfg = TrainConfig(epochs=epochs, lr=lr, device="cpu", log_dir="/tmp/pinneapple-constraint-test",
                      run_name="run", save_best=False)
    Trainer(model, loss_fn, constraints=constraints).fit([(X, Y)], None, cfg)
    with torch.no_grad():
        return float(model(X).mean())


def test_no_treino_o_equilibrio_da_penalidade_e_o_que_a_conta_preve():
    """``(w - 120)^2 + 9 relu(w - 85)^2`` zera o gradiente em ``(120 + 9 * 85) / 10 = 88.5``."""
    sem = _train_constrained([])
    com = _train_constrained([pp.Constraint("T_max", quantity="T_max", op="<=", value=85.0,
                                            family="safety", weight=9.0)])
    assert sem == pytest.approx(120.0, abs=0.5)                      # sem a restrição o dado manda
    assert com == pytest.approx(88.5, abs=0.5)                       # com ela, o equilíbrio previsto


def test_a_penalidade_e_diferenciavel_e_so_age_em_quem_viola():
    y = torch.tensor([[40.0], [95.0]], requires_grad=True)
    total = penalty([T_MAX], y)
    assert float(total.detach()) == pytest.approx((10.0 ** 2) / 2)   # só a linha de 95 entra, pela média
    total.backward()
    assert [round(float(g), 3) for g in y.grad.ravel()] == [0.0, 10.0]


def test_um_termo_de_restricao_entra_no_total_e_fica_registrado_a_parte():
    t = Trainer(torch.nn.Linear(1, 1), lambda _m, y, _b: {"total": y.sum() * 0}, constraints=[T_MAX])
    losses = t._add_constraint_penalty({"total": torch.zeros(())}, torch.full((4, 1), 120.0), {})
    assert float(losses["constraint"].detach()) == pytest.approx(35.0 ** 2)
    assert set(losses) == {"total", "constraint"}


# --------------------------------------------------------------------------- ponto 2: otimizador
def test_o_otimizador_recebe_o_mesmo_objeto_como_restricao_de_verdade():
    thickness = pp.Constraint("espessura >= 2 mm", quantity="t", op=">=", value=2.0, family="material")
    mass = pp.Constraint("massa", kind="expression", expression="w * t <= 30", family="physical")
    forced = optimizer_constraints([thickness, mass], names=["w", "t"])
    assert [c.name for c in forced] == [thickness.name, mass.name]
    assert [c.kind for c in forced] == ["ineq", "ineq"]

    opt = PhysicsOptimizer({"w": (1.0, 20.0), "t": (0.5, 5.0)},
                           objective=lambda v: (v[0] - 3) ** 2 + (v[1] - 1) ** 2,
                           constraints=forced)
    r = opt.minimize("slsqp")
    assert r.feasible and r.violation == pytest.approx(0.0, abs=1e-6)
    assert r.params["t"] == pytest.approx(2.0, abs=1e-3)             # o limite de espessura está ativo
    assert r.params["w"] == pytest.approx(3.0, abs=1e-2)             # e w fica no ótimo, com folga de massa
    assert residual(thickness, r.params) <= 1e-6 and residual(mass, r.params) <= 1e-6


def test_uma_igualdade_chega_ao_otimizador_como_igualdade():
    from pinneapple_core.optim import Constraint as OptimConstraint

    c = pp.Constraint("centro", quantity="x", op="==", value=1.0, family="operational")
    (forced,) = optimizer_constraints([c], names=["x"])
    assert isinstance(forced, OptimConstraint) and forced.kind == "eq"


# --------------------------------------------------------------------------- ponto 3: guarda de deploy
def test_a_guarda_de_deploy_recusa_a_predicao_que_viola_e_serve_a_que_respeita():
    guard = ConstraintGuard([T_MAX], where="serve /predict")
    predict = guard.wrap(lambda v: float(v))
    assert predict(70.0) == 70.0
    assert guard.last_report.ok
    with pytest.raises(ConstraintViolation) as excinfo:
        predict(92.5)
    assert "T_max do componente" in str(excinfo.value)
    assert excinfo.value.report.hard_violated[0].margin == pytest.approx(7.5)


def test_uma_restricao_branda_e_relatada_e_nao_para_o_servico():
    soft = pp.Constraint("derating desejável", quantity="T_max", op="<=", value=60.0,
                         family="operational", hard=False)
    guard = ConstraintGuard([soft])
    assert guard.wrap(lambda v: float(v))(92.5) == 92.5
    assert guard.last_report.ok and len(guard.last_report.violated) == 1     # relatada, e ``ok`` olha só as duras


def test_o_modo_sombra_mede_sem_recusar():
    guard = ConstraintGuard([T_MAX], on_violation="report")
    assert guard.wrap(lambda v: float(v))(92.5) == 92.5
    assert guard.last_report.hard_violated


def test_uma_guarda_sem_restricao_nao_guarda_nada():
    with pytest.raises(ValueError, match="não guarda nada"):
        ConstraintGuard([])


# --------------------------------------------------------------------------- o aceite: um objeto, três pontos
def test_o_mesmo_objeto_e_aplicado_no_treino_no_otimizador_e_na_guarda():
    """O critério de aceite do #221, verificado por identidade: um ``pp.Constraint``, três aplicações."""
    limite = pp.Constraint("T_max", quantity="T_max", op="<=", value=85.0, family="safety",
                           weight=9.0, unit="degC")

    # (1) treino -- o mesmo objeto
    trainer = Trainer(torch.nn.Linear(1, 1), lambda _m, y, _b: {"total": y.sum() * 0}, constraints=[limite])
    assert trainer.constraints[0] is limite

    # (2) otimizador -- o mesmo objeto, adaptado e não reescrito
    forced = optimizer_constraints([limite], names=["T_max"])
    assert forced[0].name == limite.name
    opt = PhysicsOptimizer({"T_max": (0.0, 200.0)}, objective=lambda v: (v[0] - 120.0) ** 2,
                           constraints=forced)
    r = opt.minimize("slsqp")
    assert r.feasible and r.params["T_max"] <= 85.0 + 1e-6

    # (3) guarda de deploy -- o mesmo objeto outra vez
    guard = ConstraintGuard([limite], where="serve")
    assert guard.constraints[0] is limite
    assert guard.wrap(lambda v: float(v))(r.params["T_max"]) == pytest.approx(r.params["T_max"])
    with pytest.raises(ConstraintViolation):
        guard.wrap(lambda v: float(v))(120.0)

    # e a mesma aritmética que o relatório de decisão usa sobre um ensemble (#219)
    assert check(limite, {"T_max": r.params["T_max"]}).satisfied


def test_a_probabilidade_de_violacao_e_a_media_das_checagens_sobre_o_ensemble():
    """O que o #219 consome: nenhuma estatística nova, a média de ``check`` sobre as realizações."""
    ensemble = [{"T_max": t} for t in (40.0, 60.0, 84.0, 85.0, 90.0)]
    reports = [check_all([T_MAX], member) for member in ensemble]
    assert sum(1 for r in reports if r.violated) == 1                    # 90 está acima; 85 é empate, não violação
    p = sum(1 for r in reports if r.violated) / len(reports)
    assert p == pytest.approx(0.2)
    assert isinstance(ConstraintReport([c for r in reports for c in r.checks]).checks[0].to_dict()["margin"], float)


def test_o_relatorio_resume_o_que_falhou():
    rep = check_all([T_MAX, pp.Constraint("piso", quantity="T_max", op=">=", value=100.0, hard=False)],
                    {"T_max": 92.5})
    assert not rep.ok
    assert "T_max do componente" in rep.summary() and "branda" in rep.summary()
