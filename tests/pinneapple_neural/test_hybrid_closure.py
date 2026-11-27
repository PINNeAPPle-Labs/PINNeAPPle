"""O fechamento neural reduz o erro de um solver grosso num caso que ele não viu (#211).

O aceite do #211 é literal: "um modelo de fechamento reduz o erro de um solver grosso num caso reservado e é
validado pela camada de confiança". O caso é convecção-difusão 1-D em regime permanente, que tem solução exata —

    u(x) = (exp(Pe x) - 1) / (exp(Pe) - 1),   Pe = c L / nu

— e um solver grosso de diferenças finitas com **upwind** no termo advectivo, cujo erro é a difusão numérica do
esquema: ele decai com a malha (medido aqui também) e cresce com ``Pe``. O fechamento aprende o resíduo
``u_exato - u_grosso`` como função de ``(x, log10 Pe)`` e é treinado em sete números de Péclet; o caso reservado é
``Pe = 35``, que **não está** no treino.

O que o teste cobra é o fator de redução medido no reservado, e o relatório de confiança com as duas checagens dele —
inclusive quando a checagem reprova (um limite que o fechamento não alcança tem de aparecer como *contradicts*, senão
o relatório não estaria medindo nada).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pinneapple_analysis.trust.trust_report import TrustReport
from pinneapple_neural.hybrid_closure import ClosureBlock, HybridModel, as_callable
from pinneapple_neural.physics_model import PhysicsModel

# ------------------------------------------------------------------------------------------ o caso físico
N_CELLS = 10
NODES = np.arange(1, N_CELLS) / N_CELLS          # nós interiores da malha grossa
PECLETS = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 80.0])
HELD_OUT = 35.0


def exact(x: np.ndarray, pe: float) -> np.ndarray:
    return (np.exp(pe * x) - 1.0) / (np.exp(pe) - 1.0)


def upwind(n: int, pe: float) -> np.ndarray:
    """``c u' = nu u''`` com ``c = 1`` e ``nu = 1/Pe``: upwind no advectivo, centrada na difusão."""
    h = 1.0 / n
    a = pe * h
    m = n - 1
    matrix = np.zeros((m, m))
    rhs = np.zeros(m)
    for k in range(m):
        matrix[k, k] = 2.0 + a
        if k - 1 >= 0:
            matrix[k, k - 1] = -1.0 - a
        if k + 1 < m:
            matrix[k, k + 1] = -1.0
        else:
            rhs[k] += 1.0                          # u(1) = 1
    return np.linalg.solve(matrix, rhs)


def features(pe: float | np.ndarray) -> np.ndarray:
    """``(x, log10 Pe)``: o logaritmo porque é nele que o erro do esquema varia de forma suave."""
    return np.stack([NODES, np.full_like(NODES, np.log10(pe))], axis=1)


_TABLE = {pe: upwind(N_CELLS, pe) for pe in PECLETS}
_PES = np.array(sorted(_TABLE))
_LOG_PES = np.log10(_PES)


def coarse_solver(x: torch.Tensor) -> torch.Tensor:
    """O solver grosso como um ``callable(x) -> y``: recebe ``(x, log10 Pe)`` e devolve o valor naquele nó."""
    values = x.detach().cpu().numpy()
    out = np.empty((values.shape[0], 1))
    for row, (position, log_pe) in enumerate(values):
        node = int(np.clip(np.round(position * N_CELLS - 1.0), 0, N_CELLS - 2))
        upper = int(np.clip(np.searchsorted(_LOG_PES, log_pe), 1, len(_PES) - 1))
        lo, hi = _PES[upper - 1], _PES[upper]
        weight = (log_pe - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        out[row, 0] = (1.0 - weight) * _TABLE[lo][node] + weight * _TABLE[hi][node]
    return torch.tensor(out, dtype=torch.float32)


def training_set() -> tuple[torch.Tensor, torch.Tensor]:
    x = np.concatenate([features(pe) for pe in PECLETS])
    y = np.concatenate([exact(NODES, pe)[:, None] for pe in PECLETS])
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def held_out(pe: float = HELD_OUT) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.tensor(features(pe), dtype=torch.float32),
            torch.tensor(exact(NODES, pe)[:, None], dtype=torch.float32))


@pytest.fixture(scope="module")
def fitted() -> HybridModel:
    torch.manual_seed(0)
    x, y = training_set()
    hybrid = HybridModel(physics=coarse_solver, correction=ClosureBlock(2, 1, hidden=32, layers=2),
                         name="advection-diffusion")
    hybrid.fit({"x": x, "y": y}, epochs=2000, lr=5e-3, seed=0)
    return hybrid


# ------------------------------------------------------------------------------------------ o solver é grosso
def test_the_coarse_solver_converges_so_its_error_is_discretization_not_a_bug():
    """A malha fina tem de tender à solução exata: sem isso, o "erro do solver grosso" seria um bug do solver."""
    errors = []
    for n in (N_CELLS, 2 * N_CELLS, 4 * N_CELLS):
        nodes = np.arange(1, n) / n
        errors.append(float(np.sqrt(np.mean((exact(nodes, 20.0) - upwind(n, 20.0)) ** 2))))
    assert errors[0] > errors[1] > errors[2]       # converge com a malha
    assert errors[0] / errors[2] > 1.8             # a primeira ordem esperada do upwind


# ------------------------------------------------------------------------------------------ o aceite
def test_the_closure_reduces_the_error_of_the_coarse_solver_on_a_held_out_case(fitted):
    """O erro no caso reservado (Pe = 35, fora do treino), antes e depois do fechamento."""
    x, y = held_out()
    with torch.no_grad():
        base = fitted.base(x)
        hybrid = fitted.predict(x).mean
    rmse_base = float(torch.sqrt(torch.mean((base - y) ** 2)))
    rmse_hybrid = float(torch.sqrt(torch.mean((hybrid - y) ** 2)))
    assert rmse_base == pytest.approx(0.0702, abs=1e-3)          # o solver grosso, medido
    assert rmse_hybrid < rmse_base * 0.6                          # e o híbrido, no mesmo caso
    assert float(torch.mean((hybrid - y) ** 2)) < float(torch.mean((base - y) ** 2))


def test_the_trust_layer_is_what_says_whether_the_closure_helped(fitted):
    """O relatório de confiança do conjunto reservado: duas checagens, os números medidos, e uma decisão."""
    x, y = held_out()
    report = fitted.score(x, y)
    assert isinstance(report, TrustReport)
    assert [c.name for c in report.checks] == ["closure.reduces_error", "closure.generalizes"]
    assert all(c.status == "supports" for c in report.checks)
    assert report.decisao == "APPROVED"
    assert report.score == 1.0 and report.cobertura == 1.0
    assert report.details["reduction"] >= 2.0                     # o fator medido, não uma promessa
    assert "RMSE" in report.checks[0].reason


def test_a_report_that_would_be_rigged_reproves_instead(fitted):
    """Pedindo uma redução que o fechamento não entrega, a checagem crítica reprova e a decisão é REJECT."""
    x, y = held_out()
    report = fitted.score(x, y, min_reduction=100.0)
    assert report.decisao == "REJECT"
    assert report.checks[0].status == "contradicts" and report.checks[0].critical


def test_the_prediction_is_the_base_plus_the_closure_and_says_so(fitted):
    """``predict`` devolve a conta aberta: a base e o termo de fechamento ficam em ``extras``."""
    x, _ = held_out()
    prediction = fitted.predict(x)
    assert torch.allclose(prediction.mean, prediction.extras["base"] + prediction.extras["closure"])
    assert prediction.extras["fitted"] is True
    # a base do híbrido é o solver, e ela é diferente da previsão (o fechamento não é nulo)
    assert not torch.allclose(prediction.extras["base"], prediction.mean)


# ------------------------------------------------------------------------------------------ a interface
def test_the_hybrid_is_a_case_of_the_physics_model_interface(fitted):
    """A interface do #155: ``family == "hybrid"``, e as quatro chamadas vêm do mesmo contrato."""
    assert isinstance(fitted, PhysicsModel)
    assert fitted.family == "hybrid"
    description = fitted.describe()
    assert description["family"] == "hybrid" and description["has_correction"] is True
    assert description["capabilities"]["fit"] == "yes"
    assert "state_dict" in description["capabilities"]["export"]


def test_a_physics_model_can_be_the_base_without_being_wrapped_by_hand():
    """Um ``PhysicsModel`` (o PINN do #155) entra como base pelo mesmo ``callable`` que um solver entra."""
    import pinneapple as pp
    from pinneapple_neural.physics_model import as_physics_model

    torch.manual_seed(0)
    pinn = as_physics_model(pp.build_model("vanilla_pinn", in_dim=1, out_dim=1))
    base = as_callable(pinn)
    out = base(torch.zeros(4, 1))
    assert out.shape == (4, 1) and torch.allclose(out, pinn.predict(torch.zeros(4, 1)).mean)


def test_the_capabilities_refuse_the_uncertainty_that_would_be_fake():
    """Um fechamento linear não tem camada escondida para o dropout amostrar — e o ``capabilities`` diz isso antes."""
    torch.manual_seed(0)
    plain = HybridModel(physics=coarse_solver, correction=torch.nn.Linear(2, 1), name="linear-closure")
    assert plain.capabilities()["uncertainty"].startswith("no:")
    x, y = training_set()
    plain.fit({"x": x, "y": y}, epochs=20, seed=0)
    with pytest.raises(Exception, match="zero spread"):
        plain.uncertainty(held_out()[0])


def test_uncertainty_over_the_closure_is_a_real_spread():
    """O spread vem do próprio fechamento (o hook de Monte Carlo dropout sobre as camadas escondidas); a média
    acompanha a previsão, porque a base é determinística."""
    torch.manual_seed(0)
    hybrid = HybridModel(physics=coarse_solver, correction=ClosureBlock(2, 1, hidden=16, dropout=0.2),
                         name="dropout")
    x, y = training_set()
    hybrid.fit({"x": x, "y": y}, epochs=50, seed=0)
    prediction = hybrid.uncertainty(held_out()[0], n_samples=20, seed=0)
    assert prediction.method == "mc_dropout" and prediction.std is not None
    assert float(prediction.std.mean()) > 0.0


def test_the_ensemble_of_closures_needs_two_members_and_uses_them():
    torch.manual_seed(0)
    x, y = training_set()
    members = []
    for seed in (0, 1):
        hybrid = HybridModel(physics=coarse_solver, correction=ClosureBlock(2, 1, hidden=16), name=f"m{seed}")
        hybrid.fit({"x": x, "y": y}, epochs=30, seed=seed)
        members.append(hybrid)
    prediction = members[0].uncertainty(held_out()[0], method="ensemble", members=members)
    assert prediction.method == "ensemble" and float(prediction.std.mean()) > 0.0
    with pytest.raises(Exception, match="at least two fitted hybrid members"):
        members[0].uncertainty(held_out()[0], method="ensemble", members=members[:1])


# ------------------------------------------------------------------------------------------ o que é recusado
def test_a_closure_that_exists_but_was_never_fitted_is_refused():
    """A saída de um MLP com pesos sorteados não é correção: somá-la à base seria pior que não ter fechamento."""
    hybrid = HybridModel(physics=coarse_solver, correction=ClosureBlock(2, 1, hidden=8))
    with pytest.raises(RuntimeError, match="never fitted"):
        hybrid.predict(held_out()[0])


def test_a_hybrid_without_a_correction_has_nothing_to_fit_and_says_so():
    plain = HybridModel(physics=coarse_solver)
    assert plain.capabilities()["fit"].startswith("no:")
    with pytest.raises(ValueError, match="no correction block"):
        plain.fit({"x": training_set()[0], "y": training_set()[1]})
    # mas prever é a base, e a base funciona
    prediction = plain.predict(held_out()[0])
    assert torch.allclose(prediction.mean, prediction.extras["base"])


def test_too_few_reference_points_are_refused():
    torch.manual_seed(0)
    hybrid = HybridModel(physics=coarse_solver, correction=ClosureBlock(2, 1, hidden=8))
    x, y = training_set()
    with pytest.raises(ValueError, match="at least 8 reference points"):
        hybrid.fit({"x": x[:5], "y": y[:5]})


def test_a_correction_that_is_only_a_callable_cannot_be_fitted():
    torch.manual_seed(0)
    hybrid = HybridModel(physics=coarse_solver, correction=lambda x: x[:, :1] * 0.0)
    x, y = training_set()
    with pytest.raises(TypeError, match="nn.Module"):
        hybrid.fit({"x": x, "y": y})


def test_a_torchscript_export_is_refused_because_the_base_is_python():
    """Só o fechamento é exportável; a base é um ``callable`` e nenhum ``torch.jit.trace`` a captura."""
    torch.manual_seed(0)
    hybrid = HybridModel(physics=coarse_solver, correction=ClosureBlock(2, 1, hidden=8))
    with pytest.raises(Exception, match="Python callable"):
        hybrid.export("/tmp/hybrid.pt", fmt="torchscript")


def test_the_closure_is_persisted_and_reloads_into_the_same_block(tmp_path):
    torch.manual_seed(0)
    hybrid = HybridModel(physics=coarse_solver, correction=ClosureBlock(2, 1, hidden=8))
    x, y = training_set()
    hybrid.fit({"x": x, "y": y}, epochs=20, seed=0)
    path = hybrid.export(tmp_path / "closure.pt")
    assert path.is_file()
    restored = ClosureBlock(2, 1, hidden=8)
    restored.load_state_dict(torch.load(path, weights_only=True)["correction"])
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(restored(held_out()[0]), hybrid.correction(held_out()[0]), atol=1e-6)


def test_the_closure_block_refuses_a_nonsense_shape():
    with pytest.raises(ValueError, match="in_dim and out_dim"):
        ClosureBlock(0, 1)
    with pytest.raises(ValueError, match="dropout is a probability"):
        ClosureBlock(2, 1, dropout=1.0)
    with pytest.raises(ValueError, match="layers"):
        ClosureBlock(2, 1, layers=0)
