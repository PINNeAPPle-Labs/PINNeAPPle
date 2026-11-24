"""Fechamento neural: ``pp.HybridModel(physics=..., surrogate=..., correction=...)`` (#211).

O que a física conhecida não explica tem nome: fechamento de turbulência, termo de submalha, correção
constitutiva, coeficiente de arrasto que a correlação erra. A técnica é sempre a mesma — a previsão é

    y(x)  ~=  base(x)  +  delta(x)

com ``base`` sendo o que se sabe (um solver grosso, uma ROM, um modelo de ordem reduzida) e ``delta`` um bloco
pequeno aprendido sobre o **resíduo** que sobra. Este módulo é esse arranjo, como um caso da interface comum que
entrou em ``main`` no #155 (:class:`pinneapple_neural.physics_model.PhysicsModel`): ``HybridModel.family == "hybrid"``
e ele responde ``fit``/``predict``/``uncertainty``/``export`` como qualquer PINN ou operador. Não é uma abstração
paralela — é o quinto caso dela, e por isso entra num ``benchmark`` e num otimizador que iteram sobre famílias sem
saber que ele existe.

As três peças, e o que cada uma pode ser
---------------------------------------
``physics``
    o modelo de referência: um :class:`PhysicsModel`, um ``nn.Module`` ou um simples ``callable(x) -> y`` (o solver
    grosso do caso). É a base quando não há ``surrogate``.
``surrogate``
    opcional: a aproximação barata da mesma física (a ROM do caminho CFD -> ROM -> surrogate). Quando existe, a base
    é **ela**, não o solver caro — é assim que o híbrido serve para rodar onde o solver não roda.
``correction``
    opcional: um :class:`ClosureBlock` (ou qualquer ``nn.Module``/``callable``) que aprende o resíduo. Sem ele o
    híbrido é a base, e ``fit`` diz que não há o que treinar em vez de fingir um treino.

A confiança é medida, não afirmada
----------------------------------
O aceite do #211 é que o fechamento **reduza o erro de um solver grosso num caso que ele não viu** e que isso seja
validado pela camada de confiança. :meth:`HybridModel.score` recebe um conjunto reservado e devolve um
:class:`~pinneapple_analysis.trust.TrustReport` com as duas checagens que importam: o erro do híbrido contra o da
base no reservado (com o fator de redução medido) e o erro ali contra o do conjunto que o próprio ajuste reservou —
um caso novo muito pior do que aquele reprova a segunda. O relatório decide com o que mediu; nada aqui aprova nada.

Limites declarados: o ``delta`` é aprendido sobre um conjunto finito, então ele reduz o erro **dentro do que os
dados cobrem** — a extrapolação em parâmetros vira erro de volta, e o relatório de confiança mede isso em vez de
esconder; e o híbrido não exporta um grafo torchscript, porque a base é um ``callable`` de Python (o que ele
exporta é o peso do fechamento, que é a parte treinada).

    >>> import torch
    >>> from pinneapple_neural.hybrid_closure import ClosureBlock, HybridModel
    >>> physics = lambda x: 2.0 * x            # a física conhecida (e incompleta)
    >>> correction = ClosureBlock(in_dim=1, out_dim=1, hidden=16)
    >>> hybrid = HybridModel(physics=physics, correction=correction)
    >>> x = torch.linspace(0, 1, 40)[:, None]
    >>> report = hybrid.fit({"x": x, "y": 2.0 * x + 1.0}, epochs=200)    # o resíduo a aprender é o "+1"
    >>> round(float(report.rmse_after), 3) <= round(float(report.rmse_before), 3)
    True
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pinneapple_neural.physics_model import (
    ExportNotAvailable,
    FitReport,
    PhysicsModel,
    Prediction,
    UncertaintyNotAvailable,
    _dropout_capacity,
    _primary_tensor,
)

__all__ = ["ClosureBlock", "ClosureFit", "HybridModel", "as_callable"]


def as_callable(model: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    """A base model as ``f(x) -> tensor``: a ``PhysicsModel``, an ``nn.Module`` or a plain callable.

    A ``PhysicsModel`` goes through its own ``predict`` (so a PINN, a ROM or a GNN is used exactly as its adapter
    says); a callable is called directly. The result is always a float tensor, which is what the closure sees.
    """
    if isinstance(model, PhysicsModel):
        def _via_interface(x: torch.Tensor) -> torch.Tensor:
            return _to_tensor(model.predict(x).mean)

        return _via_interface
    if isinstance(model, nn.Module):
        def _via_module(x: torch.Tensor) -> torch.Tensor:
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    out = model(x)
            finally:
                model.train(was_training)
            return _to_tensor(getattr(out, "y", out))

        return _via_module
    if callable(model):
        def _via_call(x: torch.Tensor) -> torch.Tensor:
            return _to_tensor(model(x))

        return _via_call
    raise TypeError(
        f"a hybrid needs a base that can be called with the inputs: got {type(model).__name__}, which is neither a "
        "PhysicsModel, an nn.Module, nor a callable"
    )


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


class ClosureBlock(nn.Module):
    """O bloco que aprende o que a física não explica: um MLP pequeno, com dropout opcional.

    ``in_dim`` são as variáveis de que o resíduo depende (as coordenadas, um número de Reynolds, uma fração de
    vórtice submalha) e ``out_dim`` a grandeza corrigida. O dropout existe por uma razão prática: sem ele a
    ``uncertainty(method="mc_dropout")`` do híbrido é honestamente recusada (o spread seria exatamente zero).
    """

    def __init__(self, in_dim: int, out_dim: int, *, hidden: int = 32, layers: int = 2, dropout: float = 0.0,
                 activation: type[nn.Module] = nn.Tanh):
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim and out_dim must be > 0; got {in_dim}, {out_dim}")
        if layers < 1:
            raise ValueError(f"layers must be >= 1; got {layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout is a probability; got {dropout}")
        blocks: list[nn.Module] = []
        width = int(in_dim)
        for _ in range(int(layers)):
            blocks.append(nn.Linear(width, int(hidden)))
            blocks.append(activation())
            if dropout:
                blocks.append(nn.Dropout(float(dropout)))
            width = int(hidden)
        blocks.append(nn.Linear(width, int(out_dim)))
        self.net = nn.Sequential(*blocks)
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ClosureFit(FitReport):
    """O que o ajuste do fechamento mediu, além do que qualquer família reporta.

    ``rmse_before``/``rmse_after`` são a raiz do erro quadrático médio da base e do híbrido **no conjunto reservado**
    que o ajuste separou — antes e depois do fechamento —, e ``reduction`` é o fator entre os dois. É a primeira
    metade do aceite do #211, medida no mesmo passo em que o modelo é treinado.
    """

    rmse_before: float = float("nan")
    rmse_after: float = float("nan")
    n_train: int = 0
    n_test: int = 0
    variance_explained: float = float("nan")

    @property
    def reduction(self) -> float:
        """``rmse_before / rmse_after``: quanto o fechamento reduziu (>1 é redução)."""
        if not np.isfinite(self.rmse_after) or self.rmse_after <= 0:
            return float("inf") if self.rmse_before > 0 else 1.0
        return float(self.rmse_before / self.rmse_after)

    def summary(self) -> str:  # noqa: D102 - the parent's summary with the two numbers that matter here
        return (f"hybrid: closure fitted on {self.n_train} points, held out {self.n_test}; "
                f"RMSE {self.rmse_before:.4g} -> {self.rmse_after:.4g} (x{self.reduction:.2f}), "
                f"{self.seconds:.3f} s")


class HybridModel(PhysicsModel):
    """``base(x) + correction(x)``: a física conhecida mais o que ela não explica (#211).

    A base é o ``surrogate`` quando ele existe (a ROM do caminho CFD -> ROM -> surrogate) e o ``physics`` quando não.
    Não confundir com :class:`pinneapple_neural.physics_model.HybridModel`, que é o **adaptador** da interface para o
    ``HybridSurrogatePhysics`` (surrogate + pós-modelo explícito): este é o arranjo da issue #211 — base + fechamento
    aprendido sobre o resíduo —, e os dois respondem à mesma interface ``fit``/``predict``/``uncertainty``/``export``.
    """

    family = "hybrid"

    def __init__(self, physics: Any, surrogate: Any = None, correction: Any = None, *, name: str = "",
                 x_key: str = "x", y_key: str = "y"):
        self.physics = physics
        self.surrogate = surrogate
        self.correction = correction
        base = surrogate if surrogate is not None else physics
        super().__init__(base, name=name or "hybrid")
        self._base = as_callable(base)
        self._closure = as_callable(correction) if correction is not None else None
        self.x_key, self.y_key = x_key, y_key
        self._fitted = False
        self._history: list[dict[str, float]] = []
        self._cv_rmse = float("nan")

    # -- leitura ---------------------------------------------------------------------------------------
    @property
    def n_params(self) -> int:
        """Parâmetros treináveis do fechamento (a base pode não ter nenhum: um solver é código)."""
        if isinstance(self.correction, nn.Module):
            return int(sum(p.numel() for p in self.correction.parameters()))
        return 0

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"base": type(self.surrogate if self.surrogate is not None else self.physics).__name__,
                    "has_surrogate": self.surrogate is not None, "has_correction": self.correction is not None,
                    "fitted": self._fitted})
        return out

    def capabilities(self) -> dict[str, str]:
        if self.correction is None:
            return {"fit": "no: there is no correction block to fit (the hybrid is the base alone)",
                    "predict": "yes", "uncertainty": "no: a closure is what would carry a spread", "export": "no"}
        hidden, dropouts = _dropout_capacity(self.correction) if isinstance(self.correction, nn.Module) else (0, 0)
        uncertainty = ("yes" if (hidden or dropouts) else
                       "no: the correction has no hidden nn.Linear for the Monte Carlo dropout hook to act on, so "
                       "the spread would be exactly zero")
        return {"fit": "yes", "predict": "yes", "uncertainty": uncertainty,
                "export": "state_dict only: the base is a Python callable, so there is no single torch graph to trace"}

    # -- a base e o fechamento -------------------------------------------------------------------------
    def base(self, x: Any) -> torch.Tensor:
        """A previsão da base (surrogate, se houver; senão a física), sem o fechamento."""
        return self._base(_primary_tensor(x))

    def closure(self, x: Any) -> torch.Tensor:
        """O que o fechamento acrescenta; zero antes de existir um.

        Um fechamento que existe mas nunca foi ajustado **recusa**: a saída de um MLP com pesos sorteados não é uma
        correção, é ruído, e somá-la à base seria pior do que não ter fechamento nenhum.
        """
        if self._closure is None:
            return torch.zeros_like(self.base(x))
        if not self._fitted:
            raise RuntimeError(
                f"{self.name}: the closure block exists but was never fitted -- its output would be the random "
                "initialization, not a correction. Call fit({'x': ..., 'y': ...}) first, or build the hybrid "
                "without a correction to use the base alone"
            )
        return self._closure(_primary_tensor(x))

    def predict(self, x: Any, **options: Any) -> Prediction:
        """``base(x) + correction(x)`` — as duas parcelas ficam em ``extras``, porque a conta é a explicação."""
        base = self.base(x)
        delta = self.closure(x)
        return Prediction(mean=base + delta, extras={"base": base, "closure": delta,
                                                     "fitted": self._fitted})

    # -- ajuste do fechamento --------------------------------------------------------------------------
    def fit(self, data: Any, *, epochs: int = 300, lr: float = 1e-2, batch_size: int = 64, seed: int = 0,
            test_fraction: float = 0.25, **options: Any) -> ClosureFit:
        """Ajusta o fechamento sobre o resíduo ``y - base(x)``, com um conjunto reservado separado **antes** do ajuste.

        ``data`` é ``{"x": ..., "y": ...}`` (o ``y`` é a referência: o solver fino, o experimento, o dado do
        produto). O resíduo é o alvo do MLP; o erro da base e do híbrido no reservado é reportado no
        :class:`ClosureFit`, e é isso que torna o "reduz erro" uma medida em vez de uma esperança.
        """
        if self.correction is None:
            raise ValueError("the hybrid has no correction block: pass correction=<ClosureBlock or callable> to fit "
                             "something; without it the hybrid is the base alone and there is nothing to train")
        if not isinstance(data, Mapping) or self.x_key not in data or self.y_key not in data:
            raise TypeError(f"fit takes a mapping with {self.x_key!r} and {self.y_key!r} "
                            "(the reference the closure is measured against)")
        x = _to_tensor(data[self.x_key])
        y = _to_tensor(data[self.y_key])
        n = int(x.shape[0])
        if n < 8:
            raise ValueError(f"at least 8 reference points are needed for a train/held-out split; got {n}")
        if int(test_fraction * n) < 2 or int((1 - test_fraction) * n) < 4:
            raise ValueError(f"test_fraction={test_fraction} leaves too few points on one side of {n}")

        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(n, generator=generator)
        n_test = max(2, int(round(test_fraction * n)))
        test_idx, train_idx = order[:n_test], order[n_test:]

        if not isinstance(self.correction, nn.Module):
            raise TypeError("a correction that is not an nn.Module has nothing to fit; pass a ClosureBlock")

        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        with torch.no_grad():
            residual_train = y_train - self.base(x_train)
            rmse_before = float(torch.sqrt(torch.mean((self.base(x_test) - y_test) ** 2)))

        closure = self.correction
        optimizer = torch.optim.Adam(closure.parameters(), lr=float(lr))
        loss_fn = nn.MSELoss()
        history: list[dict[str, float]] = []
        start = time.perf_counter()
        for _epoch in range(int(epochs)):
            closure.train()
            permutation = torch.randperm(int(x_train.shape[0]), generator=generator)
            total = 0.0
            for begin in range(0, int(x_train.shape[0]), int(batch_size)):
                idx = permutation[begin:begin + int(batch_size)]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(closure(x_train[idx]), residual_train[idx])
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * int(idx.numel())
            history.append({"train_total": total / int(x_train.shape[0])})
        seconds = time.perf_counter() - start

        closure.eval()
        with torch.no_grad():
            hybrid_test = self.base(x_test) + closure(x_test)
            rmse_after = float(torch.sqrt(torch.mean((hybrid_test - y_test) ** 2)))
            true_residual = y_test - self.base(x_test)
            predicted = closure(x_test)
            variance = 1.0 - float(torch.mean((predicted - true_residual) ** 2)) / max(
                float(torch.mean(true_residual ** 2)), 1e-30)

        self._fitted = True
        self._history = history
        self._cv_rmse = rmse_after
        return ClosureFit(family=self.family, method="gradient", epochs=len(history), seconds=seconds,
                          n_params=self.n_params, history=history,
                          notes="closure fitted on the residual y - base(x)", rmse_before=rmse_before,
                          rmse_after=rmse_after, n_train=int(train_idx.numel()), n_test=int(test_idx.numel()),
                          variance_explained=variance)

    # -- incerteza e persistência ----------------------------------------------------------------------
    def uncertainty(self, x: Any, *, method: str = "mc_dropout", n_samples: int = 100, seed: int | None = 0,
                    members: Sequence[Any] | None = None, **options: Any) -> Prediction:
        """O spread do fechamento, pela mesma regra das outras famílias: MC dropout quando o bloco tem o que amostrar,
        ensemble de fechamentos quando o chamador traz membros. A base é determinística, então a incerteza é do
        fechamento — e isso está dito no ``extras``."""
        if self.correction is None:
            raise UncertaintyNotAvailable("the hybrid has no correction block: nothing here carries a spread")
        if isinstance(self.correction, nn.Module):
            hidden, dropouts = _dropout_capacity(self.correction)
            if method == "mc_dropout" and not (hidden or dropouts):
                raise UncertaintyNotAvailable(
                    f"{self.name}: the closure has {hidden} hidden nn.Linear and {dropouts} nn.Dropout layers, so "
                    "Monte Carlo dropout would return an exactly zero spread; build the ClosureBlock with "
                    "dropout=0.1 or pass method='ensemble' with members fitted on different splits")
        if method == "ensemble":
            if not members or len(members) < 2:
                raise UncertaintyNotAvailable(
                    f"method='ensemble' needs at least two fitted hybrid members, got {0 if not members else len(members)}")
            stack = torch.stack([_to_tensor(m.predict(x).mean) for m in members], dim=0)
            return Prediction(mean=stack.mean(dim=0), std=stack.std(dim=0, unbiased=True), method="ensemble",
                              extras={"members": len(members)})
        if method != "mc_dropout":
            raise ValueError(f"unknown method {method!r}: use 'mc_dropout' or 'ensemble'")

        tensor = _primary_tensor(x)
        base = self.base(tensor)
        closure = self.correction
        if not isinstance(closure, nn.Module):
            raise UncertaintyNotAvailable(
                f"{self.name}: the correction is a plain callable with no stochastic layer to sample")
        from pinneapple_analysis.uncertainty.mc_dropout import MCDropoutConfig, MCDropoutWrapper

        wrapper = MCDropoutWrapper(closure, MCDropoutConfig(n_samples=int(n_samples), seed=seed))
        result = wrapper.predict_with_uncertainty(tensor)
        return Prediction(mean=base + result.mean, std=result.std, method="mc_dropout",
                          extras={"n_samples": int(n_samples), "base": base,
                                  "config": dict(result.metadata)})

    def export(self, path: str | Path, *, fmt: str = "state_dict") -> Path:
        """Só o fechamento é exportável: a base é um ``callable`` de Python, que nenhum ``torch.jit.trace`` captura."""
        if fmt != "state_dict":
            raise ExportNotAvailable(
                f"{self.name}: the base is a Python callable, so there is no single graph to trace; export with "
                "fmt='state_dict' (the trained part) and pass the base back when reloading")
        if not isinstance(self.correction, nn.Module):
            raise ExportNotAvailable(f"{self.name}: the correction is not an nn.Module, so it has no state to save")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"correction": self.correction.state_dict(), "fitted": self._fitted,
                    "class": type(self.correction).__qualname__}, target)
        return target

    # -- a camada de confiança -------------------------------------------------------------------------
    def score(self, x: Any, y: Any, *, min_reduction: float = 1.0, max_generalization_gap: float = 0.5
              ) -> Any:
        """O relatório de confiança do híbrido num conjunto que ele não viu.

        Duas checagens, ambas com número medido:

        * ``closure.reduces_error`` — RMSE do híbrido contra o da base no conjunto dado; passa quando a redução é de
          ao menos ``min_reduction`` (``1.0`` = "não piorou", que é o mínimo honesto);
        * ``closure.generalizes`` — o RMSE aqui contra o do conjunto que o próprio ``fit`` reservou, relativo. A
          referência é o reservado do ajuste, não o erro de treino: o erro de treino mede decoreba, e comparar com
          ele reprovaria qualquer fechamento que de fato aprendeu uma função. A pergunta desta checagem é a do
          aceite — o caso novo (outro número de Péclet, outra geometria) ficou muito pior do que o caso que o
          ajuste já não via?

        O relatório é o :class:`~pinneapple_analysis.trust.TrustReport` de sempre: quem decide é a política dele, não
        este método.
        """
        from pinneapple_analysis.trust.trust_report import NOT_RUN, SUPPORTS, Check, TrustReport

        tensor_x, tensor_y = _primary_tensor({self.x_key: x}), _to_tensor(y)
        with torch.no_grad():
            base = self.base(tensor_x)
            hybrid = self.predict(tensor_x).mean
        rmse_base = float(torch.sqrt(torch.mean((base - tensor_y) ** 2)))
        rmse_hybrid = float(torch.sqrt(torch.mean((hybrid - tensor_y) ** 2)))
        reduction = rmse_base / rmse_hybrid if rmse_hybrid > 0 else float("inf")

        checks = [
            Check(name="closure.reduces_error", status=SUPPORTS if reduction >= min_reduction else "contradicts",
                  critical=True, value=float(reduction), limit=float(min_reduction),
                  source="hybrid_closure.score",
                  reason=f"RMSE {rmse_base:.4g} (base) -> {rmse_hybrid:.4g} (hybrid), a factor of {reduction:.2f}"),
        ]
        if np.isfinite(self._cv_rmse):
            # a escala honesta de comparação é o conjunto que o próprio ajuste reservou, não o erro de treino (que
            # mede decoreba): a pergunta é "este caso novo ficou muito pior do que o reservado do ajuste?"
            gap = (rmse_hybrid - self._cv_rmse) / max(self._cv_rmse, 1e-30)
            checks.append(Check(name="closure.generalizes",
                                status=SUPPORTS if gap <= max_generalization_gap else "contradicts",
                                critical=False, value=float(gap), limit=float(max_generalization_gap),
                                source="hybrid_closure.score",
                                reason=f"RMSE {rmse_hybrid:.4g} here against {self._cv_rmse:.4g} on the split the "
                                       "fit reserved"))
        else:
            checks.append(Check(name="closure.generalizes", status=NOT_RUN, source="hybrid_closure.score",
                                reason="the closure was never fitted here, so there is no reserved split to compare"))
        return TrustReport(checks=checks, model=self.name,
                           details={"rmse_base": rmse_base, "rmse_hybrid": rmse_hybrid, "reduction": reduction})
