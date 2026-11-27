"""Conservação e condição de contorno calculadas a partir da especificação e da previsão (#159).

Até aqui, "conserva massa" e "respeita a condição de contorno" eram booleanos que o chamador passava — a afirmação de
quem treinou o modelo, não uma medida do resultado. Este módulo faz a pergunta ao campo previsto: lê os fluxos nas
faces da grade, integra o balanço em cada volume de controle e no domínio inteiro, e compara com a fonte e as
condições de contorno que a **especificação** declara. Sai um número com limite no vocabulário do cartão de confiança
(:class:`~pinneapple_analysis.trust.FieldCheck`), então entra no relatório de deploy como qualquer outra checagem.

As duas grandezas que o aceite pede:

* **massa**, para um campo de velocidade: ``sum_a d(rho u_a)/dx_a = fonte`` volume a volume (divergência nula no
  escoamento incompressível sem fonte) e o balanço global — o que sai pelas faces do contorno menos o que entra é a
  fonte integrada;
* **energia**, para um campo de temperatura em regime permanente: o fluxo condutivo líquido que atravessa cada volume
  é a fonte integrada (``q''' V``); sem geração, o balanço é zero.

A violação de contorno é medida contra o valor prescrito, face por face e componente por componente — não-deslizamento
na parede é ``u = 0``, entrada uniforme é ``u = U`` —, relativa à amplitude do próprio campo.

A discretização é de volume finito **centrado no nó**, que é onde os valores da grade vivem: o volume de controle do
nó ``i`` vai da face ``i`` à ``i+1``, com métricas
``[(x1-x0)/2, (x2-x0)/2, …, (xn-x_{n-2})/2, (xn-x_{n-1})/2]``. As faces internas ficam no ponto médio, onde a média das
duas células (massa) e a diferença central de temperatura (energia) dão o valor exato para perfis quadráticos, qualquer
que seja o espaçamento; as faces do contorno usam a fórmula unilateral de segunda ordem
(:func:`numpy.gradient`, ``edge_order=2``), também exata para quadráticos. É isso que deixa comparar o resíduo medido
com o exato nos dois problemas de referência, em ``tests/pinneapple_analysis/test_flow_conservation.py``: Poiseuille
em canal plano (massa) e condução 1-D numa parede com geração uniforme (energia) — o campo exato, onde o resíduo tem
de ficar no erro de arredondamento, e uma perturbação cuja violação é conhecida de antemão.

>>> import numpy as np
>>> from pinneapple_analysis.trust.flow_card import FlowSpec, flow_checks
>>> y = np.linspace(0, 1, 41)
>>> x = np.linspace(0, 2, 81)
>>> u = 4 * 1.0 * (y * (1 - y))                            # Poiseuille: u(y), nada em x
>>> spec = FlowSpec.mass(components={"u": "x", "v": "y"},
...                      boundary={"y_min": {"u": 0.0}, "y_max": {"u": 0.0}})
>>> field = np.broadcast_to(u[None, :], (x.size, y.size))          # a grade é (x, y): u varia só em y
>>> card = {c.name: c.passed for c in flow_checks({"x": x, "y": y}, {"u": field, "v": 0.0}, spec=spec)}
>>> card["conservation.mass"], card["boundary y_min (u)"]
(True, True)
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

from .field_card import DEFAULT_TOLERANCES, FieldCheck

__all__ = ["FlowSpec", "flow_checks", "mass_balance", "energy_balance", "TOLERANCES"]

Value = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]  # noqa: UP007 (runtime alias, Python 3.10)

#: ``conservation``: desequilíbrio relativo ao tamanho do fluxo que carrega a grandeza; ``local``: o mesmo, no pior
#: volume; ``boundary``: maior |campo - prescrito| relativo à amplitude do campo. ``conservation`` e ``boundary`` têm
#: os mesmos nomes e o mesmo sentido do cartão de campo escalar, para os dois relatórios serem lidos juntos.
TOLERANCES = {"conservation": DEFAULT_TOLERANCES["conservation"], "local": DEFAULT_TOLERANCES["conservation"],
              "boundary": DEFAULT_TOLERANCES["bc"]}


@dataclass(frozen=True)
class FlowSpec:
    """O que a especificação do problema diz que a previsão tem de cumprir.

    ``quantity``: ``"mass"`` (campos de velocidade) ou ``"energy"`` (campo de temperatura). ``components``: nome do
    campo → eixo da grade em que ele atua. ``source``: fonte volumétrica — massa por volume, ou calor por volume
    ``q'''`` — como constante, campo ou função dos pontos; sem fonte o balanço é zero. ``boundary``: face
    (``"x_min"``, ``"y_max"``, …) → componente → valor prescrito. ``density`` e ``thermal_conductivity`` são as
    constantes que a especificação declara.
    """

    quantity: str
    components: Mapping[str, str]
    source: Value | None = None
    boundary: Mapping[str, Mapping[str, Value]] = field(default_factory=dict)
    density: float = 1.0
    thermal_conductivity: float = 1.0
    tolerances: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity not in ("mass", "energy"):
            raise ValueError("quantity must be 'mass' or 'energy'")
        if not self.components:
            raise ValueError("components must name each field and the axis it acts along")
        for face in self.boundary:
            _split_face(face)

    @classmethod
    def mass(cls, components: Mapping[str, str], **kw: Any) -> FlowSpec:
        return cls(quantity="mass", components=dict(components), **kw)

    @classmethod
    def energy(cls, name: str = "T", axis: str = "x", **kw: Any) -> FlowSpec:
        return cls(quantity="energy", components={name: axis}, **kw)


def mass_balance(grid: Mapping[str, Sequence[float]], velocity: Mapping[str, Any], *,
                 spec: FlowSpec | None = None, **kw: Any) -> list[FieldCheck]:
    """As checagens de massa de um campo de velocidade; sem ``spec``, cada campo age no eixo da mesma posição."""
    if spec is None:
        spec = FlowSpec.mass(components={c: a for c, a in zip(velocity, grid, strict=False)}, **kw)
    return flow_checks(grid, velocity, spec=spec)


def energy_balance(grid: Mapping[str, Sequence[float]], temperature: Any, *, name: str = "T",
                   axis: str = "x", spec: FlowSpec | None = None, **kw: Any) -> list[FieldCheck]:
    """As checagens de energia de um campo de temperatura em regime permanente."""
    if spec is None:
        spec = FlowSpec.energy(name, axis, **kw)
    return flow_checks(grid, {name: temperature}, spec=spec)


# ---------------------------------------------------------------------------------------------- grid helpers
def _axes(grid: Mapping[str, Sequence[float]]) -> tuple[list[str], list[np.ndarray], tuple[int, ...]]:
    names = list(grid)
    axes = [np.asarray(grid[n], dtype=np.float64).ravel() for n in names]
    for n, a in zip(names, axes, strict=True):
        if a.size < 3 or np.any(np.diff(a) <= 0):
            raise ValueError(f"grid axis '{n}' must be strictly increasing with at least 3 points (the boundary "
                             "face needs a second-order one-sided derivative)")
    return names, axes, tuple(len(a) for a in axes)


def _points(axes: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=1)


def _evaluate(value: Value, pts: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if callable(value):
        out = np.asarray(value(pts), dtype=np.float64)
    else:
        out = np.asarray(value, dtype=np.float64)
        if out.ndim == 0:
            return np.full(shape, float(out))
    if out.size != int(np.prod(shape)):
        raise ValueError(f"a field of {out.size} values does not fit the grid {shape}")
    return out.reshape(shape)


def _split_face(face: str) -> tuple[str, int]:
    coord, _, side = str(face).rpartition("_")
    if not coord or side not in ("min", "max"):
        raise ValueError(f"face '{face}' must be '<coord>_min' or '<coord>_max'")
    return coord, 0 if side == "min" else -1


def _control_widths(axis: np.ndarray) -> np.ndarray:
    """A largura do volume de controle de cada nó: meia célula nas pontas, célula inteira no meio."""
    w = np.empty(axis.size)
    w[0] = (axis[1] - axis[0]) / 2.0
    w[-1] = (axis[-1] - axis[-2]) / 2.0
    w[1:-1] = (axis[2:] - axis[:-2]) / 2.0
    return w


def _relative(imbalance: float, scale: float) -> float:
    return float(abs(imbalance) / max(scale, 1e-300))


# ---------------------------------------------------------------------------------------------- the checks
def flow_checks(grid: Mapping[str, Sequence[float]], fields: Mapping[str, Any], *, spec: FlowSpec) -> list[FieldCheck]:
    """As checagens de conservação e de contorno de ``fields`` (campo → valores na grade) contra ``spec``.

    Cada checagem é um :class:`FieldCheck`, com ``passed=None`` quando não pôde ser perguntada; a ``detail`` traz o
    número medido e contra o que ele foi comparado.
    """
    tol = {**TOLERANCES, **dict(spec.tolerances or {})}
    names, axes, shape = _axes(grid)
    pts = _points(axes)
    values: dict[str, np.ndarray] = {}
    for comp in spec.components:
        if comp not in fields:
            raise ValueError(f"a especificação pede o campo {comp!r}, que a previsão não traz")
        raw = fields[comp]
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        values[comp] = _evaluate(np.asarray(raw, dtype=np.float64), pts, shape)
        if not np.isfinite(values[comp]).all():
            return [FieldCheck(f"finite values ({comp})", "sanity", False, critical=True,
                               detail=f"NaN or Inf in '{comp}': nothing can be measured on it")]
    source = np.zeros(shape) if spec.source is None else _evaluate(spec.source, pts, shape)
    widths = [_control_widths(a) for a in axes]
    volume = np.ones(shape)
    for a, w in enumerate(widths):
        volume = volume * w.reshape(tuple(-1 if i == a else 1 for i in range(len(shape))))
    source_cell = source * volume

    residual = np.zeros(shape)
    local_scale = np.abs(source_cell).copy()
    net_out, flux_total = 0.0, 0.0
    for comp, axis_name in spec.components.items():
        if axis_name not in names:
            raise ValueError(f"o campo {comp!r} atua no eixo {axis_name!r}, que não está na grade {names}")
        a = names.index(axis_name)
        rest = [i for i in range(len(shape)) if i != a]
        areas = np.ones(tuple(shape[i] for i in rest))
        for pos, i in enumerate(rest):
            areas = areas * widths[i].reshape(tuple(-1 if j == pos else 1 for j in range(len(rest))))
        faces = _face_flux(values[comp], axes[a], areas, a, spec)
        moved = np.moveaxis(faces, a, 0)
        residual += np.moveaxis(moved[1:] - moved[:-1], 0, a)
        low, high = np.take(faces, 0, axis=a), np.take(faces, -1, axis=a)
        net_out += float(high.sum() - low.sum())
        # a escala do balanço global é o que atravessa o contorno (mais a fonte): somar as faces internas diluiria a
        # violação numa malha fina, e uma malha mais fina não torna o balanço melhor
        flux_total += float(np.abs(low).sum() + np.abs(high).sum())
        magnitude = np.moveaxis(np.abs(faces), a, 0)
        local_scale += np.moveaxis(magnitude[1:] + magnitude[:-1], 0, a)
    residual = residual - source_cell

    source_total = float(source_cell.sum())
    imbalance = net_out - source_total
    rel_global = _relative(imbalance, max(flux_total, abs(source_total)))
    worst, worst_scale = float(np.abs(residual).max()), float(local_scale.max())
    rel_local = _relative(worst, worst_scale)
    checks = [
        FieldCheck(f"conservation.{spec.quantity}", "physics", rel_global <= tol["conservation"], value=rel_global,
                   threshold=tol["conservation"], critical=True,
                   detail=f"{spec.quantity}: o que sai pelas faces {net_out:.6g} contra a fonte integrada "
                          f"{source_total:.6g} (desequilíbrio {imbalance:.3g})"),
        FieldCheck(f"conservation.{spec.quantity}.local", "physics", rel_local <= tol["local"], value=rel_local,
                   threshold=tol["local"],
                   detail=f"pior volume: |balanço| {worst:.6g} contra a escala local {worst_scale:.6g} (um total "
                          "que fecha não esconde um volume que não fecha)"),
    ]
    checks.extend(_boundary_checks(spec, values, names, shape, pts, tol))
    return checks


def _face_flux(field_values: np.ndarray, axis: np.ndarray, areas: np.ndarray, a: int,
               spec: FlowSpec) -> np.ndarray:
    """O fluxo integrado em cada face normal ao eixo ``a``: a forma da célula com um plano a mais naquele eixo.

    Massa: ``rho u_a A`` com ``u_a`` na média dos dois nós da face — exato para um perfil linear. Energia:
    ``-k dT/dx_a A`` com a diferença central dentro e a fórmula unilateral de segunda ordem na face do contorno.
    """
    moved = np.moveaxis(field_values, a, 0)
    n = moved.shape[0]
    area_b = areas.reshape((1, *areas.shape))
    faces = np.empty((n + 1, *moved.shape[1:]))
    if spec.quantity == "mass":
        cell = spec.density * moved * area_b                     # o fluxo que a célula carrega, por unidade de face
        faces[1:-1] = 0.5 * (cell[:-1] + cell[1:])
        # a face do contorno usa o valor do próprio campo, não o prescrito: o balanço é do campo que a previsão
        # entrega. Trocar pelo prescrito fecharia a conta de um campo que sopra pela parede, e é a checagem de
        # contorno que mede essa diferença, não esta.
        faces[0], faces[-1] = cell[0], cell[-1]
    else:
        k = spec.thermal_conductivity
        grad = np.empty((n + 1, *moved.shape[1:]))
        grad[1:-1] = (moved[1:] - moved[:-1]) / np.asarray(np.diff(axis)).reshape((-1,) + (1,) * (moved.ndim - 1))
        edge = np.gradient(moved, np.asarray(axis), axis=0, edge_order=2)
        grad[0], grad[-1] = edge[0], edge[-1]
        faces = -k * grad * area_b
    return np.moveaxis(faces, 0, a)


def _boundary_target(spec: FlowSpec, face: str, comp: str, pts: np.ndarray, shape: tuple[int, ...], a: int,
                     side: int) -> np.ndarray | None:
    """O valor que a especificação prescreve para ``comp`` naquela face, achatado no plano da face; None se cala."""
    target = spec.boundary.get(face, {}).get(comp)
    if target is None:
        return None
    return np.take(_evaluate(target, pts, shape), 0 if side == 0 else -1, axis=a)


def _boundary_checks(spec: FlowSpec, values: Mapping[str, np.ndarray], names: Sequence[str],
                     shape: tuple[int, ...], pts: np.ndarray, tol: Mapping[str, float]) -> list[FieldCheck]:
    out: list[FieldCheck] = []
    if not spec.boundary:
        return [FieldCheck("boundary conditions", "physics", None,
                           detail="a especificação não declara condição de contorno")]
    for face, prescribed in spec.boundary.items():
        coord, side = _split_face(face)
        for comp in prescribed:
            if coord not in names:
                out.append(FieldCheck(f"boundary {face} ({comp})", "physics", None,
                                      detail=f"a face {face!r} não está na grade {list(names)}"))
                continue
            if comp not in values:
                out.append(FieldCheck(f"boundary {face} ({comp})", "physics", None,
                                      detail=f"a previsão não traz o campo {comp!r}"))
                continue
            a = names.index(coord)
            plane = np.take(values[comp], side, axis=a)
            want = _boundary_target(spec, face, comp, pts, shape, a, side)
            err = float(np.abs(plane - want).max())
            rng = float(np.ptp(values[comp])) or max(float(np.abs(values[comp]).max()), 1e-300)
            rel = _relative(err, rng)
            out.append(FieldCheck(f"boundary {face} ({comp})", "physics", rel <= tol["boundary"], value=rel,
                                  threshold=tol["boundary"], critical=True,
                                  detail=f"Dirichlet: maior |{comp} - prescrito| {err:.6g} em {int(plane.size)} "
                                         f"pontos, contra a amplitude {rng:.6g}"))
    return out
