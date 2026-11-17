"""Ficha de condições de um caso de CFD e comparação item a item entre duas fichas.

Antes de comparar *resultados* de dois estudos, compare as *condições* que cada um declarou. É barato, é honesto e
quase sempre é onde está a explicação de uma divergência: vazão, área de entrada, nível inicial, definição das linhas,
turbulência de entrada, malha e solver. O que não estiver declarado aparece como "não declarado" — nunca é assumido.

    from pinneapple_simulation.pipelines.case_card import CondicoesCaso, comparar, tabela_md

    nos = CondicoesCaso(nome="nosso", vetor_entrada=(0, 9.85, 1.74), area_entrada_m2=0.005632,
                        nivel_inicial_m=-2.04, linhas={"LEdir": ((-1.8, -7.429, -2.2), (-1.8, 15.305, -2.2))})
    deles = CondicoesCaso(nome="deles", vetor_entrada=(0, 9.85, 1.74), vazao_m3_s=0.04582,
                          nivel_inicial_m=-2.22)
    comparar(nos, deles)        # [{"campo": "vetor_entrada", "a": "...", "b": "...", "igual": True}, ...]
    print(tabela_md(nos, deles))

Duas contas que a ficha faz e que costumam resolver discussão:

- **área a partir da vazão e da velocidade:** ``area = vazao / |U|``. Duas fichas que declaram a mesma velocidade e
  vazões diferentes *não* têm a mesma seção de entrada, e isso muda o jato;
- **k de entrada a partir da intensidade:** ``k = 1.5 (I |U|)²``. Intensidade de 1 % com 10 m/s dá k = 0,015 m²/s².
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


def modulo(u: Sequence[float]) -> float:
    """|U| de um vetor de 3 componentes."""
    u = tuple(float(x) for x in u)
    if len(u) != 3:
        raise ValueError("vetor precisa ter 3 componentes")
    return math.sqrt(sum(x * x for x in u))


def diametro_equivalente(area: float) -> float:
    """Diâmetro (m) de um círculo com a área dada."""
    if area <= 0:
        raise ValueError(f"área precisa ser positiva; recebi {area}")
    return math.sqrt(4.0 * area / math.pi)


def area_de_vazao(vazao: float, velocidade: float) -> float:
    """Área de entrada (m²) que, com essa velocidade média, dá essa vazão."""
    if velocidade <= 0:
        raise ValueError(f"velocidade precisa ser positiva; recebi {velocidade}")
    return float(vazao) / float(velocidade)


def k_de_intensidade(intensidade: float, velocidade: float) -> float:
    """Energia cinética turbulenta de entrada a partir da intensidade (fração) e da velocidade média."""
    return 1.5 * (float(intensidade) * float(velocidade)) ** 2


@dataclass(frozen=True)
class CondicoesCaso:
    """O que um caso declara sobre a entrada, o banho, a amostragem e o solver.

    Campo que não foi declarado fica ``None`` — e é assim que aparece na comparação. ``linhas`` é
    ``{nome: (ponto_inicial, ponto_final)}``. ``extras`` guarda o que for específico do estudo (modelo de turbulência,
    condições de contorno de parede, versão de software) sem virar campo fixo aqui.
    """

    nome: str
    vetor_entrada: tuple[float, float, float] | None = None
    vazao_m3_s: float | None = None
    area_entrada_m2: float | None = None
    nivel_inicial_m: float | None = None
    turbulencia_intensidade: float | None = None
    turbulencia_escala_m: float | None = None
    linhas: dict = field(default_factory=dict)
    malha_celulas: int | None = None
    solver: str | None = None
    extras: dict = field(default_factory=dict)

    def velocidade(self) -> float | None:
        return modulo(self.vetor_entrada) if self.vetor_entrada else None

    def area(self) -> float | None:
        """Área declarada; se só houver vazão e velocidade, é derivada delas."""
        if self.area_entrada_m2 is not None:
            return float(self.area_entrada_m2)
        if self.vazao_m3_s is not None and self.velocidade():
            return area_de_vazao(self.vazao_m3_s, self.velocidade())
        return None

    def vazao(self) -> float | None:
        if self.vazao_m3_s is not None:
            return float(self.vazao_m3_s)
        if self.area() is not None and self.velocidade():
            return self.area() * self.velocidade()
        return None

    def diametro(self) -> float | None:
        a = self.area()
        return diametro_equivalente(a) if a else None

    def k_entrada(self) -> float | None:
        if self.turbulencia_intensidade is None or not self.velocidade():
            return None
        return k_de_intensidade(self.turbulencia_intensidade, self.velocidade())


def _fmt(v) -> str:
    if v is None:
        return "não declarado"
    if isinstance(v, (tuple, list)):
        return "(" + "; ".join(_fmt(x) for x in v) + ")"
    if isinstance(v, bool):
        return "sim" if v else "não"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _itens(c: CondicoesCaso) -> dict:
    """Os itens comparáveis, já com as grandezas derivadas resolvidas."""
    d = c.diametro()
    return {
        "vetor de entrada [m/s]": c.vetor_entrada,
        "|U| de entrada [m/s]": c.velocidade(),
        "vazão [m³/s]": c.vazao(),
        "área de entrada [m²]": c.area(),
        "diâmetro equivalente [mm]": (d * 1000.0) if d else None,
        "nível inicial [m]": c.nivel_inicial_m,
        "intensidade de turbulência": c.turbulencia_intensidade,
        "escala de turbulência [m]": c.turbulencia_escala_m,
        "k de entrada [m²/s²]": c.k_entrada(),
        "linhas de amostragem": len(c.linhas) or None,
        "malha [células]": c.malha_celulas,
        "solver": c.solver,
        **{f"extra: {k}": v for k, v in c.extras.items()},
    }


def _iguais(a, b, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def comparar(a: CondicoesCaso, b: CondicoesCaso, tol: float = 1e-6) -> list[dict]:
    """Lista item a item: ``{"campo", "a", "b", "igual", "declarado"}``.

    ``declarado`` é ``False`` quando pelo menos um dos lados não traz o item — o que é, em si, um achado: a comparação
    não pode ser feita de igual para igual naquele item.
    """
    ia, ib = _itens(a), _itens(b)
    out = []
    for campo in dict.fromkeys([*ia, *ib]):
        va, vb = ia.get(campo), ib.get(campo)
        out.append({"campo": campo, "a": _fmt(va), "b": _fmt(vb),
                    "igual": _iguais(va, vb, tol) if (va is not None and vb is not None) else None,
                    "declarado": va is not None and vb is not None})
    return out


def divergentes(a: CondicoesCaso, b: CondicoesCaso, tol: float = 1e-6) -> list[dict]:
    """Só os itens declarados dos dois lados que **não** coincidem — a lista de suspeitos, em ordem de aparição."""
    return [d for d in comparar(a, b, tol) if d["declarado"] and d["igual"] is False]


def tabela_md(a: CondicoesCaso, b: CondicoesCaso, tol: float = 1e-6) -> str:
    """A comparação como tabela markdown (para relatório), com a divergência explícita na última coluna."""
    linhas = [f"| condição | {a.nome} | {b.nome} | situação |", "|---|---|---|---|"]
    for d in comparar(a, b, tol):
        situacao = ("não declarado" if not d["declarado"] else ("igual" if d["igual"] else "**difere**"))
        linhas.append(f"| {d['campo']} | {d['a']} | {d['b']} | {situacao} |")
    return "\n".join(linhas)


__all__ = ["CondicoesCaso", "comparar", "divergentes", "tabela_md", "area_de_vazao", "diametro_equivalente",
           "k_de_intensidade", "modulo"]
