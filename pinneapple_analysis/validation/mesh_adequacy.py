"""A malha é suficiente para o número que se está reportando?

Três perguntas, nesta ordem, porque a terceira costuma ser a que resolve a discussão:

1. **O resultado converge com a malha?** Refinar três vezes e olhar a grandeza de interesse (não a média global):
   ordem de convergência observada ``p``, valor extrapolado e o índice de convergência de malha (GCI, Roache),
   que é a incerteza numérica do resultado na malha fina.
2. **A região que governa o número está resolvida?** Um jato precisa de células pequenas em relação ao seu diâmetro na
   zona de cisalhamento e de impacto: com ``Δx/d`` grande, a camada onde ``k`` é produzido é numérica, não física.
3. **A linha de amostragem consegue representar aquilo?** Se a linha amostra o centro da célula mais próxima e as
   células têm 0,5 m, a posição de um pico fica quantizada em passos de 0,5 m — e comparar com outro estudo ao
   milímetro é medir o ruído da malha, não a física.

    from pinneapple_analysis.validation.mesh_adequacy import indice_convergencia, resolucao_da_linha
    indice_convergencia([7.05, 7.20, 7.62])         # do mais fino para o mais grosso, razão 2 entre eles
    resolucao_da_linha(centros_das_celulas, pontos_da_linha)

Só a terceira não precisa de rodadas novas: sai do próprio campo gravado, e é a que mostra se a posição de pico pode
ser comparada com outra malha.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["Criterio", "indice_convergencia", "resolucao_da_linha", "resolver_jato", "verificar"]


@dataclass(frozen=True)
class Criterio:
    """Um critério de suficiência de malha: o valor medido, o limite e o veredito."""

    nome: str
    valor: float | None
    limite: float | None
    ok: bool | None
    nota: str = ""

    def to_dict(self) -> dict:
        return {"nome": self.nome, "valor": self.valor, "limite": self.limite, "ok": self.ok, "nota": self.nota}


def indice_convergencia(valores: Sequence[float], razoes: Sequence[float] | None = None,
                        fator_seguranca: float = 1.25) -> dict:
    """Índice de convergência de malha (GCI) a partir de três resultados, do **mais fino** para o mais grosso.

    ``razoes`` são os fatores de refinamento entre malhas consecutivas (``h_grosso / h_fino``); se não vier, assume 2.
    Devolve ``{"p", "extrapolado", "gci", "monotonico", "convergiu", "nota"}``:

    - ``p``: ordem de convergência observada; ``None`` se não há convergência monotônica;
    - ``extrapolado``: o valor que se esperaria com malha infinita (Richardson);
    - ``gci``: incerteza relativa do resultado na malha fina — é isso que se cita ao lado do número;
    - ``convergiu``: se a diferença entre as duas malhas mais finas é menor que a tolerância de 5 %.

    Referência: Roache, *Verification and Validation in Computational Science and Engineering* (1998), e o
    procedimento de três malhas de Celik et al. (2008).
    """
    f = [float(x) for x in valores]
    if len(f) < 3:
        raise ValueError("são necessários três resultados (fino, médio, grosso)")
    f1, f2, f3 = f[0], f[1], f[2]
    r = list(razoes) if razoes else [2.0, 2.0]
    if len(r) < 2:
        raise ValueError("razoes precisa de dois fatores (fino→médio e médio→grosso)")
    if min(r) <= 1.0:
        raise ValueError("razão de refinamento precisa ser maior que 1")
    d21, d32 = f2 - f1, f3 - f2
    monotonico = d21 * d32 > 0
    tol = 1e-12 * max(1.0, abs(f1), abs(f2), abs(f3))
    if abs(d21) <= tol:                                  # as duas malhas mais finas dão o mesmo número
        return {"p": None, "extrapolado": f1, "gci": 0.0, "monotonico": True, "convergiu": True,
                "nota": "as duas malhas mais finas coincidem dentro do arredondamento"}
    if not monotonico:
        return {"p": None, "extrapolado": None,
                "gci": abs(d21) / abs(f1) * fator_seguranca if f1 else None, "monotonico": False,
                "convergiu": False,
                "nota": "convergência não monotônica: refinar não leva a um valor estável — a malha não é adequada "
                        "para esta grandeza"}
    p = abs(math.log(abs(d32 / d21))) / math.log(r[0]) if d32 else float("inf")
    if not math.isfinite(p) or p <= 0:
        return {"p": p, "extrapolado": None, "gci": abs(d21) / abs(f1) * fator_seguranca if f1 else None,
                "monotonico": True, "convergiu": False,
                "nota": "ordem observada não positiva: o refino não está na faixa assintótica"}
    extrapolado = f1 + d21 / (r[0] ** p - 1.0)
    gci = fator_seguranca * abs(d21 / f1) / (r[0] ** p - 1.0) if f1 else None
    return {"p": p, "extrapolado": extrapolado, "gci": gci, "monotonico": True,
            "convergiu": bool(gci is not None and gci < 0.05),
            "nota": (f"p = {p:.2f}; GCI = {100 * gci:.1f} % na malha fina" if gci is not None else "f1 nulo")}


def resolucao_da_linha(centros: np.ndarray, pontos: np.ndarray) -> dict:
    """O que a linha de amostragem consegue representar numa malha, sem interpolação.

    ``centros`` (n_células, 3) e ``pontos`` (n_pontos, 3). Cada ponto é atribuído à célula de centro mais próximo (é o
    que a amostragem por célula faz). Devolve ``{"pontos", "celulas_unicas", "passo_medio", "passo_max", "fracao_repetida"}``:

    - ``passo_max``: o maior vão entre células distintas ao longo da linha — **é a quantização da posição do pico**;
    - ``fracao_repetida``: fração dos pontos que caíram em uma célula já visitada (linha amostrando a mesma célula).

    Um pico localizado com passo de 1 m não pode ser comparado a outro estudo ao milímetro: a diferença mínima que a
    malha permite enxergar é ``passo_max``.
    """
    c = np.asarray(centros, dtype=float)
    p = np.asarray(pontos, dtype=float)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"centros deve ser (n, 3); veio {c.shape}")
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"pontos deve ser (n, 3); veio {p.shape}")
    if len(c) == 0 or len(p) == 0:
        raise ValueError("centros e pontos não podem ser vazios")
    d = np.linalg.norm(p[:, None, :] - c[None, :, :], axis=2)      # malhas pequenas: direto é mais claro que KD-tree
    idx = d.argmin(1)
    unicos = list(dict.fromkeys(idx.tolist()))                     # ordem de percurso, sem repetir
    # a quantização é a distância entre os CENTROS das células que a linha atravessa, não entre os pontos de
    # amostragem: ela existe mesmo que se amostre mais fino
    passos = [float(np.linalg.norm(c[j] - c[i])) for i, j in zip(unicos[:-1], unicos[1:], strict=True)]
    return {"pontos": int(len(p)), "celulas_unicas": len(unicos),
            "passo_medio": float(np.mean(passos)) if passos else 0.0,
            "passo_max": float(np.max(passos)) if passos else 0.0,
            "fracao_repetida": float(1.0 - len(unicos) / len(p))}


def resolver_jato(delta_x: float, diametro: float, minimo: float = 10.0) -> Criterio:
    """Critério ``Δx/d``: quantas células cabem no diâmetro do jato na zona de cisalhamento.

    Abaixo de ~10 células por diâmetro, a camada onde ``k`` é produzida passa a ser dominada por difusão numérica, e a
    posição do máximo de ``k`` deixa de ser uma propriedade do escoamento: vira uma propriedade da malha.
    """
    if diametro <= 0:
        raise ValueError("diâmetro precisa ser positivo")
    n = float(diametro) / float(delta_x)
    return Criterio("células por diâmetro do jato", n, float(minimo), n >= minimo,
                    f"Δx/d = {1.0 / n:.3f}; abaixo de {minimo:g} células por diâmetro a produção de k é numérica")


def verificar(centros: np.ndarray | None = None, pontos: np.ndarray | None = None, valores: Sequence[float] | None = None,
              delta_x: float | None = None, diametro: float | None = None, razoes: Sequence[float] | None = None,
              minimo_por_diametro: float = 10.0) -> dict:
    """Roda os três critérios disponíveis e devolve ``{"criterios": [...], "ok": bool, "resumo": str}``.

    Todos os argumentos são opcionais: só entra o critério cujos dados foram dados. ``ok`` é ``None`` quando nenhum
    critério pôde ser avaliado — "não sei" é diferente de "está bom".
    """
    criterios: list[Criterio] = []
    if centros is not None and pontos is not None:
        r = resolucao_da_linha(centros, pontos)
        criterios.append(Criterio("passo máximo da linha de amostragem [m]", r["passo_max"], None, None,
                                  f"{r['celulas_unicas']} células distintas em {r['pontos']} pontos; um pico não pode "
                                  f"ser localizado melhor que {r['passo_max']:.3f} m"))
    if delta_x is not None and diametro is not None:
        criterios.append(resolver_jato(delta_x, diametro, minimo_por_diametro))
    if valores is not None:
        g = indice_convergencia(valores, razoes)
        criterios.append(Criterio("ordem de convergência observada", g["p"], None, None, g["nota"]))
        criterios.append(Criterio("GCI na malha fina", g["gci"], 0.05, g["convergiu"], g["nota"]))
    avaliados = [c for c in criterios if c.ok is not None]
    ok = all(c.ok for c in avaliados) if avaliados else None
    resumo = ("sem critério avaliável: informe centros+pontos, delta_x+diametro e/ou três resultados"
              if ok is None else ("malha suficiente nos critérios avaliados" if ok
                                  else "malha insuficiente em pelo menos um critério"))
    return {"criterios": [c.to_dict() for c in criterios], "ok": ok, "resumo": resumo}
