"""Linhas de amostragem de campo, janelas de tempo e posição do pico — o pós-processamento genérico de um escoamento.

Generaliza o que era código de um estudo só (o canal de corrida): qualquer caso em que se amostram linhas fixas no
espaço e se resume o campo no tempo. As linhas são segmentos em ``y`` com ``x`` e ``z`` fixos — o caso comum de uma
linha paralela ao eixo do canal, a uma distância lateral que pode ser relativa à parede ou absoluta. Nada aqui lê
solvers: entra ``(k, t, dabs)``, sai perfil, pico, média de faixa e a tabela de comparação com uma referência.

Uso típico, ligado ao :class:`~pinneapple_design.geometry.runner_ramp` (ou a qualquer geometria da lib)::

    from pinneapple_simulation.pipelines.field_lines import Linha, pico, perfil_medio, media_faixa
    linha = Linha("LEdir", x=-1.8, z=-2.2, y_inicio=-7.429, y_fim=15.305, n=101)
    dabs = linha.distancias(y_origem=-7.429)          # 0 no plano de entrada
    k, t = ...                                        # (n_instantes, n_pontos) e (n_instantes,)
    prof = perfil_medio(k, t, janela=(20.0, 30.0))    # média temporal em cada ponto
    k_pico, s_pico, i = pico(k, t, dabs, janela=(20.0, 30.0))
    media_faixa(k, t, dabs, 2.0, 8.0, janela=(20.0, 30.0))

Convenção de distância: ``dabs`` é o eixo longitudinal medido do plano de entrada (``y - y_origem``), que é o que os
relatórios chamam de "distância a partir do inlet". A janela de tempo é sempre explícita — o mesmo campo dá posições
de pico diferentes se uma pessoa lê a média de 0–30 s e outra a de 20–30 s, e essa ambiguidade já custou uma
comparação entre duas equipes.

**A lição que custou mais caro, e que este módulo agora carrega explicitamente.** Duas equipes compararam a posição do
pico de k no mesmo caso e uma encontrou 5,68 m, a outra 7,27 m. Não havia erro em nenhuma das duas simulações nem nas
medições: uma contava a distância do plano de entrada e a outra **do primeiro ponto que a linha efetivamente amostra** —
que, numa linha avaliada em centros de célula, fica 1,36 m adiante. O desvio de 1,6 m era o metro zero. Três coisas
resolvem esse tipo de armadilha, e estão aqui: :func:`origem_da_linha` mede o deslocamento a partir dos pontos,
:func:`comparar_picos` aceita ``deslocamento_ref`` para pôr as duas tabelas na mesma origem, e
:func:`deslocamento_de_origem` **descobre** o deslocamento a partir de dois perfis, quando não se sabe qual convenção o
outro lado usou. A regra prática: antes de chamar de divergência física uma diferença de posição, verifique se ela não é
um múltiplo exato do passo de amostragem e se os dois perfis não são o mesmo perfil deslocado.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Linha:
    """Uma linha de amostragem: segmento em y, com x e z fixos, em metros, no referencial do caso."""

    nome: str
    x: float
    z: float
    y_inicio: float
    y_fim: float
    n: int = 101

    @property
    def pontos(self) -> np.ndarray:
        """Os ``n`` pontos da linha, (n, 3)."""
        y = np.linspace(self.y_inicio, self.y_fim, self.n)
        return np.column_stack([np.full(self.n, self.x), y, np.full(self.n, self.z)])

    def distancias(self, y_origem: float) -> np.ndarray:
        """Distância de cada ponto ao plano de entrada (``y - y_origem``)."""
        return self.pontos[:, 1] - y_origem


def janela(t, a: float, b: float) -> np.ndarray:
    """Máscara dos instantes em ``[a, b]``, tolerante ao arredondamento de ponto flutuante."""
    t = np.asarray(t, dtype=float)
    return (t >= a - 1e-9) & (t <= b + 1e-9)


def _conferir(k, t, janela_t=None):
    k = np.asarray(k, dtype=float)
    t = np.asarray(t, dtype=float)
    if k.ndim != 2:
        raise ValueError(f"k deve ser (n_instantes, n_pontos), veio {k.shape}")
    if k.shape[0] != t.size:
        raise ValueError(f"k tem {k.shape[0]} instantes e t tem {t.size}")
    if janela_t is None:
        return k, np.ones(t.size, bool)
    m = janela(t, *janela_t)
    if m.sum() < 1:
        raise ValueError(f"janela {janela_t} não tem instante: t vai de {t.min()} a {t.max()}")
    return k, m


def perfil_medio(k: np.ndarray, t, janela_t: tuple[float, float] | None = None) -> np.ndarray:
    """Média temporal de k em cada ponto da linha (o perfil que se lê num gráfico de k contra a distância)."""
    k, m = _conferir(k, t, janela_t)
    return k[m].mean(0)


def pico(k: np.ndarray, t, dabs, janela_t: tuple[float, float] | None = None):
    """Posição do pico: ``(k, distância, índice)`` do maior valor do perfil médio.

    É a definição que os dois estudos do canal usam — e a que diverge quando muda a linha ou a janela, não o solver.
    """
    dabs = np.asarray(dabs, dtype=float)
    prof = perfil_medio(k, t, janela_t)
    if prof.size != dabs.size:
        raise ValueError(f"perfil com {prof.size} pontos e {dabs.size} distâncias")
    i = int(np.argmax(prof))
    return float(prof[i]), float(dabs[i]), i


def pico_em_liquido(k: np.ndarray, t, dabs, alpha: np.ndarray, janela_t: tuple[float, float] | None = None,
                    alpha_min: float = 0.9) -> dict:
    """Pico da média temporal de k **restrito ao líquido** — a mesma linha amostrada em duas fases não se compara.

    Onde uma linha de amostragem cruza a superfície livre, o máximo de k passa a cair em ar: o jato em voo produz k de
    outra ordem de grandeza (sem a dissipação do banho), e o argmax salta para a superfície. Isso não é física nova, é
    a métrica mudando de fase — e basta o nível do banho mudar alguns centímetros para a linha inteira trocar de lado.

    Devolve ``{"s", "k", "i", "alpha", "filtrado", "pontos_liquido", "s_sem_filtro", "k_sem_filtro",
    "alpha_sem_filtro", "motivo"}``. Se nenhum ponto for líquido na janela, ``filtrado`` é ``False`` e o pico volta a
    ser o da linha inteira, com o motivo dito.
    """
    k, m = _conferir(k, t, janela_t)
    dabs = np.asarray(dabs, dtype=float)
    a = np.asarray(alpha, dtype=float)
    if a.shape != np.shape(k):
        raise ValueError(f"alpha com formato {a.shape} e k com {np.shape(k)}")
    prof = k[m].mean(0)
    if prof.size != dabs.size:
        raise ValueError(f"perfil com {prof.size} pontos e {dabs.size} distâncias")
    a_med = a[m].mean(0)
    i = int(np.argmax(prof))
    out = {"s": float(dabs[i]), "k": float(prof[i]), "i": i, "alpha": float(a_med[i]),
           "filtrado": False, "pontos_liquido": int((a_med >= alpha_min).sum()),
           "s_sem_filtro": float(dabs[i]), "k_sem_filtro": float(prof[i]), "alpha_sem_filtro": float(a_med[i]),
           "motivo": None}
    liq = a_med >= alpha_min
    if not liq.any():
        out["motivo"] = f"nenhum ponto com alpha ≥ {alpha_min} nesta janela: a linha está em ar"
        return out
    j = int(np.argmax(np.where(liq, prof, -np.inf)))
    out.update({"s": float(dabs[j]), "k": float(prof[j]), "i": j, "alpha": float(a_med[j]), "filtrado": True})
    return out


def plato(k: np.ndarray, t, dabs, janela_t: tuple[float, float] | None = None, fracao: float = 0.85) -> dict:
    """O quanto "a posição do pico" vale: a faixa em torno do máximo que fica dentro de ``fracao`` dele.
    Devolve ``{"pico": {"k", "s", "i"}, "s_ini", "s_fim", "largura", "margem", "segundo", "passo"}``:

    - ``s_ini``/``s_fim``/``largura``: a faixa contígua em torno do máximo com perfil ≥ ``fracao`` × máximo;
    - ``margem``: quanto o segundo maior valor fica abaixo do maior, em fração (0,004 = 0,4 %);
    - ``segundo``: ``(k, s)`` do segundo ponto mais alto da linha;
    - ``passo``: o passo de amostragem da linha (m), que é a resolução da própria métrica.

    Serve para não comparar posições de pico ao milímetro entre estudos: quando a margem é menor que o ruído do
    campo, um passo de amostragem move o argmax, e a posição deixa de ser um número com significado.
    """
    dabs = np.asarray(dabs, dtype=float)
    prof = perfil_medio(k, t, janela_t)
    if prof.size != dabs.size:
        raise ValueError(f"perfil com {prof.size} pontos e {dabs.size} distâncias")
    i = int(np.argmax(prof))
    ordem = np.argsort(prof)[::-1]
    seg = int(ordem[1]) if ordem.size > 1 else i
    lim = fracao * prof[i]
    lo = hi = i
    while lo - 1 >= 0 and prof[lo - 1] >= lim:
        lo -= 1
    while hi + 1 < prof.size and prof[hi + 1] >= lim:
        hi += 1
    passo = float(np.median(np.diff(dabs))) if dabs.size > 1 else 0.0
    return {"pico": {"k": float(prof[i]), "s": float(dabs[i]), "i": i},
            "s_ini": float(dabs[lo]), "s_fim": float(dabs[hi]), "largura": float(dabs[hi] - dabs[lo]),
            "margem": float(1.0 - prof[seg] / prof[i]) if prof[i] else None,
            "segundo": (float(prof[seg]), float(dabs[seg])), "passo": passo}


def joelho(k: np.ndarray, t, dabs, s_ref: float, fator: float = 2.0,
           janela_t: tuple[float, float] | None = None) -> float | None:
    """Onde o perfil médio cruza ``fator`` × o valor em ``s_ref`` (m).

    Num caso de jato entrando num banho, o perfil de k sobe em degrau quando o jato toca a superfície; o ponto em que
    ele dobra em relação ao nível de montante marca esse toque com mais fidelidade do que a geometria, porque inclui o
    freio do ar. Devolve ``None`` se o perfil não cruza.
    """
    dabs = np.asarray(dabs, dtype=float)
    prof = perfil_medio(k, t, janela_t)
    i0 = int(np.argmin(np.abs(dabs - s_ref)))
    alvo = fator * prof[i0]
    for i in range(i0, prof.size - 1):
        if prof[i] < alvo <= prof[i + 1]:
            return float(dabs[i] + (dabs[i + 1] - dabs[i]) * (alvo - prof[i]) / (prof[i + 1] - prof[i]))
    return None


def quantizacao(dabs) -> dict:
    """O quanto a linha consegue representar: ``{"pontos", "passo_medio", "passo_max", "fracao_em_passos_grandes"}``.

    A posição do pico é lida numa sequência de pontos amostrados (centros de célula), então o maior vão entre pontos
    consecutivos é a **resolução da métrica**: uma diferença de posição menor que isso não é distinguível do nada. Sai
    do próprio arquivo de resultados — não precisa de rodada nova.
    """
    d = np.sort(np.asarray(dabs, dtype=float))
    if d.size < 2:
        return {"pontos": int(d.size), "passo_medio": None, "passo_max": None, "fracao_em_passos_grandes": None}
    p = np.diff(d)
    return {"pontos": int(d.size), "passo_medio": float(p.mean()), "passo_max": float(p.max()),
            "fracao_em_passos_grandes": float((p > 2.0 * np.median(p)).mean())}


def media_faixa(k: np.ndarray, t, dabs, s0: float, s1: float,
                janela_t: tuple[float, float] | None = None) -> float | None:
    """k médio na faixa de distância ``[s0, s1]`` e na janela de tempo (``None`` se a faixa não tem ponto)."""
    dabs = np.asarray(dabs, dtype=float)
    m = (dabs >= s0) & (dabs <= s1)
    if not m.any():
        return None
    prof = perfil_medio(k, t, janela_t)
    return float(prof[m].mean())


def picos_por_linha(series: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                    janela_t: tuple[float, float] | None = None) -> dict:
    """``{linha: (k, t, dabs)}`` -> ``{linha: {"k": ..., "s": ..., "i": ...}}``; linhas sem série ficam de fora."""
    out = {}
    for nome, (k, t, dabs) in series.items():
        k_pico, s_pico, i = pico(k, t, dabs, janela_t)
        out[nome] = {"k": k_pico, "s": s_pico, "i": i}
    return out


def origem_da_linha(centros, inicio: float = 0.0) -> float:
    """Quanto o **primeiro ponto efetivamente amostrado** está adiante do início nominal da linha.

    Uma linha amostrada em centros de célula começa onde existe célula, não onde a linha começa: no caso do canal, os
    primeiros 1,36 m das linhas de escória e 1,82 m das de gusa não têm ponto válido. Quem zera a distância no primeiro
    ponto (uma convenção comum) publica posições 1,36 m menores que quem conta do início do domínio — e a diferença some
    da análise, virando "divergência física" de mais de um metro.

    Devolve ``primeiro_ponto - inicio``.
    """
    import numpy as np

    c = np.asarray(centros, dtype=float)
    if c.size == 0:
        raise ValueError("linha sem pontos")
    return float(c.min() - inicio)


def deslocamento_de_origem(s_a, k_a, s_b, k_b, faixa: tuple[float, float] | None = None,
                          passo: float | None = None) -> dict:
    """Quanto o perfil **b** está deslocado em relação ao perfil **a** — o metro zero de cada tabela, medido dos dados.

    Recebe dois perfis já médios no tempo (``s``, ``k``) e procura o deslocamento que sobrepõe um ao outro. É o teste
    que distingue "os dois escoamentos diferem" de "as duas tabelas contam a distância de pontos diferentes": no segundo
    caso, um único deslocamento alinha os perfis com erro pequeno, e esse deslocamento é a diferença de convenção.

    ``faixa`` limita a busca (padrão: ±10 % da extensão do perfil); ``passo``, se dado, arredonda o resultado para a
    grade de amostragem — o deslocamento entre duas convenções é sempre um número inteiro de passos.

    Devolve ``{"deslocamento", "erro_rms", "erro_sem_deslocar", "sobreposicao", "e_o_mesmo_perfil"}``. ``sobreposicao``
    é a fração do perfil que cai dentro dos dois trechos depois de deslocar; ``e_o_mesmo_perfil`` é ``True`` quando o
    deslocamento reduz o erro pela metade e sobra pouco erro — o sinal de que os dois lados mediram a mesma coisa.
    """
    s_a, k_a = np.asarray(s_a, float), np.asarray(k_a, float)
    s_b, k_b = np.asarray(s_b, float), np.asarray(k_b, float)
    if s_a.size != k_a.size or s_b.size != k_b.size:
        raise ValueError("cada perfil precisa do mesmo número de distâncias e de valores")
    if s_a.size < 3 or s_b.size < 3:
        raise ValueError("perfil com menos de três pontos: não dá para alinhar")
    extensao = float(min(s_a.max() - s_a.min(), s_b.max() - s_b.min()))
    lo, hi = faixa if faixa else (-0.1 * extensao, 0.1 * extensao)
    # os dois perfis precisam estar definidos no mesmo trecho depois de deslocar
    ini = max(s_a.min(), s_b.min() + lo)
    fim = min(s_a.max(), s_b.max() + hi)
    if fim <= ini:
        raise ValueError("os perfis não se sobrepõem: nenhum deslocamento na faixa os alinha")
    grade = np.linspace(ini, fim, 400)
    a_grade = np.interp(grade, s_a, k_a)

    def _erro(desloc: float) -> float:
        return float(np.sqrt(np.mean((np.interp(grade - desloc, s_b, k_b) - a_grade) ** 2)))

    grade_desl = np.arange(lo, hi + 1e-12, max((hi - lo) / 800, 1e-4))
    erros = np.array([_erro(d) for d in grade_desl])
    i = int(np.argmin(erros))
    desloc = float(grade_desl[i])
    if passo:
        desloc = round(desloc / passo) * passo
    sem = _erro(0.0)
    com = _erro(desloc)
    escala = float(np.std(a_grade)) or 1.0
    return {"deslocamento": desloc, "erro_rms": com, "erro_sem_deslocar": sem,
            "sobreposicao": float((fim - ini) / extensao) if extensao else 0.0,
            "e_o_mesmo_perfil": bool(com <= 0.5 * sem and com / escala <= 0.1)}


def comparar_picos(picos: Mapping[str, Mapping[str, float]], referencia: Mapping[str, Mapping[str, Sequence[float]]],
                   passo: float | None = None, deslocamento_ref: Mapping[str, float] | float | None = None) -> list[dict]:
    """Compara os picos com uma referência publicada, linha a linha.

    ``picos``: ``{linha: {"k": ..., "s": ...}}`` (o que sai de :func:`picos_por_linha`).
    ``referencia``: ``{linha: (k, s)}`` — a tabela do outro estudo ou a medição.
    ``passo`` (m): se dado, também devolve o desvio em número de pontos de amostragem, que é a unidade útil quando as
    duas malhas usam a mesma grade de y (todo Δs múltiplo do passo é o sinal de que a diferença é de trajetória, não
    de eixo).
    ``deslocamento_ref`` (m): soma à posição da referência antes de comparar. Serve para pôr as duas tabelas na mesma
    origem de distância quando uma delas conta do primeiro ponto amostrado e a outra do início da linha — ver
    :func:`origem_da_linha`. Pode ser um número (todas as linhas) ou ``{linha: valor}``.

    Devolve uma lista de dicionários com ``linha``, ``k``, ``s``, ``k_ref``, ``s_ref``, ``ds``, ``razao_k`` e,
    quando ``passo`` é dado, ``passos``. Linha que falta de um dos lados entra com ``None`` nos campos ausentes.
    """
    linhas = list(dict.fromkeys([*picos, *referencia]))
    out = []
    for nome in linhas:
        p = picos.get(nome) or {}
        ref = referencia.get(nome)
        k_ref, s_ref = (float(ref[0]), float(ref[1])) if ref else (None, None)
        if s_ref is not None and deslocamento_ref is not None:
            s_ref += (float(deslocamento_ref[nome]) if isinstance(deslocamento_ref, Mapping)
                      else float(deslocamento_ref))
        s = p.get("s")
        ds = (float(s) - s_ref) if (s is not None and s_ref is not None) else None
        out.append({"linha": nome, "k": p.get("k"), "s": s, "k_ref": k_ref, "s_ref": s_ref, "ds": ds,
                    "razao_k": (p["k"] / k_ref) if (p.get("k") is not None and k_ref) else None,
                    "passos": (ds / passo) if (ds is not None and passo) else None})
    return out


def desvio_mediano_passos(comparacao: Iterable[dict]) -> float | None:
    """|Δs| mediano em passos de amostragem, sobre as linhas que têm os dois lados."""
    v = [abs(c["passos"]) for c in comparacao if c.get("passos") is not None]
    return float(np.median(v)) if v else None
