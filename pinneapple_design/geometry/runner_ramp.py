"""Rampa do canal de corrida (família Capture02): geometria paramétrica e sólido fechado.

Um só lugar para os parâmetros da rampa. Nasceu de uma divergência real: dois estudos simularam "a mesma rampa" e
chegaram a escoamentos diferentes porque cada equipe materializou os parâmetros à sua maneira — a nossa subia do
piso a partir de 3,5 m, 100/250/400 mm acima dele; o sólido do CAD desce da crista, que fica acima do nível do banho,
começa em 4,228 m e tem pé residual de 472 a 772 mm. Fixar a definição aqui (coordenadas globais, sem altura relativa
ao piso) é a correção de fundo.

Convenções, todas em metros e no referencial do CFD do canal (`canal.stp`):

- ``s`` é a distância ao plano de entrada, ``s = y - y_origem`` com ``y_origem = -7.429``; ``s = 0`` é o plano de
  entrada, que é o "centroide da entrada" da outra equipe a 1 mm (o centroide fica em ``y = -7.427945``).
- ``z_crista`` e ``z_pe`` são cotas **globais** do CFD, não alturas acima do piso.
- a rampa é centrada no eixo lateral do canal, ``x_eixo = -2.111`` (paredes em ``x = -2.544`` e ``-1.678``).

Forma (a do CAD, descrita pela outra equipe): face vertical a montante na crista; face reta descendente com a
inclinação publicada; pé curvo, ocupando 35 % do trecho inclinado, com tangência contínua no patamar; patamar
horizontal; face vertical a jusante. O trecho inclinado é resolvido para cair exatamente ``z_crista - z_pe``, o que
reproduz o patamar de 0,25 m das cinco variantes.

O piso é o do CAD, linear em ``s`` (3,5 m: -2,848; 5 m: -2,883), conferido contra o STEP até 9 m dentro de 2 mm.

Uso::

    from pinneapple_design.geometry.runner_ramp import CASOS, perfil, solido, validar
    perfil(CASOS["V02"], 5.0)                 # cota do topo da rampa em s = 5,0 m
    verts, faces = solido(CASOS["V02"])       # sólido fechado, pronto para STL/STL binário
    validar()                                 # comparação com as cotas publicadas do CAD

``validar()`` é o teste de referência desta geometria: as alturas da crista e do pé acima do fundo, medidas no
sólido, têm de bater com as do CAD publicado dentro de 3 mm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict


def piso(s):
    """Cota do piso do canal em s (m). Linear no trecho do CAD, conferido contra canal.stp até 9 m (2 mm)."""
    import numpy as np
    s = np.asarray(s, dtype=float)
    z = -2.848 + (s - 3.5) * (-2.883 + 2.848) / 1.5
    return float(z) if z.ndim == 0 else z


def perfil(r: "Rampa", s):
    """Cota do topo da rampa em s, vetorizado (``nan`` fora do trecho). Ordem das variantes de ``CASOS``."""
    import numpy as np
    a, reto, pe, _ = r.trechos()
    s = np.asarray(s, dtype=float)
    z = np.full(s.shape, np.nan)
    m = (s >= r.s_inicio) & (s < r.s_inicio + reto)
    z[m] = r.z_crista - a * (s[m] - r.s_inicio)
    m2 = (s >= r.s_inicio + reto) & (s < r.s_inicio + reto + pe)
    u = (s[m2] - r.s_inicio - reto) / pe
    z[m2] = r.z_crista - a * reto - a * pe * (u - u * u / 2)
    m3 = (s >= r.s_inicio + reto + pe) & (s <= r.s_fim)
    z[m3] = r.z_pe
    return float(z) if z.ndim == 0 else z


def dentro(r: "Rampa", x, y, z) -> object:
    """Máscara vetorizada: os pontos (x, y, z) estão dentro do sólido? (mesma regra do ``Rampa.contem``)"""
    import numpy as np
    s = np.asarray(y, dtype=float) - r.y_origem
    return ((s >= r.s_inicio) & (s <= r.s_fim) & (np.abs(np.asarray(x, dtype=float) - r.x_eixo) <= r.largura / 2)
            & (np.asarray(z, dtype=float) >= piso(s) - 1e-9) & (np.asarray(z, dtype=float) <= perfil(r, s) + 1e-12))


@dataclass(frozen=True)
class Rampa:
    """Uma variante da rampa, em coordenadas globais do CFD (metros)."""

    caso: str
    comprimento: float      # L: do início ao fim em s (inclui o patamar)
    largura: float          # largura lateral, centrada no eixo
    face_reta_deg: float    # inclinação da face reta (não é dZ/L total)
    s_inicio: float         # início em s (face vertical a montante, na crista)
    s_fim: float            # fim em s (fim do patamar)
    z_crista: float         # cota global do topo a montante
    z_pe: float             # cota global do patamar a jusante
    y_origem: float = -7.429
    x_eixo: float = -2.111
    fracao_pe_curvo: float = 0.35

    # --- derivados -------------------------------------------------------------------------------------------
    @property
    def dz(self) -> float:
        return self.z_crista - self.z_pe

    def trechos(self) -> tuple[float, float, float, float]:
        """(inclinação da face reta, comprimento reto, comprimento do pé curvo, patamar)."""
        a = math.tan(math.radians(self.face_reta_deg))
        inclinado = self.dz / (a * (1 - self.fracao_pe_curvo + self.fracao_pe_curvo / 2))
        reto = (1 - self.fracao_pe_curvo) * inclinado
        pe = self.fracao_pe_curvo * inclinado
        return a, reto, pe, self.comprimento - inclinado

    def topo(self, s: float) -> float:
        """Cota do topo do sólido em s (``nan`` fora de [s_inicio, s_fim])."""
        return float(perfil(self, s))

    def contem(self, x: float, y: float, z: float) -> bool:
        """O ponto (x, y, z) está dentro do sólido?"""
        s = y - self.y_origem
        if s < self.s_inicio or s > self.s_fim or abs(x - self.x_eixo) > self.largura / 2:
            return False
        return piso(s) - 1e-9 <= z <= self.topo(s) + 1e-12

    def alturas(self) -> tuple[float, float]:
        """(altura da crista sobre o fundo, altura residual do pé), em mm — o que o CAD publica."""
        return ((self.z_crista - piso(self.s_inicio)) * 1000, (self.z_pe - piso(self.s_fim)) * 1000)


# As cinco variantes do CAD, com as cotas publicadas pelo estudo paralelo (seção 03 do relatório de 06/11).
CASOS: dict[str, Rampa] = {
    "V02": Rampa("V02", 2.500, 0.600, 10.68, 4.228, 6.728, -2.100, -2.450),
    "V03": Rampa("V03", 2.500, 0.500, 10.68, 4.228, 6.728, -1.800, -2.150),
    "V04": Rampa("V04", 3.750, 0.600, 6.91, 4.228, 7.978, -1.950, -2.300),
    "V05": Rampa("V05", 1.875, 0.600, 14.63, 4.228, 6.103, -1.950, -2.300),
    "V06": Rampa("V06", 2.500, 0.600, 10.68, 4.978, 7.478, -1.950, -2.300),
}

# alturas publicadas (mm): altura da crista sobre o fundo, altura residual do pé
PUBLICADO = {"V02": (764.9, 472.1), "V03": (1064.9, 772.1), "V04": (914.9, 650.6),
             "V05": (914.9, 607.8), "V06": (932.1, 639.2)}
TOLERANCIA_MM = 3.0


def solido(r: Rampa, n_s: int = 60) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Sólido fechado da rampa: (vértices, faces triangulares), pronto para STL.

    Grade em ``s`` (``n_s`` estações) por duas colunas laterais, com as superfícies de topo, fundo, laterais e as
    duas faces verticais (montante e jusante). Cada aresta aparece em exatamente duas faces (fechado).
    """
    xs = (r.x_eixo - r.largura / 2, r.x_eixo + r.largura / 2)
    ss = [r.s_inicio + i * (r.s_fim - r.s_inicio) / n_s for i in range(n_s + 1)]
    verts: list[tuple[float, float, float]] = []
    topo_i: list[list[int]] = []            # [i][j]
    piso_i: list[list[int]] = []
    for s in ss:
        y = r.y_origem + s
        fila_t, fila_p = [], []
        for x in xs:
            fila_t.append(len(verts)); verts.append((x, y, r.topo(s)))
        for x in xs:
            fila_p.append(len(verts)); verts.append((x, y, piso(s)))
        topo_i.append(fila_t); piso_i.append(fila_p)
    faces: list[tuple[int, int, int]] = []

    def quad(a: int, b: int, c: int, d: int):
        """Quad a-b-c-d (sentido anti-horário visto de fora) em dois triângulos."""
        faces.extend([(a, b, c), (a, c, d)])

    for i in range(n_s):                    # topo (normal para cima) e fundo (para baixo)
        quad(topo_i[i][0], topo_i[i][1], topo_i[i + 1][1], topo_i[i + 1][0])
        quad(piso_i[i][0], piso_i[i + 1][0], piso_i[i + 1][1], piso_i[i][1])
    for i in range(n_s):                    # laterais
        quad(topo_i[i][0], topo_i[i + 1][0], piso_i[i + 1][0], piso_i[i][0])
        quad(topo_i[i][1], piso_i[i][1], piso_i[i + 1][1], topo_i[i + 1][1])
    quad(topo_i[0][0], piso_i[0][0], piso_i[0][1], topo_i[0][1])                    # face de montante
    quad(topo_i[-1][1], piso_i[-1][1], piso_i[-1][0], topo_i[-1][0])                # face de jusante
    return verts, faces


def facetas_abertas(faces: list[tuple[int, int, int]]) -> int:
    """Quantas arestas não aparecem em exatamente duas faces (0 = sólido fechado)."""
    from collections import Counter
    c = Counter()
    for a, b, d in faces:
        for e in ((a, b), (b, d), (d, a)):
            c[tuple(sorted(e))] += 1
    return sum(1 for v in c.values() if v != 2)


def validar() -> dict:
    """Compara o sólido reconstruído com as cotas publicadas do CAD.

    Devolve ``{"ok": bool, "casos": {...}, "pior_desvio_mm": float}``. É a referência fechada desta geometria: o
    desvio de cada altura tem de ficar dentro de ``TOLERANCIA_MM``.
    """
    out, pior = {}, 0.0
    for k, r in CASOS.items():
        h_crista, h_pe = r.alturas()
        pub_c, pub_p = PUBLICADO[k]
        d_c, d_p = h_crista - pub_c, h_pe - pub_p
        pior = max(pior, abs(d_c), abs(d_p))
        _, reto, pe, patamar = r.trechos()
        out[k] = {"altura_crista_mm": round(h_crista, 1), "publicado_crista_mm": pub_c, "desvio_crista_mm": round(d_c, 1),
                  "altura_pe_mm": round(h_pe, 1), "publicado_pe_mm": pub_p, "desvio_pe_mm": round(d_p, 1),
                  "patamar_m": round(patamar, 3), "reto_m": round(reto, 3), "pe_curvo_m": round(pe, 3)}
    return {"ok": pior <= TOLERANCIA_MM, "tolerancia_mm": TOLERANCIA_MM, "pior_desvio_mm": round(pior, 1), "casos": out}


def parametros(r: Rampa) -> dict:
    """Os parâmetros da variante como dicionário (para deck, JSON de caso ou relatório)."""
    return asdict(r)
