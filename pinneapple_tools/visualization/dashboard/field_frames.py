"""Quadros de um campo salvo (superfície livre ou plano de corte) para o ``Dashboard``, e tabelas de relatório.

Duas peças que faltavam para um painel ou relatório montado com o :class:`~.spec.Dashboard`:

* :func:`frames_from_field` — transforma o ``.npz`` que as rodadas gravam (``sup_xyz_i``/``sup_k_i`` e
  ``corte_k_i`` por instante) numa lista de quadros RGB, um por instante, projetando os pontos no plano
  longitudinal (y, z) e colorindo por k. É a "animação" do relatório, sem depender de renderizador 3-D: cada quadro é
  a vista lateral do domínio com a superfície (ou o corte) colorida pelo campo.
* :class:`Table` — tabela HTML de relatório (cabeçalho, linhas, alinhamento, destaque), para pôr no painel ao lado do
  ``Dashboard``, que só tem tiles e gráficos.

Nada aqui lê solvers nem conhece um caso: entram arrays e um ``.npz``, sai ``(t, quadros)`` e HTML.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

STOPS = np.array([[0.10, 0.18, 0.42], [0.10, 0.52, 0.45], [0.85, 0.72, 0.15], [0.75, 0.15, 0.12]])


def cor_por_valor(v: np.ndarray) -> np.ndarray:
    """Colormap azul→verde→amarelo→vermelho para valores em [0, 1] (sem dependência de matplotlib)."""
    v = np.clip(np.asarray(v, float), 0.0, 1.0)
    x = v * (len(STOPS) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(STOPS) - 2)
    f = (x - i)[..., None]
    return (STOPS[i] * (1 - f) + STOPS[i + 1] * f).astype(np.float32)


def _imagem(y: np.ndarray, z: np.ndarray, k: np.ndarray, lim: tuple[float, float, float, float],
            larg: int = 240, alt: int = 110) -> np.ndarray:
    """Projeta os pontos (y, z, k) numa grade e devolve a imagem RGB."""
    y0, y1, z0, z1 = lim
    iy = np.clip(((y - y0) / max(y1 - y0, 1e-9) * (larg - 1)).astype(int), 0, larg - 1)
    iz = np.clip(((z1 - z) / max(z1 - z0, 1e-9) * (alt - 1)).astype(int), 0, alt - 1)
    soma = np.zeros((alt, larg))
    cnt = np.zeros((alt, larg))
    np.add.at(soma, (iz, iy), k)
    np.add.at(cnt, (iz, iy), 1.0)
    val = np.where(cnt > 0, soma / np.maximum(cnt, 1), np.nan)
    vazio = np.isnan(val)
    lo, hi = np.nanmin(val), np.nanmax(val)
    z01 = (val - lo) / (hi - lo) if hi > lo else np.zeros_like(val)
    img = cor_por_valor(np.nan_to_num(z01))
    img[vazio] = 0.97                                   # onde não há ponto: branco (não é k = 0)
    return (img * 255).astype(np.uint8)


def frames_from_field(npz: str | Path, campo: str = "superficie", larg: int = 240, alt: int = 110,
                      limite: tuple[float, float, float, float] | None = None):
    """``(t, quadros)`` a partir de um ``.npz`` de campos gravados.

    ``campo="superficie"`` usa ``sup_xyz_i``/``sup_k_i``; ``campo="corte"`` usa ``corte_xyz``/``corte_k_i``. A escala
    de cor é comum a todos os quadros (o ``limite`` é calculado uma vez sobre todos os instantes), senão a cor mudaria
    de quadro para quadro e a animação mentiria.
    """
    d = np.load(Path(npz), allow_pickle=True)
    t = np.asarray(d["tempos"], float)
    xs, ks = [], []
    for i in range(len(t)):
        chave_k = f"sup_k_{i}" if campo == "superficie" else f"corte_k_{i}"
        chave_xyz = f"sup_xyz_{i}" if campo == "superficie" else "corte_xyz"
        if chave_k not in d or chave_xyz not in d:
            continue
        xs.append(np.asarray(d[chave_xyz], float))
        ks.append(np.asarray(d[chave_k], float))
    if not xs:
        raise ValueError(f"{npz}: sem campos de {campo!r} (chaves sup_k_* / corte_k_*)")
    todos = np.vstack(xs)
    if limite is None:
        limite = (float(todos[:, 1].min()), float(todos[:, 1].max()),
                  float(todos[:, 2].min()), float(todos[:, 2].max()))
    quadros = [_imagem(x[:, 1], x[:, 2], k, limite, larg, alt) for x, k in zip(xs, ks)]
    return t[:len(quadros)], quadros


@dataclass
class Table:
    """Tabela HTML de relatório: cabeçalho, linhas e alinhamento por coluna."""

    titulo: str
    colunas: Sequence[str]
    linhas: Sequence[Sequence[object]] = field(default_factory=list)
    alinhamento: Sequence[str] | None = None          # "l" | "r" | "c" por coluna
    nota: str = ""
    destaque: Sequence[int] = field(default_factory=list)   # índices de linha em negrito

    def to_html(self) -> str:
        al = list(self.alinhamento or ["l"] + ["r"] * (len(self.colunas) - 1))
        mapa = {"l": "left", "r": "right", "c": "center"}
        th = "".join(f'<th align="{mapa.get(a, "left")}">{html.escape(str(c))}</th>' for c, a in zip(self.colunas, al))

        def cel(v):
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.4g}"
            return html.escape(str(v))

        tr = []
        for i, ln in enumerate(self.linhas):
            peso = " style='font-weight:600'" if i in self.destaque else ""
            tr.append(f"<tr{peso}>" + "".join(f'<td align="{mapa.get(a, "left")}">{cel(v)}</td>'
                                              for v, a in zip(ln, al)) + "</tr>")
        nota = f"<p style='font-size:12px;color:#666'>{html.escape(self.nota)}</p>" if self.nota else ""
        return (f"<h3>{html.escape(self.titulo)}</h3>"
                "<table cellpadding='5' cellspacing='0' border='0' "
                "style='font:13px system-ui;border-collapse:collapse'>"
                f"<thead><tr style='border-bottom:1px solid #bbb'>{th}</tr></thead>"
                f"<tbody>{''.join(tr)}</tbody></table>{nota}")
