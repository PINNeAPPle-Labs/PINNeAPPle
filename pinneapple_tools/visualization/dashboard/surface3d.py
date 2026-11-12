"""Quadros 3D de um escoamento com superfície livre: contorno do CAD, superfície colorida por um campo e a pluma.

Generaliza a animação que nasceu no estudo do canal de corrida (``estudos/canal_jato_doe/ferramentas/animacao_3d.py``)
para qualquer caso que grave o mesmo formato de ``campos.npz``:

- ``tempos`` — um instante por quadro, em segundos;
- ``sup_xyz_<i>`` / ``sup_k_<i>`` — pontos da superfície livre (0,1 < alpha < 0,9) e o valor do campo neles;
- ``corte_xyz`` / ``corte_alpha_<i>`` / ``corte_umag_<i>`` — um plano do jato, com alpha e |U| em cada ponto.

O que é do caso (contorno, escala de cor, nível do banho, câmera) entra por parâmetro numa :class:`Cena3D`; nada aqui
sabe de canal, rampa ou metal líquido.

    from pinneapple_tools.visualization.dashboard import (Cena3D, contorno_de_step, quadros_de_campos,
                                                          animacao_de_campos)
    linhas = contorno_de_step("canal.stp", "figuras/contorno.npz")   # lido uma vez, depois do cache
    cena = Cena3D(linhas=linhas, kmax=0.6)
    animacao_de_campos("resultados/campos/R0.npz", "relatorios/R0", rid="R0", cena=cena)

Escreve ``<saida>/<rid>_animacao.mp4`` (H.264, ``fps`` quadros por segundo simulado) e ``<rid>_animacao.png`` (o painel
de instantes). Nada é interpolado: cada quadro é um instante gravado.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

FUNDO = "#575757"
PONTO = 10.0


@dataclass(frozen=True)
class Cena3D:
    """O que é específico do caso: o desenho de fundo, os limites do eixo e as escalas.

    ``linhas`` são polilinhas em metros (contorno do CAD, rampa, o que se quiser desenhar em branco no fundo).
    ``lims`` é ``((x0, x1), (y0, y1), (z0, z1))``; se ``None``, sai do próprio contorno. ``nivel`` é a altura
    (em z) acima da qual a pluma é desenhada — normalmente a superfície livre em t inicial; se ``None``, sai da
    mediana do primeiro quadro. ``campo`` é o nome do campo colorido (só rótulo) e ``kmax`` o topo da escala.
    """

    linhas: list = field(default_factory=list)
    lims: tuple | None = None
    nivel: float | None = None
    kmax: float = 1.0
    umag_min: float = 0.0
    campo: str = "k"
    unidade: str = "m²/s²"
    vista: dict = field(default_factory=lambda: {"elev": 16.0, "azim": -58.0})
    zoom: float = 1.6
    fundo: str = FUNDO
    alpha_corte: float = 0.5
    ponto: float = PONTO

    def limites(self) -> tuple:
        if self.lims is not None:
            return self.lims
        if not self.linhas:
            raise ValueError("Cena3D sem linhas e sem lims: não há de onde tirar os limites")
        todos = np.concatenate(self.linhas)
        return tuple((float(todos[:, i].min()), float(todos[:, i].max())) for i in range(3))


def _expandir(lims: tuple, folga: float = 0.5) -> tuple:
    """Garante que nenhum eixo tenha extensão zero: um contorno plano (tudo em x = 0, por exemplo) deixaria a
    projeção 3D singular e o quadro sairia em branco."""
    out = []
    for lo, hi in lims:
        if abs(hi - lo) < 1e-9:
            lo, hi = lo - folga, hi + folga
        out.append((float(lo), float(hi)))
    return tuple(out)


def contorno_de_step(stp: str | Path, cache: str | Path | None = None, n_curva: int = 40) -> list:
    """Arestas de um STEP como polilinhas em metros. Com ``cache``, lê o primeiro e guarda o resultado.

    O STEP costuma vir em milímetros, então as coordenadas são divididas por 1000. Reta vira 2 pontos, curva vira
    ``n_curva``.
    """
    stp = Path(stp)
    if cache and Path(cache).exists():
        z = np.load(cache)
        return [z[k] for k in sorted(z.files, key=lambda s: int(s[1:]))]
    import cadquery as cq  # só quando não há cache

    sh = cq.importers.importStep(str(stp)).val()
    linhas = []
    for e in sh.Edges():
        n = 2 if e.geomType() == "LINE" else n_curva
        pts = np.array([e.positionAt(t).toTuple() for t in np.linspace(0, 1, n)]) / 1000.0
        linhas.append(pts.astype(np.float32))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{f"e{i}": p for i, p in enumerate(linhas)})
    return linhas


def quadros_de_campos(campos: str | Path) -> dict:
    """``{"quadros": [{"t", "sup", "campo", "alpha", "umag"}, ...], "corte": ndarray}`` a partir do ``campos.npz``.

    Instante sem superfície gravada é pulado; se nenhum sobrar, devolve lista vazia (e quem chamar decide o erro).
    O campo colorido é lido de ``sup_k_<i>`` — o nome vem do formato do arquivo, não do significado.
    """
    z = np.load(campos)
    quadros = []
    for i, t in enumerate(z["tempos"]):
        if f"sup_xyz_{i}" not in z.files:
            continue
        alpha = z[f"corte_alpha_{i}"]
        quadros.append({
            "t": float(t),
            "sup": z[f"sup_xyz_{i}"].astype(np.float32),
            "campo": z[f"sup_k_{i}"].astype(np.float32),
            "alpha": alpha.astype(np.float32) / (255.0 if alpha.dtype == np.uint8 else 1.0),
            "umag": z[f"corte_umag_{i}"].astype(np.float32),
        })
    return {"quadros": quadros, "corte": z["corte_xyz"].astype(np.float32)}


def _mapa(cena: Cena3D):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(f"{cena.campo}_branco_vermelho",
                                             ["#ffffff", "#ffe9a8", "#f4a23c", "#d6402b", "#8e1212"])


def _pluma(d: dict, q: dict, cena: Cena3D) -> np.ndarray:
    c = d["corte"]
    m = (q["alpha"] > cena.alpha_corte) & (c[:, 2] > (cena.nivel or 0.0)) & (q["umag"] > cena.umag_min)
    return c[m]


def _eixos(fig, pos, cena: Cena3D, lims):
    ax = fig.add_axes(pos, projection="3d")
    ax.set_facecolor(cena.fundo)
    for p in cena.linhas:
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="white", lw=0.6, alpha=0.9)
    (x0, x1), (y0, y1), (z0, z1) = lims
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, (z1 - z0)), zoom=cena.zoom)
    ax.set_proj_type("persp", focal_length=0.5)
    ax.view_init(**cena.vista)
    ax.set_axis_off()
    return ax


def _quadro(ax, d: dict, q: dict, cena: Cena3D, cmap):
    s = q["sup"]
    k = np.clip(q["campo"], 0, cena.kmax)
    o = np.argsort(k)                                      # os pontos de valor alto ficam por cima
    ax.scatter(s[o, 0], s[o, 1], s[o, 2], c=k[o], cmap=cmap, vmin=0, vmax=cena.kmax, s=cena.ponto, linewidths=0,
               alpha=0.95, depthshade=False)
    p = _pluma(d, q, cena)
    if len(p):
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color="#c0392b", s=cena.ponto * 1.6, linewidths=0, alpha=0.06,
                   depthshade=False)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def gerar(rid: str, campos: str | Path, saida: str | Path, cena: Cena3D, fps: int = 4,
          painel=(1, 5, 10, 30), titulo: str | None = None) -> dict:
    """MP4 com um quadro por instante gravado + PNG com até quatro instantes do ``painel``."""
    plt = _pyplot()
    d = quadros_de_campos(campos)
    if not d["quadros"]:
        raise ValueError(f"{campos}: nenhum instante com superfície gravada")
    lims = _expandir(cena.limites())
    if cena.nivel is None:
        cena = replace(cena, nivel=float(np.median(d["quadros"][0]["sup"][:, 2])))
    cmap = _mapa(cena)
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    legenda = titulo or rid

    tmp = Path(tempfile.mkdtemp(prefix=f"anim_{rid}_"))
    try:
        for n, q in enumerate(d["quadros"]):
            fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=cena.fundo)
            ax = _eixos(fig, [0.0, 0.0, 1.0, 1.0], cena, lims)
            _quadro(ax, d, q, cena, cmap)
            fig.text(0.02, 0.95, f"{legenda} · t = {q['t']:.0f} s", color="white", fontsize=14, weight="bold")
            fig.text(0.02, 0.91, f"superfície livre colorida por {cena.campo} (0 a {cena.kmax:g} {cena.unidade}); "
                                 "pluma do jato em vermelho", color="#e0e0e0", fontsize=9)
            fig.savefig(tmp / f"q_{n:03d}.png", facecolor=cena.fundo)
            plt.close(fig)
        mp4 = saida / f"{rid}_animacao.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "q_%03d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        str(mp4)], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    sel = [q for t in painel for q in d["quadros"] if abs(q["t"] - t) < 0.01]
    fig = plt.figure(figsize=(12, 6.6), dpi=130, facecolor=cena.fundo)
    for i, q in enumerate(sel[:4]):
        ax = _eixos(fig, [0.5 * (i % 2), 0.5 * (1 - i // 2), 0.5, 0.48], cena, lims)
        _quadro(ax, d, q, cena, cmap)
        fig.text(0.5 * (i % 2) + 0.02, 0.5 * (1 - i // 2) + 0.44, f"t = {q['t']:.0f} s", color="white",
                 fontsize=11, weight="bold")
    png = saida / f"{rid}_animacao.png"
    fig.savefig(png, facecolor=cena.fundo)
    plt.close(fig)
    return {"mp4": mp4, "png": png, "quadros": len(d["quadros"]), "tempos": [q["t"] for q in d["quadros"]]}


def animacao_de_campos(campos: str | Path, saida: str | Path, rid: str, cena: Cena3D, **kw) -> dict:
    """:func:`gerar` com o contorno já na cena; atalho para o caso comum (um ``campos.npz`` por rodada)."""
    return gerar(rid, campos, saida, cena, **kw)


__all__ = ["Cena3D", "contorno_de_step", "quadros_de_campos", "gerar", "animacao_de_campos"]
