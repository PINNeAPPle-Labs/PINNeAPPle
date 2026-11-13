"""Geração e verificação de malha com o OpenFOAM.

Três camadas, cada uma fazendo uma coisa só:

- :mod:`.registry` — **o que existe nesta instalação**: ``blockMesh``, ``snappyHexMesh``, ``extrudeMesh``,
  ``refineMesh``, ``topoSet``, ``createPatch``, ``checkMesh`` e, se instalados, ``cartesianMesh``/``tetMesh`` do
  cfMesh. Nada é prometido antes de ser detectado;
- :mod:`.blockmesh` — o **fundo estruturado** (``blockMeshDict``) a partir de uma caixa, com expansão por direção e
  fronteiras nomeadas;
- :mod:`.snappy` — o **refino em volta da geometria** (``snappyHexMeshDict``) a partir de STL, regiões e camadas.

O estudo de refino (``estudo_de_refino``) monta as três malhas que alimentam o GCI de
``pinneapple_analysis.validation.mesh_adequacy``: sem ele, "a malha está boa" é opinião.

    from pinneapple_simulation.meshing import Caixa, SnappyConfig, detectar, estudo_de_refino
    [m["nome"] for m in detectar() if m["disponivel"]]
    plano = estudo_de_refino(caso_base, niveis=(0, 1, 2))     # 3 malhas, razão 2, para o GCI

Os caminhos de CAD → malha continuam nos módulos que já existiam: ``pinneapple_design.fitmesh`` (gmsh/snappy a partir
de um CAD paramétrico) e ``pinneapple_simulation.pipelines.freecad_openfoam`` (caso pronto a partir de STL). Este pacote
é a camada de baixo — dicionários e detecção — usada por eles e disponível para quem quer montar a malha à mão.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .blockmesh import Caixa, dict_blockmesh
from .registry import CATALOGO, FerramentaMalha, bashrc_padrao, detectar, disponiveis, tem
from .snappy import Geometria, Refino, SnappyConfig

__all__ = ["Caixa", "dict_blockmesh", "Geometria", "Refino", "SnappyConfig", "FerramentaMalha", "CATALOGO",
           "detectar", "disponiveis", "tem", "bashrc_padrao", "estudo_de_refino", "nivel_para_resolver"]


def nivel_para_resolver(delta_x_fundo: float, delta_x_alvo: float) -> int:
    """Quantos níveis de refino (cada um divide a célula por 2) para ir de ``delta_x_fundo`` a ``delta_x_alvo``.

    É a ponte entre o critério de engenharia e o dicionário: se o jato precisa de ``Δx ≤ d/10``, este é o nível que
    entrega isso a partir do fundo que se tem.
    """
    if delta_x_fundo <= 0 or delta_x_alvo <= 0:
        raise ValueError("tamanhos de célula precisam ser positivos")
    if delta_x_alvo >= delta_x_fundo:
        return 0
    n = 0
    while delta_x_fundo / 2 ** n > delta_x_alvo:
        n += 1
    return n


def estudo_de_refino(caso_base: str | Path, niveis=(0, 1, 2), destino: str | Path | None = None,
                     padrao: str = "snappyHexMeshDict") -> dict:
    """Prepara as malhas de um estudo de convergência: um caso por nível, com os refinos somados de ``nivel``.

    Copia ``caso_base`` para ``<destino>/malha_<n>/`` e, em cada cópia, aumenta em ``n`` o nível de todas as superfícies
    e regiões do ``snappyHexMeshDict`` (ou de outro dicionário indicado em ``padrao``). **Não roda nada**: devolve o
    plano, com os comandos que faltam, para que a execução seja explícita.

    Devolve ``{"casos": [{"nivel", "pasta", "dict"}], "comandos": [...], "razoes": [...], "aviso": ...}``. As razões
    entre malhas consecutivas são o que o GCI precisa — com elas em 2, a ordem observada é comparável entre estudos.
    """
    base = Path(caso_base)
    if not base.is_dir():
        raise FileNotFoundError(f"caso base não encontrado: {base}")
    niveis = [int(n) for n in niveis]
    if len(niveis) < 3:
        raise ValueError("estudo de refino precisa de ao menos três níveis (é o que o GCI exige)")
    if niveis != sorted(niveis) or len(set(niveis)) != len(niveis):
        raise ValueError(f"níveis devem ser crescentes e distintos; recebi {niveis}")
    out = Path(destino) if destino else base.parent / f"{base.name}_refino"
    out.mkdir(parents=True, exist_ok=True)

    casos, dicts = [], []
    for n in niveis:
        pasta = out / f"malha_{n}"
        if pasta.exists():
            shutil.rmtree(pasta)
        shutil.copytree(base, pasta)
        d = pasta / "system" / padrao
        if not d.exists():
            raise FileNotFoundError(f"{padrao} não existe em {pasta / 'system'}")
        texto = d.read_text()
        # cada "level (a b)" e cada "levels ((d n))" ganham o nível extra
        import re

        novo = re.sub(r"level \((\d+)\s+(\d+)\)",
                      lambda m, k=n: f"level ({int(m.group(1)) + k} {int(m.group(2)) + k})", texto)
        novo = re.sub(r"levels \(\(([\d.eE+-]+)\s+(\d+)\)\)",
                      lambda m, k=n: f"levels (({m.group(1)} {int(m.group(2)) + k}))", novo)
        d.write_text(novo)
        casos.append({"nivel": n, "pasta": str(pasta), "dict": str(d)})
        dicts.append(novo)

    # a razão de refino entre malhas consecutivas vem do próprio dicionário (níveis somados)
    razoes = [2.0 ** (niveis[i + 1] - niveis[i]) for i in range(len(niveis) - 1)]
    return {"casos": casos, "razoes": razoes,
            "comandos": ["blockMesh", "snappyHexMesh -overwrite", "checkMesh"],
            "aviso": "as malhas foram escritas, não geradas: rode os comandos em cada pasta e confira o checkMesh "
                     "antes de simular"}
