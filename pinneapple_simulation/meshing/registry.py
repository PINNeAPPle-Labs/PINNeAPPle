"""Quais geradores de malha existem nesta instalação do OpenFOAM.

A lib já tem dois caminhos de malha montados sobre o OpenFOAM — ``pinneapple_design.fitmesh`` (CAD → malha, com gmsh ou
snappy) e ``pinneapple_simulation.pipelines.freecad_openfoam`` (caso pronto a partir de STL). O que faltava era a
pergunta anterior a qualquer um deles: **o que está instalado aqui, e o que cada um serve para fazer?** Este módulo
responde isso e é o que os escritores de dicionário (:mod:`.blockmesh`, :mod:`.snappy`) consultam antes de prometer
qualquer coisa.

    from pinneapple_simulation.meshing import detectar, disponiveis
    detectar()                       # [{"nome": "snappyHexMesh", "comando": ..., "disponivel": True, ...}, ...]
    [m["nome"] for m in disponiveis()]

A detecção é feita no shell do OpenFOAM (``source <bashrc>; command -v <comando>``), então vale tanto para um OpenFOAM
de sistema quanto para o aplicativo do macOS. Sem ``bashrc``, usa o ``FOAM_BASHRC`` do ambiente ou procura os caminhos
conhecidos.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["FerramentaMalha", "CATALOGO", "bashrc_padrao", "detectar", "disponiveis", "tem"]

BASHRC_CONHECIDOS = (
    "/Applications/OpenFOAM-v2606.app/Contents/Resources/etc/bashrc",
    "/usr/lib/openfoam/openfoam/etc/bashrc",
    "/opt/openfoam/etc/bashrc",
    "/usr/share/openfoam/etc/bashrc",
)


@dataclass(frozen=True)
class FerramentaMalha:
    """Uma ferramenta de malha do OpenFOAM (ou de um pacote que se instala ao lado dele)."""

    nome: str
    comando: str
    tipo: str
    para: str
    extra: bool = False          # não vem no OpenFOAM padrão (cfMesh)

    def to_dict(self) -> dict:
        return {"nome": self.nome, "comando": self.comando, "tipo": self.tipo, "para": self.para, "extra": self.extra}


CATALOGO: tuple[FerramentaMalha, ...] = (
    FerramentaMalha("blockMesh", "blockMesh", "estruturada",
                    "malha hexaédrica de fundo a partir de um dicionário de blocos; também o fundo do snappyHexMesh"),
    FerramentaMalha("snappyHexMesh", "snappyHexMesh", "cartesian+geo",
                    "malha hexaédrica adaptada a uma superfície STL, com refino por região e camadas de parede"),
    FerramentaMalha("extrudeMesh", "extrudeMesh", "extrusão",
                    "extrusão de uma malha 2D (ou de um patch) na direção normal, com camadas e expansão"),
    FerramentaMalha("refineMesh", "refineMesh", "adaptação",
                    "refino/de-refino por região de uma malha que já existe, sem remalhar tudo"),
    FerramentaMalha("topoSet", "topoSet", "seleção",
                    "seleciona células/faces por caixa, cilindro, superfície; alimenta refino, patches e campos"),
    FerramentaMalha("createPatch", "createPatch", "patches",
                    "cria ou divide patches a partir de conjuntos de faces (entrada do jato, paredes)"),
    FerramentaMalha("checkMesh", "checkMesh", "verificação",
                    "qualidade e validade da malha: não conformidade, ortogonalidade, skewness, aspect ratio"),
    FerramentaMalha("cartesianMesh", "cartesianMesh", "cartesian (cfMesh)",
                    "malha cartesiana com refino por superfície e camadas; costuma dar menos células que o snappy "
                    "para a mesma qualidade", extra=True),
    FerramentaMalha("tetMesh", "tetMesh", "tetraédrica (cfMesh)",
                    "malha tetraédrica a partir de superfície, com camadas de parede", extra=True),
)


def bashrc_padrao() -> str | None:
    """O bashrc do OpenFOAM desta máquina: ``PP_FOAM_BASHRC``/``FOAM_BASHRC`` ou um dos caminhos conhecidos."""
    for var in ("PP_FOAM_BASHRC", "FOAM_BASHRC"):
        v = os.environ.get(var)
        if v and Path(v).exists():
            return v
    for c in BASHRC_CONHECIDOS:
        if Path(c).exists():
            return c
    return None


def _roda(comando: str, bashrc: str | None, timeout: float) -> str:
    """Roda um comando no shell do OpenFOAM e devolve a saída (vazia se falhar)."""
    pre = f'source "{bashrc}" >/dev/null 2>&1; ' if bashrc else ""
    try:
        r = subprocess.run(["bash", "-c", pre + comando], capture_output=True, text=True, timeout=timeout, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


@lru_cache(maxsize=4)
def _detectar_cache(rc: str | None) -> tuple:
    """A detecção custa uma chamada de shell por ferramenta (cada uma abrindo o bashrc): fica em cache."""
    out = []
    for f in CATALOGO:
        caminho = _roda(f"command -v {f.comando}", rc, 60.0) if rc else ""
        if not caminho:
            caminho = shutil.which(f.comando) or ""
        out.append((f, caminho))
    return tuple(out)


def detectar(bashrc: str | None = None, timeout: float = 60.0) -> list[dict]:
    """Uma entrada por ferramenta do catálogo, com ``disponivel`` conforme a instalação.

    Sem nenhum OpenFOAM localizado, cai no ``PATH`` do sistema (``shutil.which``) — o que também encontra um
    OpenFOAM carregado à mão.
    """
    rc = bashrc or bashrc_padrao()
    return [{**f.to_dict(), "disponivel": bool(caminho), "caminho": caminho or None, "bashrc": rc,
             "motivo": None if caminho else ("não está instalado" if not f.extra else
                                             "pacote opcional (cfMesh) não instalado")}
            for f, caminho in _detectar_cache(rc)]


def disponiveis(bashrc: str | None = None) -> list[dict]:
    """Só as ferramentas que existem nesta máquina."""
    return [m for m in detectar(bashrc) if m["disponivel"]]


def tem(nome: str, bashrc: str | None = None) -> bool:
    """Se uma ferramenta específica está disponível (``tem("snappyHexMesh")``)."""
    return any(m["nome"] == nome for m in disponiveis(bashrc))
