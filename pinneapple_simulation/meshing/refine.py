"""Refino local de uma malha que já existe: ``topoSet`` + ``refineMesh``, que é o caminho do OpenFOAM quando não se
quer remalhar tudo.

O ``snappyHexMesh`` (:mod:`.snappy`) constrói a malha em volta da geometria; o ``refineMesh`` faz outra coisa: pega uma
malha pronta e divide as células de uma **região** — uma caixa, um cilindro, um conjunto de células — em cada direção,
quantas vezes se pedir. É o que se usa para um estudo de convergência quando refinar o domínio inteiro é inviável: no
caso do canal, a malha base tem 182 mil células, e refinar tudo por dois níveis daria 11,7 milhões — enquanto refinar
só a região do impacto custa uma fração disso.

São dois utilitários, e o segundo depende do primeiro:

1. ``topoSet`` cria o conjunto ``cellSet`` com as células dentro da caixa (``boxToCell``);
2. ``refineMesh`` divide cada célula do conjunto em 8 (uma vez por nível), com ``useHexTopology`` ligado.

    from pinneapple_simulation.meshing import RefinoCaixa, dict_toposet, dict_refinemesh
    caixa = RefinoCaixa("impacto", minimo=(-2.55, -2.0, -2.6), maximo=(-0.85, 3.0, -1.0), nivel=2)
    (caso / "system" / "topoSetDict").write_text(dict_toposet(caixa))
    (caso / "system" / "refineMeshDict").write_text(dict_refinemesh(conjunto=nome_do_set))

O nível é o do OpenFOAM: cada passada divide a célula pela metade em cada direção, então ``nível`` é quantas passadas
o :func:`aplicar_refino` roda — e a célula final na região é ``Δx / 2^nivel``. A razão entre duas malhas do estudo é,
portanto, ``2 ** (nível_b − nível_a)``, que é o que o GCI precisa.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .registry import bashrc_padrao

__all__ = ["RefinoCaixa", "dict_toposet", "dict_refinemesh", "aplicar_refino"]

_CAB_TOPO = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}
"""

_CAB_REFINO = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      refineMeshDict;
}}
"""


@dataclass(frozen=True)
class RefinoCaixa:
    """Uma região a refinar: a caixa alinhada aos eixos e quantos níveis de refino dentro dela."""

    nome: str
    minimo: tuple[float, float, float]
    maximo: tuple[float, float, float]
    nivel: int = 1

    def validar(self) -> None:
        if not self.nome or not self.nome.replace("_", "").isalnum():
            raise ValueError(f"nome de refino inválido: {self.nome!r} (use letras, números e _)")
        if len(self.minimo) != 3 or len(self.maximo) != 3:
            raise ValueError("a caixa precisa de três coordenadas em mínimo e em máximo")
        if any(lo >= hi for lo, hi in zip(self.minimo, self.maximo, strict=True)):
            raise ValueError(f"caixa invertida ou degenerada em {self.nome}: {self.minimo} → {self.maximo}")
        if self.nivel < 0:
            raise ValueError("o nível de refino não pode ser negativo")

    @property
    def volume(self) -> float:
        return abs((self.maximo[0] - self.minimo[0]) * (self.maximo[1] - self.minimo[1])
                   * (self.maximo[2] - self.minimo[2]))

    def to_dict(self) -> dict:
        return {"nome": self.nome, "minimo": list(self.minimo), "maximo": list(self.maximo), "nivel": self.nivel}


def _vetor(v) -> str:
    return "(" + " ".join(f"{float(x):.10g}" for x in v) + ")"


def dict_toposet(*caixas: RefinoCaixa, nome_do_set: str = "refino") -> str:
    """``topoSetDict`` que seleciona as células dentro das caixas, em ordem, num único ``cellSet``."""
    if not caixas:
        raise ValueError("informe ao menos uma caixa de refino")
    for c in caixas:
        c.validar()

    def _acao(c: RefinoCaixa, acao: str) -> str:
        # dentro de `actions (...)` cada entrada é um dicionário: fecha com `}` e **sem** `;`
        return ("    {\n"
                f"        name    {nome_do_set};\n"
                "        type    cellSet;\n"
                f"        action  {acao};\n"
                "        source  boxToCell;\n"
                "        sourceInfo\n        {\n"
                f"            box     {_vetor(c.minimo)} {_vetor(c.maximo)};\n"
                "        }\n"
                "    }")

    entradas = [_acao(caixas[0], "new")] + [_acao(c, "add") for c in caixas[1:]]
    return _CAB_TOPO.format() + "\nactions\n(\n" + "\n".join(entradas) + "\n);\n"


def dict_refinemesh(direcoes: tuple[int, int, int] = (1, 1, 1), conjunto: str = "refino",
                    usar_topologia_hexa: bool = True, corte_geometrico: bool = False) -> str:
    """``refineMeshDict`` que divide cada célula do conjunto em ``2^(soma das direções)``.

    O nome do conjunto vai no próprio dicionário (entrada ``set``): a partir do OpenFOAM v2506 o ``refineMesh`` não
    aceita mais ``-set`` na linha de comando. ``direcoes`` em ``(1, 1, 1)`` é refino isotrópico — é o que um estudo de
    convergência quer, porque mantém a razão de aspecto e faz a razão entre malhas ser exatamente ``2`` por nível.
    """
    if any(d not in (0, 1) for d in direcoes) or not any(direcoes):
        raise ValueError(f"direções inválidas: {direcoes} (use 0 ou 1 em cada eixo, ao menos uma)")
    nomes = ("tan1", "tan2", "normal")
    ativas = [n for n, d in zip(nomes, direcoes, strict=True) if d]
    return (_CAB_REFINO.format() + "\n"
            f"set             {conjunto};\n\n"
            "coordinateSystem global;\n\n"
            "globalCoeffs\n{\n    tan1 (1 0 0);\n    tan2 (0 1 0);\n}\n\n"
            "directions\n(\n" + "\n".join(f"    {n}" for n in ativas) + "\n);\n\n"
            f"useHexTopology  {'true' if usar_topologia_hexa else 'false'};\n"
            f"geometricCut    {'true' if corte_geometrico else 'false'};\n"
            "writeMesh       false;\n\n"
            "// mantém as fronteiras existentes e não move a malha: só divide células\n")


def _roda(caso: Path, comando: str, bashrc: str | None, log: Path) -> int:
    pre = f'source "{bashrc}" >/dev/null 2>&1; ' if bashrc else ""
    with log.open("a") as f:
        f.write(f"$ {comando}\n")
        f.flush()
        r = subprocess.run(["bash", "-c", f"{pre}cd {caso} && {comando}"], stdout=f, stderr=subprocess.STDOUT,
                           check=False)
    return r.returncode


def aplicar_refino(caso: str | Path, *caixas: RefinoCaixa, bashrc: str | None = None,
                   nome_do_set: str = "refino") -> dict:
    """Escreve os dicionários e roda ``topoSet`` + ``refineMesh`` de verdade, nível por nível.

    A cada passada o conjunto é **recriado** (``topoSet`` de novo) antes de refinar: as células filhas ficam dentro da
    mesma caixa, então o conjunto cresce sozinho para elas, e nenhuma célula é refinada duas vezes por engano.

    Devolve ``{"niveis", "passadas", "log", "ok"}``. Não silencia falha: ``ok`` é ``False`` e o log fica no caso.
    """
    caso = Path(caso)
    if not (caso / "constant" / "polyMesh").is_dir():
        raise FileNotFoundError(f"{caso} não tem constant/polyMesh: não é um caso com malha")
    for c in caixas:
        c.validar()
    niveis = max((c.nivel for c in caixas), default=0)
    if niveis == 0:
        raise ValueError("todas as caixas com nível 0: nada a refinar")
    if bashrc is None:
        bashrc = bashrc_padrao()
    log = caso / "log.refino"
    if log.exists():
        log.unlink()

    (caso / "system" / "topoSetDict").write_text(dict_toposet(*caixas, nome_do_set=nome_do_set))
    (caso / "system" / "refineMeshDict").write_text(dict_refinemesh(conjunto=nome_do_set))
    passadas = 0
    for _ in range(niveis):
        if _roda(caso, "topoSet", bashrc, log) != 0:
            return {"niveis": niveis, "passadas": passadas, "log": str(log), "ok": False,
                    "erro": "topoSet falhou"}
        if _roda(caso, "refineMesh -dict system/refineMeshDict -overwrite", bashrc, log) != 0:
            return {"niveis": niveis, "passadas": passadas, "log": str(log), "ok": False,
                    "erro": "refineMesh falhou"}
        passadas += 1
    return {"niveis": niveis, "passadas": passadas, "log": str(log), "ok": True,
            "nivel_efetivo": {c.nome: passadas for c in caixas if c.nivel}}
