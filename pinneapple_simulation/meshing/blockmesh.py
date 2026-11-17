"""``blockMeshDict`` a partir de uma caixa: o fundo estruturado de qualquer malha hexaédrica.

O ``blockMesh`` é o único gerador que vem sempre com o OpenFOAM e não precisa de CAD: descreve-se o domínio em blocos,
com número de células e razão de expansão por direção, e o ``snappyHexMesh`` refina depois em volta da geometria. Este
módulo escreve esses dicionários a partir de uma :class:`Caixa` (domínio alinhado aos eixos), com as seis fronteiras
nomeadas — e valida antes de escrever, para o erro sair aqui e não num ``checkMesh`` dez minutos depois.

    from pinneapple_simulation.meshing import Caixa, dict_blockmesh
    caixa = Caixa(x=(-2.55, -0.88), y=(-7.44, 15.32), z=(-3.25, -1.00), nx=20, ny=300, nz=12)
    print(dict_blockmesh(caixa))          # texto do blockMeshDict, pronto para system/

O que este módulo **não** faz: escolher o tamanho de célula por você. Isso é decisão de engenharia e há critério para
ela em ``pinneapple_analysis.validation.mesh_adequacy`` (células por diâmetro do jato, GCI).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Caixa", "dict_blockmesh"]

_CAB = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
"""

# faces de um hexa com vértices 0-3 na base e 4-7 no topo, na ordem do OpenFOAM
_FACES = {
    "xmin": (0, 4, 7, 3),
    "xmax": (1, 2, 6, 5),
    "ymin": (0, 1, 5, 4),
    "ymax": (3, 7, 6, 2),
    "zmin": (0, 3, 2, 1),
    "zmax": (4, 5, 6, 7),
}
_TIPO_PADRAO = {"xmin": "patch", "xmax": "patch", "ymin": "patch", "ymax": "patch", "zmin": "wall", "zmax": "patch"}


@dataclass(frozen=True)
class Caixa:
    """Um bloco hexaédrico alinhado aos eixos, com os nomes das seis fronteiras.

    ``grading`` é a razão de expansão por direção (``1`` é uniforme; ``2`` dobra o tamanho da última célula em relação
    à primeira). ``fronteiras`` mapeia ``xmin/xmax/ymin/ymax/zmin/zmax`` para ``(nome, tipo)``; o que não for dado usa
    ``paredes``/``aberto`` conforme o tipo padrão.
    """

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    nx: int
    ny: int
    nz: int
    grading: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fronteiras: dict = field(default_factory=dict)
    parede_tipo: str = "wall"
    aberto_tipo: str = "patch"

    # --- geometria ---------------------------------------------------------------------------------------------
    def vertices(self) -> list[tuple[float, float, float]]:
        (x0, x1), (y0, y1), (z0, z1) = self.x, self.y, self.z
        return [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]

    def tamanho_celula(self) -> tuple[float, float, float]:
        return ((self.x[1] - self.x[0]) / self.nx, (self.y[1] - self.y[0]) / self.ny,
                (self.z[1] - self.z[0]) / self.nz)

    def volumes(self) -> float:
        dx, dy, dz = self.tamanho_celula()
        return dx * dy * dz * self.nx * self.ny * self.nz

    def celulas(self) -> int:
        return self.nx * self.ny * self.nz

    def nome_e_tipo(self, face: str) -> tuple[str, str]:
        if face in self.fronteiras:
            v = self.fronteiras[face]
            return (v, self.aberto_tipo) if isinstance(v, str) else (v[0], v[1])
        tipo = self.parede_tipo if _TIPO_PADRAO.get(face) == "wall" else self.aberto_tipo
        return f"{face}_{'parede' if tipo == 'wall' else 'contorno'}", tipo

    # --- validação ---------------------------------------------------------------------------------------------
    def validar(self) -> None:
        """Levanta ``ValueError`` no que o ``blockMesh`` recusaria — antes de escrever o dicionário."""
        for nome, (a, b) in (("x", self.x), ("y", self.y), ("z", self.z)):
            if not b > a:
                raise ValueError(f"extensão em {nome} precisa ser crescente; recebi {a} → {b}")
        for nome, n in (("nx", self.nx), ("ny", self.ny), ("nz", self.nz)):
            if int(n) < 1:
                raise ValueError(f"{nome} precisa ser ≥ 1; recebi {n}")
        for g in self.grading:
            if g <= 0:
                raise ValueError(f"razão de expansão precisa ser positiva; recebi {g}")
        desconhecidas = set(self.fronteiras) - set(_FACES)
        if desconhecidas:
            raise ValueError(f"fronteiras desconhecidas: {sorted(desconhecidas)}; use {sorted(_FACES)}")

    # --- dicionário --------------------------------------------------------------------------------------------
    def dict_texto(self, convert_to_meters: float = 1.0) -> str:
        self.validar()
        v = self.vertices()
        linhas = [_CAB.format(), f"convertToMeters {convert_to_meters};\n", "vertices\n("]
        linhas += [f"    ({p[0]:.6g} {p[1]:.6g} {p[2]:.6g})" for p in v]
        linhas += [");\n", "blocks\n(", f"    hex (0 1 2 3 4 5 6 7) ({self.nx} {self.ny} {self.nz}) simpleGrading "
                                        f"({self.grading[0]:g} {self.grading[1]:g} {self.grading[2]:g})", ");\n",
                   "edges\n(\n);\n", "boundary\n("]
        for face, idx in _FACES.items():
            nome, tipo = self.nome_e_tipo(face)
            linhas += [f"    {nome}\n    {{", f"        type {tipo};", f"        faces ( ({' '.join(map(str, idx))}) );",
                       "    }"]
        linhas += [");\n"]
        return "\n".join(linhas)


def dict_blockmesh(caixa: Caixa, convert_to_meters: float = 1.0) -> str:
    """Texto do ``blockMeshDict`` para a caixa (mesma coisa que ``caixa.dict_texto()``, em forma de função)."""
    return caixa.dict_texto(convert_to_meters)
