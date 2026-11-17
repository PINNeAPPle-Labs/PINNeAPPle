"""``snappyHexMeshDict`` a partir de superfícies STL e regiões de refino.

É o caminho padrão do OpenFOAM para malha em volta de geometria: um fundo hexaédrico (``blockMesh``), o refinamento
"castellated" conforme as superfícies, o encaixe nos contornos (snap) e as camadas de parede. Este módulo monta o
dicionário a partir de uma lista de :class:`Geometria` e de :class:`Refino`, com validação antes de escrever.

    from pinneapple_simulation.meshing import Geometria, Refino, SnappyConfig
    cfg = SnappyConfig(
        geometrias=[Geometria("canal", "canal.stl", nivel=(2, 3))],
        refinos=[Refino("jato", "box", minimo=(-2.6, -7.5, -2.4), maximo=(-0.8, -2.0, -1.0), nivel=3)],
        camadas=3, ponto_dentro=(-2.1, 0.0, -2.0))
    (caso / "system" / "snappyHexMeshDict").write_text(cfg.dict_texto())

O nível de refino é o do OpenFOAM: cada nível divide a célula pela metade em cada direção, então a célula final é
``Δx_fundo / 2^nivel``. Quem decide o nível é o critério de resolução — ``mesh_adequacy.resolver_jato`` diz quantas
células o jato precisa por diâmetro — e não o tamanho da máquina.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Geometria", "Refino", "SnappyConfig"]

_CAB = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}
"""


@dataclass(frozen=True)
class Geometria:
    """Uma superfície STL que o snappy vai refinar e encaixar.

    ``nivel`` é ``(mínimo, máximo)`` de refino na superfície; ``regiao`` permite usar só uma região do STL
    (arquivos multi-região).
    """

    nome: str
    arquivo: str
    nivel: tuple[int, int] = (2, 3)
    tipo_patch: str = "wall"
    regiao: str | None = None

    def validar(self) -> None:
        a, b = self.nivel
        if not (0 <= a <= b):
            raise ValueError(f"{self.nome}: nível precisa ser (mínimo ≤ máximo, ambos ≥ 0); recebi {self.nivel}")


@dataclass(frozen=True)
class Refino:
    """Região de refino: uma caixa ou esfera, dentro (``inside``) ou fora (``outside``) dela."""

    nome: str
    forma: str                      # "box" ou "sphere"
    nivel: int = 2
    modo: str = "inside"
    minimo: tuple[float, float, float] | None = None
    maximo: tuple[float, float, float] | None = None
    centro: tuple[float, float, float] | None = None
    raio: float | None = None
    distancia: float | None = None   # refino por distância à superfície, em metros

    def validar(self) -> None:
        if self.forma not in ("box", "sphere"):
            raise ValueError(f"{self.nome}: forma {self.forma!r} não suportada (use 'box' ou 'sphere')")
        if self.modo not in ("inside", "outside"):
            raise ValueError(f"{self.nome}: modo {self.modo!r} não suportado (use 'inside' ou 'outside')")
        if int(self.nivel) < 0:
            raise ValueError(f"{self.nome}: nível não pode ser negativo")
        if self.forma == "box" and (self.minimo is None or self.maximo is None):
            raise ValueError(f"{self.nome}: caixa precisa de minimo e maximo")
        if self.forma == "box" and any(b <= a for a, b in zip(self.minimo, self.maximo, strict=True)):
            raise ValueError(f"{self.nome}: maximo precisa ser maior que minimo em todas as direções")
        if self.forma == "sphere" and (self.centro is None or not self.raio or self.raio <= 0):
            raise ValueError(f"{self.nome}: esfera precisa de centro e raio positivo")


@dataclass(frozen=True)
class SnappyConfig:
    """O dicionário completo: superfícies, refinos, camadas e ponto que está dentro do domínio."""

    geometrias: list = field(default_factory=list)
    refinos: list = field(default_factory=list)
    camadas: int = 3
    expansao_camada: float = 1.2
    espessura_ultima_camada: float = 0.3
    espessura_minima: float = 0.1
    angulo_camada: float = 60.0
    ponto_dentro: tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_celulas_entre_niveis: int = 2
    max_celulas_globais: int = 2_000_000
    max_celulas_locais: int = 1_000_000
    resolve_feature_angle: float = 30.0
    qualidade: dict = field(default_factory=lambda: {
        "maxNonOrtho": 65.0, "maxBoundarySkewness": 20.0, "maxInternalSkewness": 4.0, "maxConcave": 80.0,
        "minVol": 1e-13, "minTetQuality": 1e-30, "minArea": -1.0, "minTwist": 0.01, "minDeterminant": 0.001,
        "minFaceWeight": 0.02, "minVolRatio": 0.01, "minTriangleTwist": -1.0, "nSmoothScale": 4,
        "errorReduction": 0.75})

    def validar(self) -> None:
        if not self.geometrias:
            raise ValueError("snappy sem geometria não refina nada: informe ao menos uma Geometria")
        for g in self.geometrias:
            g.validar()
        for r in self.refinos:
            r.validar()
        if self.camadas < 0:
            raise ValueError("número de camadas não pode ser negativo")
        if self.camadas and not 0 < self.expansao_camada <= 2:
            raise ValueError("expansão de camada deve ficar em (0, 2]")
        nomes = [g.nome for g in self.geometrias] + [r.nome for r in self.refinos]
        if len(set(nomes)) != len(nomes):
            raise ValueError(f"nomes repetidos em geometrias/refinos: {nomes}")

    def dict_texto(self) -> str:
        self.validar()
        L = [_CAB.format(), "castellatedMesh true;", "snap true;", f"addLayers {'true' if self.camadas else 'false'};\n",
             "geometry", "{"]
        for g in self.geometrias:
            L += [f"    {g.nome}.stl", "    {", "        type triSurfaceMesh;", f"        name {g.nome};",
                  "    }"]
        for r in self.refinos:
            L += [f"    {r.nome}", "    {",
                  "        type searchableBox;" if r.forma == "box" else "        type searchableSphere;",
                  *self._forma(r), "    }"]
        L += ["}\n", "castellatedMeshControls", "{",
              f"    maxLocalCells {self.max_celulas_locais};", f"    maxGlobalCells {self.max_celulas_globais};",
              "    minRefinementCells 10;", "    maxLoadUnbalance 0.10;",
              f"    nCellsBetweenLevels {self.n_celulas_entre_niveis};", "    features ();",
              "    refinementSurfaces", "    {"]
        for g in self.geometrias:
            L += [f"        {g.nome}", "        {", f"            level ({g.nivel[0]} {g.nivel[1]});",
                  "            patchInfo", "            {", f"                type {g.tipo_patch};", "            }",
                  "        }"]
        L += ["    }", f"    resolveFeatureAngle {self.resolve_feature_angle:g};", "    refinementRegions", "    {"]
        for r in self.refinos:
            alvo = f"((1e15 {r.nivel}))" if r.distancia is None else f"(({r.distancia} {r.nivel}))"
            L += [f"        {r.nome}", "        {", f"            mode {r.modo};", f"            levels {alvo};", "        }"]
        L += ["    }", f"    locationInMesh ({self.ponto_dentro[0]:g} {self.ponto_dentro[1]:g} "
                       f"{self.ponto_dentro[2]:g});", "    allowFreeStandingZoneFaces true;", "}\n",
              "snapControls", "{", "    nSmoothPatch 3;", "    tolerance 2.0;", "    nSolveIter 30;", "    nRelaxIter 5;",
              "    nFeatureSnapIter 10;", "    implicitFeatureSnap false;", "    explicitFeatureSnap true;",
              "    multiRegionFeatureSnap false;", "}\n"]
        L += ["addLayersControls", "{", "    relativeSizes true;", "    layers", "    {"]
        if self.camadas:
            L += ['        ".*"', "        {", f"            nSurfaceLayers {self.camadas};", "        }"]
        L += ["    }", f"    expansionRatio {self.expansao_camada:g};",
              f"    finalLayerThickness {self.espessura_ultima_camada:g};",
              f"    minThickness {self.espessura_minima:g};", "    nGrow 0;",
              f"    featureAngle {self.angulo_camada:g};", "    slipFeatureAngle 30;", "    nRelaxIter 3;",
              "    nSmoothSurfaceNormals 1;", "    nSmoothNormals 3;", "    nSmoothThickness 10;",
              "    maxFaceThicknessRatio 0.5;", "    maxThicknessToMedialRatio 0.3;", "    minMedialAxisAngle 90;",
              "    nBufferCellsNoExtrude 0;", "    nLayerIter 50;", "}\n",
              "meshQualityControls", "{"]
        L += [f"    {k} {v:g};" for k, v in self.qualidade.items()]
        L += ["}\n", "debug 0;", "mergeTolerance 1e-6;\n"]
        return "\n".join(L)

    @staticmethod
    def _forma(r: Refino) -> list[str]:
        if r.forma == "box":
            return [f"        min ({r.minimo[0]:g} {r.minimo[1]:g} {r.minimo[2]:g});",
                    f"        max ({r.maximo[0]:g} {r.maximo[1]:g} {r.maximo[2]:g});"]
        return [f"        centre ({r.centro[0]:g} {r.centro[1]:g} {r.centro[2]:g});", f"        radius {r.raio:g};"]
