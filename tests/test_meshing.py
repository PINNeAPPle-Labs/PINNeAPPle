"""Geração de malha com o OpenFOAM: registro de ferramentas, blockMeshDict, snappyHexMeshDict e estudo de refino."""
from __future__ import annotations

import pytest

from pinneapple_simulation.meshing import (
    Caixa,
    Geometria,
    Refino,
    SnappyConfig,
    detectar,
    dict_blockmesh,
    estudo_de_refino,
    nivel_para_resolver,
    tem,
)


# --- registro ------------------------------------------------------------------------------------------------------
def test_registro_traz_o_catalogo_e_um_veredito_por_ferramenta():
    r = detectar(bashrc="/caminho/que/nao/existe")
    nomes = [m["nome"] for m in r]
    assert {"blockMesh", "snappyHexMesh", "refineMesh", "checkMesh"} <= set(nomes)
    for m in r:
        assert isinstance(m["disponivel"], bool)
        if not m["disponivel"]:
            assert m["motivo"]


def test_registro_detecta_o_openfoam_desta_maquina():
    """Se houver OpenFOAM instalado, o blockMesh aparece — sem inventar quando não houver."""
    if tem("blockMesh"):
        assert any(m["nome"] == "blockMesh" and m["disponivel"] for m in detectar())


# --- blockMesh -----------------------------------------------------------------------------------------------------
def _caixa() -> Caixa:
    return Caixa(x=(-2.55, -0.88), y=(-7.44, 15.32), z=(-3.25, -1.00), nx=20, ny=300, nz=12)


def test_caixa_gera_os_oito_vertices_na_ordem_do_openfoam():
    v = _caixa().vertices()
    assert len(v) == 8
    assert v[0] == (-2.55, -7.44, -3.25) and v[6] == (-0.88, 15.32, -1.00)


def test_blockmesh_tem_blocos_fronteiras_e_expansao():
    t = dict_blockmesh(_caixa())
    assert "hex (0 1 2 3 4 5 6 7) (20 300 12) simpleGrading (1 1 1)" in t
    assert t.count("type ") == 6                        # seis fronteiras
    assert "convertToMeters 1" in t
    assert t.count("(") == t.count(")")                 # parênteses balanceados


def test_blockmesh_nomeia_as_fronteiras_pedidas():
    c = Caixa(x=(0, 1), y=(0, 2), z=(0, 3), nx=2, ny=4, nz=6,
              fronteiras={"ymin": ("entrada", "patch"), "zmin": ("fundo", "wall")})
    t = dict_blockmesh(c)
    assert "entrada" in t and "fundo" in t and "type wall" in t


def test_caixa_recusa_extensao_invalida_e_celulas_negativas():
    with pytest.raises(ValueError):
        dict_blockmesh(Caixa(x=(1, 0), y=(0, 1), z=(0, 1), nx=1, ny=1, nz=1))
    with pytest.raises(ValueError):
        dict_blockmesh(Caixa(x=(0, 1), y=(0, 1), z=(0, 1), nx=0, ny=1, nz=1))
    with pytest.raises(ValueError):
        dict_blockmesh(Caixa(x=(0, 1), y=(0, 1), z=(0, 1), nx=1, ny=1, nz=1, fronteiras={"w": "parede"}))


def test_caixa_conta_celulas_e_volume():
    c = Caixa(x=(0, 1), y=(0, 2), z=(0, 3), nx=2, ny=4, nz=6)
    assert c.celulas() == 48 and c.volumes() == pytest.approx(6.0)


# --- snappy --------------------------------------------------------------------------------------------------------
def _snappy() -> SnappyConfig:
    return SnappyConfig(
        geometrias=[Geometria("canal", "canal.stl", nivel=(2, 3))],
        refinos=[Refino("jato", "box", minimo=(-2.6, -7.5, -2.4), maximo=(-0.8, -2.0, -1.0), nivel=4),
                 Refino("longe", "sphere", centro=(0, 0, 0), raio=1.0, nivel=1, modo="outside")],
        camadas=3, ponto_dentro=(-2.1, 0.0, -2.0))


def test_snappy_escreve_as_secoes_obrigatorias():
    t = _snappy().dict_texto()
    for secao in ("castellatedMeshControls", "snapControls", "addLayersControls", "meshQualityControls"):
        assert secao in t
    assert "castellatedMesh true;" in t and "addLayers true;" in t
    assert "locationInMesh (-2.1 0 -2)" in t


def test_snappy_leva_os_niveis_e_as_formas():
    t = _snappy().dict_texto()
    assert "level (2 3)" in t                     # superfície
    assert "levels ((1e15 4))" in t               # região por caixa, nível absoluto
    assert "searchableSphere" in t and "radius 1" in t
    assert "nSurfaceLayers 3;" in t


def test_snappy_sem_geometria_falha():
    with pytest.raises(ValueError):
        SnappyConfig().dict_texto()


def test_snappy_recusa_nome_repetido_e_nivel_invertido():
    with pytest.raises(ValueError):
        SnappyConfig(geometrias=[Geometria("a", "a.stl"), Geometria("a", "b.stl")]).dict_texto()
    with pytest.raises(ValueError):
        SnappyConfig(geometrias=[Geometria("a", "a.stl", nivel=(3, 2))]).dict_texto()


def test_snappy_recusa_caixa_invertida():
    with pytest.raises(ValueError):
        SnappyConfig(geometrias=[Geometria("a", "a.stl")],
                     refinos=[Refino("r", "box", minimo=(1, 0, 0), maximo=(0, 1, 1))]).dict_texto()


# --- estudo de refino ----------------------------------------------------------------------------------------------
def _caso(tmp_path):
    caso = tmp_path / "caso"
    (caso / "system").mkdir(parents=True, exist_ok=True)
    (caso / "system" / "snappyHexMeshDict").write_text(
        "castellatedMeshControls\n{\n    refinementSurfaces\n    {\n        canal\n        {\n"
        "            level (2 3);\n        }\n    }\n    refinementRegions\n    {\n        jato\n        {\n"
        "            mode inside;\n            levels ((1e15 3));\n        }\n    }\n}\n")
    return caso


def test_estudo_de_refino_soma_os_niveis_em_cada_copia(tmp_path):
    plano = estudo_de_refino(_caso(tmp_path), niveis=(0, 1, 2), destino=tmp_path / "out")
    assert len(plano["casos"]) == 3
    assert plano["razoes"] == [2.0, 2.0]
    textos = [open(c["dict"]).read() for c in plano["casos"]]
    assert "level (2 3)" in textos[0] and "level (3 4)" in textos[1] and "level (4 5)" in textos[2]
    assert "levels ((1e15 3))" in textos[0] and "levels ((1e15 5))" in textos[2]
    assert plano["comandos"][0] == "blockMesh" and "não geradas" in plano["aviso"]


def test_estudo_de_refino_exige_tres_niveis_crescentes(tmp_path):
    c = _caso(tmp_path)
    with pytest.raises(ValueError):
        estudo_de_refino(c, niveis=(0, 1))
    with pytest.raises(ValueError):
        estudo_de_refino(c, niveis=(2, 1, 0))
    with pytest.raises(FileNotFoundError):
        estudo_de_refino(tmp_path / "nao_existe")


def test_nivel_para_resolver_liga_o_criterio_ao_dicionario():
    assert nivel_para_resolver(0.08, 0.008) == 4          # 0,08 / 2^3 = 0,01; 2^4 = 0,005 ≤ 0,008
    assert nivel_para_resolver(0.08, 0.08) == 0
    assert nivel_para_resolver(0.08, 0.2) == 0
    with pytest.raises(ValueError):
        nivel_para_resolver(0.0, 0.1)


# --- validação contra o OpenFOAM de verdade (pulado se não houver instalação) ---------------------------------------
def _caso_foam(tmp_path, dict_txt: str, nome="blockMeshDict"):
    caso = tmp_path / "caso"
    (caso / "system").mkdir(parents=True, exist_ok=True)
    (caso / "constant").mkdir(exist_ok=True)
    (caso / "system" / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "application blockMesh;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\n"
        "deltaT 1;\nwriteControl timeStep;\nwriteInterval 1;\n")
    (caso / "system" / nome).write_text(dict_txt)
    return caso


def _foam(caso, cmd):
    import subprocess

    from pinneapple_simulation.meshing import bashrc_padrao
    rc = bashrc_padrao()
    pre = f'source "{rc}" >/dev/null 2>&1; ' if rc else ""
    return subprocess.run(["bash", "-c", f"{pre}cd {caso} && {cmd}"], capture_output=True, text=True, check=False)


@pytest.mark.skipif(not tem("blockMesh"), reason="OpenFOAM não instalado nesta máquina")
def test_blockmesh_dict_e_aceito_pelo_openfoam(tmp_path):
    """O dicionário gerado é lido e a malha sai com as fronteiras nomeadas."""
    c = Caixa(x=(0, 1), y=(0, 2), z=(0, 0.5), nx=10, ny=20, nz=5,
              fronteiras={"ymin": ("entrada", "patch"), "ymax": ("saida", "patch")})
    caso = _caso_foam(tmp_path, dict_blockmesh(c))
    r = _foam(caso, "blockMesh")
    assert r.returncode == 0, r.stdout[-800:]
    assert "End" in r.stdout
    assert "name: entrada" in r.stdout and "name: saida" in r.stdout
    assert (caso / "constant" / "polyMesh" / "points").exists()


@pytest.mark.skipif(not tem("blockMesh"), reason="OpenFOAM não instalado nesta máquina")
def test_snappy_dict_e_lido_pelo_openfoam(tmp_path):
    """Sintaxe válida: o leitor do OpenFOAM abre o dicionário e devolve a entrada esperada."""
    caso = _caso_foam(tmp_path, _snappy().dict_texto(), "snappyHexMeshDict")
    r = _foam(caso, "foamDictionary system/snappyHexMeshDict -entry castellatedMeshControls/refinementSurfaces")
    assert r.returncode == 0, r.stdout[-500:]
    assert "canal" in r.stdout and "level" in r.stdout
    r2 = _foam(caso, "foamDictionary system/snappyHexMeshDict -entry castellatedMeshControls/locationInMesh")
    assert r2.returncode == 0 and "(-2.1 0 -2)" in r2.stdout
