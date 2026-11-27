"""O registro de algoritmos: uma entrada por algoritmo que este deployment executa, com a fonte e o estado.

O que estes testes protegem: o registro é completo *como declaração* (cada entrada tem nome, implementação, ao menos
uma referência e um estado), as duas únicas leituras — lista e busca — não inventam, e o estado de verificação é um
recorte honesto: o que aparece como verificado aponta para um teste que existe neste repositório.
"""
from __future__ import annotations

import pathlib

import pytest

from pinneapple_analysis import registry as R

RAIZ = pathlib.Path(__file__).resolve().parents[1]


def test_cada_entrada_declara_o_minimo_para_ser_conferida() -> None:
    for a in R.REGISTRY:
        assert a.id and a.name and a.category and a.implementation and a.summary, a.id
        assert a.references, f"{a.id}: sem referência publicada"
        assert a.verification.get("status") in (R.VERIFIED, R.NOT_EVALUATED), a.id
        assert a.verification.get("how"), f"{a.id}: diz o estado mas não como foi conferido"


def test_toda_referencia_tem_citacao_legivel() -> None:
    for a in R.REGISTRY:
        for r in a.references:
            assert r.kind in ("paper", "book", "standard", "documentation"), (a.id, r.kind)
            assert len(r.citation) > 30, (a.id, r.citation)


def test_o_verificado_aponta_para_um_teste_que_existe() -> None:
    """``verified`` é uma afirmação sobre este repositório: o caminho do teste tem de existir."""
    for a in R.REGISTRY:
        if a.verification["status"] != R.VERIFIED:
            continue
        for caminho in a.verification.get("where", "").split(","):
            caminho = caminho.strip()
            # um caminho sem prefixo é deste repositório; "engineering-api:" é do Studio, que fica no outro repo
            if caminho and ":" not in caminho:
                assert (RAIZ / caminho).exists(), f"{a.id}: aponta para {caminho}, que não existe"


def test_o_nao_avaliado_nao_promete() -> None:
    for a in R.lacunas():
        assert a["verification"]["status"] == R.NOT_EVALUATED
        assert "não" in a["verification"]["how"].lower() or "sem" in a["verification"]["how"].lower()


def test_listar_e_obter_nao_inventam() -> None:
    assert {a["id"] for a in R.listar()} == {a.id for a in R.REGISTRY}
    assert R.obter("ridge")["category"] == "surrogate"
    assert R.obter("nao_existe") is None
    assert [a["id"] for a in R.listar(categoria="meshing")] == ["block_mesh", "gmsh", "meshless_cloud"]
    assert [a["id"] for a in R.listar(busca="blasius")] == ["blasius_plate"]


def test_as_categorias_saem_na_ordem_do_registro() -> None:
    assert R.categorias() == ["analytical", "numerical", "surrogate", "solver", "meshing", "verification"]
