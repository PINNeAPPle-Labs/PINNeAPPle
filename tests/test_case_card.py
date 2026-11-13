"""Ficha de condições de um caso e comparação item a item entre duas fichas (``case_card``)."""
from __future__ import annotations

import pytest

from pinneapple_simulation.pipelines.case_card import (
    CondicoesCaso,
    area_de_vazao,
    comparar,
    diametro_equivalente,
    divergentes,
    k_de_intensidade,
    tabela_md,
)


def _nos() -> CondicoesCaso:
    return CondicoesCaso(
        nome="nosso", vetor_entrada=(0.0, 9.85, 1.74), area_entrada_m2=0.005632, nivel_inicial_m=-2.04,
        turbulencia_intensidade=0.01, turbulencia_escala_m=0.10,
        linhas={"LEdir": ((-1.8, -7.429, -2.2), (-1.8, 15.305, -2.2))}, solver="v2606")


def _deles() -> CondicoesCaso:
    return CondicoesCaso(
        nome="deles", vetor_entrada=(0.0, 9.85, 1.74), vazao_m3_s=0.04582, nivel_inicial_m=-2.22,
        turbulencia_intensidade=0.01, turbulencia_escala_m=0.10, solver="v2212")


def test_area_derivada_da_vazao_e_da_velocidade():
    c = _deles()
    assert c.velocidade() == pytest.approx(10.0, abs=0.01)
    assert c.area() == pytest.approx(0.004582, rel=1e-3)
    assert c.diametro() * 1000 == pytest.approx(76.4, abs=0.1)


def test_vazao_derivada_da_area():
    c = _nos()
    assert c.vazao() == pytest.approx(0.005632 * 10.0, rel=1e-2)


def test_k_de_entrada_bate_com_a_definicao():
    assert k_de_intensidade(0.01, 10.0) == pytest.approx(0.015)
    assert _nos().k_entrada() == pytest.approx(0.015, rel=1e-2)


def test_areas_e_diametros():
    assert area_de_vazao(0.04582, 10.0) == pytest.approx(0.004582, rel=1e-4)
    assert diametro_equivalente(0.004582) * 1000 == pytest.approx(76.38, abs=0.02)
    with pytest.raises(ValueError):
        diametro_equivalente(0.0)
    with pytest.raises(ValueError):
        area_de_vazao(1.0, 0.0)


def test_comparar_marca_igual_e_diferente():
    d = {x["campo"]: x for x in comparar(_nos(), _deles())}
    assert d["vetor de entrada [m/s]"]["igual"] is True
    assert d["nível inicial [m]"]["igual"] is False
    assert d["intensidade de turbulência"]["igual"] is True
    assert d["k de entrada [m²/s²]"]["igual"] is True          # mesma intensidade e mesmo |U|: mesmo k


def test_item_nao_declarado_nao_vira_igualdade():
    d = {x["campo"]: x for x in comparar(_nos(), _deles())}
    assert d["malha [células]"]["declarado"] is False
    assert d["malha [células]"]["igual"] is None
    assert d["malha [células]"]["a"] == "não declarado"


def test_divergentes_lista_so_o_que_difere_e_esta_declarado():
    campos = [x["campo"] for x in divergentes(_nos(), _deles())]
    assert "nível inicial [m]" in campos
    assert "solver" in campos
    assert "vetor de entrada [m/s]" not in campos


def test_tabela_md_marca_a_divergencia():
    t = tabela_md(_nos(), _deles())
    assert t.splitlines()[0].startswith("| condição | nosso | deles |")
    assert "**difere**" in t and "não declarado" in t


def test_modulo_recusa_vetor_errado():
    from pinneapple_simulation.pipelines.case_card import modulo
    with pytest.raises(ValueError):
        modulo((1.0, 2.0))
