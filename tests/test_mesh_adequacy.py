"""A malha é suficiente para o número reportado? (``mesh_adequacy``)"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.validation.mesh_adequacy import (
    Criterio,
    indice_convergencia,
    resolucao_da_linha,
    resolver_jato,
    verificar,
)


def test_gci_de_convergencia_monotonica():
    """Diferenças caindo pela metade a cada refino com razão 2: p = 1 e extrapolação conhecida."""
    f = [7.00, 7.50, 8.50]                       # f2 - f1 = 0,5; f3 - f2 = 1,0 -> razão 2 -> p = 1
    g = indice_convergencia(f)
    assert g["monotonico"] is True
    assert g["p"] == pytest.approx(1.0, abs=0.02)
    assert g["extrapolado"] == pytest.approx(7.0 + 0.5 / (2 ** 1 - 1), rel=1e-6)
    assert g["gci"] == pytest.approx(1.25 * 0.5 / 7.0 / (2 ** 1 - 1), rel=1e-6)


def test_gci_de_ordem_dois():
    f = [7.00, 7.25, 8.00]                       # 0,25 e 0,75 -> razão 3 -> p = log2(3)
    g = indice_convergencia(f)
    assert g["p"] == pytest.approx(np.log2(3), abs=0.02)
    assert g["gci"] < 0.15


def test_convergencia_nao_monotonica_nao_convergiu():
    g = indice_convergencia([7.0, 7.5, 7.4])
    assert g["monotonico"] is False and g["convergiu"] is False and g["p"] is None


def test_malhas_mais_finas_coincidindo_e_convergencia():
    g = indice_convergencia([7.05, 7.05, 7.60])
    assert g["convergiu"] is True and g["gci"] == 0.0


def test_razoes_invalidas_falham():
    with pytest.raises(ValueError):
        indice_convergencia([1, 2])
    with pytest.raises(ValueError):
        indice_convergencia([1, 2, 3], razoes=[1.0, 2.0])


def test_resolucao_da_linha_acha_a_quantizacao():
    """Linha de 0,1 m sobre células de 0,5 m: só quatro células distintas, passo de 0,5 m."""
    centros = np.stack([np.zeros(20), np.arange(20) * 0.5, np.zeros(20)], axis=1)
    pontos = np.stack([np.zeros(10), np.arange(10) * 0.1, np.zeros(10)], axis=1)
    r = resolucao_da_linha(centros, pontos)
    assert r["celulas_unicas"] == 3                     # os 10 pontos caem em 3 células (y = 0,0 / 0,5 / 1,0)
    assert r["passo_max"] == pytest.approx(0.5, abs=1e-9)
    assert r["passo_medio"] == pytest.approx(0.5, abs=1e-9)
    assert r["fracao_repetida"] == pytest.approx(0.7)


def test_linha_fina_tem_passo_do_proprio_espacamento():
    centros = np.stack([np.zeros(101), np.arange(101) * 0.227, np.zeros(101)], axis=1)
    pontos = centros.copy()
    r = resolucao_da_linha(centros, pontos)
    assert r["celulas_unicas"] == 101 and r["passo_max"] == pytest.approx(0.227, abs=1e-6)


def test_resolver_jato_marca_insuficiente():
    assert resolver_jato(0.005, 0.08).ok is True         # 16 células por diâmetro: passa
    assert resolver_jato(0.05, 0.08).ok is False         # 1,6 célula por diâmetro: reprova
    assert resolver_jato(0.008, 0.08, minimo=10.0).ok is True      # exatamente 10


def test_verificar_sem_dados_nao_diz_que_esta_bom():
    v = verificar()
    assert v["ok"] is None and "sem critério" in v["resumo"]


def test_verificar_junta_os_criterios():
    centros = np.stack([np.zeros(50), np.arange(50) * 0.2, np.zeros(50)], axis=1)
    pontos = np.stack([np.zeros(50), np.arange(50) * 0.2, np.zeros(50)], axis=1)
    v = verificar(centros=centros, pontos=pontos, valores=[7.00, 7.02, 7.08], delta_x=0.005, diametro=0.08)
    nomes = [c["nome"] for c in v["criterios"]]
    assert any("passo máximo" in n for n in nomes) and any("GCI" in n for n in nomes)
    assert v["ok"] is True                              # linha resolvida, 16 células/diâmetro, GCI abaixo de 5 %

    ruim = verificar(centros=centros, pontos=pontos, valores=[7.0, 7.5, 8.5], delta_x=0.05, diametro=0.08)
    assert ruim["ok"] is False                          # mesh grosseira e sem convergência


def test_criterio_serializa():
    assert Criterio("x", 1.0, 2.0, True).to_dict()["ok"] is True
