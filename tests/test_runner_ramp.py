"""A rampa do canal de corrida: parâmetros globais, sólido fechado e cotas publicadas do CAD."""
import numpy as np
import pytest

from pinneapple_design.geometry.runner_ramp import (CASOS, PUBLICADO, Rampa, facetas_abertas, piso, solido,
                                                    validar)


def test_as_alturas_batem_com_o_cad_publicado():
    """Referência fechada desta geometria: crista e pé acima do fundo, contra a tabela do CAD (3 mm)."""
    r = validar()
    assert r["ok"], r
    assert r["pior_desvio_mm"] <= r["tolerancia_mm"]
    for k, (pub_crista, pub_pe) in zip(CASOS, PUBLICADO.values()):
        assert abs(r["casos"][k]["altura_crista_mm"] - pub_crista) <= r["tolerancia_mm"]
        assert abs(r["casos"][k]["altura_pe_mm"] - pub_pe) <= r["tolerancia_mm"]


def test_o_patamar_sai_com_um_quarto_de_metro():
    """Assinatura da reconstrução: o trecho inclinado cai exatamente dZ e sobra o mesmo patamar nas cinco."""
    for k, c in CASOS.items():
        *_, patamar = c.trechos()
        assert abs(patamar - 0.25) < 0.01, (k, patamar)


def test_o_perfil_desce_e_termina_no_pe():
    c = CASOS["V02"]
    assert c.topo(c.s_inicio) == pytest.approx(c.z_crista)
    assert c.topo(c.s_fim) == pytest.approx(c.z_pe)
    assert np.isnan(c.topo(c.s_inicio - 0.1)) and np.isnan(c.topo(c.s_fim + 0.1))
    t = np.array([c.topo(x) for x in np.linspace(c.s_inicio, c.s_fim, 400)])
    assert np.all(np.diff(t) <= 1e-12)                      # desce sempre, nunca sobe


def test_o_solido_e_fechado_e_vai_do_piso_a_crista():
    for k, c in CASOS.items():
        v, f = solido(c, n_s=24)
        assert facetas_abertas(f) == 0, k
        z = np.array([p[2] for p in v])
        assert z.max() == pytest.approx(c.z_crista)
        assert z.min() == pytest.approx(piso(c.s_fim))
        x = np.array([p[0] for p in v])
        assert x.min() == pytest.approx(c.x_eixo - c.largura / 2)
        assert x.max() == pytest.approx(c.x_eixo + c.largura / 2)


def test_dentro_e_fora():
    c = CASOS["V03"]
    y = c.y_origem + 5.0
    assert c.contem(c.x_eixo, y, c.topo(y - c.y_origem) - 0.05)   # dentro, sob o topo daquele s
    assert not c.contem(c.x_eixo, y, c.z_crista + 0.05)      # acima do topo
    assert not c.contem(c.x_eixo + c.largura, y, c.z_crista - 0.05)   # fora da largura
    assert not c.contem(c.x_eixo, c.y_origem + c.s_inicio - 0.05, c.z_crista - 0.05)  # a montante


def test_rampa_e_imutavel():
    with pytest.raises(Exception):
        CASOS["V02"].z_crista = -1.0                          # dataclass frozen: parâmetro não muda por engano
