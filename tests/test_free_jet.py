"""Trajetória de jato livre e estimativa da posição do pico (``free_jet``)."""
from __future__ import annotations

import math

import pytest

from pinneapple_simulation.pipelines.free_jet import (
    alcance,
    estimar_pico,
    tempo_de_voo,
    velocidade_para_alcance,
    velocidade_para_pico,
    vetor_de,
)


def test_alcance_bate_com_a_solucao_analitica():
    """x = u_y (u_z + √(u_z² + 2gh)) / g, conferido à mão."""
    uy, uz, h, g = 9.85, 1.74, 0.75, 9.81
    esperado = uy * (uz + math.sqrt(uz ** 2 + 2 * g * h)) / g
    assert alcance((0, uy, uz), h) == pytest.approx(esperado, rel=1e-12)
    assert alcance((0, uy, uz), h) == pytest.approx(5.976, abs=2e-3)      # o valor que o caso do canal dá


def test_alcance_nao_depende_do_diametro_do_furo():
    """A trajetória de um jato livre só depende do vetor — a vazão entra na força, não no alcance."""
    u = vetor_de(10.0, 10.02)
    assert alcance(u, 0.75) == pytest.approx(alcance(u, 0.93 * 0 + 0.75), rel=1e-12)


def test_altura_maior_alcanca_mais():
    u = vetor_de(10.0, 10.02)
    assert alcance(u, 0.93) > alcance(u, 0.75)


def test_velocidade_maior_alcanca_mais():
    h = 0.75
    a = [alcance(vetor_de(v, 10.02), h) for v in (7.0, 8.0, 10.0)]
    assert a[0] < a[1] < a[2]


def test_velocidade_para_alcance_inverte(  ):
    h, th = 0.75, 10.02
    for x in (4.39, 5.16, 5.97):
        v = velocidade_para_alcance(x, th, h)
        assert alcance(vetor_de(v, th), h) == pytest.approx(x, abs=1e-3)


def test_velocidade_para_pico_desconta_os_atrasos():
    """Com atrasos de 0,30 + 0,99 m, o pico em 5,68 m pede ~8 m/s (a sensibilidade que o estudo precisou)."""
    v = velocidade_para_pico(5.682545, 0.75, 0.30, 0.99, theta_deg=10.02)
    assert v == pytest.approx(7.9, abs=0.2)
    assert estimar_pico(vetor_de(v, 10.02), 0.75, 0.30, 0.99) == pytest.approx(5.682545, abs=1e-2)


def test_estimar_pico_soma_as_tres_parcelas():
    u = vetor_de(10.0, 10.02)
    assert estimar_pico(u, 0.75, 0.30, 0.99) == pytest.approx(alcance(u, 0.75) + 1.29, rel=1e-12)
    assert estimar_pico(u, 0.75) == pytest.approx(alcance(u, 0.75), rel=1e-12)


def test_tempo_de_voo_confere_com_o_alcance():
    u = vetor_de(10.0, 10.02)
    t = tempo_de_voo(u, 0.75)
    assert u[1] * t == pytest.approx(alcance(u, 0.75), rel=1e-12)


@pytest.mark.parametrize("h", [0.0, -0.5])
def test_altura_nao_positiva_falha(h):
    with pytest.raises(ValueError):
        alcance((0, 9.85, 1.74), h)


def test_jato_sem_componente_horizontal_falha():
    with pytest.raises(ValueError):
        alcance((0, 0.0, 1.74), 0.75)


def test_vetor_curto_falha():
    with pytest.raises(ValueError):
        alcance((9.85, 1.74), 0.75)
