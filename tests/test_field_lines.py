"""Linhas de amostragem, janelas e posição do pico: o pós-processamento genérico, sem solver."""
import numpy as np
import pytest

from pinneapple_simulation.pipelines.field_lines import (Linha, comparar_picos, desvio_mediano_passos, janela,
                                                        media_faixa, perfil_medio, pico, picos_por_linha)


def _serie(pico_em=20, valor=0.8, base=0.1, n_t=300, n_p=101, t_ini_pico=20.0):
    """Série sintética: um pico espacial que só existe depois de t_ini_pico (transiente inicial separado)."""
    t = np.arange(1, n_t + 1) * 0.1
    k = np.full((n_t, n_p), base)
    k[:, pico_em] = valor
    k[t < t_ini_pico, pico_em] = 3.0          # pico de partida, mais alto e no mesmo ponto: não deve dominar 20-30 s
    k[t < t_ini_pico, pico_em + 10] = 5.0     # e um pico de partida em outro ponto, que domina 0-30 s
    return k, t


def test_linha_pontos_e_distancias():
    ln = Linha("LEdir", x=-1.8, z=-2.2, y_inicio=-7.429, y_fim=15.305, n=101)
    p = ln.pontos
    assert p.shape == (101, 3) and np.allclose(p[:, 0], -1.8) and np.allclose(p[:, 2], -2.2)
    d = ln.distancias(y_origem=-7.429)
    assert d[0] == pytest.approx(0.0) and d[-1] == pytest.approx(22.734)
    assert np.allclose(np.diff(d), 0.22734)


def test_janela_e_tolerancia():
    t = np.arange(1, 301) * 0.1
    assert janela(t, 20, 30).sum() == 101        # 20,0 a 30,0 inclusive, sem depender do arredondamento
    assert janela(t, 0, 30).sum() == 300


def test_o_pico_depende_da_janela_e_isso_e_o_ponto():
    k, t = _serie()
    d = np.arange(101) * 0.22734
    assert pico(k, t, d, (0, 30))[2] == 30        # 0-30 s: o transiente de partida manda
    assert pico(k, t, d, (20, 30))[2] == 20       # 20-30 s: o regime
    assert pico(k, t, d)[2] == 30                 # sem janela explícita, é tudo
    with pytest.raises(ValueError):
        pico(k, t, d, (40, 50))                   # janela sem instante falha alto, não devolve lixo


def test_perfil_e_media_de_faixa():
    k, t = _serie()
    d = np.arange(101) * 0.22734
    prof = perfil_medio(k, t, (20, 30))
    assert prof.shape == (101,) and prof[20] == pytest.approx(0.8)
    assert media_faixa(k, t, d, 2.0, 8.0, (20, 30)) == pytest.approx(prof[(d >= 2) & (d <= 8)].mean())
    assert media_faixa(k, t, d, 50.0, 60.0, (20, 30)) is None


def test_comparacao_com_referencia_e_passos():
    k, t = _serie()
    d = np.arange(101) * 0.22734
    picos = picos_por_linha({"LEdir": (k, t, d)}, (20, 30))
    ref = {"LEdir": (0.7, d[20] - 2 * 0.22734)}    # referência dois passos a montante
    c = comparar_picos(picos, ref, passo=0.22734)
    assert c[0]["ds"] == pytest.approx(2 * 0.22734) and c[0]["passos"] == pytest.approx(2.0)
    assert c[0]["razao_k"] == pytest.approx(0.8 / 0.7)
    assert desvio_mediano_passos(c) == pytest.approx(2.0)
    assert comparar_picos({}, ref, passo=0.22734)[0]["passos"] is None   # linha ausente não inventa número
