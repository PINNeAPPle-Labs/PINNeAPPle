"""Linhas de amostragem, janelas e posição do pico: o pós-processamento genérico, sem solver."""
import numpy as np
import pytest

from pinneapple_simulation.pipelines.field_lines import (
    Linha,
    comparar_picos,
    desvio_mediano_passos,
    janela,
    joelho,
    media_faixa,
    perfil_medio,
    pico,
    pico_em_liquido,
    picos_por_linha,
    plato,
    quantizacao,
)


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


def _perfil(alturas, n_t=30):
    """Série com o perfil espacial dado, constante no tempo (para testar forma do perfil, não transiente)."""
    k = np.tile(np.asarray(alturas, float), (n_t, 1))
    return k, np.arange(1, n_t + 1) * 1.0


def test_plato_mede_a_faixa_e_a_margem():
    """Dois pontos quase empatados no topo: a margem é pequena e a faixa cobre os dois."""
    s = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    k, t = _perfil([0.1, 0.2, 0.9, 0.902, 0.5, 0.2])
    r = plato(k, t, s, fracao=0.85)
    assert r["pico"]["s"] == pytest.approx(3.0)
    assert r["margem"] == pytest.approx(1 - 0.9 / 0.902, rel=1e-6)      # 0,2 % de margem
    assert r["s_ini"] == pytest.approx(2.0) and r["s_fim"] == pytest.approx(3.0)
    assert r["largura"] == pytest.approx(1.0)
    assert r["passo"] == pytest.approx(1.0)


def test_plato_de_perfil_estreito_tem_largura_de_um_ponto():
    s = np.arange(6, dtype=float)
    k, t = _perfil([0.1, 0.1, 0.1, 5.0, 0.1, 0.1])
    r = plato(k, t, s, fracao=0.85)
    assert r["largura"] == pytest.approx(0.0)
    assert r["margem"] == pytest.approx(1 - 0.1 / 5.0)


def test_joelho_acha_o_degrau_do_perfil():
    """Perfil que sobe em degrau: o joelho é onde k alcança o dobro do nível de montante (s_ref = 1 m)."""
    s = np.arange(0.0, 8.1, 1.0)
    k, t = _perfil([0.10, 0.10, 0.10, 0.20, 0.40, 0.60, 0.30, 0.20, 0.10])
    r = joelho(k, t, s, s_ref=1.0, fator=2.0)
    assert r == pytest.approx(3.0, abs=0.05)         # 0,20 é o dobro de 0,10, e o perfil o alcança em s = 3 m


def test_joelho_sem_cruzamento_devolve_none():
    s = np.arange(0.0, 5.0, 1.0)
    k, t = _perfil([1.0, 0.9, 0.8, 0.7, 0.6])        # perfil decrescente: nunca dobra
    assert joelho(k, t, s, s_ref=0.0, fator=2.0) is None


def test_quantizacao_mede_o_maior_vao():
    """Linha com um vão grande no meio: a resolução é o maior vão, não a média."""
    d = np.array([0.0, 0.2, 0.4, 1.6, 1.8])
    q = quantizacao(d)
    assert q["pontos"] == 5
    assert q["passo_max"] == pytest.approx(1.2, abs=1e-9)
    assert q["passo_medio"] == pytest.approx(0.45, abs=1e-9)
    assert q["fracao_em_passos_grandes"] == pytest.approx(0.25)


def test_quantizacao_de_linha_regular():
    d = np.arange(101) * 0.227
    q = quantizacao(d)
    assert q["passo_max"] == pytest.approx(0.227, abs=1e-9)
    assert q["fracao_em_passos_grandes"] == 0.0


def test_quantizacao_de_um_ponto_nao_inventa_passo():
    assert quantizacao(np.array([1.0]))["passo_max"] is None


# --- pico por fase -------------------------------------------------------------------------------------------------
def _linha_com_ar():
    """Quatro pontos: o maior k está em ar (alpha 0); em líquido o maior é o último."""
    k = np.tile(np.array([0.1, 0.2, 9.0, 0.5]), (10, 1))
    a = np.tile(np.array([1.0, 1.0, 0.0, 1.0]), (10, 1))
    return k, np.arange(1, 11.0), np.array([0.0, 1.0, 2.0, 3.0]), a


def test_pico_em_liquido_ignora_o_maximo_que_caiu_em_ar():
    k, t, d, a = _linha_com_ar()
    r = pico_em_liquido(k, t, d, a)
    assert r["filtrado"] and r["s"] == 3.0 and r["k"] == pytest.approx(0.5)
    assert r["s_sem_filtro"] == 2.0 and r["k_sem_filtro"] == pytest.approx(9.0)
    assert r["pontos_liquido"] == 3


def test_pico_em_liquido_sem_liquido_devolve_o_da_linha_e_diz_por_que():
    k, t, d, _ = _linha_com_ar()
    r = pico_em_liquido(k, t, d, np.zeros_like(k))
    assert r["filtrado"] is False and r["s"] == 2.0 and "ar" in r["motivo"]


def test_pico_em_liquido_recusa_alpha_de_outro_formato():
    k, t, d, _ = _linha_com_ar()
    with pytest.raises(ValueError):
        pico_em_liquido(k, t, d, np.ones((3, 4)))


# --- origem da distância -------------------------------------------------------------------------------------------
def test_origem_da_linha_mede_o_primeiro_ponto_amostrado():
    """No canal, a linha começa em 0 mas o primeiro ponto amostrado está 1,36 m adiante."""
    from pinneapple_simulation.pipelines.field_lines import origem_da_linha
    assert origem_da_linha(np.array([1.364, 1.591, 1.819])) == pytest.approx(1.364)


def test_deslocamento_ref_põe_as_duas_tabelas_na_mesma_origem():
    picos = {"LEdir": {"k": 1.0, "s": 6.820}}
    ref = {"LEdir": (0.6, 5.683)}
    sem = comparar_picos(picos, ref, passo=0.22734)[0]
    com = comparar_picos(picos, ref, passo=0.22734, deslocamento_ref={"LEdir": 1.364})[0]
    assert sem["ds"] == pytest.approx(1.137) and com["ds"] == pytest.approx(-0.227, abs=1e-6)
    assert com["passos"] == pytest.approx(-1.0, abs=1e-2)


def test_deslocamento_ref_aceita_um_numero_para_todas_as_linhas():
    r = comparar_picos({"A": {"k": 1.0, "s": 2.0}}, {"A": (1.0, 1.0)}, deslocamento_ref=1.0)[0]
    assert r["ds"] == pytest.approx(0.0)


# --- descoberta do deslocamento de origem ---------------------------------------------------------------------------
def _gauss(s, centro=7.0, largura=2.0):
    return np.exp(-((np.asarray(s) - centro) ** 2) / largura)


def test_deslocamento_de_origem_acha_o_metro_zero_entre_duas_tabelas():
    """É o teste que separou 'duas convenções' de 'dois escoamentos' no caso do canal."""
    from pinneapple_simulation.pipelines.field_lines import deslocamento_de_origem
    s = np.arange(0, 20, 0.22734)
    r = deslocamento_de_origem(s, _gauss(s), s - 1.364, _gauss(s), passo=0.22734)
    assert r["deslocamento"] == pytest.approx(1.364, abs=0.05)
    assert r["e_o_mesmo_perfil"] is True
    assert r["erro_rms"] < 0.05 * r["erro_sem_deslocar"]


def test_deslocamento_de_origem_nao_chama_de_convencao_um_caso_diferente():
    from pinneapple_simulation.pipelines.field_lines import deslocamento_de_origem
    s = np.arange(0, 20, 0.22734)
    r = deslocamento_de_origem(s, _gauss(s, 7.0, 2.0), s, _gauss(s, 8.5, 0.8), passo=0.22734)
    assert r["e_o_mesmo_perfil"] is False


def test_deslocamento_de_origem_recusa_perfil_curto_demais():
    from pinneapple_simulation.pipelines.field_lines import deslocamento_de_origem
    with pytest.raises(ValueError):
        deslocamento_de_origem([0, 1], [1, 2], [0, 1], [1, 2])
