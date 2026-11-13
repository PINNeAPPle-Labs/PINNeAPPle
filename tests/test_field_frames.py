"""Quadros de campo salvo e tabela de relatório para o Dashboard."""
import numpy as np
import pytest

from pinneapple_tools.visualization.dashboard import Table, frames_from_field


def _npz(tmp_path, n_t=3, n_p=400):
    rng = np.random.default_rng(0)
    y = np.linspace(-7.4, 15.3, n_p)
    z = -2.0 - rng.random(n_p)
    d = {"tempos": np.arange(1, n_t + 1, dtype=float)}
    for i in range(n_t):
        d[f"sup_xyz_{i}"] = np.column_stack([np.full(n_p, -2.1), y, z])
        d[f"sup_k_{i}"] = rng.random(n_p) * (i + 1)        # escala cresce com o tempo
    f = tmp_path / "campos.npz"
    np.savez_compressed(f, **d)
    return f


def test_quadros_do_campo(tmp_path):
    t, quadros = frames_from_field(_npz(tmp_path), larg=80, alt=40)
    assert len(t) == len(quadros) == 3
    for q in quadros:
        assert q.shape == (40, 80, 3) and q.dtype == np.uint8
        assert q.max() > 0


def test_a_escala_de_cor_e_comum_aos_quadros(tmp_path):
    """Sem limite comum a cor mudaria de quadro para quadro: os quadros têm de ser diferentes entre si."""
    _, quadros = frames_from_field(_npz(tmp_path), larg=60, alt=30)
    assert not np.array_equal(quadros[0], quadros[-1])


def test_npz_sem_o_campo_falha_alto(tmp_path):
    f = tmp_path / "vazio.npz"
    np.savez_compressed(f, tempos=np.array([1.0]))
    with pytest.raises(ValueError, match="sem campos"):
        frames_from_field(f)


def test_tabela_do_relatorio():
    t = Table("k nas linhas", ["linha", "nosso k", "deles k"],
              [["LEdir", 1.3369, 0.6001], ["LEesq", None, 0.5731]], alinhamento=["l", "r", "r"])
    h = t.to_html()
    assert "<th align=\"right\">nosso k</th>" in h and "LEdir" in h
    assert "—" in h and "1.337" in h          # None vira travessão; float com 4 algarismos


def test_tabela_escapa_html():
    h = Table("t", ["a"], [["<script>x</script>"]]).to_html()
    assert "<script>" not in h and "&lt;script&gt;" in h
