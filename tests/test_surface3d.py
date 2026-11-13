"""Quadros 3D de um escoamento com superfície livre (``Cena3D``, contorno de STEP, MP4/PNG)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from pinneapple_tools.visualization.dashboard import (
    Cena3D,
    animacao_de_campos,
    contorno_de_step,
    gerar_animacao3d,
    quadros_de_campos,
)


def _campos(tmp_path: Path, n: int = 3) -> Path:
    """Um ``campos.npz`` no formato do rodar_caso: superfície, plano do jato e o campo."""
    rng = np.random.default_rng(0)
    sup = rng.normal(size=(40, 3)).astype(np.float32) + np.array([0.0, 5.0, -2.0], np.float32)
    corte = rng.normal(size=(30, 3)).astype(np.float32) + np.array([0.0, 6.0, -1.5], np.float32)
    dados = {"tempos": np.arange(1, n + 1, dtype=float), "corte_xyz": corte}
    for i in range(n):
        dados[f"sup_xyz_{i}"] = sup
        dados[f"sup_k_{i}"] = np.abs(rng.normal(size=len(sup))).astype(np.float32)
        dados[f"corte_alpha_{i}"] = (rng.random(len(corte)) > 0.5).astype(np.uint8) * 255
        dados[f"corte_umag_{i}"] = np.abs(rng.normal(size=len(corte))).astype(np.float32) * 5
    f = tmp_path / "campos.npz"
    np.savez_compressed(f, **dados)
    return f


def _cena() -> Cena3D:
    # um caixote de contorno, como o canal: precisa de extensão nos três eixos, senão a projeção 3D é singular
    linhas = [np.array([[x, y, z] for x, y, z in ((0, 0, -3), (2, 0, -3), (2, 12, -3), (0, 12, -3), (0, 0, -3),
                                                  (0, 0, -2), (2, 0, -2), (2, 12, -2), (0, 12, -2), (0, 0, -2))],
                       np.float32)]
    return Cena3D(linhas=linhas, kmax=2.0, umag_min=4.0, campo="k", unidade="m²/s²")


def test_quadros_de_campos_le_todos_os_instantes(tmp_path):
    d = quadros_de_campos(_campos(tmp_path, 3))
    assert [q["t"] for q in d["quadros"]] == [1.0, 2.0, 3.0]
    assert d["corte"].shape == (30, 3)
    assert d["quadros"][0]["alpha"].max() <= 1.0                 # alpha em uint8 é normalizado para [0, 1]
    assert d["quadros"][0]["sup"].shape == (40, 3)


def test_instante_sem_superficie_e_pulado(tmp_path):
    f = _campos(tmp_path, 2)
    z = dict(np.load(f))
    z.pop("sup_xyz_1")                                           # segundo instante incompleto
    np.savez_compressed(f, **z)
    d = quadros_de_campos(f)
    assert [q["t"] for q in d["quadros"]] == [1.0]


def test_cena_sem_lims_tira_limites_do_contorno():
    lims = _cena().limites()
    assert lims[0] == (0.0, 2.0) and lims[1] == (0.0, 12.0) and lims[2] == (-3.0, -2.0)


def test_contorno_plano_nao_quebra_a_projecao():
    from pinneapple_tools.visualization.dashboard.surface3d import _expandir
    assert _expandir(((0.0, 0.0), (0.0, 1.0), (-3.0, -2.0)))[0] == (-0.5, 0.5)


def test_cena_sem_linhas_e_sem_lims_falha():
    with pytest.raises(ValueError):
        Cena3D().limites()


def test_campos_sem_superficie_falha_em_vez_de_gerar_video_em_branco(tmp_path):
    f = tmp_path / "vazio.npz"
    np.savez_compressed(f, tempos=np.array([1.0]), corte_xyz=np.zeros((3, 3), np.float32),
                        corte_alpha_0=np.zeros(3, np.uint8), corte_umag_0=np.zeros(3, np.float32))
    with pytest.raises(ValueError):
        gerar_animacao3d("X", f, tmp_path, _cena())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg não instalado")
def test_gerar_escreve_mp4_e_png(tmp_path):
    r = animacao_de_campos(_campos(tmp_path), tmp_path / "saida", "R0", _cena(), painel=(1, 2))
    assert r["mp4"].exists() and r["mp4"].stat().st_size > 0
    assert r["png"].exists() and r["png"].stat().st_size > 0
    assert r["quadros"] == 3


def test_contorno_de_step_le_cache(tmp_path):
    """Com cache presente, não toca no cadquery (que é opcional)."""
    cache = tmp_path / "contorno.npz"
    np.savez_compressed(cache, e0=np.zeros((2, 3), np.float32), e1=np.ones((2, 3), np.float32))
    linhas = contorno_de_step("inexistente.stp", cache)
    assert len(linhas) == 2 and linhas[0].shape == (2, 3)
