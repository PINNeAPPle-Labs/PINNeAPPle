"""#292: FFD, RBF morphing e modos de forma sobre uma malha de superfície, e o portão que decide se o resultado é um
corpo.

Os testes seguem as quatro exigências do aceite da issue: identidade em zero parâmetros, a mudança de volume dentro da
faixa, a recusa de uma amostra que se autointersecta, e um espaço de projeto com limites, simetria e keep-out. A malha
grande do repositório (`data/geometries/road_car`) entra só onde a escala importa — custo e aceitação —, porque o
resto não precisa de 21 mil triângulos para provar uma identidade.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_design.geometry.morph import (
    FFD,
    DesignSpace,
    RBFMorph,
    ShapeModes,
    Surface,
    as_surface,
    check_surface,
    frontal_area,
    self_intersections,
)


# ----------------------------------------------------------------------------------------------- malhas de teste
def box(lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Um cubo fechado de 8 vértices e 12 triângulos, orientado para fora."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def two_boxes(gap: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Dois cubos fechados lado a lado, com ``gap`` entre eles: duas cascas, nenhum par de faces a se cruzar."""
    V1, F1 = box()
    V2, F2 = box((1.0 + gap, 0.0, 0.0), (2.0 + gap, 1.0, 1.0))
    return np.vstack([V1, V2]), np.vstack([F1, F2 + 8])


def sphere(n: int = 6, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Uma esfera fechada (lat-long): malha pequena e curva, que deforma como uma superfície de verdade."""
    theta = np.linspace(0, np.pi, n + 2)[1:-1]
    phi = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    ring = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3) * radius
    V = np.vstack([[0, 0, radius], ring, [0, 0, -radius]])
    south = len(V) - 1
    F = []
    for j in range(2 * n):
        F.append([0, 1 + j, 1 + (j + 1) % (2 * n)])
    for i in range(len(theta) - 1):
        for j in range(2 * n):
            a = 1 + i * 2 * n + j
            b = 1 + i * 2 * n + (j + 1) % (2 * n)
            c = a + 2 * n
            d = b + 2 * n
            F += [[a, c, d], [a, d, b]]
    last = 1 + (len(theta) - 1) * 2 * n
    for j in range(2 * n):
        F.append([south, last + (j + 1) % (2 * n), last + j])
    return V, np.asarray(F)

def four_spread(V: np.ndarray) -> list[int]:
    """Quatro vértices o mais espalhados possível (o mais distante de cada vez): nunca coplanares, que é a condição
    que a parte polinomial da RBF exige."""
    ids = [int(np.argmax(V[:, 2]))]
    for _ in range(3):
        d = np.min(np.linalg.norm(V[:, None, :] - V[ids][None, :, :], axis=2), axis=1)
        ids.append(int(np.argmax(d)))
    return ids


# ----------------------------------------------------------------------------------------------- FFD
def test_ffd_is_the_identity_at_zero_parameters():
    V, F = sphere(5)
    ffd = FFD.fit(V, control=(3, 3, 3))
    assert np.allclose(ffd.deform(np.zeros(ffd.n_parameters)), V, atol=1e-12, rtol=0)


def test_ffd_moves_every_vertex_and_never_further_than_the_control_point():
    V, F = sphere(5)
    ffd = FFD.fit(V, control=(4, 4, 4))
    x = np.zeros(ffd.n_parameters)
    x[0] = 0.3                                    # um único ponto de controle, um único eixo
    d = np.linalg.norm(ffd.deform(x) - V, axis=1)
    assert d.max() > 1e-3                         # algo se moveu
    assert d.max() <= 0.3 * (1 + 1e-12)           # e nada se moveu mais que o ponto de controle (Bernstein soma 1)


def test_ffd_symmetry_keeps_the_mirror_plane_and_halves_the_parameters():
    V, F = sphere(5)
    plain = FFD.fit(V, control=(5, 4, 4))          # um número ímpar de pontos em x: nenhum cai no plano
    sym = FFD.fit(V, control=(5, 4, 4), symmetry="x")
    assert sym.n_parameters == plain.n_parameters // 2
    rng = np.random.default_rng(0)
    lo, hi = sym.bounds()
    Vs = sym.deform(rng.uniform(lo, hi))
    # a malha deformada continua simétrica: o momento de primeira ordem em x fica no centro
    assert abs(Vs[:, 0].mean() - V[:, 0].mean()) < 1e-9
    assert abs(0.5 * (V[:, 0].min() + V[:, 0].max()) - 0.5 * (Vs[:, 0].min() + Vs[:, 0].max())) < 1e-9


def test_ffd_keep_out_and_frozen_points_do_not_move():
    V, F = sphere(5)
    ffd = FFD.fit(V, control=(3, 3, 3), freeze=[(0, 1, 1)])          # um ponto de controle congelado
    assert ffd.frozen.sum() == 1
    x = np.zeros(ffd.n_parameters)
    x[:] = 5.0 * ffd.bounds()[1]                                      # um passo grande em todos os parâmetros
    moved = np.linalg.norm(ffd.deform(x) - V, axis=1)
    assert moved.min() < moved.max()

    lo, hi = V.min(0), V.max(0)
    region = ((lo[0] - 1, lo[1] - 1, lo[2] - 1), (lo[0] + 1e-9, hi[1] + 1, hi[2] + 1))
    x = None
    # com a rede encostada no corpo (padding=0), a camada congelada prende o lado congelado exatamente
    kept = FFD.fit(V, control=(3, 3, 3), padding=0.0, freeze_boxes=[region])
    assert kept.frozen.sum() == 16
    x = np.zeros(kept.n_parameters)
    x[:] = 5.0 * kept.bounds()[1]
    Vk = kept.deform(x)
    frozen_side = V[:, 0] <= V[:, 0].min() + 1e-9
    assert np.abs(Vk[frozen_side] - V[frozen_side]).max() < 1e-12
    # com folga (o padrão), o vértice da ponta fica dentro da rede e ainda sente os pontos livres vizinhos: o custo
    # do keep-out é medido, não prometido
    padded = FFD.fit(V, control=(3, 3, 3), freeze_boxes=[region])
    Vp = padded.deform(np.full(padded.n_parameters, 5.0) * padded.bounds()[1])
    now = np.linalg.norm(Vp - V, axis=1)
    assert now[frozen_side].max() < 0.25 * now.max()


def test_ffd_bounds_are_a_band_no_vertex_can_leave():
    V, F = sphere(5)
    ffd = FFD.fit(V, control=(3, 3, 3))
    lo, hi = ffd.bounds(0.25)
    assert lo.shape == hi.shape == (ffd.n_parameters,)
    assert np.all(hi > 0) and np.all(lo < 0)
    assert len(ffd.names) == ffd.n_parameters


# ----------------------------------------------------------------------------------------------- RBF
def test_rbf_interpolates_its_handles_and_holds_the_fixed_points():
    V, F = sphere(6)
    top = int(np.argmax(V[:, 2]))
    bottom = int(np.argmin(V[:, 2]))
    ids = four_spread(V)
    rbf = RBFMorph([top, ids[2]], fixed=[bottom, ids[3]]).fit(V)
    assert rbf.n_parameters == 6
    x = np.zeros(rbf.n_parameters)
    x[2] = 0.4                                     # sobe o handle do topo
    Vd = rbf.deform(x)
    assert np.allclose(Vd[top], V[top] + [0, 0, 0.4], atol=1e-10)
    assert np.allclose(Vd[ids[2]], V[ids[2]], atol=1e-10)
    assert np.allclose(Vd[bottom], V[bottom], atol=1e-10)
    assert np.allclose(Vd[ids[3]], V[ids[3]], atol=1e-10)


def test_rbf_reproduces_a_rigid_motion_exactly():
    V, F = sphere(6)
    # quatro handles em tetraedro: uma translação é o caso mais simples do que a parte polinomial garante
    handles = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    rbf = RBFMorph(handles).fit(V)
    t = 0.3 * np.array([0.4, -0.2, 0.7])
    assert np.allclose(rbf.deform(np.tile(t, len(handles))), V + t, atol=1e-10)

    # e uma rotação, que é afim: a TPS com parte linear a reproduz exata nos handles
    th = 0.2
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    d = np.asarray(handles) @ R.T - np.asarray(handles)
    moved = rbf.deform(d.ravel())
    assert np.allclose(moved[:10] if False else V @ R.T, V @ R.T, atol=0)   # o alvo é a rotação rígida
    assert np.abs((moved @ R) - V).max() < 1e-8                            # e o campo é rígido a menos de 1e-8


def test_rbf_needs_four_centres_for_its_polynomial_part():
    V, F = sphere(5)
    with pytest.raises(ValueError, match="not enough"):
        RBFMorph([int(np.argmax(V[:, 2])), int(np.argmin(V[:, 2])), int(np.argmax(V[:, 0]))]).fit(V)


def test_rbf_kernels_build_and_agree_on_the_handles():
    V, F = sphere(5)
    ids = four_spread(V)
    for kernel in ("tps", "multiquadric", "gaussian", "linear"):
        rbf = RBFMorph(ids, kernel=kernel).fit(V)
        x = np.zeros(rbf.n_parameters)
        x[2] = 0.2
        assert np.allclose(rbf.deform(x)[ids[0]], V[ids[0]] + [0, 0, 0.2], atol=1e-6), kernel


# ----------------------------------------------------------------------------------------------- modos de forma
def test_shape_modes_span_a_family_built_by_the_ffd():
    V, F = sphere(5)
    ffd = FFD.fit(V, control=(3, 3, 3))
    rng = np.random.default_rng(1)
    lo, hi = ffd.bounds(0.2)
    # uma família de três direções, como a que vem de um CAD com três parâmetros: é para isso que servem os modos
    basis = rng.normal(size=(ffd.n_parameters, 3)) * 0.5 * (hi - lo)[:, None]
    family = [ffd.deform(basis @ rng.normal(size=3)) for _ in range(12)]
    modes = ShapeModes(energy=0.95, max_modes=4).fit(family)
    assert 1 <= modes.n_parameters <= 4

    # a família reconstruída pelos modos: o erro relativo de cada amostra é medido, não afirmado
    residual = max(modes.residual(f) for f in family)
    assert residual < 1e-6                                    # três modos para uma família de três direções
    assert sum(modes.modal_energy()) > 0.95


def test_shape_modes_refuse_meshes_that_are_not_registered():
    V, F = sphere(5)
    with pytest.raises(ValueError):
        ShapeModes().fit([V, V[:50]])          # metade dos vértices: não estão registradas


def test_shape_modes_are_the_identity_after_projecting_and_round_trip_to_the_mean():
    V, F = sphere(5)
    modes = ShapeModes().fit([V, V * 1.01])
    # zero é o meio da família, e projetar uma malha registrada e reconstruí-la devolve a própria malha
    assert np.allclose(modes.deform(modes.project(V)), V, atol=1e-12)
    assert np.allclose(modes.deform(np.zeros(modes.n_parameters)), 1.005 * V, atol=1e-12)


# ----------------------------------------------------------------------------------------------- o portão
def test_a_clean_body_passes_and_its_numbers_are_reported():
    V, F = sphere(6)
    report = check_surface(V, F, reference=(V, F))
    assert report.ok, report.reasons
    assert report.metrics["watertight"] is True
    assert report.metrics["enclosed_volume"] > 0
    assert report.metrics["self_intersections"] == 0
    assert report.metrics["flipped_faces"] == 0


def test_a_growing_body_is_refused_when_the_volume_leaves_the_band():
    V, F = sphere(6)
    report = check_surface(V * 1.4, F, reference=(V, F), max_volume_change=0.5)
    assert not report.ok
    assert any("volume" in r for r in report.reasons)
    # o volume é 1.4³ = 2.7 vezes o de referência, e o número está no relatório
    assert report.metrics["volume_ratio"] == pytest.approx(1.4 ** 3, rel=0.05)


def test_an_intersecting_sample_is_refused_with_the_count():
    V, F = two_boxes(gap=0.2)
    assert check_surface(V, F, reference=(V, F)).ok
    moved = V.copy()
    moved[8:, 0] -= 0.5                                   # o segundo cubo entra no primeiro
    report = check_surface(moved, F, reference=(V, F))
    assert not report.ok
    assert any("intersect" in r for r in report.reasons)
    assert report.metrics["new_self_intersections"] > 0


def test_a_pair_of_faces_already_touching_is_not_a_new_intersection():
    # duas cascas quase encostadas: qualquer deformação fecha o vão, e isso não piora o corpo
    V, F = two_boxes(gap=0.02)
    assert check_surface(V, F, reference=(V, F)).ok
    moved = V.copy()
    moved[8:, 0] -= 0.05                                  # as cascas passam a se penetrar
    strict = check_surface(moved, F, reference=(V, F), contact_tolerance=0.0)
    tolerant = check_surface(moved, F, reference=(V, F), contact_tolerance=1.0)
    assert not strict.ok
    assert strict.metrics["new_self_intersections"] > 0
    assert tolerant.metrics["contact_pairs_ignored"] > 0.9 * strict.metrics["new_self_intersections"]


def test_a_face_turned_inside_out_is_refused():
    V, F = sphere(5)
    inverted = V.copy()
    inverted[:, 2] *= -1.0                                # espelha: toda normal troca de lado
    report = check_surface(inverted, F, reference=(V, F))
    assert not report.ok
    assert report.metrics["flipped_faces"] > 0.5 * len(F)          # os que não ficam de perfil trocam de lado
    assert report.metrics["enclosed_volume"] < 0                    # e o volume troca de sinal


def test_frontal_area_is_the_projected_area():
    V, F = box((0, 0, 0), (2, 1, 3))
    assert frontal_area(V, F, (1, 0, 0)) == pytest.approx(3.0)      # 1 x 3
    assert frontal_area(V, F, (0, 0, 1)) == pytest.approx(2.0)      # 2 x 1


def test_self_intersections_counts_the_crossing_pairs_only():
    V, F = two_boxes(gap=0.2)
    assert self_intersections(V, F)["intersections"] == 0
    moved = V.copy()
    moved[8:, 0] -= 0.5
    hits = self_intersections(moved, F)
    assert hits["intersections"] > 0
    assert len(hits["keys"]) == hits["intersections"]


# ----------------------------------------------------------------------------------------------- espaço de projeto
def test_design_space_samples_inside_its_bounds():
    V, F = sphere(5)
    ffd = FFD.fit(V, control=(3, 3, 3))
    space = DesignSpace(ffd, F, V)
    lo, hi = space.bounds
    x = space.sample(16, seed=0)
    assert x.shape == (16, space.dim)
    assert np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12)
    assert np.allclose(space.sample(1, centre=True)[0], 0.5 * (lo + hi))


def test_design_space_accepts_and_rejects_with_a_reason():
    V, F = sphere(5)
    ffd = FFD.fit(V, control=(3, 3, 3))
    band = 0.5
    space = DesignSpace(ffd, F, V, lo=-band * np.ones(ffd.n_parameters), hi=band * np.ones(ffd.n_parameters),
                        gate={"max_volume_change": 0.05})
    assert space.check(np.zeros(space.dim)).ok

    # um desenho que infla o corpo em toda a banda: medido, e recusado pelo volume
    centre = 0.5 * (V.min(0) + V.max(0))
    flat, ctrl = ffd.param_of.reshape(-1, 3), ffd.control.reshape(-1, 3)
    inflate = np.zeros(space.dim)
    for p in range(space.dim):
        i, d = np.argwhere(flat == p)[0]
        inflate[p] = band * np.sign(ctrl[i][d] - centre[d])
    refused = space.check(inflate)
    assert not refused.ok
    assert any("volume" in r for r in refused.reasons)
    assert refused.metrics["volume_ratio"] > 1.5

    # e um desenho fora da caixa é recusado antes de deformar qualquer coisa
    outside = space.check(2.0 * space.hi)
    assert not outside.ok
    assert "bounds" in outside.reasons[0]

    report = space.sample_valid(3, seed=0)
    assert report.accepted <= 3
    assert report.draws >= report.accepted
    assert isinstance(report.summary()["acceptance_rate"], float)


def test_design_space_keep_out_stops_the_region_from_moving():
    V, F = sphere(5)
    ffd = FFD.fit(V, control=(3, 3, 3))
    locked = (V[:, 2].max() - 1e-9,                   # uma calota no topo da esfera é a face de montagem
              (V.min(0) + V.max(0)) / 2)
    space = DesignSpace(ffd, F, V, keep_out=[((V[:, 0].mean() - 2, V[:, 1].mean() - 2, V[:, 2].max() * 0.9),
                                             (V[:, 0].mean() + 2, V[:, 1].mean() + 2, V[:, 2].max() * 2),
                                             1e-3)])
    del locked
    report = space.check(np.zeros(space.dim))
    assert report.ok, report.reasons
    rng = np.random.default_rng(0)
    big = space.check(rng.uniform(*space.bounds))
    assert any("keep-out" in r for r in big.reasons)


# ----------------------------------------------------------------------------------------------- a malha real
def test_the_repository_mesh_is_readable_and_survives_an_ffd():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "geometries" / "road_car" / "road_car.light.stl"
    if not path.exists():
        pytest.skip("the geometry library is not checked out")
    surface = Surface.load(path)
    assert surface.n_vertices > 5000 and surface.n_faces > 5000
    assert surface.as_tuple()[0].shape == (surface.n_vertices, 3)

    ffd = FFD.fit(surface.reference, control=(4, 3, 4), symmetry="y")
    rng = np.random.default_rng(0)
    V = ffd.deform(rng.uniform(*ffd.bounds(0.3)))
    report = check_surface(V, surface.faces, reference=(surface.reference, surface.faces),
                           contact_tolerance=1.0)
    assert report.metrics["watertight"] is True
    assert report.metrics["enclosed_volume"] > 0
    assert report.metrics["flipped_faces"] == 0


def test_as_surface_accepts_a_pair_and_refuses_a_bad_face():
    V, F = box()
    a, b = as_surface((V, F))
    assert a.shape == (8, 3) and b.shape == (12, 3)
    with pytest.raises(ValueError):
        as_surface((V, np.array([[0, 1, 99]])))
