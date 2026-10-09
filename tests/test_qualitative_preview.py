"""Qualitative physics preview: descriptors, physical models against known results, objectives, mechanisms,
confidence, part sensitivity and the quantitative check."""
import math

import numpy as np
import pytest

from pinneapple_design.geometry.bodies import box, sphere
from pinneapple_design.qualitative import (
    Assembly,
    Cantilever,
    ConvectiveCooling,
    ExternalFlow,
    ScalingModel,
    describe,
    parse_objective,
    part_sensitivity,
    preview,
    section_profile,
)
from pinneapple_design.qualitative.quantitative import (
    hex8_incompatible_stiffness,
    voxel_fem_cantilever,
)


def ellipsoid(L, D, n=48):
    V, F = sphere(0.5, n=n)
    return V * np.array([L, D, D]), F


def test_descriptors_are_exact_for_simple_bodies():
    d = describe(box((2.0, 1.0, 0.5)))
    assert d.volume == pytest.approx(1.0)
    assert d.wetted_area == pytest.approx(7.0)
    assert d.frontal_area == pytest.approx(0.5)
    assert d.sharp_edge_length == pytest.approx(14.0)
    np.testing.assert_allclose(np.diag(d.second_moments), [2 ** 3 * 0.5 / 12, 2 * 0.5 / 12, 2 * 0.5 ** 3 / 12 * 1.0], rtol=1e-9)
    s = describe(sphere(0.5, n=64))
    assert s.volume == pytest.approx(4 / 3 * math.pi * 0.125, rel=0.01)
    assert s.sphericity == pytest.approx(1.0, abs=0.01)
    p = section_profile(box((1.0, 0.05, 0.1), (0.5, 0, 0)), axis=0, n_slices=10, n_grid=(10, 20))
    np.testing.assert_allclose(p["I_a"], 0.05 * 0.1 ** 3 / 12, rtol=1e-6)


@pytest.mark.parametrize("text, quantity, sense", [
    ("minimizar arrasto", "drag", "min"), ("maximize heat rejection", "heat_rate", "max"),
    ("aumentar rigidez por massa", "stiffness_to_mass", "max"), ("reduzir a deflexão", "tip_deflection", "min"),
    ("more downforce", "lift", "min"), ("menor resistência térmica", "thermal_resistance", "min"),
    ("aumentar a primeira frequência", "first_frequency", "max")])
def test_objectives_in_portuguese_and_english(text, quantity, sense):
    o = parse_objective(text)
    assert (o.quantity, o.sense) == (quantity, sense)
    with pytest.raises(ValueError):
        parse_objective("deixar bonito")


def test_drag_ranking_and_mechanisms():
    m = ExternalFlow(U=30)
    cd = {k: m.evaluate(g).quantities["drag_coefficient"] for k, g in
          {"cube": box((1, 1, 1)), "sphere": sphere(0.5, n=48), "ell4": ellipsoid(4, 1), "ell8": ellipsoid(8, 1)}.items()}
    assert cd["cube"] > cd["sphere"] > cd["ell4"] > cd["ell8"]
    assert 0.9 < cd["cube"] < 1.5 and cd["ell8"] < 0.15        # Hoerner: cube ~1.05, slender bodies ~0.05-0.1
    ev = m.evaluate(box((1, 1, 1)))
    assert sum(ev.terms["drag"].values()) == pytest.approx(ev.quantities["drag"])
    assert ev.terms["drag"]["stagnation"] > ev.terms["drag"]["separation"] > ev.terms["drag"]["friction"] > 0


def test_tandem_body_is_sheltered():
    m = ExternalFlow(U=30)
    one = m.evaluate(Assembly({"a": box((0.2, 1, 1), (0, 0, 0))})).quantities["drag"]
    two = m.evaluate(Assembly({"a": box((0.2, 1, 1), (0, 0, 0)), "b": box((0.2, 1, 1), (0.6, 0, 0))})).quantities["drag"]
    assert two < 1.5 * one                                     # the second plate sits in the first one's wake


def test_cantilever_matches_beam_theory():
    m = Cantilever(E=70e9, load=1000.0)
    for h in (0.05, 0.1):
        q = m.evaluate(box((1.0, 0.05, h), (0.5, 0, 0))).quantities
        inertia = 0.05 * h ** 3 / 12
        assert q["tip_deflection"] == pytest.approx(1000 / (3 * 70e9 * inertia), rel=0.01)
        assert q["max_stress"] == pytest.approx(1000 * h / 2 / inertia, rel=0.03)
        f1 = 1.875 ** 2 / (2 * math.pi) * math.sqrt(70e9 * inertia / (2700 * 0.05 * h))
        assert q["first_frequency"] == pytest.approx(f1, rel=0.06)


def test_voxel_fem_matches_timoshenko_without_locking():
    K = hex8_incompatible_stiffness((1.0, 1.0, 1.0), 1.0, 0.3)
    assert np.sum(np.abs(np.linalg.eigvalsh(K)) < 1e-9) == 6           # exactly the rigid-body modes
    E, P, L, b, h = 70e9, 1000.0, 1.0, 0.05, 0.1
    inertia = b * h ** 3 / 12
    timo = P * L ** 3 / (3 * E * inertia) + P * L / (5 / 6 * E / 2.6 * b * h)
    r = voxel_fem_cantilever(box((L, b, h), (L / 2, 0, 0)), E=E, load=P, n_slices=30, cells_across=6)
    assert r["tip_deflection"] == pytest.approx(timo, rel=0.03)


def heat_sink(n_fins=8, height=0.03, t=0.0015):
    W, D, TB = 0.06, 0.08, 0.004
    parts = {"base": box((D, W, TB), (D / 2, 0, TB / 2))}
    for i, y in enumerate(np.linspace(-W / 2 + t / 2, W / 2 - t / 2, n_fins)):
        parts[f"fin{i + 1}"] = box((D, t, height), (D / 2, y, TB + height / 2))
    return Assembly(parts)


def test_cooling_more_area_helps_with_diminishing_returns():
    m = ConvectiveCooling(U=2.0, base_part="base")
    q = [m.evaluate(heat_sink(n)).quantities["heat_rate"] for n in (4, 8, 16, 32)]
    assert q[0] < q[1] < q[2] < q[3]
    gains = np.diff(q) / np.array([4, 8, 16])
    assert gains[0] > gains[1] > gains[2]                     # per added fin
    ev = m.evaluate(heat_sink(32))
    assert ev.terms["heat_rate"]["air_heating"] < 0
    assert any("camadas-limite" in c[0] for c in ev.caveats)


def test_preview_explains_ranks_and_flags():
    p = preview({"base": heat_sink(8), "more": heat_sink(12), "taller": heat_sink(8, 0.045), "fewer": heat_sink(4)},
                "maximizar dissipação de calor", {"U": 2.0, "base_part": "base"})
    assert p.ranking()[-1] == "fewer"
    assert p.variants["more"].good and not p.variants["fewer"].good
    total = sum(s for _, s in p.variants["more"].mechanisms)
    assert total == pytest.approx(p.variants["more"].change["heat_rate"], rel=0.02, abs=0.5)
    txt = p.text()
    assert "Por quê" in txt and "Simular primeiro" in txt
    assert "Why" in p.text(lang="en")
    assert p.to_simulate(2)[0] == "base"


def test_part_sensitivity_root_matters_more_than_tip():
    beam = Assembly({"root": box((0.5, 0.05, 0.08), (0.25, 0, 0)), "tip": box((0.5, 0.05, 0.08), (0.75, 0, 0))})
    s = part_sensitivity(beam, "reduzir a deflexão", step=0.2, axes="z", about="centroid")
    eff = {r["part"]: r["pct"] for r in s.rows if r["change_en"].endswith("z")}
    assert eff["root"] < eff["tip"] < 0                       # deeper root reduces the deflection most
    assert "Alavancas" in s.text()


def test_quantify_checks_direction_and_rank():
    geoms = {"h=0.05": box((1, 0.05, 0.05), (0.5, 0, 0)), "h=0.08": box((1, 0.05, 0.08), (0.5, 0, 0)),
             "h=0.12": box((1, 0.05, 0.12), (0.5, 0, 0)), "h=0.03": box((1, 0.05, 0.03), (0.5, 0, 0))}
    p = preview(geoms, "reduzir a deflexão", {"load": 1000.0})
    exact = lambda name, g: 1000 / (3 * 70e9 * 0.05 * float(name[2:]) ** 3 / 12)  # noqa: E731
    c = p.quantify(exact, variants=list(geoms))
    assert c.direction_agreement == 1.0 and c.rank_correlation == pytest.approx(1.0)
    assert "✔" in c.text()


def test_open_section_lowers_confidence():
    L, B, H = 1.0, 0.05, 0.08
    cut = Assembly({"before": box((0.1, B, H), (0.05, 0, 0)), "top": box((0.3, B, 0.02), (0.25, 0, H / 2 - 0.01)),
                    "bottom": box((0.3, B, 0.02), (0.25, 0, -H / 2 + 0.01)), "after": box((0.6, B, H), (0.7, 0, 0))})
    p = preview({"solid": box((L, B, H), (L / 2, 0, 0)), "cut": cut}, "aumentar rigidez por massa")
    assert p.variants["cut"].confidence == "baixa"


def test_scaling_model_attributes_to_descriptors():
    m = ScalingModel({"resistance": lambda d, c: d.length / (c["k"] * d.frontal_area)}, better={"resistance": "min"},
                     k=200.0)
    p = preview({"a": box((0.1, 0.05, 0.05)), "b": box((0.2, 0.05, 0.05))}, ("resistance", "min"), model=m)
    assert p.variants["b"].change["resistance"] == pytest.approx(100.0, rel=1e-6)
    assert p.variants["b"].mechanisms[0][0] == "length"


def test_figure(tmp_path):
    p = preview({"cube": box((1, 1, 1)), "sphere": sphere(0.5, n=24)}, "minimizar arrasto", {"U": 20})
    p.figure(tmp_path / "f.png")
    assert (tmp_path / "f.png").stat().st_size > 10_000
