"""Advanced CadQuery builders: valid solids, closed-form volumes, registry integration, meshing."""
import math

import pytest

cq = pytest.importorskip("cadquery")

from pinneapple_design.geometry.gen import cadquery_advanced as A  # noqa: E402
from pinneapple_design.geometry.gen.cadquery_gen import CadQueryRegistry, ParametricCadSpec  # noqa: E402


def _defaults(name, **over):
    p = {k: v["default"] for k, v in A.ADVANCED_TEMPLATES[name][1].items() if "default" in v}
    p.update(over)
    return p


def _solid(w):
    vals = w.solids().vals()
    assert len(vals) == 1, f"expected one solid, got {len(vals)}"
    assert vals[0].isValid()
    return vals[0]


def test_pipe_bend_volume_matches_torus_section_plus_flanges():
    p = _defaults("pipe_bend_flanged", n_bolts=4)
    ro, ri = p["od"] / 2, p["od"] / 2 - p["wall"]
    tube = math.pi * (ro ** 2 - ri ** 2) * p["bend_radius"] * math.radians(p["angle_deg"])
    flange = (math.pi * ((p["flange_od"] / 2) ** 2 - ri ** 2) - 4 * math.pi * (p["bolt_d"] / 2) ** 2) * p["flange_t"]
    assert _solid(A.pipe_bend_flanged(p)).Volume() == pytest.approx(tube + 2 * flange, rel=1e-3)


def test_reducer_volume_is_the_hollow_frustum():
    p = _defaults("concentric_reducer")
    fr = lambda a, b: math.pi * p["length"] / 3 * (a * a + a * b + b * b)
    r1, r2, t = p["d1"] / 2, p["d2"] / 2, p["wall"]
    assert _solid(A.concentric_reducer(p)).Volume() == pytest.approx(fr(r1, r2) - fr(r1 - t, r2 - t), rel=1e-4)


def test_helical_coil_volume_is_area_times_helix_length():
    p = _defaults("helical_coil")
    L = p["turns"] * math.hypot(2 * math.pi * p["coil_radius"], p["pitch"])
    assert _solid(A.helical_coil(p)).Volume() == pytest.approx(math.pi * (p["tube_d"] / 2) ** 2 * L, rel=5e-3)


def test_pin_fin_heatsink_volume():
    p = _defaults("pin_fin_heatsink")
    v = p["base_x"] * p["base_y"] * p["base_t"] + p["nx"] * p["ny"] * math.pi * (p["pin_d"] / 2) ** 2 * p["pin_h"]
    assert _solid(A.pin_fin_heatsink(p)).Volume() == pytest.approx(v, rel=1e-6)


def test_spur_gear_area_close_to_pitch_circle_and_teeth_count():
    p = _defaults("spur_gear", bore_d=0, key_w=0)
    g = _solid(A.spur_gear(p))
    area = g.Volume() / p["width"]
    rp = p["module"] * p["teeth"] / 2
    assert area == pytest.approx(math.pi * rp * rp, rel=0.05)
    assert g.BoundingBox().xlen == pytest.approx(2 * (rp + p["module"]), rel=0.01)  # tip circle


def test_tee_and_fan_are_valid_single_solids():
    _solid(A.pipe_tee(_defaults("pipe_tee")))
    fan = _solid(A.axial_fan(_defaults("axial_fan", n_blades=3)))
    r_max = max(math.hypot(v.X, v.Y) for v in fan.Vertices())
    assert r_max == pytest.approx(0.15, rel=0.05)  # blades reach the tip radius (bbox would not: 3 blades at 120 deg)


def test_documentation_reproductions():
    pb = _solid(A.pillow_block({}))
    holes = 4 * (math.pi * 1.2 ** 2 * 10 + math.pi * (2.2 ** 2 - 1.2 ** 2) * 2.1)
    assert pb.Volume() == pytest.approx(30 * 40 * 10 - math.pi * 11 ** 2 * 10 - holes, rel=1e-6)
    assert _solid(A.occ_bottle({})).Volume() > 0
    lego = _solid(A.lego_brick({"lbumps": 2, "wbumps": 2}))
    assert lego.BoundingBox().xlen == pytest.approx(2 * 8.0 - 0.2, rel=1e-6)


def test_registry_builds_from_a_parametric_spec_and_rejects_bad_params():
    reg = A.register_advanced_templates(CadQueryRegistry())
    w = reg.build(ParametricCadSpec("concentric_reducer", {"d1": 0.2, "d2": 0.1}, schema=reg.get_schema("concentric_reducer")))
    assert _solid(w).Volume() > 0
    with pytest.raises(ValueError):
        reg.build(ParametricCadSpec("spur_gear", {"teeth": 3}, schema=reg.get_schema("spur_gear")))


def test_meshes_for_simulation():
    from pinneapple_design.geometry.gen.cadquery_gen import cadquery_to_trimesh
    tm = cadquery_to_trimesh(A.concentric_reducer(_defaults("concentric_reducer")))
    assert tm.is_watertight and tm.volume == pytest.approx(_solid(A.concentric_reducer(_defaults("concentric_reducer"))).Volume(), rel=0.02)
