"""Core coverage (#1129, toward 1.0) for the industrial presets' support code and each case class.

* 3-D shapes are checked against exact geometry: a box's area and volume (divergence theorem on the closed mesh), a
  capped prism's volume, a polygonal tube's lateral area.
* The case modules' closed forms against independent references: the rigid-column surge of a linear valve closure
  against its closed form (frictionless), the Lamé stresses against their boundary conditions, EN 1993 class limits.
* The report, catalogue, surrogate and variant tools run on a small user preset with an exact answer (cantilever
  tip deflection F L^3 / 3 E I).
"""
import json
import math
import os

import numpy as np
import pytest

import pinneapple_presets.core as PC
import pinneapple_presets.registry as PR
import pinneapple_presets.report as RP
import pinneapple_presets.surrogate as SG
import pinneapple_presets.viz3d as V3
from pinneapple_presets.cases.airfoil import AirfoilSection, PresetCache
from pinneapple_presets.cases.eddy_current import EddyCurrentInspection
from pinneapple_presets.cases.footbridge import SteelFootbridge, section_class_ok
from pinneapple_presets.cases.heatsink import PlateFinHeatsink
from pinneapple_presets.cases.insulation import SteamLineInsulation
from pinneapple_presets.cases.ipm_traction import IPMTraction
from pinneapple_presets.cases.pcb import PowerPCB
from pinneapple_presets.cases.pipe_network import WaterNetwork
from pinneapple_presets.cases.pressure_vessel import PressureVessel, lame, ring_mesh
from pinneapple_presets.cases.pump_npsh import PumpNPSH
from pinneapple_presets.cases.shell_tube import ShellTubeCooler
from pinneapple_presets.cases.spm_motor import SPMMotor
from pinneapple_presets.cases.water_hammer import WaterHammer, rigid_column_max_head


def _area_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    return 0.5 * np.linalg.norm(cr, axis=1).sum(), abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6


# ------------------------------------------------------------------------------------------------ 3-D shapes
def test_box_extrude_tube_merge_and_fields():
    V, F = V3.box([0, 0, 0], [1, 2, 3], n=(2, 3, 4))
    area, vol = _area_volume(V, F)
    assert area == pytest.approx(2 * (2 + 6 + 3)) and vol == pytest.approx(6.0)        # exact: closed box
    sq = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    Ve, Fe, st = V3.extrude(sq, [0.0, 1.0, 3.0])
    assert _area_volume(Ve, Fe)[1] == pytest.approx(3.0) and set(st) >= {0, 1, 2}       # section 1 x length 3
    n = 24
    Vt, Ft, s = V3.tube(np.c_[np.linspace(0, 2, 11), np.zeros(11), np.zeros(11)], 0.1, n_around=n)
    lateral = _area_volume(Vt, Ft)[0]
    assert lateral == pytest.approx(n * 2 * 0.1 * math.sin(math.pi / n) * 2.0, rel=1e-6)  # polygonal tube
    assert np.allclose(V3.field_on(s, np.arange(11.0)), s)
    Vm, Fm, vals = V3.merge([(V, F, np.zeros(len(V))), (Ve, Fe, np.ones(len(Ve)))])
    assert len(Vm) == len(V) + len(Ve) and Fm.max() == len(Vm) - 1 and vals.sum() == len(Ve)


def test_stills_write_the_self_contained_viewer(tmp_path):
    V, F = V3.box([0, 0, 0], [1, 1, 1])
    sc = V3.Scene([V3.Surface("cube", V, F, "steel", {"T": V[:, 2]})], title="cube")
    out = V3.stills(sc, str(tmp_path), "cube", modes=())
    assert os.path.isfile(out["viewer"]) and open(out["viewer"]).read().lstrip().lower().startswith("<!doctype")


# ------------------------------------------------------------------------------------------------ case closed forms
def test_rigid_column_matches_the_closed_form_of_a_linear_closure():
    """Frictionless, linear valve-area closure: H_max / H0 = 1 + K^2 / 2 + K sqrt(1 + K^2 / 4), K = L V0 / (g H0 T)
    (rigid water column, Wylie & Streeter sec. 1.3)."""
    L, D, q0, H0, tc = 500.0, 0.5, 0.4, 50.0, 20.0
    K = L * q0 / (math.pi * D * D / 4) / (9.80665 * H0 * tc)
    h = rigid_column_max_head(length=L, diameter=D, friction_factor=0.0, q0=q0, h_reservoir=H0, h_downstream=0.0,
                              t_close=tc)
    assert h / H0 == pytest.approx(1 + K * K / 2 + K * math.sqrt(1 + K * K / 4), rel=1e-6)   # analytic


def test_lame_boundary_conditions_and_the_ring_mesh():
    p, ri, ro = 10e6, 0.5, 0.6
    assert lame(p, ri, ro, ri, 200e9, 0.3)["s_rr"] == pytest.approx(-p)                   # exact: inner pressure
    assert lame(p, ri, ro, ro, 200e9, 0.3)["s_rr"] == pytest.approx(0.0, abs=1e-6)         # free outer surface
    m = ring_mesh(ri, ro, 0.1, 4, 8, 2)
    assert m.nodes.shape == (5 * 9 * 3, 3) and m.elements.shape == (4 * 8 * 2, 8)
    r = np.hypot(m.nodes[:, 0], m.nodes[:, 1])
    assert r.min() == pytest.approx(ri) and r.max() == pytest.approx(ro)


def test_eurocode_section_class_limits():
    ok, cw, lw, cf, lf = section_class_ok(1.0, 0.3, 0.02, 0.01, 235e6)
    assert lw == pytest.approx(124.0) and lf == pytest.approx(14.0)                       # eps = 1 at 235 MPa
    assert cw == pytest.approx(96.0) and cf == pytest.approx(7.25) and ok
    assert not section_class_ok(1.0, 0.3, 0.02, 0.005, 355e6)[0]                          # slender web


def test_every_case_class_is_registered_with_a_data_sheet():
    cases = [AirfoilSection(), EddyCurrentInspection(), SteelFootbridge(), PlateFinHeatsink(),
             SteamLineInsulation(), IPMTraction(), PowerPCB(), WaterNetwork(), PressureVessel(), PumpNPSH(),
             ShellTubeCooler(), SPMMotor(), WaterHammer()]
    names = set(PR.names())
    for c in cases:
        card = c.card()
        assert c.name in names and type(PR.load(c.name)) is type(c)
        assert card["inputs"] and card["outputs"] and all(i["source"] for i in card["inputs"])
        assert all(ref.source and ref.tol > 0 for ref in c.references())
    assert hasattr(PresetCache(), "model")                                                # class-level cache slot


# ------------------------------------------------------------------------------------------------ a user preset
class Cantilever(PC.Preset):
    name = "_test_cantilever"
    title = "Cantilever tip deflection"
    sector = "test"
    problem = "tip load on a cantilever"
    question = "how far does the tip move?"
    params = (PC.Param("F", "N", 100.0, "test", 10.0, 1000.0), PC.Param("L", "m", 1.0, "test", 0.5, 2.0))
    outputs = (PC.Output("delta", "m", "tip deflection"),)
    surrogate_spec = PC.SurrogateSpec(inputs=("F", "L"), outputs=("delta",), n_train=16, n_test=6, log=("F", "L"),
                                      log_outputs=("delta",), bounds={"L": (0.5, 2.0)})   # log delta: linear
    EI = 2e5

    def solve(self, p):
        return {"delta": p["F"] * p["L"] ** 3 / (3 * self.EI)}

    def checks(self, p, out):
        return [PC.check("deflection below L/100", out["delta"] < p["L"] / 100, out["delta"], p["L"] / 100)]

    def references(self):
        return [PC.Reference("tip load", {"F": 300.0, "L": 2.0}, "delta", 300.0 * 8 / 6e5, "m", "F L^3 / 3 EI",
                             "analytic", 1e-12)]

    def variants(self):
        return {"short": {"L": 0.5}, "long": {"L": 2.0}}


@pytest.fixture()
def cantilever():
    PR.register(Cantilever)
    try:
        yield Cantilever()
    finally:
        PR._EXTRA.pop(Cantilever.name, None)


def test_register_rejects_what_is_not_a_named_preset():
    with pytest.raises(TypeError):
        PR.register(object)


def test_surrogate_of_a_smooth_preset(cantilever, tmp_path, monkeypatch):
    assert SG.surrogate_path("x").endswith(os.path.join("surrogates", "x.json"))
    monkeypatch.setattr(SG, "DATA", str(tmp_path))
    sur = SG.build_surrogate(cantilever, seed=0, save=True)
    assert sur.meta["holdout"]["max_rel_err"] < 0.05 and os.path.isfile(SG.surrogate_path(cantilever.name))
    spec = SG.serve_spec(cantilever.name)
    assert spec["inputs"]["L"]["bounds"] == [0.5, 2.0] and spec["outputs"] == {"delta": "m"}
    assert SG.serve_spec("_no_such_preset") is None
    with pytest.raises(ValueError):
        SG.build_surrogate(WaterHammer.__new__(type("NoSpec", (PC.Preset,), {"name": "n"})))


def test_report_readme_variants_and_catalogue(cantilever, tmp_path):
    root = tmp_path / cantilever.name
    s = RP.write_report(cantilever, str(root), surrogate="skip", three_d=False, lab=False)
    assert s["validation"]["all_pass"] and s["validation"]["references"][0]["rel_error"] < 1e-12   # exact
    assert set(s["variants"]["rows"]) == {"short", "long"}
    md = (root / "README.md").read_text(encoding="utf-8")
    assert md == RP.readme(cantilever, json.loads((root / "results" / "summary.json").read_text()))
    assert md.startswith("# Cantilever tip deflection")
    cmp = RP.compare_variants(cantilever, {"a": {"F": 10.0}, "b": {"F": 20.0}}, str(root / "results"), str(root),
                              three_d=False)
    deltas = {k: v["outputs"]["delta"] for k, v in cmp["rows"].items()}
    assert deltas["b"] == pytest.approx(2 * deltas["a"])                                  # linear in the load
    cat = RP.write_catalogue(str(tmp_path))
    assert os.path.isfile(cat) and cantilever.name in open(cat, encoding="utf-8").read()


def test_core_value_objects():
    o = PC.Output("delta", "m")
    r = PC.Reference("x", {}, "delta", 1.0, "m", "s", "analytic", 0.01)
    ss = PC.SurrogateSpec(inputs=("a",), outputs=("b",))
    assert o.help == "" and r.tol == 0.01 and ss.n_train == 60 and ss.bounds == {}
