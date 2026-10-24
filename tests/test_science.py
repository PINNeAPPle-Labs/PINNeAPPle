"""pinneapple_science (pp.science): the atlas, the experiment abstraction, the virtual instruments and every
electromagnetism experiment at reduced size (each must pass its own comparisons and checks)."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

import pinneapple_science as S
from pinneapple_science.core import Experiment, Reference, Result

FAST = ["em.coulomb_superposition", "em.gauss_law", "em.biot_savart", "em.ampere_law", "em.hall_effect",
        "em.circuits_rc_rl_rlc", "em.dipole_radiation", "em.lossy_plane_wave", "em.waveguide_modes",
        "em.mie_scattering", "em.bench_mu0_solenoid", "em.bench_rc_discharge", "em.cavity_resonance",
        "em.research_uq_waveguide", "em.research_inverse_permittivity", "em.research_surrogate_hall"]


# ------------------------------------------------------------------------------------------------ atlas
def test_atlas_points_only_to_code_that_exists():
    assert S.validate() == []


def test_atlas_covers_the_mapped_areas_and_renders(tmp_path):
    a = S.atlas()
    ids = {x.id for x in a.areas}
    for need in ("physics.classical_mechanics", "physics.electromagnetism", "physics.thermodynamics",
                 "physics.fluids", "physics.continuum_mechanics", "physics.acoustics", "physics.quantum_mechanics",
                 "physics.relativity", "physics.nuclear_particle", "physics.condensed_matter", "physics.plasma",
                 "physics.astrophysics", "physics.geophysics", "chemistry", "materials", "biology", "neuroscience",
                 "ecology_epidemiology", "earth_climate", "applied_math"):
        assert need in ids
    assert a.area("electromagnetism").status == "available"
    assert set(a.counts()) == {"available", "partial", "planned"}
    page = a.to_html(str(tmp_path / "atlas.html"))
    assert "em.faraday_induction" in page and os.path.isfile(tmp_path / "atlas.html")
    assert "| Electromagnetism and optics |" in a.to_markdown()


def test_atlas_validation_catches_a_bad_pointer():
    a = S.atlas()
    a.areas[0].subareas[0]["covered_by"].append({"module": "pinneapple_physics.no_such_module"})
    assert any("no_such_module" in p for p in S.validate(a, check_presets=False, check_lab=False))


# ------------------------------------------------------------------------------------------- abstraction
def test_experiment_declarations_are_checked():
    with pytest.raises(ValueError):
        class Bad(Experiment):  # noqa: F841
            name, level, kind, dims = "bad", "expert", "theory", ("3D",)
    with pytest.raises(ValueError):
        class Bad2(Experiment):  # noqa: F841
            name, level, kind, dims = "bad2", "academic", "theory", ("5D",)
    e = S.get("em.gauss_law")
    with pytest.raises(ValueError):
        e.run("1D")
    with pytest.raises(TypeError):
        e.run("3D", wrong=1)


def test_result_records_and_decides():
    r = Result("x", "X", "physics.electromagnetism", "t", "academic", "theory", "1D", {})
    with pytest.raises(ValueError):
        r.compare("q", 1.0, 1.0, tol=0.1)                   # a reference needs its source
    with pytest.raises(ValueError):
        Reference(1.0, "")
    with pytest.raises(ValueError):
        r.value("v", float("nan"))
    r.compare("q", 1.001, Reference(1.0, "a book"), tol=1e-2)
    r.check("law", True, value=0.01, limit=0.05)
    r.build_trust()
    assert r.passed and r.decision == "APPROVED"
    r.compare("q2", 2.0, 1.0, tol=1e-2, source="a book")
    r.build_trust()
    assert not r.passed and r.decision == "REJECT"          # a failed reference comparison is critical


def test_filters_and_cards():
    em = S.experiments(area="electromagnetism")
    assert len(em) >= 16
    dims = {d for e in em for d in e["dims"]}
    assert {"1D", "2D", "3D", "3D+t"} <= dims                 # at least one experiment per dimension
    assert {e["kind"] for e in em} == {"theory", "practice"}
    assert all(e["dims"] for e in S.experiments(dim="3D+t"))


# ------------------------------------------------------------------------------------------- instruments
def test_instrument_and_fit_uncertainty():
    from pinneapple_science.instruments import FieldProbe, linear_fit

    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 30)
    z = []
    for _ in range(200):
        y = 2.0 * x + 0.5 + 0.01 * rng.standard_normal(30)
        f = linear_fit(x, y, sigma_y=0.01)
        z.append((f.slope - 2.0) / f.slope_u)
    assert 0.8 < np.std(z) < 1.2                               # the stated slope uncertainty is right
    p = FieldProbe(noise=0.0, gain_u=0.0, offset_u=0.0, resolution=1e-3, seed=1)
    assert p.read(0.12345) == pytest.approx(0.123)


# ----------------------------------------------------------------------------------------- library pieces
def test_vector_potential_curl_is_biot_savart_and_flux_is_maxwell():
    from pinneapple_simulation.electromagnetics import statics as st

    s, e = st.circle_segments(0.006, 256)
    P0 = np.array([[0.002, 0.001, 0.004]])
    h = 1e-6
    A = lambda p: st.vector_potential(p, s, e, 1.0)  # noqa: E731
    curl_x = (A(P0 + [0, h, 0])[0, 2] - A(P0 - [0, h, 0])[0, 2] - A(P0 + [0, 0, h])[0, 1] + A(P0 - [0, 0, h])[0, 1]) / (2 * h)
    assert curl_x == pytest.approx(st.biot_savart(P0, s, e, 1.0)[0, 0], rel=1e-6)
    ps, pe = st.circle_segments(0.015, 256)
    for z in (0.0, 0.02):
        assert st.flux_through_path(ps, pe, s + [0, 0, z], e + [0, 0, z], 1.0) == pytest.approx(
            st.mutual_inductance_coaxial_loops(0.006, 0.015, z), rel=2e-4)


def test_strip_capacitor_approaches_palmer():
    from pinneapple_science.em.statics import palmer_strip_capacitance, strip_capacitor_2d

    c = [strip_capacitor_2d(1.0, 0.1, 0.1 / k)[0] for k in (6, 12)]
    ref = palmer_strip_capacitance(1.0, 0.1)
    assert abs(c[1] / ref - 1) < abs(c[0] / ref - 1) < 0.03
    with pytest.raises(ValueError):
        strip_capacitor_2d(1.0, 0.1, 0.1 / 5)                 # the plate must lie on grid nodes


# ------------------------------------------------------------------------------------------ experiments
@pytest.mark.parametrize("name", FAST)
def test_experiment_passes_its_references(name):
    e = S.get(name)
    for d in e.dims:
        r = e.run(d, quick=True)
        assert r.comparisons, f"{name} [{d}] compares nothing"
        bad = [c.quantity for c in r.comparisons if not c.passed] + [k["name"] for k in r.checks if k["passed"] is False]
        assert not bad, f"{name} [{d}]: {bad}"
        assert r.decision == "APPROVED"
        for c in r.comparisons:
            assert c.source and math.isfinite(c.computed)


def test_capacitor_quick():
    e = S.get("em.parallel_plate_capacitor")
    r = e.run("2D", quick=True)
    assert r.passed, r


def test_faraday_3d_t_quick():
    r = S.run("em.faraday_induction", quick=True)
    assert r.dim == "3D+t" and r.passed, r


def test_fresnel_quick():
    r = S.run("em.fresnel_snell", quick=True, angles_deg=[30.0], cells_per_wavelength=[30])
    assert r.passed, r


def test_save_writes_result_and_figures(tmp_path):
    r = S.run("em.circuits_rc_rl_rlc", quick=True)
    paths = r.save(str(tmp_path))
    assert os.path.isfile(paths["result"]) and os.path.isfile(paths["circuits"])
    import json
    d = json.load(open(paths["result"]))
    assert d["schema"] == "pinneapple.science.result/1" and d["trust"]["decision"] == "APPROVED"


def test_pp_namespace():
    import pinneapple as pp
    assert pp.science.run is S.run and "em.hall_effect" in pp.science.names()


def test_flow_nodes_list_every_experiment_and_run(tmp_path, monkeypatch):
    from pinneapple_flow.nodes.science import EXPERIMENTS
    assert EXPERIMENTS == S.names()
    import pinneapple_flow as pf
    monkeypatch.setenv("PINNEAPPLE_FLOW_HOME", str(tmp_path / "home"))
    f = pf.Flow("science", nodes=[{"id": "e", "type": "science.run", "params": {"experiment": "em.hall_effect"}},
                                  {"id": "a", "type": "science.atlas", "params": {}}], edges=[])
    r = pf.run(f, workdir=tmp_path / "w")
    assert r.status == "done", {k: v["error"] for k, v in r.nodes.items()}
    assert any(k.startswith("rel_error[") for k in r.report("e")["metrics"])


def test_fdtd_3d_t_research_study(tmp_path):
    """The 3D+t FDTD study at its committed ladder (about two minutes): order, Richardson, 4-D Maxwell checks,
    the animation frames and the three.js viewer."""
    r = S.run("em.research_fdtd_3d", quick=True, snapshots=3, viewer=True)
    assert r.passed, r
    assert r.values["observed_order_vacuum"]["value"] > 1.5
    paths = r.save(str(tmp_path))
    assert os.path.getsize(paths["fdtd_viewer_html"]) > 10_000 and os.path.isfile(paths["fdtd_3d_t_mp4"])


def test_fdtd_config_rejects_unknown_keys():
    from pinneapple_science.em.fdtd_study import FDTDConfig
    with pytest.raises(ValueError):
        FDTDConfig.from_dict({"cells": 10})
    with pytest.raises(ValueError):
        FDTDConfig.from_dict({"materials": [{"shape": "torus"}]})


def test_gp_loo_scale_is_one_for_a_well_specified_gp():
    from pinneapple_presets.surrogate import ARDGP
    from pinneapple_science.numerics import gp_loo_scale
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (30, 1))
    gp = ARDGP().fit(X, np.sin(4 * X[:, 0]) + 0.01 * rng.standard_normal(30))   # noise the GP can learn
    assert 0.8 < gp_loo_scale(gp) < 1.25
