"""The compressible-passage pipeline: geometry, mesh, gas, and the checks that gate a result.

The solver itself is exercised separately (``test_compressible_passage_solver_roda``, skipped when
OpenFOAM is not on the machine): what runs here is everything that must be right *before* a solver
can be trusted — a passage that cannot be meshed is refused, the mesh is conforming and one region,
the gas properties match an independent real-gas reference, and each check in :mod:`.checks` says
"fail" on the numbers the reference document produced.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from pinneapple_simulation.pipelines.compressible_internal import (
    GASES,
    PassageSpec,
    Ring,
    Span,
    converging_nozzle,
    globe_valve,
    orifice_plate,
    peng_robinson,
    sutherland_mu,
)
from pinneapple_simulation.pipelines.compressible_internal.checks import (
    critical_pressure_ratio,
    iec_flow_coefficient,
    stagnation_enthalpy,
)
from pinneapple_simulation.pipelines.compressible_internal.passage import build_mesh

CFD08 = "PINNeAPPle-CFD/casos/cfd08-valvula-globo"
NEEDS_FOAM = pytest.mark.skipif(shutil.which("openfoam") is None, reason="OpenFOAM is not on PATH")


# --------------------------------------------------------------------------------- geometry
def test_the_three_shipped_passages_can_be_meshed():
    for passage in (globe_valve(lift=4.0), orifice_plate(hole_ratio=0.5), converging_nozzle(area_ratio=4.0)):
        assert passage.problems() == [], passage.title
        assert len(passage.spans) >= 3


def test_a_span_that_does_not_match_its_neighbour_is_refused():
    """Conformity is the one thing the mesher cannot fix afterwards, so the spec is refused instead."""
    passage = PassageSpec(spans=(Span("a", 0.0, 10.0, (Ring(0.5, 10.0, 4),), 1.0),
                                 Span("b", 10.0, 20.0, (Ring(0.5, 4.0, 4),), 1.0)))
    bad = passage.problems()
    assert any("do not share their radii" in b for b in bad), bad


def test_a_gap_between_spans_is_refused():
    passage = PassageSpec(spans=(Span("a", 0.0, 10.0, (Ring(0.5, 10.0, 4),), 1.0),
                                 Span("b", 11.0, 20.0, (Ring(0.5, 10.0, 4),), 1.0)))
    assert any("starts at" in b for b in passage.problems())


def test_a_tapered_ring_is_a_cone_and_stays_conforming():
    """The band carries its radii at both ends, so a nozzle is one span and the wall is the cone."""
    passage = converging_nozzle(area_ratio=4.0, n_theta=4, n_radial=4)
    assert passage.problems() == []
    cone = passage.spans[1]
    assert cone.radii_at(0.0)[-1] > cone.radii_at(1.0)[-1]


# --------------------------------------------------------------------------------- mesh
def test_mesh_is_conforming_and_has_every_patch(tmp_path):
    passage = globe_valve(lift=4.0, n_theta=6, radial_cells=(3, 3, 2), h_pipe=4.0, h_bowl=3.0, h_gap=2.0)
    summary = build_mesh(passage, str(tmp_path))
    assert summary.n_cells > 0
    assert summary.n_internal > summary.n_cells          # a hexahedral cell has 6 faces, mostly shared
    assert summary.patch_faces["inlet"] > 0 and summary.patch_faces["outlet"] > 0
    assert summary.patch_faces["sym0"] == summary.patch_faces["sym1"]
    text = (tmp_path / "constant" / "polyMesh" / "boundary").read_text()
    for name in ("inlet", "outlet", "wall", "axis", "sym0", "sym1"):
        assert name in text
    assert "symmetryPlane" in text and "symmetry;" in text


@NEEDS_FOAM
def test_checkmesh_accepts_the_valve_mesh_as_one_region(tmp_path, monkeypatch):
    """The bug this guards: a point shared by two blocks must be *the same* point, or the mesh is 12
    disconnected regions that pass ``checkMesh``'s geometry tests and solve to nothing."""
    from pinneapple_simulation.pipelines.compressible_internal import Conditions
    from pinneapple_simulation.pipelines.compressible_internal.case import run_case, write_case
    passage = globe_valve(lift=4.0, n_theta=6, radial_cells=(3, 3, 2), h_pipe=4.0, h_bowl=3.0, h_gap=2.0)
    write_case(passage, str(tmp_path), GASES["methane"], Conditions(iterations=2))
    run = run_case(str(tmp_path), timeout=300)
    assert run.mesh_ok, run.checkmesh_log
    assert "Number of regions: 1" in (tmp_path / "log.checkMesh").read_text()


# --------------------------------------------------------------------------------- gas
def test_peng_robinson_matches_the_reference_density_of_methane():
    """51.74 kg/m3 at 70 bar and 300 K; CoolProp's PR gives 51.73 and GERG-2008 50.52."""
    gas = GASES["methane"]
    rho = peng_robinson(gas, 70e5, 300.0)["rho"]
    assert rho == pytest.approx(51.74, abs=0.05)
    ideal = 70e5 / (gas.Rs * 300.0)
    assert ideal == pytest.approx(45.02, abs=0.05)
    assert rho / ideal > 1.10            # 13 % heavier than the ideal gas: not a detail


def test_the_equation_of_state_reproduces_the_isenthalpic_cooling():
    """The departure enthalpy is what makes a throttle cool: solving ``h(70 bar, 300 K) = h(60 bar, T)``
    with this EOS gives 295.81 K, against 295.84 K from CoolProp's Peng-Robinson. An ideal gas (no
    departure term) would give exactly 300 K, which is the mistake the reference document's setup
    would have made had it used ``perfectGas``."""
    gas = GASES["methane"]

    def h(p, T):
        return gas.Cp_ig * (T - 298.15) + peng_robinson(gas, p, T)["h_dep"]

    lo, hi, target = 280.0, 320.0, h(70e5, 300.0)
    for _ in range(80):                       # bisection: no scipy needed for one monotone root
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if h(60e5, mid) < target else (lo, mid)
    assert 295.6 < 0.5 * (lo + hi) < 296.0


def test_sutherland_matches_the_quoted_viscosity():
    assert sutherland_mu(GASES["methane"], 300.0) == pytest.approx(1.11e-5, rel=0.02)


# --------------------------------------------------------------------------------- checks
def test_critical_pressure_ratio_of_methane_is_about_a_half():
    """0.857 (the document's ratio) is well above it: the valve is not choked, so Mach 0.9 is impossible."""
    crit = critical_pressure_ratio(GASES["methane"])
    assert 0.5 < crit < 0.6
    assert 60.0 / 70.0 > crit


def test_stagnation_enthalpy_is_conserved_across_a_throttle():
    """Adiabatic and no work: h + u^2/2 is the same before and after, which is what the check asserts."""
    gas = GASES["methane"]
    p1, T1, u1 = 70e5, 300.0, 4.0
    p2, T2, u2 = 60e5, 295.81, 19.5          # the isenthalpic outlet this EOS itself gives
    h0_in = stagnation_enthalpy(gas, p1, T1, u1)
    h0_out = stagnation_enthalpy(gas, p2, T2, u2)
    assert abs(h0_out - h0_in) / (gas.Cp_ig * T1) < 0.005
    # an outlet 3.8 K warmer than the isenthalpic one (i.e. no Joule-Thomson at all) must not pass
    warm = stagnation_enthalpy(gas, p2, 300.0, u2)
    assert abs(warm - h0_in) / (gas.Cp_ig * T1) > 0.01


def test_iec_flow_coefficient_reports_the_choked_branch():
    gas = GASES["methane"]
    not_choked = iec_flow_coefficient(1.7, 70e5, 60e5, 51.7, gas)
    assert not not_choked["choked"] and not_choked["Y"] < 1.0
    choked = iec_flow_coefficient(1.7, 70e5, 5e5, 51.7, gas)
    assert choked["choked"] and choked["Y"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------- contract
def test_the_first_mass_flow_is_the_orifice_estimate_and_lands_near_the_answer():
    """``mdot = Cd A sqrt(2 rho dP)`` with Cd = 0.6: 1.76 kg/s for the valve at 10 bar, which is where
    the outer loop then finds it. The guess is documented because it is the first number a reader sees."""
    from pinneapple_simulation.pipelines.compressible_internal import Conditions
    from pinneapple_simulation.pipelines.compressible_internal.study import (
        StudySpec,
        initial_mass_flow,
    )
    spec = StudySpec(globe_valve(lift=4.0), gas="methane",
                     conditions=Conditions(p_out_bar=60.0, T_in_K=300.0, p_in_target_bar=70.0))
    m = initial_mass_flow(spec, 70.0)
    area = spec.passage.meta["throttling_area_mm2"] * 1e-6
    rho = peng_robinson(GASES["methane"], 60e5, 300.0)["rho"]
    assert m == pytest.approx(0.6 * area * np.sqrt(2 * rho * 10e5), rel=1e-6)
    assert 1.0 < m < 3.0
    from pinneapple_simulation.pipelines.compressible_internal.case import resolved_mass_flow
    assert resolved_mass_flow(spec.passage, spec.resolved_gas, spec.conditions) == pytest.approx(m)


def test_the_flow_capability_is_registered_with_the_words_a_person_writes():
    from pinneapple_flow.planner import extensions as X
    X.capabilities()
    cap = X.CAPABILITIES.get("cfd.compressible_passage")
    assert cap is not None
    for text in ("válvula globo de gás natural com 70 bar na entrada e 60 na saída",
                 "compressible flow through a globe valve with a 10 bar pressure drop"):
        score, hits = X.score(cap, text)
        assert score >= cap.min_score, (text, score, sorted(hits))


def test_globe_valve_solid_has_the_body_the_bonnet_and_two_ports():
    """O domínio 3-D: um corpo arredondado com pescoço — o que um sólido de revolução não faz.

    A válvula globo do documento tem o pescoço num eixo perpendicular ao escoamento. Nenhuma malha
    de revolução em torno do eixo do escoamento tem esse ramo, então o caso do cliente só pode ser
    resolvido neste backend. O teste confere o que o construtor promete: um único volume de fluido,
    com dois bocais identificados e o resto como parede.
    """
    pytest.importorskip("gmsh")
    from pinneapple_simulation.pipelines.compressible_internal.solid3d import globe_valve_domain

    p = globe_valve(lift=4.0, solid=True)
    assert p.meta["solid_builder"] == "globe_valve"
    assert p.sector_deg == 360.0

    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        tag, groups = globe_valve_domain(dict(p.meta["solid"], lift=4.0))
        volume_cm3 = gmsh.model.occ.getMass(3, tag) / 1e3
        bbox = gmsh.model.getBoundingBox(3, tag)
    finally:
        gmsh.finalize()

    assert 80.0 < volume_cm3 < 250.0, volume_cm3          # o volume de fluido de uma válvula DN20
    assert len(groups["inlet"]) == 1 and len(groups["outlet"]) == 1
    assert len(groups["wall"]) >= 3
    assert bbox[4] > 60.0, "o pescoço tem de subir acima do corpo"     # y_max, mm
