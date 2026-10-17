"""Interior-PM machine FEA (``pinneapple_simulation.electromagnetics.machines``): parametric geometry, the
conforming sector mesher, catalogue steel and loss data, the IPM model's torque / flux-linkage / inductance
post-processing, and the verification cases (closed forms and the published Prius torque curve)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pinneapple_simulation.electromagnetics.fem import Geometry
from pinneapple_simulation.electromagnetics.machines import (
    M400_50A,
    IPMMachine,
    MachineMesh,
    TableIronLoss,
    fit_iron_loss,
    mesh_sector,
    prius_2004,
    stacked,
)
from pinneapple_simulation.electromagnetics.machines import verification as mv


@pytest.fixture(scope="module")
def prius():
    return IPMMachine(prius_2004(), n_gap=288)


def test_prius_geometry_matches_pyleecan_parameters():
    """Magnet 18.9 mm x 6.5 mm, 145 deg V, slot area of the closed form (trapezoid + half disc)."""
    g = prius_2004()
    s = g.rotor.sector(5e-4)
    mags = [p for p in s.polygons if p.name.startswith("magnet")]
    assert len(mags) == 2 and all(abs(p.area - 0.0189 * 0.0065) < 1e-15 for p in mags)
    assert math.degrees(g.rotor.v_angle) == pytest.approx(145.07, abs=0.01)
    assert g.airgap == pytest.approx(0.75e-3)
    assert g.stator.slot_area() == pytest.approx(0.5 * (5e-3 + 8e-3) * 29.3e-3 + 0.5 * math.pi * 4e-3 ** 2)


def test_sector_mesh_conforms_and_copies_close_the_circle():
    g = prius_2004()
    mm = MachineMesh(g, n_gap=288, smooth=1)
    mesh = mm.mesh(0.123)
    edges = np.sort(np.vstack([mesh.tris[:, [0, 1]], mesh.tris[:, [1, 2]], mesh.tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(counts) == {1, 2}  # conforming, no hanging nodes
    area = Geometry.of(mesh).area
    assert area.min() > 0
    ann = math.pi * (g.stator.r_out ** 2 - g.rotor.r_in ** 2)
    assert abs(area.sum() - ann) / ann < 5e-3  # circles are polygons: O(h^2) short of the annulus
    for k in range(g.rotor.poles):  # every pole has the same magnet area
        assert area[mesh.mask(f"magnet{k}")].sum() == pytest.approx(2 * 0.0189 * 0.0065, rel=1e-12)


def test_mesh_sector_rejects_nothing_and_has_fair_angles():
    g = prius_2004()
    sm = mesh_sector(g.rotor.sector(1e-3), 1e-3, 30, 60)
    assert sm.min_angle() > 10.0
    assert set(sm.region) == {"rotor_iron", "pocket_a", "pocket_b", "pocket_c", "magnet_a0", "magnet_b0"}


def test_catalogue_steel_and_losses():
    """The table model reproduces the catalogue; the Bertotti fit stays within its measured error."""
    tab = TableIronLoss(M400_50A)
    assert np.allclose(tab.specific_loss(M400_50A.loss_f, M400_50A.loss_B), M400_50A.loss_W_per_kg, rtol=1e-12)
    assert tab.specific_loss(50.0, 1.5) == pytest.approx(3.57)  # datasheet value, below the 4.00 W/kg grade limit
    fit = fit_iron_loss(M400_50A, f_max=1000.0)
    assert fit.rms_rel_error < 0.09 and fit.max_rel_error < 0.25
    kf = stacked(M400_50A.curve(), 0.95)
    b = np.interp(5000.0, kf.H, kf.B)
    assert b == pytest.approx(0.95 * np.interp(5000.0, M400_50A.H, M400_50A.B) + 0.05 * 4e-7 * math.pi * 5000.0,
                              rel=2e-3)


def test_park_round_trip_and_current_angle(prius):
    th = 0.0371
    i = prius.phase_currents(th, 100.0, math.radians(30.0))
    assert sum(i) == pytest.approx(0.0, abs=1e-12)
    d, q = prius.park(np.array(i), th)
    assert d == pytest.approx(-50.0, abs=1e-9) and q == pytest.approx(100 * math.cos(math.radians(30)), abs=1e-9)


def test_frozen_permeability_superposition_is_exact(prius):
    ind = prius.inductances(-100.0, 150.0)
    assert ind["superposition_error"] < 1e-8
    assert 0 < ind["L_dd"] < ind["L_qq"]  # saliency of an interior-PM rotor
    assert ind["psi_pm_d"] > 0.15


def test_no_load_flux_is_on_the_d_axis_and_cogging_is_periodic(prius):
    d, q = prius.park(prius.flux_linkage(prius.solve(0.0)), 0.0)
    assert abs(q) < 1e-9 * abs(d) and d > 0
    T0 = prius.torque(prius.solve(0.0))
    T1 = prius.torque(prius.solve(prius.cogging_period))
    assert T1 == pytest.approx(T0, abs=1e-6 * max(1.0, abs(T0)))


@pytest.mark.parametrize("name", [n for n in mv.CASES if not n.startswith(("Drive", "Prius flux map"))])
def test_machine_verification_case(name):
    """Each case meets its tolerance: closed forms at second order, torque consistency, the published curve."""
    r = mv.run([name], quick=True)[0]
    assert r.passed, r.row() + " " + r.detail


def test_loss_components_reproduce_the_direct_loss(prius):
    """One period of field solutions gives the iron loss at any speed: the speed-free sums of
    ``loss_components`` times the Bertotti coefficients equal ``losses`` with the same model."""
    fit = fit_iron_loss(M400_50A, f_max=1000.0)
    k_h, alpha, beta, k_c, k_e = fit.fitted
    c = prius.loss_components(-100.0, 150.0, alpha, beta, steps=12)
    for rpm in (1000.0, 4000.0):
        direct = prius.losses(rpm, -100.0, 150.0, steps=12, loss_model=fit.model)
        f = rpm / 60.0 * prius.pole_pairs
        S = c["stator"] + c["rotor"]
        from_sums = k_h * f ** alpha * S[0] + k_c * f ** 2 * S[1] + k_e * f ** 1.5 * S[2]
        assert from_sums == pytest.approx(direct["stator_iron_W"] + direct["rotor_iron_W"], rel=1e-10)
        assert c["magnet_per_f2"] * f ** 2 == pytest.approx(direct["magnet_W"], rel=1e-10)


def test_geometry_errors_accept_the_prius_and_reject_broken_rotors():
    import dataclasses

    from pinneapple_simulation.electromagnetics.machines.geometry import geometry_errors

    g = prius_2004()
    assert geometry_errors(g.rotor.sector(1e-3)) == [] and geometry_errors(g.stator.sector(1e-3)) == []
    assert any("leaves the sector" in e for e in geometry_errors(dataclasses.replace(g.rotor, W4=0.026).sector(1e-3)))
    assert any("cross" in e or "itself" in e
               for e in geometry_errors(dataclasses.replace(g.rotor, H3=0.012).sector(1e-3)))
