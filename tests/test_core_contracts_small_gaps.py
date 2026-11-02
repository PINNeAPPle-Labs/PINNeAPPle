"""Core coverage (#1129, toward 1.0): core tools that were exercised only indirectly (through another function, by
getattr or in a subprocess) are now called by name, each against what it promises."""
import json
import math
import os
import warnings

import numpy as np
import pytest

import pinneapple_stability as ST
from pinneapple_data.cae import foam as F
from pinneapple_data.cae import io as IO
from pinneapple_data.cae import metrics as MT
from pinneapple_data.cae.upload import expand
from pinneapple_physics.closed_form import (
    eddy_coil,
    ideal_reactors,
    pipe_flow,
    plate_fin_heatsink,
    rocket_nozzle,
)
from pinneapple_stability import api as API
from pinneapple_stability import deprecation as DEP

EX = os.path.join(os.path.dirname(__file__), "..", "apps", "mesh_quality", "examples")


# ------------------------------------------------------------------------------------------------ stability
def test_warn_experimental_from_user_code_warns_once(monkeypatch):
    monkeypatch.delenv(ST.QUIET_ENV, raising=False)
    monkeypatch.setattr(ST, "_warned", set())
    monkeypatch.setattr(ST, "_importer", lambda: "user_script")             # the import comes from user code
    with pytest.warns(ST.ExperimentalWarning, match="pinneapple_x is experimental"):
        ST.warn_experimental("pinneapple_x")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ST.warn_experimental("pinneapple_x")                                  # once per package and process
        monkeypatch.setattr(ST, "_importer", lambda: "pinneapple_presets.cases")
        ST.warn_experimental("pinneapple_y")                                  # library-internal import: silent
    assert issubclass(ST.ExperimentalWarning, UserWarning)
    w = ST.ExperimentalWarning("m")
    assert str(w) == "m"


def test_deprecation_warning_class_is_shown_by_default():
    w = DEP.PinneappleDeprecationWarning("old")
    assert isinstance(w, FutureWarning) and str(w) == "old"


def test_api_cli_check_and_diff(tmp_path, capsys):
    assert API.main(["--check"]) == 0
    old = tmp_path / "old.json"
    snap = API.snapshot()
    snap["pp"].pop("solve")
    old.write_text(json.dumps(snap))
    API.main(["--diff", str(old)])
    assert "pp.solve" in capsys.readouterr().out                             # added relative to the old file


# ------------------------------------------------------------------------------------------------ CAE
def test_boundary_values_of_a_field_by_patch_type():
    with open(os.path.join(EX, "openfoam_pitzDaily.zip"), "rb") as f:
        fs = expand([("pitz.zip", f.read())])
    mesh = IO.read_any(fs, fields=False)
    p = mesh.poly
    mesh.cell_data_tmp = np.full(mesh.n_cells, 7.0)                           # what read_time sets
    field = (b'FoamFile { class volScalarField; object T; }\ndimensions [0 0 0 1 0 0 0];\ninternalField uniform 7;\n'
             b'boundaryField {\n inlet { type fixedValue; value uniform 300; }\n outlet { type zeroGradient; }\n'
             b' upperWall { type noSlip; }\n lowerWall { type fixedValue; value uniform 350; }\n'
             b' frontAndBack { type empty; }\n}\n')
    bv = F.read_boundary_values(field, mesh, 1)[:, 0]
    sizes = {pt["name"]: (pt["startFace"] - p.n_internal, pt["nFaces"]) for pt in p.patches}

    def on(name):
        a, n = sizes[name]
        return bv[a:a + n]
    assert np.all(on("inlet") == 300.0) and np.all(on("lowerWall") == 350.0)    # written values
    assert np.all(on("upperWall") == 0.0)                                        # noSlip
    assert np.all(on("outlet") == 7.0)                                           # zeroGradient: owner cell value
    assert np.all(np.isnan(on("frontAndBack")))                                  # empty patch
    assert F.read_boundary_values(b"FoamFile { class volScalarField; }\n", mesh, 1) is None


def test_courant_number_of_a_cube_is_u_dt_over_dx():
    """One cube of side dx in uniform flow along x: Co = |U| dt / dx (the docstring's reference case)."""
    dx, u, dt = 0.1, 3.0, 0.01
    S = dx * dx * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    co = MT.courant_number(S, np.zeros(6, int), np.zeros(0, int), np.tile([u, 0.0, 0.0], (6, 1)), [dx ** 3], dt,
                           where="face")
    assert co[0] == pytest.approx(u * dt / dx, rel=1e-12)                       # exact


# ------------------------------------------------------------------------------------------------ trainer metric
def test_metric_base_class_is_a_protocol():
    torch = pytest.importorskip("torch")
    from pinneapple_neural.trainer.metrics import Metric
    with pytest.raises(NotImplementedError):
        Metric("base")(torch.zeros(1), torch.zeros(1))

    class Bias(Metric):
        def __call__(self, y_hat, y):
            return float((y_hat - y).mean())
    assert Bias("bias").name == "bias" and Bias("bias")(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 1.0])) == 2.0


# ------------------------------------------------------------------------------------------------ closed forms
def test_friction_factor_laminar_and_colebrook():
    assert pipe_flow.friction_factor(1000.0) == pytest.approx(0.064)                       # 64 / Re
    f = pipe_flow.friction_factor(1e5, 1e-4)
    colebrook = -2.0 * math.log10(1e-4 / 3.7 + 2.51 / (1e5 * math.sqrt(f)))
    assert 1 / math.sqrt(f) == pytest.approx(colebrook, rel=1e-8)                         # satisfies Colebrook
    assert f == pytest.approx(0.0185, rel=0.02)                                           # Moody chart


def test_skin_depth_of_copper_at_mains_frequency():
    d = eddy_coil.skin_depth(50.0, 5.8e7)
    assert d == pytest.approx(1 / math.sqrt(math.pi * 50 * 4e-7 * math.pi * 5.8e7), rel=1e-6)
    assert d == pytest.approx(9.35e-3, rel=5e-3)                                           # published: 9.3 mm


def test_specific_impulse_definition():
    assert rocket_nozzle.specific_impulse(1.6, 1500.0) == pytest.approx(1.6 * 1500.0 / 9.80665)


def test_pfr_result_is_a_first_order_pfr():
    D, L, k, Q = 0.05, 2.0, 0.3, 1e-3
    r = ideal_reactors.pfr_conversion(D, L, k, Q)
    assert ideal_reactors.PFRResult(**r) == r
    V = math.pi * D * D / 4 * L
    assert r["V"] == pytest.approx(V) and r["tau"] == pytest.approx(V / Q)
    assert r["X"] == pytest.approx(1 - math.exp(-k * V / Q), rel=1e-12)                    # exact, first order


def test_heatsink_geometry_errors_and_the_elenbaas_limits():
    g = plate_fin_heatsink.HeatSinkGeometry(0.1, 0.1, 0.005, 10, 0.0015, 0.03)
    op = plate_fin_heatsink.OperatingPoint(20.0, 25.0, 0.0, 0.03, 0.03)
    assert plate_fin_heatsink.geometry_errors(g, op) == []
    bad = plate_fin_heatsink.HeatSinkGeometry(0.01, 0.1, 0.005, 10, 0.0015, 0.03, material="cheese")
    errs = plate_fin_heatsink.geometry_errors(bad, plate_fin_heatsink.OperatingPoint(-1.0, 25.0, -1.0, 0.03, 0.03))
    assert len(errs) == 5                                    # material, fins don't fit, power, source, velocity
    air = {"beta": 1 / 300.0, "nu": 1.6e-5, "alpha": 2.2e-5, "k": 0.026}
    nc = plate_fin_heatsink.natural_convection(g, 30.0, air)
    el = nc["elenbaas"]
    assert nc["nu_b"] == pytest.approx((576 / el ** 2 + 2.873 / math.sqrt(el)) ** -0.5)
    assert nc["h"] == pytest.approx(nc["nu_b"] * 0.026 / g.fin_gap) and nc["fan_power_w"] == 0.0
    narrow = plate_fin_heatsink.HeatSinkGeometry(0.1, 0.1, 0.005, 40, 0.0015, 0.03)       # tiny gaps: El << 1
    lo = plate_fin_heatsink.natural_convection(narrow, 30.0, air)
    assert lo["nu_b"] == pytest.approx(lo["elenbaas"] / 24, rel=1e-2)                       # fully developed: El/24
    wide = plate_fin_heatsink.HeatSinkGeometry(0.5, 0.01, 0.005, 2, 0.0015, 0.03)          # isolated plates
    hi = plate_fin_heatsink.natural_convection(wide, 30.0, air)
    assert hi["nu_b"] == pytest.approx(0.59 * hi["elenbaas"] ** 0.25, rel=2e-2)            # 0.59 El^(1/4)

