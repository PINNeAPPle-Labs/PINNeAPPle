"""The universal physics data schema: what it refuses, what it keeps, and what the gates decide."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pinneapple_data.schema import (
    AxisSpec,
    Field,
    Geometry,
    Physics,
    PhysicsDataBundle,
    Provenance,
    Verification,
    run_gates,
)
from pinneapple_data.schema.bundle import content_digest
from pinneapple_data.schema.gates import conservation_drift, relative_l2
from pinneapple_data.schema.models import ArrayRef, unit_or_dimensionless


def wave(n_t: int = 12, n_x: int = 20, *, decay: float = 0.3) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n_x)
    t = np.linspace(0.0, 1.0, n_t)
    return (np.sin(2 * np.pi * (x[None, :] - decay * t[:, None])) * np.exp(-0.5 * t[:, None])).astype(np.float32)


def density(n_t: int = 12, n_x: int = 20) -> np.ndarray:
    """A travelling wave on a constant background: its spatial integral is a constant, so conservation means
    something (a pure sine integrates to zero and has no drift to measure)."""
    x = np.linspace(0.0, 1.0, n_x)
    t = np.linspace(0.0, 1.0, n_t)
    return (1.0 + 0.1 * np.sin(2 * np.pi * (x[None, :] - 0.3 * t[:, None]))).astype(np.float32)


def make_bundle(**overrides) -> PhysicsDataBundle:
    u = overrides.pop("u", density())
    t = np.linspace(0.0, 1.0, u.shape[0])
    x = np.linspace(0.0, 1.0, u.shape[1])
    kwargs = dict(
        arrays={"field:u": u, "axis:t": t, "axis:x": x},
        fields=[Field.from_array("u", u, unit="m/s", axes=("t", "x"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")],
        physics=Physics(governing="u_t + c u_x = 0", symbols={"u": "velocity", "c": "speed"}),
        source="unit test wave", license="CC0-1.0", citation="none, synthetic", title="a wave",
        kind="synthetic",
    )
    kwargs.update(overrides)
    return PhysicsDataBundle.from_arrays(**kwargs)


# ── the manifest ───────────────────────────────────────────────────────────────────────────────────────────
def test_units_are_normalised_and_rejected():
    assert unit_or_dimensionless("-") == "1"
    assert unit_or_dimensionless("dimensionless") == "1"
    assert unit_or_dimensionless("kg/m^3") == "kg/m^3"
    with pytest.raises(ValueError, match="not a unit"):
        unit_or_dimensionless("furlongs per fortnight")


def test_a_field_must_match_the_axes_it_declares():
    u = wave()
    with pytest.raises(ValidationError, match="axes"):
        PhysicsDataBundle.from_arrays(
            {"field:u": u, "axis:t": np.arange(u.shape[0]), "axis:x": np.arange(u.shape[1] + 1)},
            fields=[Field.from_array("u", u, axes=("t", "x"))],
            axes=[AxisSpec("t", "s"), AxisSpec("x", "m")], source="x")


def test_a_field_cannot_reference_an_absent_array():
    u = wave()
    with pytest.raises(KeyError, match="field:u"):
        PhysicsDataBundle.from_arrays({"axis:t": np.arange(u.shape[0])}, fields=[Field.from_array("u", u)],
                                      axes=[AxisSpec("t", "s")], source="x")


def test_provenance_requires_a_licence_and_a_source():
    with pytest.raises(ValidationError):
        Provenance(source="")
    with pytest.raises(ValidationError):
        Provenance(source="x", license="")


def test_unknown_keys_are_errors_not_silences():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Geometry(kind="cad", unit="m")           # the field is `units`


# ── persistence and integrity ──────────────────────────────────────────────────────────────────────────────
def test_roundtrip_through_a_directory(tmp_path: Path):
    bundle = make_bundle()
    path = bundle.save(tmp_path / "b")
    assert (path / "bundle.json").exists() and (path / "arrays.npz").exists()
    loaded = PhysicsDataBundle.load(path)
    assert loaded == bundle
    assert loaded.manifest.digest == bundle.digest()
    assert np.allclose(loaded.field_array("u"), bundle.field_array("u"))
    assert json.loads((path / "bundle.json").read_text())["schema_version"]


def test_a_tampered_array_is_refused(tmp_path: Path):
    bundle = make_bundle()
    path = bundle.save(tmp_path / "b")
    arrays = dict(np.load(path / "arrays.npz"))
    arrays["field:u"] = arrays["field:u"] + 1.0
    np.savez_compressed(path / "arrays.npz", **arrays)
    with pytest.raises(ValueError, match="not intact"):
        PhysicsDataBundle.load(path)
    # and the report says exactly what changed
    with pytest.raises(ValueError):
        PhysicsDataBundle.load(path)
    report = PhysicsDataBundle(manifest=bundle.manifest, arrays=arrays).verify()
    assert not report["ok"] and "digest mismatch" in "; ".join(report["problems"])


def test_html_safe_json_and_digest_order(tmp_path: Path):
    bundle = make_bundle()
    assert bundle.digest() == bundle.digest()                     # stable
    same = make_bundle()
    assert same.digest() == bundle.digest()                       # same content, same digest
    other = make_bundle(u=wave(decay=0.4))
    assert other.digest() != bundle.digest()


def test_pack_and_unpack(tmp_path: Path):
    bundle = make_bundle()
    archive = bundle.pack(tmp_path / "b.zip")
    assert zipfile.is_zipfile(archive)
    assert PhysicsDataBundle.unpack(archive) == bundle


def test_content_digest_is_order_independent():
    a = {"x": np.zeros(3), "y": np.ones(2)}
    b = {"y": np.ones(2), "x": np.zeros(3)}
    assert content_digest(a) == content_digest(b)


# ── the gates ─────────────────────────────────────────────────────────────────────────────────────────────
def test_gates_pass_on_a_clean_bundle():
    report = run_gates(make_bundle(), conserved="u")
    assert report.ok, report.summary()
    assert report.get("integrity").passed
    assert report.get("frame").passed
    assert report.get("provenance").passed
    assert report.get("conservation").passed


def test_an_unknown_licence_fails_provenance():
    bundle = make_bundle(license="unknown")
    report = run_gates(bundle, gates=("provenance",))
    assert report.get("provenance").passed is False
    assert "licence" in report.get("provenance").detail


def test_a_non_monotonic_axis_fails_the_frame_gate():
    u = wave()
    x = np.linspace(0.0, 1.0, u.shape[1])
    x[[3, 4]] = x[[4, 3]]
    bundle = PhysicsDataBundle.from_arrays(
        {"field:u": u, "axis:t": np.linspace(0, 1, u.shape[0]), "axis:x": x},
        fields=[Field.from_array("u", u, axes=("t", "x"))], axes=[AxisSpec("t", "s"), AxisSpec("x", "m")],
        source="shuffled", license="CC0-1.0", citation="x")
    report = run_gates(bundle, gates=("frame",))
    assert report.get("frame").passed is False
    assert "monotonic" in report.get("frame").detail


def test_a_cell_centred_field_of_the_right_length_passes():
    report = run_gates(make_bundle(fields=[Field.from_array("u", wave(), location="cell", axes=("t", "x"))]),
                       gates=("location",))
    assert report.get("location").passed


def test_a_field_whose_shape_does_not_fit_its_axes_is_refused_at_construction():
    u = wave(12, 20)
    bad = np.zeros((u.shape[0], u.shape[1] - 3))
    with pytest.raises(ValidationError, match="field 'bad' has shape"):
        PhysicsDataBundle.from_arrays(
            {"field:u": u, "field:bad": bad, "axis:t": np.arange(u.shape[0]), "axis:x": np.arange(u.shape[1])},
            fields=[Field.from_array("u", u, axes=("t", "x")),
                    Field.from_array("bad", bad, location="cell", axes=("t", "x"))],
            axes=[AxisSpec("t", "s"), AxisSpec("x", "m")], source="x", license="CC0-1.0")


def test_the_location_gate_catches_a_cell_field_that_fits_no_convention():
    u = wave(12, 20)
    bad = np.zeros((u.shape[0], u.shape[1] - 3))          # neither nx nor nx-1
    bundle = PhysicsDataBundle.from_arrays(
        {"field:u": u, "field:bad": bad, "axis:t": np.arange(u.shape[0]), "axis:x": np.arange(u.shape[1])},
        fields=[Field.from_array("u", u, axes=("t", "x")),
                Field.from_array("bad", bad, location="cell")],   # no axes declared, so the gate is what checks
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")], source="x", license="CC0-1.0", kind="synthetic")
    bundle.manifest.fields.variables[1].axes = ["t", "x"]
    report = run_gates(bundle, gates=("location",))
    assert report.get("location").passed is False


def test_conservation_gate_reports_not_run_when_it_cannot_be_asked():
    report = run_gates(make_bundle(), conserved=None)
    assert report.get("conservation").passed is None
    assert report.get("conservation").status == "not run"
    assert not report.failed


def test_conservation_drift_finds_a_leak():
    t = np.linspace(0.0, 1.0, 20)
    x = np.linspace(0.0, 1.0, 32)
    tight = np.ones((20, 32)) * (1.0 - 0.5 * t)[:, None]        # mass decaying by half
    bundle = PhysicsDataBundle.from_arrays(
        {"field:rho": tight, "axis:t": t, "axis:x": x},
        fields=[Field.from_array("rho", tight, axes=("t", "x"))], axes=[AxisSpec("t", "s"), AxisSpec("x", "m")],
        source="leaky", license="CC0-1.0")
    drift = conservation_drift(bundle, "rho")
    assert drift == pytest.approx(0.5, abs=1e-9)
    report = run_gates(bundle, gates=("conservation",), conserved="rho")
    assert report.get("conservation").passed is False


def test_relative_l2_is_the_documented_metric():
    a = np.array([1.0, 2.0])
    assert relative_l2(a, a) == 0.0
    assert relative_l2(np.array([0.0, 0.0]), a) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="identically zero"):
        relative_l2(a, np.zeros(2))


def test_gate_report_applies_to_the_manifest_and_seals_into_a_ledger():
    from pinneapple_veriphysics.ledger import EvidenceLedger

    bundle = make_bundle()
    report = run_gates(bundle, conserved="u")
    report.apply(bundle)
    assert bundle.manifest.verification.passed
    ledger = EvidenceLedger()
    report.record(ledger)
    assert ledger.status("bundle/integrity") == "passed"
    assert ledger.status("bundle/conservation") == "passed"
    assert ledger.status("bundle/reference") == "not run"       # never ran, so never claimed


# ── interop ───────────────────────────────────────────────────────────────────────────────────────────────
def test_to_xarray_carries_the_units():
    xr = pytest.importorskip("xarray")
    ds = make_bundle().to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert ds["u"].attrs["units"] == "m/s"
    assert ds["x"].attrs["units"] == "m"
    assert ds.attrs["license"] == "CC0-1.0"
    assert make_bundle().to_xarray("u").dims == ("t", "x")


def test_summary_mentions_the_essentials():
    text = make_bundle().summary()
    assert "unit test wave" in text and "CC0-1.0" in text and "field" in text


def test_a_bundle_with_a_check_records_the_verdict():
    bundle = make_bundle()
    bundle.check("residual", value=0.2, tolerance=0.1, tool="t")
    assert bundle.manifest.verification.status("residual") == "failed"
    bundle.check("residual", value=0.05, tolerance=0.1, tool="t")
    assert bundle.manifest.verification.status("residual") == "passed"
    assert bundle.manifest.verification.get("residual").value == 0.05


def test_array_ref_rejects_an_impossible_shape():
    with pytest.raises(ValidationError):
        ArrayRef(key="k", shape=(), dtype="float32")


def test_verification_requires_the_named_gates():
    verification = Verification(required=["units", "reference"])
    assert not verification.passed and verification.missing == ["units", "reference"]


def test_saving_and_packing_the_same_bundle_twice_gives_the_same_bytes(tmp_path):
    """A digest over an archive is only meaningful if the archive is reproducible: numpy stamps the clock into
    ``arrays.npz`` and zipfile stamps the file's mtime, so both are written with a fixed timestamp."""
    bundle = make_bundle()
    digests = []
    for i in range(2):
        bundle.pack(tmp_path / f"ship{i}.zip")
        digests.append(hashlib.sha256((tmp_path / f"ship{i}.zip").read_bytes()).hexdigest())
    assert digests[0] == digests[1]

    first = bundle.save(tmp_path / "one")
    second = bundle.save(tmp_path / "two")
    assert (first / "arrays.npz").read_bytes() == (second / "arrays.npz").read_bytes()
    assert (first / "bundle.json").read_bytes() == (second / "bundle.json").read_bytes()
    assert PhysicsDataBundle.unpack(tmp_path / "ship0.zip").digest() == bundle.digest()
