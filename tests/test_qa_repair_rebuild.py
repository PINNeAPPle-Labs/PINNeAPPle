"""Repair that changes the data must be judged on the data it produced.

``QualityPipeline(rebuild_from_source=...)`` exists because a repair can be
right about the code and wrong about the output: re-validating the corrected
source against the bundle it has not yet produced judges the fix on stale
numbers. These tests pin both paths -- with the callable the scientific
validators must see the rebuilt bundle, without it they must see exactly the
caller's bundle -- and pin what a failed rebuild does (fails the attempt,
never the run).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pinneapple_llm.qa as qa
from pinneapple_llm.qa.pipeline import _context_from_bundle
from pinneapple_llm.qa.validators import register

#: Every context the probe validator was handed, in order.
SEEN: list = []

#: The invariant the probe enforces, in the artifact's own unit (mm).
LIMIT_MM = 8.0


@register("probe_rebuild")
def _probe(ctx: qa.ValidationContext) -> qa.ValidatorResult:
    """A scientific validator that records its input and enforces one cap."""
    SEEN.append(ctx)
    values = ctx.fields.get("wear_depth")
    if values is None:
        return qa.ValidatorResult(
            name="probe_rebuild", status=qa.Status.BLOCKED,
            findings=[qa.Finding(id="probe.unsatisfied", status=qa.Status.BLOCKED,
                                 severity="warning", message="no wear_depth field")])
    worst = float(np.max(np.asarray(values, dtype=float)))
    if worst > LIMIT_MM:
        return qa.ValidatorResult(
            name="probe_rebuild", status=qa.Status.FAIL,
            findings=[qa.Finding(id="invariant.wear_cap", status=qa.Status.FAIL, severity="error",
                                 message=f"wear_depth reaches {worst} mm, above {LIMIT_MM} mm")],
            evidence={"max_mm": worst})
    return qa.ValidatorResult(name="probe_rebuild", status=qa.Status.PASS,
                              evidence={"max_mm": worst})


def _spec():
    return qa.PhysicalSpecification(
        objective="repair rebuild test",
        phenomenon="synthetic wall thinning",
        geometry=qa.GeometrySpec(kind="hollow_pipe",
                                 dimensions={"length": qa.Quantity(1.0, "m")}, hollow=True,
                                 wall_thickness=qa.Quantity(0.01, "m")),
        independent_variables=["t"],
        dependent_variables=["wear_depth"],
        invariants=[qa.Invariant("wear_cap", "max", "wear_depth", params={"limit": LIMIT_MM})],
        spatial_validity={"y": (qa.Quantity(0.0, "m"), qa.Quantity(1.0, "m"))},
        temporal_validity=(qa.Quantity(0.0, "year"), qa.Quantity(5.0, "year")),
        required_validations=["probe_rebuild"],
    )


def _bundle(worst_mm: float, *, times=None):
    times = np.linspace(0, 5, 4) if times is None else times
    values = np.tile(np.linspace(0.0, worst_mm, 3), (len(times), 1))
    return {
        "meshes": {"wall": (np.zeros((4, 3)), np.array([[0, 1, 2], [0, 2, 3]]))},
        "fields": {"wear_depth": values},
        "field_units": {"wear_depth": "mm"},
        "times": times,
        "sensors": [qa.SensorReading("S1", (0.0, 0.0, 0.0), "mm", "wear_depth",
                                     np.linspace(0, worst_mm, len(times)), holdout=True)],
        "payload": {"note": "rebuilt"},
    }


def _pipeline(**kwargs):
    kwargs.setdefault("validators", ("probe_rebuild",))
    return qa.QualityPipeline(_spec(), required_validations=("probe_rebuild",),
                              max_repair_attempts=3, **kwargs)


SOURCE_BAD = "def wear_rate():\n    return 9.0\n"
SOURCE_FIXED = "def wear_rate():\n    return 6.0\n"


@pytest.fixture(autouse=True)
def _clear_seen():
    SEEN.clear()
    yield
    SEEN.clear()


def _attempt_evidence(report):
    step = report.step("repair")
    assert step is not None
    return [f.evidence for f in step.findings if f.id.startswith("repair.attempt.")]


# ---------------------------------------------------------------------------
# (a) without rebuild_from_source nothing changes
# ---------------------------------------------------------------------------
def test_without_rebuild_the_validators_see_the_callers_bundle():
    bundle = _bundle(worst_mm=6.0)

    def repair(_artifact, _diagnosis, n):
        return f"def wear_rate():\n    return {6.0 + n}\n"

    pipeline = _pipeline(repair=repair)
    report = pipeline.run(source=SOURCE_BAD, **bundle)

    # the pre-repair validator pass plus one pass per repair attempt, all on
    # the very same caller-supplied context -- no data was rebuilt.
    assert len(SEEN) >= 2
    assert all(ctx is SEEN[0] for ctx in SEEN)
    assert SEEN[0].fields["wear_depth"] is bundle["fields"]["wear_depth"]
    assert SEEN[0].payload is not bundle["payload"] and SEEN[0].payload == bundle["payload"]

    attempts = _attempt_evidence(report)
    assert attempts and all(a["bundle_source"] == "stale" for a in attempts)
    assert all(a["rebuilt"] is False for a in attempts)
    assert report.steps and any("re-execute the repaired source" in lim
                               for lim in report.limitations)


# ---------------------------------------------------------------------------
# (b) with rebuild_from_source the validators see the new data
# ---------------------------------------------------------------------------
def test_rebuild_lets_a_repair_that_changed_the_data_be_accepted():
    rebuilt_from = []

    def repair(_artifact, _diagnosis, _n):
        return SOURCE_FIXED

    def rebuild_from_source(source):
        rebuilt_from.append(source)
        # the original code produced 9 mm; the fix produces 6 mm
        return _bundle(worst_mm=9.0 if source == SOURCE_BAD else 6.0)

    pipeline = _pipeline(repair=repair, rebuild_from_source=rebuild_from_source)
    report = pipeline.run(source=SOURCE_BAD, **_bundle(worst_mm=6.0))

    assert rebuilt_from == [SOURCE_BAD, SOURCE_FIXED]
    # pre-repair pass sees the caller's bundle; each attempt sees rebuilt data
    assert len(SEEN) == 3
    assert float(np.max(SEEN[1].fields["wear_depth"])) == pytest.approx(9.0)
    assert float(np.max(SEEN[2].fields["wear_depth"])) == pytest.approx(6.0)
    assert SEEN[1].sensors[0].id == "S1" and SEEN[1].payload == {"note": "rebuilt"}

    step = report.step("repair")
    assert step.status is qa.Status.PASS
    attempts = _attempt_evidence(report)
    assert len(attempts) == 2
    assert attempts[0]["accepted"] is False and attempts[1]["accepted"] is True
    assert all(a["bundle_source"] == "fresh" and a["rebuilt"] is True for a in attempts)


def test_rebuild_that_produces_object_bundles_is_accepted():
    bundle = SimpleNamespace(**_bundle(worst_mm=5.0))

    pipeline = _pipeline(repair=lambda _a, _d, _n: SOURCE_FIXED,
                         rebuild_from_source=lambda _source: bundle)
    report = pipeline.run(source=SOURCE_BAD, **_bundle(worst_mm=6.0))

    assert report.step("repair").status is qa.Status.PASS
    assert SEEN[-1].fields["wear_depth"].shape == bundle.fields["wear_depth"].shape


def test_rebuilt_data_that_still_violates_the_invariant_is_rejected():
    def rebuild_from_source(_source):
        return _bundle(worst_mm=9.5)

    pipeline = _pipeline(repair=lambda _a, _d, n: f"def wear_rate():\n    return {7 + n}.0\n",
                         rebuild_from_source=rebuild_from_source)
    report = pipeline.run(source=SOURCE_BAD, **_bundle(worst_mm=1.0))

    step = report.step("repair")
    assert step.status is qa.Status.FAIL
    attempts = _attempt_evidence(report)
    assert attempts and all(a["accepted"] is False for a in attempts)
    # judged on rebuilt data that fails, not on the caller's passing bundle
    assert all(a["bundle_source"] == "fresh" for a in attempts)
    diagnosed = [f["id"] for a in attempts for f in a["diagnosis"]["findings"]]
    assert "invariant.wear_cap" in diagnosed


# ---------------------------------------------------------------------------
# (c) a failing rebuild fails the attempt, not the run
# ---------------------------------------------------------------------------


def test_sandbox_runs_the_candidate_before_the_rebuild():
    ran = []
    rebuilt = []

    class FakeSandbox:
        def __init__(self, status):
            self.status = status

        def run_python(self, source, files=None, output_dir=None):
            ran.append(source)
            passed = self.status is qa.Status.PASS
            return qa.SandboxResult(status=self.status,
                                    isolation=qa.IsolationLevel.SUBPROCESS,
                                    exit_code=0 if passed else 2,
                                    stdout="", stderr="" if passed else "boom")

    pipeline = _pipeline(repair=lambda _a, _d, n: f"def wear_rate():\n    return {6 + n}.0\n",
                         rebuild_from_source=lambda source: (rebuilt.append(source),
                                                             _bundle(worst_mm=6.0))[1],
                         sandbox=FakeSandbox(qa.Status.PASS))
    report = pipeline.run(source=SOURCE_BAD, **_bundle(worst_mm=6.0))
    assert report.step("repair").status is qa.Status.PASS
    assert ran and rebuilt and ran[0] == rebuilt[0]

    ran.clear()
    rebuilt.clear()
    pipeline = _pipeline(repair=lambda _a, _d, n: f"def wear_rate():\n    return {6 + n}.0\n",
                         rebuild_from_source=lambda source: (rebuilt.append(source),
                                                             _bundle(worst_mm=6.0))[1],
                         sandbox=FakeSandbox(qa.Status.FAIL))
    report = pipeline.run(source=SOURCE_BAD, **_bundle(worst_mm=6.0))
    assert report.step("repair").status is qa.Status.FAIL
    assert ran and not rebuilt  # nothing to rebuild from a failed execution
    attempts = _attempt_evidence(report)
    assert all(a["execution_status"] == "FAIL" and a["rebuilt"] is False for a in attempts)


def test_rebuild_that_raises_is_recorded_and_the_loop_continues():
    def rebuild_from_source(_source):
        raise RuntimeError("the repaired script wrote no fields")

    pipeline = _pipeline(repair=lambda _a, _d, n: f"def wear_rate():\n    return {6 + n}.0\n",
                         rebuild_from_source=rebuild_from_source)
    report = pipeline.run(source=SOURCE_BAD, **_bundle(worst_mm=6.0))

    step = report.step("repair")
    assert step.status is qa.Status.FAIL
    attempts = _attempt_evidence(report)
    assert len(attempts) >= 2
    assert all(a["accepted"] is False for a in attempts)
    assert all("the repaired script wrote no fields" in a["rebuild_error"] for a in attempts)
    assert all(a["bundle_source"] == "stale" and a["rebuilt"] is False for a in attempts)
    # the scientific validator never ran on data the rebuild failed to produce
    assert len(SEEN) == 1


# ---------------------------------------------------------------------------
# (d) a malformed bundle is refused with the missing key named
# ---------------------------------------------------------------------------
def test_malformed_bundle_raises_value_error_naming_the_missing_key():
    base = qa.ValidationContext(spec=_spec())
    bundle = _bundle(worst_mm=6.0)
    del bundle["times"]

    with pytest.raises(ValueError) as excinfo:
        _context_from_bundle(bundle, base=base, spec=_spec())
    assert "times" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        _context_from_bundle(object(), base=base, spec=_spec())
    assert "meshes" in str(excinfo.value)


def test_malformed_bundle_is_reported_as_a_failed_attempt():
    broken = _bundle(worst_mm=6.0)
    del broken["sensors"]

    pipeline = _pipeline(repair=lambda _a, _d, n: f"def wear_rate():\n    return {6 + n}.0\n",
                         rebuild_from_source=lambda _source: broken)
    report = pipeline.run(source=SOURCE_BAD, **_bundle(worst_mm=6.0))

    step = report.step("repair")
    assert step.status is qa.Status.FAIL
    attempts = _attempt_evidence(report)
    assert attempts and all("'sensors'" in a["rebuild_error"] for a in attempts)
    assert all("ValueError" in a["rebuild_error"] for a in attempts)
