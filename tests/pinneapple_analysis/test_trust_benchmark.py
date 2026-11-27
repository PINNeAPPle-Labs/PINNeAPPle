"""Tests for ``pinneapple_analysis.trust.benchmark`` -- the certificate measured on five physics families and one CFD
surrogate (#341).

The "Done when" of the issue is one sentence: *the certificate separates in-domain from out-of-domain predictions on
the five-family benchmark and on one CFD surrogate*. A sentence like that is worth nothing unless something checks it,
so what is asserted here is the separation itself, group by group, and the fact that the JSON the guide prints is the
JSON this code produces:

* **both classes, every group** -- a family with only in-domain cases measures nothing, and the test refuses to read a
  rate over an empty population as a result.
* **no false alarms** -- every in-domain field (the exact solution and two noise levels) is certified, in all six
  groups. A benchmark whose in-domain cases fail is measuring its own tolerance, not the model.
* **the misses are named, not averaged away** -- four groups leave one out-of-domain kind uncertified-but-not-flagged
  (the sign inversion on the three linear homogeneous laws, the undamped direction of the viscosity error, and one
  small extrapolation of the surrogate below its envelope). The test fixes the set: a change in what the certificate
  catches has to show up as a change here, with the note that says why the row is what it is.
* **the committed document** -- ``results/certificate_five_families.json`` is compared with a fresh measurement, so
  the numbers in the guide cannot drift from the code that produced them.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pinneapple_analysis.trust.benchmark import CFD_TRAIN, SCHEMA, family_cases, measure, table, write
from pinneapple_analysis.trust.certificate import verify

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "results" / "certificate_five_families.json"

#: The five families plus the CFD surrogate: the six groups the issue asks the separation to be measured on.
GROUPS = ("advection-diffusion", "heat", "incompressible-flow", "shallow-water", "elasticity", "cfd-surrogate")

#: The out-of-domain cases the certificate does **not** flag, as measured, with the reason each one is expected.
#: ``sign``: the residual of a linear, homogeneous law is invariant under ``C -> -C``. ``parameter-undamped``: at this
#: Reynolds number the viscous term that the error changes is small next to the advection, and 0.0279 stays under the
#: 0.05 the case is judged against. ``nu=0.02``: an extrapolation just below the fitted envelope whose momentum
#: residual lands at 0.0412, also under the limit -- the surrogate extrapolates gracefully there.
EXPECTED_MISSES = {
    "advection-diffusion": {"advection-diffusion/sign"},
    "heat": {"heat/sign"},
    "incompressible-flow": {"incompressible-flow/parameter-undamped"},
    "shallow-water": set(),
    "elasticity": {"elasticity/sign"},
    "cfd-surrogate": {"cfd-surrogate/nu=0.02"},
}


@pytest.fixture(scope="module")
def report() -> dict:
    """One measurement for the whole module: the benchmark takes a few seconds (the POD surrogate is fitted once)."""
    return measure()


def _group(report: dict, name: str) -> dict:
    return report["cfd_surrogate"] if name == "cfd-surrogate" else report["families"][name]


def _split(body: dict) -> tuple[list[dict], list[dict]]:
    return ([c for c in body["cases"] if c["in_domain"]], [c for c in body["cases"] if not c["in_domain"]])


def test_every_group_carries_both_classes(report):
    """A rate over an empty population is not a measurement: each group needs in-domain *and* out-of-domain cases."""
    for name in GROUPS:
        inside, outside = _split(_group(report, name))
        assert inside, f"{name} has no in-domain case: its false-alarm rate would be a rate over nothing"
        assert outside, f"{name} has no out-of-domain case: there is nothing to detect"
        assert _group(report, name)["cases_in"] == len(inside)
        assert _group(report, name)["cases_out"] == len(outside)


def test_no_in_domain_prediction_is_flagged(report):
    """Zero false alarms in all six groups -- the other half of the separation."""
    for name in GROUPS:
        inside, _ = _split(_group(report, name))
        flagged = [c["label"] for c in inside if c["verdict"] != "certified"]
        assert not flagged, f"{name}: in-domain fields the certificate refused: {flagged}"
        assert _group(report, name)["verdict"]["false_alarm"] == 0.0


def test_the_out_of_domain_cases_flagged_are_the_measured_ones(report):
    """The detection side, case by case: which out-of-domain kinds the certificate catches, and which it does not.

    The second set is the finding, not an omission: it is asserted so that a certificate that becomes stricter (or
    blinder) fails here instead of quietly changing the number the guide prints.
    """
    for name in GROUPS:
        _, outside = _split(_group(report, name))
        missed = {c["label"] for c in outside if c["verdict"] == "certified"}
        detected = {c["label"] for c in outside if c["verdict"] != "certified"}
        assert detected, f"{name}: no out-of-domain case was flagged -- the group does not separate"
        assert missed == EXPECTED_MISSES[name], f"{name}: the misses changed, {missed ^ EXPECTED_MISSES[name]}"
        assert detected | missed == {c["label"] for c in outside}


def test_the_detection_rate_is_what_the_counts_say(report):
    """The rate printed next to the cases is the count, not an independent claim."""
    for name in GROUPS:
        body = _group(report, name)
        _, outside = _split(body)
        flagged = [c for c in outside if c["verdict"] != "certified"]
        assert body["verdict"]["detection"] == pytest.approx(len(flagged) / len(outside))


def test_the_surrogate_is_judged_inside_and_outside_its_own_envelope(report):
    """The CFD group is a real surrogate: POD over the reference fields it was fitted on, four queries each side."""
    body = report["cfd_surrogate"]
    inside, outside = _split(body)
    assert body["train"] == list(CFD_TRAIN)
    assert len(inside) == len(outside) == 4
    assert all(c["kind"] == "inside" for c in inside) and all(c["kind"] == "outside" for c in outside)
    assert all(c["checks"]["domain"]["status"] == "no_skill" for c in outside if c["label"] != "cfd-surrogate/nu=0.02")


def test_the_table_names_the_cases_it_let_through(report):
    """The printed table is readable on its own: a rate below 1 comes with the labels behind it."""
    printed = table(report)
    assert "detection" in printed and "false alarms" in printed
    assert "out of domain, certified: advection-diffusion/sign" in printed
    assert "out of domain, certified: cfd-surrogate/nu=0.02" in printed
    assert "in domain, flagged" not in printed


def test_the_committed_document_is_what_this_code_produces(report):
    """``results/certificate_five_families.json`` is the report the guide prints: it has to be this one."""
    assert COMMITTED.exists(), "run python scripts/certificate_benchmark.py"
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    assert committed["schema"] == SCHEMA
    fresh = json.loads(json.dumps(report, allow_nan=False))
    assert committed == fresh, "results/certificate_five_families.json is out of date: rerun the script"


def test_the_report_round_trips_through_the_file_it_writes(report, tmp_path):
    path = write(tmp_path / "benchmark.json", report=report)
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(report, allow_nan=False))


def test_a_declared_boundary_catches_the_sign_the_residual_cannot_see():
    """The blind spot of the ``sign`` rows has a cure that costs one argument: declare the boundary.

    The residual of a linear, homogeneous law is invariant under ``C -> -C``, which is why the benchmark reports the
    sign case as *not flagged* rather than pretending it was detected. A problem that says what the field has to be at
    one face is a different measurement, and it does separate the two: the exact solution passes the boundary check at
    0.0 and the inverted one fails it at 1.0, relative to the field's range, against the 0.02 a boundary is judged at.
    This test holds the two halves of that sentence together, so neither is read without the other.
    """
    x = np.linspace(0.0, 2.0 * np.pi, 65)[:-1]
    exact = family_cases("advection-diffusion")[0]                       # the analytic solution of the family
    sign = next(c for c in family_cases("advection-diffusion") if c.kind == "sign")
    assert sign.in_domain is False
    without = replace(sign.problem, boundary={"t_min": {"C": lambda pts: np.sin(pts[:, 0])}})

    good = verify(exact.fields, without)
    bad = verify(sign.fields, without)

    assert good.certified() is True
    assert bad.certified() is False, "the inverted field passes the residual and must fail the declared boundary"
    boundary = {c.name: c for c in bad.checks if c.name.startswith("boundary")}
    assert boundary["boundary:t_min.C"].status == "fail"
    assert boundary["boundary:t_min.C"].value == pytest.approx(1.0)
    assert verify(sign.fields, sign.problem).certified() is True, "the residual alone stays blind -- as reported"
