"""Tests for ``pinneapple_analysis.trust.certificate`` -- verification as a service (#341).

The question the issue asks is the one a consumer of somebody else's surrogate has: *does this field satisfy the
physics it claims, and where does it stop doing so?* So what is asserted here is the decision, not the plumbing:

* the **law**: a field that solves ``dC/dt + c dC/dx = alpha d2C/dx2`` to round-off is certified; the same field
  with the advection speed off by 60 % is not, and the certificate says which point in the domain is worst.
* the **limit**: the same number is a pass, a failure or nothing at all depending on what it is judged against --
  the model's own in-domain level first, the caller's declared tolerance second, and never a silent pass third.
* the **domain**: a field inside the envelope is judged on its residual; a field that left the envelope is
  ``no_skill``, which is a different sentence from "the residual is high".
* the **reading**: arrays, an ``.npz`` file, a flat shallow-water lake and a shallow-water lake that is not flat
  (zero velocity under a bumpy free surface is not a solution of the equations, and the certificate says so).
* the **calibration**: detection rate and false alarms, measured case by case over in-domain and out-of-domain
  predictions -- the two numbers the issue asks for, printed as a table so they can be read instead of assumed.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis.trust.certificate import (
    Calibration,
    Certificate,
    Problem,
    calibrate,
    read_fields,
    verify,
)
from pinneapple_analysis.trust.record import Trust

# --------------------------------------------------------------------------------------------------- the case
# C(x, t) = exp(-alpha t) sin(x - c t) solves the advection-diffusion equation exactly, so the truth is known by
# construction and the residual the certificate reports is the prediction's, not the reference's.


def _grid(n: int = 65, m: int = 41):
    x = np.linspace(0.0, 2.0 * np.pi, n)[:-1]          # periodic in x: the last point repeats the first
    t = np.linspace(0.0, 1.0, m)
    return x, t


def _field(*, c: float = 1.0, alpha: float = 0.05, noise: float = 0.0, seed: int = 0):
    x, t = _grid()
    X, T = np.meshgrid(x, t, indexing="ij")
    C = np.exp(-alpha * T) * np.sin(X - c * T)
    if noise:
        C = C + noise * C.std() * np.random.default_rng(seed).standard_normal(C.shape)
    return {"C": C}, {"x": x, "t": t}


def _problem(axes, **overrides):
    body = dict(physics="advection-diffusion", axes=axes, parameters={"c": 1.0, "alpha": 0.05},
                units={"C": "1", "x": "m", "t": "s"}, tolerance={"advection-diffusion": 0.05})
    body.update(overrides)
    return Problem(**body)


def _named(cert: Certificate) -> dict[str, str]:
    return {c.name: c.status for c in cert.checks}


# ------------------------------------------------------------------------------------------- the law, measured
def test_the_exact_solution_is_certified():
    fields, axes = _field()
    cert = verify(fields, _problem(axes))

    assert cert.certified() is True
    assert cert.badge() == "good"
    assert cert.verdict() == "certified"
    assert cert.failures() == ()
    assert cert.unmeasured() == ()
    residual = next(c for c in cert.checks if c.name.startswith("residual:"))
    assert residual.value < 0.01, "the exact solution leaves round-off, not a modelling residual"
    assert residual.passed is True


def test_a_field_that_does_not_solve_the_equations_is_rejected_and_the_worst_point_is_named():
    fields, axes = _field(c=1.6)                        # the field advects at 1.6, the problem declares 1.0
    cert = verify(fields, _problem(axes))

    assert cert.badge() == "fading"
    assert cert.verdict() == "qualified"
    failing = cert.failures()
    assert [c.name for c in failing] == ["residual:advection-diffusion"]
    assert failing[0].value > failing[0].threshold
    assert failing[0].where, "a failing check has to say where in the domain it is worst"
    assert cert.separating().startswith("failing:")


def test_the_residual_with_no_limit_is_never_a_pass():
    fields, axes = _field()
    cert = verify(fields, _problem(axes, tolerance={}))

    assert _named(cert)["residual:advection-diffusion"] == "unmeasured"
    assert cert.badge() == "fading"
    assert cert.certified() is False
    assert cert.unmeasured(), "a residual nobody can judge is listed as unmeasured"
    assert cert.separating().startswith("not measured:")


def test_the_in_domain_level_wins_over_the_declared_tolerance():
    fields, axes = _field(c=1.2)                        # residual lands between the two limits
    loose = verify(fields, _problem(axes, tolerance={"advection-diffusion": 0.5}))
    strict = verify(fields, _problem(axes, tolerance={"advection-diffusion": 0.5},
                                     reference={"advection-diffusion": 1e-3},
                                     reference_source="the ensemble's own in-domain median"))

    assert loose.certified() is True
    residual = next(c for c in strict.checks if c.name.startswith("residual:"))
    assert residual.threshold == pytest.approx(1e-3)
    assert strict.certified() is False
    assert "in-domain median" in residual.source, "the certificate says where its limit came from"


def test_units_the_equation_uses_and_nobody_declared_are_not_measured():
    fields, axes = _field()
    cert = verify(fields, _problem(axes, units={"x": "m", "t": "s"}))

    assert _named(cert)["units:C"] == "unmeasured"
    assert cert.badge() == "fading"


def test_a_parameter_with_the_wrong_dimension_stops_before_the_residual():
    fields, axes = _field()
    cert = verify(fields, _problem(axes, parameters={"c": "1 kg", "alpha": 0.05}))

    statuses = _named(cert)
    assert statuses["parameter:c"] == "fail"
    assert not any(name.startswith("residual:") for name in statuses), \
        "a residual computed from a differently-scaled number is not reported"
    assert "m·s^-1" in cert.separating() or "m/s" in cert.separating()


def test_a_field_that_goes_negative_where_it_cannot_fails_positivity():
    fields, axes = _field()
    fields["C"] = fields["C"] - 1.5
    cert = verify(fields, _problem(axes, positivity=("C",)))

    check = next(c for c in cert.checks if c.name == "positivity:C")
    assert check.status == "fail"
    assert check.value < 0 and check.where


# ------------------------------------------------------------------------------------------------ the domain
def test_leaving_the_declared_domain_is_no_skill_and_says_so():
    fields, axes = _field()
    cert = verify(fields, _problem(axes, domain={"t": (0.0, 0.5)}))     # the field runs to t = 1

    assert _named(cert)["domain"] == "no_skill"
    assert cert.badge() == "no_skill"
    assert cert.verdict() == "rejected"
    assert cert.separating().startswith("no_skill: domain")


def test_a_field_inside_the_domain_is_judged_on_the_residual():
    fields, axes = _field()
    cert = verify(fields, _problem(axes, domain={"t": (-1.0, 2.0)}))

    assert _named(cert)["domain"] == "pass"
    assert cert.badge() == "good"


# --------------------------------------------------------------------------------------------- what it reads
def test_an_npz_file_is_read_and_certified(tmp_path):
    fields, axes = _field()
    path = tmp_path / "case.npz"
    np.savez(path, x=axes["x"], t=axes["t"], C=fields["C"])

    fd = read_fields(path)
    assert fd.kind == "grid"
    cert = verify(path, _problem(axes))
    assert cert.badge() == "good"


def test_flat_water_at_rest_satisfies_shallow_water():
    x, y, t = np.linspace(0, 1, 21), np.linspace(0, 1, 17), np.linspace(0, 0.2, 9)
    shape = (len(x), len(y), len(t))
    cert = verify({"h": np.full(shape, 2.0), "u": np.zeros(shape), "v": np.zeros(shape)},
                  Problem(physics="shallow-water", axes={"x": x, "y": y, "t": t}, parameters={"g": 9.81},
                          units={"h": "m", "u": "m/s", "v": "m/s", "x": "m", "y": "m", "t": "s"},
                          tolerance={"shallow-water": 1e-6}))

    assert cert.badge() == "good"
    assert {"residual:continuity", "residual:momentum"} <= set(_named(cert))


def test_a_bumpy_surface_at_rest_is_not_a_solution_of_shallow_water():
    x, y, t = np.linspace(0, 1, 21), np.linspace(0, 1, 17), np.linspace(0, 0.2, 9)
    X, Y, _ = np.meshgrid(x, y, t, indexing="ij")
    h = 2.0 + 0.1 * np.sin(2 * np.pi * X) * np.cos(np.pi * Y)     # gravity with nowhere to go
    cert = verify({"h": h, "u": np.zeros_like(h), "v": np.zeros_like(h)},
                  Problem(physics="shallow-water", axes={"x": x, "y": y, "t": t}, parameters={"g": 9.81},
                          units={"h": "m", "u": "m/s", "v": "m/s", "x": "m", "y": "m", "t": "s"},
                          tolerance={"shallow-water": 1e-3}))

    assert _named(cert)["residual:momentum"] == "fail"
    assert _named(cert)["residual:continuity"] == "pass"


# ------------------------------------------------------------------------------------------- the document
def test_the_certificate_round_trips_through_json():
    fields, axes = _field()
    cert = verify(fields, _problem(axes, domain={"t": (0.0, 0.5)}))

    again = Certificate.from_json(cert.to_json())
    assert again.badge() == cert.badge()
    assert again.verdict() == cert.verdict()
    assert [(c.name, c.status) for c in again.checks] == [(c.name, c.status) for c in cert.checks]
    assert json.loads(cert.to_json())["schema"] == cert.schema


def test_a_document_of_another_schema_is_refused():
    with pytest.raises(ValueError):
        Certificate.from_dict({"schema": "something-else/1", "trust": {}})


def test_the_certificate_can_be_written_and_read_back(tmp_path):
    fields, axes = _field()
    cert = verify(fields, _problem(axes))
    path = cert.write(tmp_path / "cert.json")

    assert path.exists()
    assert Certificate.from_json(path.read_text(encoding="utf-8")).certified() is True


def test_a_supplied_trust_record_keeps_its_own_gates():
    fields, axes = _field()
    cert = verify(fields, _problem(axes), trust=Trust.empty(kind="field"))

    names = {g.name for g in cert.gates()}
    assert cert.trust_gates is True
    assert {"horizon", "measurement"} <= names, "the sections the certificate did not measure are kept, not dropped"
    assert cert.badge() == "fading", "a section nobody filled is not a pass"


# -------------------------------------------------------------------------------------------- the calibration
def _calibration() -> Calibration:
    """Six predictions of the same field: three that obey the equation, three that do not.

    The out-of-domain ones are what a surrogate does when it is asked outside its range: it returns something
    smooth, plausible and wrong. The in-domain ones carry the perturbation a model that obeys the law still has --
    measured, not assumed: 0.1 % and 0.2 % of the field's own standard deviation land at residual 0.019 and 0.039,
    under the 0.05 declared, while 0.5 % lands at 0.095 and is flagged (see the boundary test below).
    """
    cases = []
    for i, noise in enumerate((0.0, 0.001, 0.002)):
        fields, axes = _field(noise=noise, seed=i)
        cases.append((f"in-domain-{i}", fields, _problem(axes), True))
    for i, c in enumerate((1.3, 1.6, 2.0)):
        fields, axes = _field(c=c, noise=0.01, seed=100 + i)
        cases.append((f"out-of-domain-{i}", fields, _problem(axes), False))
    return calibrate(cases)


def test_the_calibration_separates_in_domain_from_out_of_domain():
    cal = _calibration()

    detection, false_alarms = cal.rate("residual:advection-diffusion")
    assert detection == 1.0, "every out-of-domain prediction has to be flagged by the residual"
    assert false_alarms == 0.0, "a prediction that obeys the law must not be flagged"
    assert {"residual:advection-diffusion"} <= set(cal.gates())


def test_the_calibration_table_prints_the_counts_it_was_computed_from():
    table = _calibration().table()

    assert "detection" in table and "false alarms" in table
    assert "3/3 out" in table and "0/3 in" in table


def test_the_calibration_document_is_json():
    document = _calibration().to_dict()

    assert document["schema"] == "pinneapple.calibration/1"
    assert document["rates"]["residual:advection-diffusion"]["detection"] == 1.0
    assert len(document["cases"]) == 6
    json.dumps(document, allow_nan=False)


def test_a_rate_over_no_case_is_not_zero():
    fields, axes = _field()
    cal = calibrate([("only-in-domain", fields, _problem(axes), True)])

    detection, false_alarms = cal.rate("residual:advection-diffusion")
    assert detection != detection              # nan: no out-of-domain case to detect
    assert false_alarms == 0.0


def test_the_measured_boundary_of_the_residual_gate():
    """Where the separation actually sits, so the 1.0 / 0.0 above is not read as a general law.

    A residual of 0.05 separates 0.2 % of noise (0.039, a pass) from 0.5 % (0.095, flagged) on this case. That is a
    boundary of *this* grid and *this* tolerance, and it is the reason the certificate takes the in-domain level
    from the model rather than a constant: a whichever-the-caller-likes tolerance moves the boundary.
    """
    tolerated, axes = _field(noise=0.002, seed=7)
    flagged, _ = _field(noise=0.005, seed=7)

    assert verify(tolerated, _problem(axes)).certified() is True
    assert verify(flagged, _problem(axes)).certified() is False

    stricter = verify(tolerated, _problem(axes, tolerance={"advection-diffusion": 0.02}))
    assert stricter.certified() is False, "the same field fails a stricter limit -- the limit is the caller's"
