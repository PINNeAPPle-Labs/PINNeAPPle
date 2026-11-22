"""Which physical regimes a dataset reaches, and which ones it never does (#171).

"Coverage" that only measures how spread the inputs are answers the wrong question: a dataset can be perfectly spread
in Reynolds and still contain no turbulent case if the values cluster just below the threshold. These tests hold the
regime audit to the answer that matters — a named band with zero samples is reported, named, and comes back as an
interval a sampler can fill.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.audit import audit_dataset
from pinneapple_data.regimes import PRESETS, Band, RegimeSpec, regime_audit, regime_coverage

RE = PRESETS["reynolds_pipe"]


def test_a_band_labels_its_own_interval_and_the_boundary_belongs_to_the_upper_band():
    values = [0.0, 2299.9, 2300.0, 3999.9, 4000.0, 1e6, float("nan")]
    assert [RE.label(v) for v in values] == ["laminar", "laminar", "transitional", "transitional", "turbulent",
                                             "turbulent", "missing"]


def test_the_counts_and_the_share_add_up():
    cov = regime_coverage([1200, 1800, 9500, 5000], RE)
    assert cov.counts == {"laminar": 2, "transitional": 0, "turbulent": 2}
    assert cov.n == 4 and cov.share("laminar") == pytest.approx(0.5)
    assert sum(cov.share(label) for label in RE.labels) == pytest.approx(1.0)


def test_a_regime_with_no_sample_is_named_and_fails():
    cov = regime_coverage([1200, 1800, 9500], RE)
    assert cov.missing == ["transitional"] and cov.status == "fail"
    assert "transitional" in str(cov)


def test_a_regime_with_a_sample_or_two_is_thin_not_covered():
    cov = regime_coverage([1200, 1800, 2500, 5000], RE)    # each upper band has one sample: present, not learnable
    assert cov.counts == {"laminar": 2, "transitional": 1, "turbulent": 1}
    assert cov.thin == ["transitional", "turbulent"] and cov.covered == ["laminar"]
    assert cov.missing == [] and cov.status == "warn"


def test_the_empty_band_comes_back_as_the_interval_to_sample_in():
    cov = regime_coverage([1200, 9500], RE)
    assert cov.missing_regions() == [{"column": "reynolds", "low": 2300.0, "high": 4000.0, "label": "transitional"}]


def test_the_report_is_serialisable_and_readable():
    cov = regime_coverage([1200, 9500], RE)
    payload = cov.to_dict()
    assert payload["regime"] == "reynolds_pipe" and payload["status"] == "fail"
    assert payload["sample_here"][0]["label"] == "transitional"
    assert "**missing**" in cov.to_markdown()


def test_a_spec_with_a_hole_is_refused_instead_of_reporting_unknown():
    with pytest.raises(ValueError, match="gap or an overlap"):
        RegimeSpec("buraco", "x", (Band(None, 1.0, "a"), Band(2.0, None, "b")))
    with pytest.raises(ValueError, match="reach -inf"):
        RegimeSpec("comeca", "x", (Band(0.0, 1.0, "a"), Band(1.0, None, "b")))
    with pytest.raises(ValueError, match="reach \\+inf"):
        RegimeSpec("termina", "x", (Band(None, 1.0, "a"), Band(1.0, 2.0, "b")))
    with pytest.raises(ValueError, match="repeats a label"):
        RegimeSpec("repete", "x", (Band(None, 1.0, "a"), Band(1.0, None, "a")))


def test_the_presets_say_what_they_mean():
    assert PRESETS["mach"].label(0.1) == "incompressible"
    assert PRESETS["mach"].label(2.5) == "supersonic"
    assert PRESETS["reynolds_flat_plate"].label(1e6) == "transitional"
    assert PRESETS["rayleigh"].note and PRESETS["reynolds_pipe"].reference


def test_the_audit_of_a_dataset_says_which_regime_is_missing():
    card = audit_dataset({"train": {"reynolds": np.array([1000.0, 2000.0, 5000.0]),
                                    "u": np.array([0.1, 0.2, 0.3])}}, regimes=["reynolds_pipe"])
    check = card["regime.reynolds_pipe"]
    assert check.status == "fail"
    assert "transitional" in check.summary
    assert check.evidence["sample_here"][0]["column"] == "reynolds"
    assert card.decision == "fail"


def test_a_dataset_that_covers_every_regime_passes_that_check():
    values = np.array([1000.0, 1500.0, 2500.0, 3000.0, 9000.0, 12000.0])
    card = audit_dataset({"train": {"reynolds": values, "u": np.linspace(0, 1, len(values))}},
                         regimes=["reynolds_pipe"])
    assert card["regime.reynolds_pipe"].status == "pass"


def test_a_missing_column_is_reported_as_not_applicable_not_as_covered():
    card = audit_dataset({"train": {"u": np.array([0.1, 0.2])}}, regimes=["reynolds_pipe"])
    check = card["regime.reynolds_pipe"]
    assert check.status == "not-applicable" and check.passed is None
    assert "no 'reynolds' column" in check.summary


def test_a_spec_of_my_own_works_the_same_way():
    spec = RegimeSpec("temperature", "T", (Band(None, 300.0, "cold"), Band(300.0, None, "hot")))
    cov = regime_coverage([280, 290], spec)
    assert cov.missing == ["hot"]
    rows = regime_audit({"T": [280.0, 310.0]}, [spec])
    assert rows[0].spec.name == "temperature" and rows[0].missing == []
