"""pinneapple_analysis.inverse_problems.heat_diffusivity (ported from the tests of PINNeAPPle-apps
poc_parameter_discovery; small runs). The full 2,500-step run needs PINNEAPPLE_SLOW=1."""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("torch")
from pinneapple_analysis.inverse_problems import heat_diffusivity as hd  # noqa: E402

SMALL = dict(n_observations=64, n_collocation=1000, n_condition=300)


def test_identifiability_follows_the_fisher_information():
    t = np.full(64, 0.5)
    x = np.linspace(0.05, 0.95, 64)
    lo, hi = hd.identifiability_report(t, x, 0.15, 0.01), hd.identifiability_report(t, x, 0.15, 0.10)
    assert lo["is_identifiable"] and hi["cramer_rao_standard_error"] == pytest.approx(
        10 * lo["cramer_rao_standard_error"], rel=1e-6)
    # closed form: dU/dalpha = -pi^2 t U, FIM = sum (dU/dalpha)^2 / sigma^2
    g = -(np.pi**2) * t * hd.u_exact(t, x, 0.15)
    assert lo["fisher_information"] == pytest.approx(float(np.sum(g.astype(float) ** 2) / 0.01**2), rel=1e-5)


def test_alpha_moves_towards_the_truth_from_a_wrong_guess():
    r = hd.run_inverse_discovery(true_alpha=0.15, initial_guess=0.5, n_epochs=400, seed=0, **SMALL)
    a = r["history"]["alpha_estimate"]
    assert abs(a[0] - 0.5) < 0.02
    assert abs(a[-1] - 0.15) < abs(0.5 - 0.15) * 0.6
    assert len(r["field_fit"]["x_grid"]) == 200 and len(r["field_fit"]["x_observed"]) == 64
    assert r["final"]["abs_error"] == abs(r["final"]["alpha_recovered"] - 0.15)


def test_seed_reproduces_and_none_is_fresh():
    a = hd.run_inverse_discovery(n_epochs=5, seed=7, **SMALL)
    b = hd.run_inverse_discovery(n_epochs=5, seed=7, **SMALL)
    assert a["final"]["alpha_recovered"] == b["final"]["alpha_recovered"]
    c = hd.run_inverse_discovery(n_epochs=1, **SMALL)
    d = hd.run_inverse_discovery(n_epochs=1, **SMALL)
    assert c["field_fit"]["u_observed_noisy"] != d["field_fit"]["u_observed_noisy"]


@pytest.mark.skipif(not os.environ.get("PINNEAPPLE_SLOW"), reason="full 2,500-step run (minutes); PINNEAPPLE_SLOW=1")
def test_full_run_recovers_alpha_within_five_percent():
    r = hd.run_inverse_discovery(seed=0)
    assert r["final"]["pct_error"] < 5.0 and r["final"]["rel_l2_field_error"] < 0.02
