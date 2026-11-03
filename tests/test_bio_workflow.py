"""The standard bio workflow (#1205): closed forms (SIR final size, PK AUC, HH threshold and Hopf onset, FHN Hopf,
Michaelis-Menten QSSA), the SBML round trip, identification, and designs re-verified under uncertainty."""
import math

import numpy as np
import pytest

from pinneapple_physics.biology import read_sbml
from pinneapple_physics.biology.standard_models import (
    fitzhugh_nagumo,
    hodgkin_huxley,
    pk_two_compartment,
    sir,
)
from pinneapple_physics.biology.workflow import (
    bio_workflow,
    fhn_hopf_current,
    sir_final_size,
    wilks_order,
)

QUICK = {"n_sobol": 0, "identify": [], "uncertainty": {}}


def _status(r, code):
    return next(c["status"] for c in r["checks"] if c["code"] == code)


def test_sir_final_size_conservation_and_robust_intervention():
    r = bio_workflow(model="sir", n_sobol=8, identify=[])
    assert r["verdict"] == "pass", r["failed"]
    assert r["closed_form"]["error_over_N"] < 1e-6
    assert _status(r, "model.conservation") == "pass"
    d = r["design"]
    assert d["wilks_k"] == 32 and 0.4 < d["u"] < 0.9                    # the max of 32 draws bounds the 90 % quantile
    assert d["peak_nominal"] <= 0.05 < d["peak_without"]
    assert d["reverify_probability"] >= 0.8


def test_sir_final_size_root_matches_the_relation():
    N, S0, R0 = 1.0e6, 1.0e6 - 10, 3.0
    s = sir_final_size(N, S0, 0.0, R0)
    assert abs(math.log(S0 / s) - R0 * (N - s) / N) < 1e-10
    assert sir_final_size(N, S0, 0.0, 0.9) == S0                          # below threshold: no epidemic


def test_identification_recovers_the_generating_parameters():
    r = bio_workflow(model="sir", n_sobol=0, uncertainty={})
    i = r["identification"]
    assert i["estimated"] and all(i["truth_inside"].values())
    assert _status(r, "identify.recovery") == "pass"


def test_seir_with_three_free_parameters_names_the_combination():
    r = bio_workflow(model="seir", n_sobol=0, uncertainty={}, identify=["beta", "sigma", "gamma"])
    amb = r["identification"]["identifiability"]["ambiguities"]
    assert amb and r["identification"]["identifiability"]["condition_number"] > 1e4   # sigma and beta trade off


def test_pk_auc_closed_form_mass_balance_and_dosing_regimen():
    r = bio_workflow(model="pk_two_compartment", n_sobol=0, identify=[])
    assert r["verdict"] == "pass", r["failed"]
    cf = r["closed_form"]
    assert abs(cf["AUC_simulated"] - cf["AUC_closed_form"]) / cf["AUC_closed_form"] < 1e-6
    assert _status(r, "model.conservation") == "pass"
    d = r["design"]
    assert d["superposition_error"] < 1e-3
    assert d["interval_h"] in (4.0, 6.0, 8.0, 12.0, 24.0)
    assert d["reverify_trough_p05"] > 0.9 and d["reverify_peak_p95"] < 10.0


def test_hh_threshold_is_all_or_none_and_hopf_matches_rinzel_miller():
    r = bio_workflow(model="hodgkin_huxley", **QUICK)
    assert r["verdict"] == "pass", r["failed"]
    cf = r["closed_form"]
    assert abs(cf["hopf_current"] - 9.78) / 9.78 < 0.01
    assert 6.5 < cf["pulse_threshold"] < 7.5                               # 1 ms pulse, uA/cm^2
    below, above = cf["V_peak_below_above"]
    assert below < -40 and above > 0
    assert _status(r, "model.bounds") == "pass" and _status(r, "sbml.round_trip") == "pass"


def test_fhn_hopf_from_eigenvalues_equals_the_closed_form():
    r = bio_workflow(model="fitzhugh_nagumo", **QUICK)
    assert r["verdict"] == "pass", r["failed"]
    assert abs(r["closed_form"]["hopf_current"] - fhn_hopf_current(0.7, 0.8, 0.08)) < 1e-6


def test_michaelis_menten_invariants_and_qssa():
    r = bio_workflow(model="michaelis_menten", **QUICK)
    assert r["verdict"] == "pass", r["failed"]
    assert "2 linear invariant" in next(c["detail"] for c in r["checks"] if c["code"] == "model.conservation")
    assert r["closed_form"]["max_error_over_S0"] < 5 * r["closed_form"]["epsilon"]


@pytest.mark.parametrize("mid", ["12", "28"])
def test_sbml_biomodels_round_trip_and_reference(mid):
    r = bio_workflow(model="sbml", sbml=mid, **QUICK)
    assert _status(r, "sbml.round_trip") == "pass" and _status(r, "sbml.reference") == "pass"
    assert r["verdict"] == "pass", r["failed"]


def test_sbml_writes_steps_as_piecewise_and_reads_them_back():
    for s in (fitzhugh_nagumo(), hodgkin_huxley(), pk_two_compartment(interval=12, n_doses=3),
              sir(u=0.5, t_on=10, t_off=50)):
        s2 = read_sbml(s.to_sbml())
        assert s2.breakpoints() == s.breakpoints()
        t = np.linspace(0, 60, 61)
        a, b = s.simulate(t, outputs=[]), s2.simulate(t, outputs=[])
        for x in s.states:
            np.testing.assert_allclose(a[x], b[x], rtol=1e-10, atol=1e-12)


def test_wilks_order_statistic():
    assert wilks_order(29, 0.9) == 29 and wilks_order(28, 0.9) == 29     # 28 draws are too few
    assert wilks_order(100, 0.9) < 100


def test_too_few_design_draws_are_refused():
    r = bio_workflow(model="hodgkin_huxley", n_sobol=0, identify=[], n_design=10)
    assert _status(r, "design.feasible") == "fail"
    assert "at least 29" in next(c["detail"] for c in r["checks"] if c["code"] == "design.feasible")


def test_the_planner_proposes_the_workflow_in_portuguese_and_english():
    import pinneapple_flow.nodes_ext  # noqa: F401
    from pinneapple_flow.planner.extensions import match
    for text in ("regime de dose com farmacocinética e farmacodinâmica",
                 "SIR epidemic intervention under uncertainty", "limiar do potencial de ação de Hodgkin-Huxley"):
        assert match(text)[0][1].id == "bio.workflow"
