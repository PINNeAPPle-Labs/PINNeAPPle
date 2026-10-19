"""Lab curation evidence (the headline line, the verification kind) and the repo_results sources added for the
electric machines, electromagnetics, aero, rover, forecasting and design-app results."""
import json
import os

import pytest

from pinneapple_lab import Experiment, LabStore, register, run
from pinneapple_lab.curation import check_line, curate, headline


@register
class _SecondCodeOnly(Experiment):
    name = "_test_second_code_only"
    description = "A result whose only comparison is a second code that shares its inputs."
    params = {"agree": True}

    def run(self, ctx):
        ctx.check("torque_vs_second_code", bool(ctx.params["agree"]), value=0.001, kind="verification")


@register
class _Referenced(Experiment):
    name = "_test_referenced"
    description = "A result compared with an exact solution and checked against a conservation law, with figures."
    params = {"tier_hint": "A"}

    def run(self, ctx):
        ctx.check("deflection_vs_beam_theory", value=1.001, reference=1.0, rtol=0.01, kind="reference",
                  label="deflection vs Timoshenko")
        ctx.check("energy_balance", True, kind="physics")


# ---------------------------------------------------------------------------------------------- curation evidence
def test_headline_comes_from_passing_checks_only():
    checks = [{"name": "drag_vs_wind_tunnel", "passed": False, "kind": "reference", "detail": "Ahmed 1984: 0.285"},
              {"name": "mass_balance", "passed": True, "kind": "physics", "detail": "in = out"}]
    h = headline(checks, "B")
    assert h["kind"] == "benchmark" and h["text"].startswith("mass balance") and h["missing"] == "independent reference"
    checks[0]["passed"] = True
    h = headline(checks, "A")
    assert h["kind"] == "reference" and h["text"] == "drag vs wind tunnel · Ahmed 1984"
    assert check_line({"name": "x", "label": "drag vs Ahmed 1984"}) == "drag vs Ahmed 1984"
    assert check_line({"name": "harmonic stack: closed form", "detail": "tolerance 1e-6"}) == "harmonic stack: closed form"
    assert headline([], "C")["text"] == ""


def test_second_code_verification_never_lifts_a_tier(tmp_path):
    root = str(tmp_path)
    assert run("_test_second_code_only", {}, root=root).ok
    assert run("_test_referenced", {}, root=root).ok
    out = curate(LabStore(root))
    weak = out["experiments"]["_test_second_code_only"]
    assert weak["tier"] == "C"                                 # agreeing with a code that shares the inputs is not B
    assert weak["stages"]["model"]["status"] == "answered"     # ... but it does verify the solver
    assert weak["headline"]["kind"] == "verification"          # printed as what it is, never as a reference
    good = out["items"]["_test_referenced"]
    assert good["headline"] == {"kind": "reference", "text": "deflection vs Timoshenko",
                                "check": "deflection_vs_beam_theory", "missing": "flagship gates (see gaps)",
                                "label": "deflection vs Timoshenko"}      # the label recorded on the check


# ---------------------------------------------------------------------------------------------- new repo sources
@pytest.mark.parametrize("source", ["prius_pmsm", "prius_end_effects", "prius_thermal_demag", "prius_drive", "em_verification", "motor_spm",
                                    "airfoil_handoff", "airliner_cfd", "lunar_rover", "near_random_forecasting",
                                    "design_from_intent", "topology_optimization", "free_surface_trough",
                                    "weather_era5"])
def test_repo_sources_import(tmp_path, source):
    r = run("repo_results", {"source": source}, root=str(tmp_path))
    assert r.status == "completed", (source, r.status, r.error)
    checks = json.load(open(os.path.join(r.dir, "validation.json")))
    assert all(c["kind"] in ("reference", "verification", "physics", "baseline", "generalization", "sanity")
               for c in checks)
    if source == "prius_pmsm":
        ref = [c for c in checks if c["kind"] == "reference"]
        assert [c["name"] for c in ref] == ["peak_torque_vs_yang_2013"]       # GetDP is verification, not reference
        assert any(c["name"] == "torque_vs_getdp_second_code" and c["kind"] == "verification" for c in checks)
    if source == "airliner_cfd":
        assert not any(c["kind"] == "reference" for c in checks)              # no published reference: stays B
    if source == "weather_era5":
        assert checks == []                                                    # scores not committed: nothing checked


def test_benchmark_arena_records_its_failed_audit(tmp_path):
    r = run("repo_results", {"source": "benchmark_arena"}, root=str(tmp_path))
    assert r.status == "failed_validation"                                     # the dataset audit fails: reported
    assert r.metrics["interpolation_mlp_rel_l2"] < r.metrics["interpolation_pinn_rel_l2"]


def test_check_labels_only_when_every_claimed_check_passes():
    checks = [{"name": "settling_vs_schiller_naumann", "passed": True, "kind": "reference", "detail": ""},
              {"name": "speed_vs_zwietering", "passed": True, "kind": "reference", "detail": ""}]
    labels = {"settling_vs_schiller_naumann": {"en": "vs Zwietering and Schiller-Naumann",
                                               "pt": "vs Zwietering e Schiller-Naumann",
                                               "also": ["speed_vs_zwietering"]}}
    h = headline(checks, "A", labels)
    assert h["label"] == "vs Zwietering and Schiller-Naumann" and h["label_pt"] == "vs Zwietering e Schiller-Naumann"
    checks[1]["passed"] = False                               # the line would claim a check that failed
    h = headline(checks, "B", labels)
    assert "label" not in h and h["text"].startswith("settling vs schiller naumann")
