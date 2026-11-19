from pinneapple_analysis.verification.pipeline_assurance import Artifact, Check, Status, assess, conservation, metamorphic, diagnose
import pytest


def test_conservation_and_failure():
    assert conservation("mass", "solver", 10, 10).status == Status.PASS
    assert conservation("mass", "solver", 10, 9).status == Status.FAIL


def test_metamorphic_assumptions_and_violation():
    assert metamorphic("shift", "training", [1], [2], [2], assumptions=()).status == Status.INCONCLUSIVE
    assert metamorphic("shift", "training", [1], [3], [2], assumptions=("linear constant-coefficient diffusion",)).status == Status.FAIL


def test_lineage_and_no_fabricated_confidence():
    a = [Artifact("cad", "geometry"), Artifact("mesh", "mesh", ("cad",))]
    c = [Check("volume", "geometry", Status.PASS), Check("convergence", "mesh", Status.NOT_RUN)]
    r = assess(a, c)
    assert r["verdict"] == "inconclusive"
    assert r["coverage"] == 0.5
    assert diagnose(["scale_factor"], c)["auto_repair_authorized"] is False


def test_invalid_lineage():
    with pytest.raises(ValueError):
        assess([Artifact("a", "geometry", ("b",))], [])
