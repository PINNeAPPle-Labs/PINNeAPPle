from pinneapple_veriphysics.scientific_intent import IntentSpec
from pinneapple_veriphysics.surrogate_gates import SurrogateGateReport, SurrogateGateResult

def test_intent():
    s = IntentSpec("orbit", ("r''=-mu*r/|r|^3",), ("two bodies",), ("SI",), ("energy", "accuracy"))
    r = SurrogateGateReport(results=[SurrogateGateResult("energy", True)])
    assert s.evaluate(r)["status"] == "unknown"
    r.results.append(SurrogateGateResult("accuracy", False))
    assert s.evaluate(r)["status"] == "failed"
    r.results[-1] = SurrogateGateResult("accuracy", True)
    assert s.evaluate(r)["status"] == "passed"
