"""Small thermal surrogate demonstration; requires the trust PR series."""
import hashlib
import json

import numpy as np
import torch

from pinneapple_veriphysics.trust_policy import CheckEvidence, TrustPolicy
from pinneapple_analysis.uncertainty.interval_admission import assess_intervals
from pinneapple_veriphysics.fault_campaign import FaultCase, run_campaign
from pinneapple_veriphysics.attestation import Attestation, sign, permits


def run(fault="none"):
    # Steady conduction through a unit slab, T(0)=0, T(1)=q.
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 200)
    q = rng.uniform(1, 2, 200)
    y = x * q
    if fault == "dataset":
        y[0] = np.nan
    if not np.isfinite(y).all():
        return False
    # Linear surrogate in the physics-inspired feature x*q.
    feature = x * q
    coefficient = float(np.linalg.lstsq(feature[:, None], y, rcond=None)[0][0])
    digest = hashlib.sha256(np.float64(coefficient).tobytes()).hexdigest()
    xt = rng.uniform(0, 1, 200)
    qt = rng.uniform(1, 2, 200)
    truth = xt * qt
    prediction = coefficient * truth
    if fault == "model":
        prediction += 0.2
    physical = bool(np.max(np.abs(prediction - truth)) <= 1e-6)
    width = 0.01 if fault != "uncertainty" else 10.0
    uq = assess_intervals(truth, prediction-width, prediction+width,
                          min_coverage=0.9, max_mean_width=0.1, min_samples=100)
    policy = TrustPolicy("thermal-v1", ("physics", "uq"))
    evidence = [
        CheckEvidence("physics", "thermal-physics", digest, "pass" if physical else "fail"),
        CheckEvidence("uq", "thermal-uq", digest, "pass" if uq.approved else "fail"),
    ]
    decision = policy.evaluate(artifact_hash=digest, scope_of_use="thermal-demo",
                               evidence=evidence)
    if not decision.permits(digest, "thermal-demo", "thermal-v1"):
        return False
    evidence_hash = hashlib.sha256(repr(evidence).encode()).hexdigest()
    attestation = Attestation("demo", digest, "thermal-demo", "thermal-v1",
                              evidence_hash, "demo-only", 100, 200)
    # Fixed key and clock are ONLY for a deterministic local demonstration.
    key = b"d" * 32
    return permits(attestation, sign(attestation, key), key=key,
                   artifact_hash=digest, scope="thermal-demo",
                   policy_version="thermal-v1", evidence_hash=evidence_hash,
                   issuer="demo-only", now=150,
                   revoked_ids=frozenset({"demo"}) if fault == "revoked" else frozenset())


def main():
    report = run_campaign([
        FaultCase("thermal", name, name == "none", lambda name=name: run(name))
        for name in ("none", "dataset", "model", "uncertainty", "revoked")
    ])
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
