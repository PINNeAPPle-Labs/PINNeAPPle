"""Integration tests for the reference-blind optimizer's curse study."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from pinneapple_veriphysics.design_harness import DesignVerificationHarness
from pinneapple_veriphysics.ledger import EvidenceLedger


def test_screening_does_not_call_reference():
    calls = []
    def reference(x):
        calls.append(tuple(x))
        return 2.0
    h = DesignVerificationHarness([[0, 1]], [[0.5]], 1.0, threshold=0.1)
    d = h.check([0.5], 1.4, 0.01, check_id="screen")
    assert d.status == "provisional"
    assert calls == []
    d = h.check([0.5], 1.4, 0.01, reference=reference,
                reference_baseline=1.0, check_id="reference")
    assert d.status == "verified"
    assert len(calls) == 1


def test_sealed_evidence_detects_tampering():
    h = DesignVerificationHarness([[0, 1]], [[0.5]], 1.0, threshold=0.1)
    h.check([0.5], 1.4, 0.01, check_id="screen")
    payload = h.ledger.to_dict()
    assert EvidenceLedger.from_dict(payload).latest("screen").valid
    payload["entries"][0]["outputs"]["status"] = "verified"
    tampered = EvidenceLedger.from_dict(payload)
    assert tampered.invalid()
    assert tampered.status("screen") == "not run"


def test_benchmark_cli_smoke(tmp_path):
    runner = Path(__file__).resolve().parents[1] / "examples" / "optimizer_curse" / "run.py"
    subprocess.run([sys.executable, str(runner), "--seeds", "2",
                    "--budget", "32", "--train", "16",
                    "--out", str(tmp_path)], check=True)
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["experiment"] == "optimizer_curse_synthetic_pilot"
    assert len(result["runs"]) == 2
    for row in result["runs"]:
        assert row["reference_calls_selection"] == 0
        assert row["reference_calls_post_selection"] == (
            len(row["baseline_indices"]) + len(row["screened_indices"])
        )
        assert row["reference"] == "manufactured_synthetic_not_CFD"
    assert (tmp_path / "confirmation.png").exists()
    assert (tmp_path / "REPORT.md").exists()
