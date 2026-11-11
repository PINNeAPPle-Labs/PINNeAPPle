"""Evidence import acceptance and tampering regression tests."""
import hashlib
import json

from pinneapple_llm.verified_examples import import_verified_examples


def test_reviewed_evidence_is_imported(tmp_path):
    evidence = tmp_path / "solver-output.json"
    evidence.write_text('{"residual": 0.01}')
    row = {
        "prompt": "Solve the verified reference problem",
        "response": "The measured residual was 0.01",
        "command": "python example.py",
        "exit_code": 0,
        "evidence_path": evidence.name,
        "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "reviewed": True,
        "physics_checks": ["residual"],
        "source_revision": "test-sha",
    }
    inp = tmp_path / "input.jsonl"
    inp.write_text(json.dumps(row) + "\n")
    output = tmp_path / "training.jsonl"
    report = import_verified_examples(str(inp), str(output), root=str(tmp_path))
    assert report["accepted"] == 1
    assert report["rejected"] == 0
    assert json.loads(output.read_text())["provenance"]["evidence_sha256"] == row["evidence_sha256"]


def test_tampered_and_unreviewed_evidence_are_rejected(tmp_path):
    evidence = tmp_path / "output.txt"
    evidence.write_text("actual output")
    record = {
        "prompt": "p", "response": "r", "command": "python example.py",
        "exit_code": 0, "evidence_path": "output.txt",
        "evidence_sha256": "0" * 64, "reviewed": True,
        "physics_checks": ["conservation"], "source_revision": "sha",
    }
    inp = tmp_path / "input.jsonl"
    inp.write_text(json.dumps(record) + "\n")
    output = tmp_path / "training.jsonl"
    report = import_verified_examples(str(inp), str(output), root=str(tmp_path))
    assert report["accepted"] == 0
    assert "checksum mismatch" in report["rejections"][0]["reason"]
    record["evidence_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    record["reviewed"] = False
    inp.write_text(json.dumps(record) + "\n")
    report = import_verified_examples(str(inp), str(output), root=str(tmp_path))
    assert report["accepted"] == 0
    assert "reviewed" in report["rejections"][0]["reason"]
