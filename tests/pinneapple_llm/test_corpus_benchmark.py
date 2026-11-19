"""Offline mixed-corpus and benchmark tests."""
import hashlib
import json

import pytest

from pinneapple_llm.code_training import CodeDatasetConfig, generate_code_dataset
from pinneapple_llm.corpus import assemble_training_corpus
from pinneapple_llm.benchmark import evaluate_predictions


def test_assemble_preserves_provenance_and_checksums(tmp_path):
    pkg = tmp_path / "demo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "api.py").write_text(
        'def solve(x):\n    """Return the value of x without changing its units."""\n    return x\n'
    )
    source = tmp_path / "source"
    generate_code_dataset(CodeDatasetConfig(
        root=str(tmp_path), output_dir=str(source),
        packages=("demo",), validation_fraction=0.0
    ))
    evidence = tmp_path / "reviewed.jsonl"
    evidence.write_text(json.dumps({
        "prompt": "What did the solver report?", "response": "residual 0.01",
        "provenance": {
            "reviewed": True, "source_revision": "abc",
            "evidence_path": "run.json", "evidence_sha256": "0" * 64,
        },
    }) + "\n")
    target = tmp_path / "mixed"
    result = assemble_training_corpus(str(source), str(evidence), str(target),
                                      validation_fraction=0.0)
    assert result["counts"] == {"train": 2, "validation": 0}
    assert result["counts_by_origin"]["train:reviewed_execution"] == 1
    assert hashlib.sha256((target / "train.jsonl").read_bytes()).hexdigest() == result["sha256"]["train"]
    assert assemble_training_corpus(str(source), str(evidence), str(target),
                                    validation_fraction=0.0)["sha256"] == result["sha256"]
    (source / "train.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        assemble_training_corpus(str(source), str(evidence), str(target))


def test_benchmark_reports_proxy_scores(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    tasks.write_text(json.dumps({
        "id": "task-1", "prompt": "Use PINNeAPPle", "reference": "call pp.solve",
        "required_api": ["pp.solve", "pp.verify"]
    }) + "\n")
    predictions.write_text(json.dumps({
        "id": "task-1", "response": "call pp.solve"
    }) + "\n")
    result = evaluate_predictions(str(tasks), str(predictions), str(tmp_path / "report.json"))
    assert result["exact_match_rate"] == 1
    assert result["required_api_coverage"] == 0.5
    assert (tmp_path / "report.json").is_file()
