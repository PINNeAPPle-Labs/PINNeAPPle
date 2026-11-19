"""Offline generation benchmark runner for PINNeAPPle API tasks.

The benchmark scores exact references and API token presence. These are
lightweight proxy metrics, NOT evidence of executable or physical correctness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def evaluate_predictions(tasks_jsonl: str, predictions_jsonl: str,
                         report_path: str) -> dict:
    """Score precomputed model predictions without loading or calling an LLM.

    Tasks: {"id": str, "prompt": str, "reference": str,
            "required_api": [str, ...]}
    Predictions: {"id": str, "response": str}
    """
    def read(path):
        result = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("id"), str) or row["id"] in result:
                raise ValueError("Missing or duplicate benchmark task ID")
            result[row["id"]] = row
        return result

    tasks = read(tasks_jsonl)
    predictions = read(predictions_jsonl)
    if not tasks or set(tasks) != set(predictions):
        raise ValueError("Benchmark requires nonempty identical task/prediction IDs")
    results = []
    for key, task in sorted(tasks.items()):
        response = predictions[key].get("response")
        reference = task.get("reference")
        api = task.get("required_api", [])
        if not isinstance(response, str) or not isinstance(reference, str):
            raise ValueError("Missing response or reference")
        if not isinstance(api, list) or any(not isinstance(x, str) or not x for x in api):
            raise ValueError("Invalid required_api list")
        exact = response.strip() == reference.strip()
        coverage = sum(token in response for token in api) / len(api) if api else None
        results.append({
            "id": key, "exact_match": exact,
            "required_api_coverage": coverage,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        })
    coverage_scores = [row["required_api_coverage"] for row in results
                       if row["required_api_coverage"] is not None]
    report = {
        "task_count": len(results),
        "exact_match_rate": sum(r["exact_match"] for r in results) / len(results),
        "required_api_coverage": sum(coverage_scores) / len(coverage_scores)
        if coverage_scores else None,
        "results": results,
        "limitations": [
            "Exact-match and API-coverage are proxy metrics only",
            "No execution, physical residual, or conservation checks are performed",
        ],
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report
