"""Ingest externally executed PINNeAPPle tasks as evidence-backed SFT records.

Never runs untrusted commands or accepts a success claim without an evidence
file whose content hash is checked. Human review is still required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def import_verified_examples(evidence_jsonl: str, output_jsonl: str, *,
                             root: str = ".") -> dict:
    """Accept only explicitly reviewed, successful, hash-matched runs.

    Each input line must include:
      prompt, response, command, exit_code, evidence_path, evidence_sha256,
      reviewed, physics_checks (nonempty list), source_revision.
    The evidence file must be inside root. This does not independently
    establish the correctness of the claimed physical checks.
    """
    base = Path(root).resolve()
    source = Path(evidence_jsonl).resolve()
    output = Path(output_jsonl).resolve()
    if source == output:
        raise ValueError("Input and output paths must differ")
    accepted = []
    rejected = []
    seen = set()
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            required = (
                "prompt", "response", "command", "exit_code", "evidence_path",
                "evidence_sha256", "reviewed", "physics_checks", "source_revision",
            )
            if not isinstance(record, dict) or any(key not in record for key in required):
                raise ValueError("missing required fields")
            if record["exit_code"] != 0 or record["reviewed"] is not True:
                raise ValueError("not a reviewed successful execution")
            if not all(isinstance(record[key], str) and record[key].strip()
                       for key in ("prompt", "response", "command", "source_revision")):
                raise ValueError("invalid text fields")
            if not isinstance(record["physics_checks"], list) or not record["physics_checks"]:
                raise ValueError("missing physics check evidence references")
            if not all(isinstance(check, str) and check.strip()
                       for check in record["physics_checks"]):
                raise ValueError("invalid physics check references")
            evidence = (base / record["evidence_path"]).resolve()
            if not evidence.is_relative_to(base) or not evidence.is_file() or evidence.is_symlink():
                raise ValueError("evidence file is missing or outside root")
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            if digest != record["evidence_sha256"]:
                raise ValueError("evidence checksum mismatch")
            key = hashlib.sha256(
                (record["prompt"] + "\0" + record["response"]).encode()
            ).hexdigest()
            if key in seen:
                raise ValueError("duplicate prompt/response")
            seen.add(key)
            accepted.append({
                "id": key, "prompt": record["prompt"], "response": record["response"],
                "provenance": {
                    "command": record["command"],
                    "source_revision": record["source_revision"],
                    "evidence_path": evidence.relative_to(base).as_posix(),
                    "evidence_sha256": digest,
                    "physics_checks": record["physics_checks"],
                    "reviewed": True,
                }
            })
        except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            rejected.append({"line": number, "reason": str(exc)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for item in accepted:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "accepted": len(accepted), "rejected": len(rejected),
        "rejections": rejected,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "warning": "Checksums and human review are not independent scientific verification.",
    }
    output.with_suffix(output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
