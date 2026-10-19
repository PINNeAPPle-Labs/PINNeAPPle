"""The validation report as an artifact (#706): a versioned JSON schema, provenance hashes, the difference between two
reports (what a new model version lost), a section for a Hugging Face model card, a badge, and signatures.

``REPORT_SCHEMA``
    JSON schema of ``validation-report/v1`` (the output of ``pp.validate(...).to_dict()``); :func:`check_report`
    validates a report against it and against the check-ID vocabulary of :mod:`.schema`.
``hash_model`` / ``hash_data``
    SHA-256 provenance of the model (weights, ONNX file, zoo weights) and of the data the report used.
``diff(old, new)``
    Checks whose status changed (a check that passed and no longer does is a regression), metrics that moved beyond a
    relative tolerance, and changes in the validity envelope. :attr:`ReportDiff.regressed` is what a release gate
    blocks on (``scripts/zoo/validation_reports.py check``, run by the zoo publishing workflow).
``model_card_section`` / ``badge`` / ``shields_endpoint``
    A markdown block for a Hugging Face card; an SVG badge ("physics-validated: 9/9 checks") and the JSON of a
    shields.io endpoint badge.
``sign_report`` / ``verify_report``
    An Ed25519 (or HMAC) envelope of the report with ``pinneapple_security.signing`` (#644).
"""
from __future__ import annotations

import hashlib
import html
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import schema

__all__ = ["REPORT_SCHEMA", "check_report", "hash_model", "hash_data", "diff", "ReportDiff", "model_card_section",
           "badge", "shields_endpoint", "sign_report", "verify_report", "as_dict"]

SCHEMA_ID = "validation-report/v1"

_CHECK = {
    "type": "object",
    "required": ["id", "factor", "status", "summary", "critical"],
    "properties": {
        "id": {"type": "string", "pattern": r"^[a-z_]+(\.[^\s]+)*$"},
        "factor": {"enum": list(schema.FACTORS)},
        "status": {"enum": list(schema.STATUSES)},
        "summary": {"type": "string"},
        "value": {},
        "threshold": {},
        "critical": {"type": "boolean"},
        "evidence": {"type": "object"},
    },
    "additionalProperties": False,
}

REPORT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://pinneapple-labs.github.io/PINNeAPPle/schemas/validation-report-v1.json",
    "title": "PINNeAPPle validation report",
    "type": "object",
    "required": ["schema", "model", "problem", "level", "decision", "trust_decision", "verdict", "credibility",
                 "checks", "provenance"],
    "properties": {
        "schema": {"const": SCHEMA_ID},
        "model": {"type": "string"},
        "problem": {"type": "string"},
        "level": {"enum": ["quick", "standard", "thorough"]},
        "decision": {"enum": ["pass", "warn", "fail"]},
        "trust_decision": {"enum": ["APPROVED", "REVIEW", "REJECT"]},
        "verdict": {"type": "string"},
        "credibility": {
            "type": "object",
            "required": list(schema.FACTORS),
            "additionalProperties": {
                "type": "object", "required": ["level", "meaning", "checks"],
                "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 4},
                               "meaning": {"type": "string"}, "checks": {"type": "array", "items": {"type": "string"}}},
            },
        },
        "checks": {"type": "array", "items": _CHECK},
        "provenance": {
            "type": "object",
            "required": ["library_version", "level", "source", "model_hash", "data_hash", "physics"],
            "properties": {
                "library_version": {"type": "string"},
                "model_hash": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
                "data_hash": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
                "physics": {"type": "object"},
            },
        },
    },
}


def as_dict(report) -> dict[str, Any]:
    """A report object, a dict or a JSON file path -> the report dict."""
    if isinstance(report, Mapping):
        return dict(report)
    if isinstance(report, (str, Path)):
        return json.loads(Path(report).read_text())
    return report.to_dict()


def check_report(report) -> list[str]:
    """Problems of a report against :data:`REPORT_SCHEMA` and the check-ID vocabulary; empty when it is valid."""
    d = as_dict(report)
    errors: list[str] = []
    try:
        import jsonschema
        v = jsonschema.Draft202012Validator(REPORT_SCHEMA)
        errors += [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in v.iter_errors(d)]
    except ImportError:                                           # minimal checks without jsonschema
        for k in REPORT_SCHEMA["required"]:
            if k not in d:
                errors.append(f"<root>: '{k}' is a required property")
    for c in d.get("checks", []):
        try:
            spec = schema.spec_of(c.get("id", ""))
            if c.get("factor") != spec.factor:
                errors.append(f"checks/{c.get('id')}: factor {c.get('factor')!r}, the vocabulary says {spec.factor!r}")
        except KeyError as e:
            errors.append(f"checks/{c.get('id')}: {e.args[0]}")
    if d.get("decision") and schema.DECISIONS.get(d["decision"]) != d.get("trust_decision"):
        errors.append("trust_decision does not match decision")
    return errors


# ---------------------------------------------------------------------------------------------- provenance
def _sha(chunks) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def hash_model(model) -> str | None:
    """SHA-256 of a model's parameters (torch module, zoo model) or of its file (ONNX); None for a plain callable."""
    if isinstance(model, (str, Path)) and Path(model).is_file():
        return _sha([Path(model).read_bytes()])
    module = getattr(model, "module", None) if hasattr(model, "card") else model
    if hasattr(module, "state_dict"):
        sd = module.state_dict()
        parts = []
        for k in sorted(sd):
            t = sd[k].detach().cpu()
            parts += [k.encode(), str(tuple(t.shape)).encode(), str(t.dtype).encode()]
            parts.append(t.contiguous().view(-1).view(_byte_dtype()).numpy().tobytes() if t.numel() else b"")
        return _sha(parts)
    return None


def _byte_dtype():
    import torch
    return torch.uint8


def _array_bytes(x) -> list[bytes]:
    if x is None:
        return [b"none"]
    if isinstance(x, Mapping):
        out = []
        for k in sorted(x):
            out += [str(k).encode()] + _array_bytes(x[k])
        return out
    if isinstance(x, (list, tuple)) and x and not np.isscalar(x[0]):
        return [b for v in x for b in _array_bytes(v)]
    a = np.ascontiguousarray(np.asarray(x))
    if a.dtype.kind not in "fiub":
        return [repr(a.tolist()).encode()]
    return [str(a.shape).encode(), str(a.dtype).encode(), a.tobytes()]


def hash_data(*objs) -> str | None:
    """SHA-256 of the data a report used (arrays, dicts of arrays, nested); None when there was none."""
    if all(o is None for o in objs):
        return None
    return _sha([b for o in objs for b in _array_bytes(o)])


# ---------------------------------------------------------------------------------------------- diff
@dataclass
class ReportDiff:
    old: str
    new: str
    lost: list[dict[str, Any]] = field(default_factory=list)       # passed before, not now
    gained: list[dict[str, Any]] = field(default_factory=list)     # fail/warn before, pass now
    changed: list[dict[str, Any]] = field(default_factory=list)    # any other status change
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    moved: list[dict[str, Any]] = field(default_factory=list)      # metrics beyond the relative tolerance
    envelope: list[dict[str, Any]] = field(default_factory=list)   # validity range changes
    decision: tuple[str, str] = ("", "")

    @property
    def regressed(self) -> bool:
        """A check that passed no longer passes, a passing check disappeared, or the decision got worse."""
        order = {"pass": 0, "warn": 1, "fail": 2}
        worse = order.get(self.decision[1], 0) > order.get(self.decision[0], 0)
        return bool(self.lost or [r for r in self.removed if r.startswith("!")] or worse)

    def to_dict(self) -> dict[str, Any]:
        return {"old": self.old, "new": self.new, "regressed": self.regressed, "decision": list(self.decision),
                "lost": self.lost, "gained": self.gained, "changed": self.changed, "added": self.added,
                "removed": [r.lstrip("!") for r in self.removed], "moved": self.moved, "envelope": self.envelope}

    def to_markdown(self) -> str:
        L = [f"### Validation diff: {self.old} -> {self.new}", "",
             f"Decision {self.decision[0]} -> {self.decision[1]}; "
             + ("**regressed**" if self.regressed else "no regression"), ""]
        if self.lost:
            L += ["Lost (passed before):", ""] + [f"- `{c['id']}`: {c['old']} -> {c['new']} ({c['summary']})"
                                                  for c in self.lost] + [""]
        if self.gained:
            L += ["Gained:", ""] + [f"- `{c['id']}`: {c['old']} -> {c['new']}" for c in self.gained] + [""]
        if self.moved:
            L += ["Metrics moved:", ""] + [f"- `{m['id']}`: {m['old']:.4g} -> {m['new']:.4g} ({m['rel']:+.1%})"
                                           for m in self.moved] + [""]
        if self.envelope:
            L += ["Validity envelope:", ""] + [f"- {e['what']}: {e['old']} -> {e['new']}" for e in self.envelope]
        removed = [r.lstrip("!") for r in self.removed]
        if self.added or removed:
            L += ["", f"Checks added: {', '.join(self.added) or '-'}; removed: {', '.join(removed) or '-'}"]
        return "\n".join(L).rstrip() + "\n"


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _envelope(d: dict) -> dict[str, Any]:
    out = {}
    for c in d.get("checks", []):
        if c["id"].startswith("applicability"):
            rng = (c.get("evidence") or {}).get("ranges")
            if rng:
                out.update({f"{c['id']}:{k}": v for k, v in rng.items()})
    vr = (d.get("provenance") or {}).get("validity_range")
    if vr:
        out.update({f"validity_range:{k}": v for k, v in vr.items()})
    return out


def diff(old, new, *, rtol: float = 0.1) -> ReportDiff:
    """What changed between two validation reports (objects, dicts or JSON paths)."""
    a, b = as_dict(old), as_dict(new)
    ca = {c["id"]: c for c in a.get("checks", [])}
    cb = {c["id"]: c for c in b.get("checks", [])}
    rd = ReportDiff(a.get("model", "old"), b.get("model", "new"), decision=(a.get("decision", ""),
                                                                            b.get("decision", "")))
    for k in ca.keys() & cb.keys():
        so, sn = ca[k]["status"], cb[k]["status"]
        row = {"id": k, "old": so, "new": sn, "summary": cb[k].get("summary", "")}
        if so == sn:
            vo, vn = _num(ca[k].get("value")), _num(cb[k].get("value"))
            if vo is not None and vn is not None and abs(vn - vo) > rtol * max(abs(vo), 1e-12):
                rd.moved.append({"id": k, "old": vo, "new": vn, "rel": (vn - vo) / max(abs(vo), 1e-12)})
        elif so == schema.PASS:
            rd.lost.append(row)
        elif sn == schema.PASS:
            rd.gained.append(row)
        else:
            rd.changed.append(row)
    rd.added = sorted(cb.keys() - ca.keys())
    rd.removed = sorted(("!" if ca[k]["status"] == schema.PASS else "") + k for k in ca.keys() - cb.keys())
    ea, eb = _envelope(a), _envelope(b)
    for k in sorted(ea.keys() | eb.keys()):
        if ea.get(k) != eb.get(k):
            rd.envelope.append({"what": k, "old": ea.get(k), "new": eb.get(k)})
    return rd


# ---------------------------------------------------------------------------------------------- presentation
def _counts(d: dict) -> tuple[int, int]:
    ran = [c for c in d["checks"] if c["status"] != schema.NA]
    return sum(c["status"] == schema.PASS for c in ran), len(ran)


def model_card_section(report, *, title: str = "Physics validation") -> str:
    """A markdown block for a Hugging Face model card: verdict, credibility per factor, the checks."""
    d = as_dict(report)
    ok, n = _counts(d)
    prov = d.get("provenance", {})
    L = [f"## {title}", "",
         f"**{d['verdict']}** ({ok}/{n} applicable checks pass; level `{d['level']}`; generated by `pp.validate` "
         f"with PINNeAPPle {prov.get('library_version', '?')}, schema `{d['schema']}`).", "",
         "| Factor | Credibility | Meaning |", "|---|---|---|"]
    L += [f"| {f} | {v['level']} | {v['meaning']} |" for f, v in d["credibility"].items()]
    L += ["", "| Check | Status | Result |", "|---|---|---|"]
    L += [f"| `{c['id']}` | {c['status']} | {c['summary'].replace('|', '/')} |" for c in d["checks"]]
    if prov.get("model_hash"):
        L += ["", f"Model SHA-256 `{prov['model_hash']}`."]
    return "\n".join(L) + "\n"


_COLORS = {"pass": "#2e7d32", "warn": "#b26a00", "fail": "#c62828"}


def shields_endpoint(report, label: str = "physics-validated") -> dict[str, Any]:
    """JSON for a shields.io endpoint badge (https://shields.io/badges/endpoint-badge)."""
    d = as_dict(report)
    ok, n = _counts(d)
    color = {"pass": "brightgreen", "warn": "orange", "fail": "red"}[d["decision"]]
    return {"schemaVersion": 1, "label": label, "message": f"{ok}/{n} checks", "color": color}


def badge(report, label: str = "physics-validated") -> str:
    """A self-contained SVG badge, e.g. "physics-validated | 9/9 checks"."""
    d = as_dict(report)
    ok, n = _counts(d)
    msg = f"{ok}/{n} checks"
    w1, w2 = 7 * len(label) + 12, 7 * len(msg) + 12
    color = _COLORS[d["decision"]]
    lab, m = html.escape(label), html.escape(msg)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w1 + w2}" height="20" role="img" '
            f'aria-label="{lab}: {m}"><title>{lab}: {m}</title>'
            f'<rect width="{w1}" height="20" fill="#555"/><rect x="{w1}" width="{w2}" height="20" fill="{color}"/>'
            f'<g fill="#fff" text-anchor="middle" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">'
            f'<text x="{w1 / 2}" y="14">{lab}</text><text x="{w1 + w2 / 2}" y="14">{m}</text></g></svg>')


# ---------------------------------------------------------------------------------------------- signing (#644)
PAYLOAD_TYPE = "application/vnd.pinneapple.validation-report+json"


def sign_report(report, key) -> dict[str, Any]:
    """A signed envelope of the report (``pinneapple_security.signing.sign_json``)."""
    from pinneapple_security.signing import sign_json
    return sign_json(as_dict(report), key, payload_type=PAYLOAD_TYPE)


def verify_report(envelope: dict[str, Any], key) -> dict[str, Any]:
    """The report inside a signed envelope; raises when the signature does not verify or the envelope holds another
    kind of document (the type is part of the signed data, so it cannot be swapped)."""
    from pinneapple_security.signing import verify_json
    if envelope.get("payloadType") != PAYLOAD_TYPE:
        raise ValueError(f"not a validation report envelope (payloadType {envelope.get('payloadType')!r})")
    return verify_json(envelope, key)
