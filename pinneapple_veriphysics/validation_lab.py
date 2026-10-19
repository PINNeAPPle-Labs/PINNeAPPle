"""Validation Lab: run the physics checks of ``pinneapple check`` on predictions, results or a zoo model and
report them in three separate groups, with an evidence record for every check that ran.

=========================  ==============================================================================
``verification``           Numerical verification: were the equations solved right? PDE residuals, finite
                           values, readable and intact artefacts, method status in the catalog.
``validation``             Validation against data: does the answer match a trusted reference, and is the
                           model used where data backs it? Reference regression, QoIs, applicability (OOD).
``physics``                Physical consistency: conservation, boundary and initial conditions, bounds and
                           physical constraints, dimensional consistency (units), dataset physics.
=========================  ==============================================================================

No single test, nor all of them together, guarantees that a model represents reality: the report says so
(:data:`DISCLAIMER`) and lists, per target, the groups in which nothing ran.

::

    from pinneapple_veriphysics.validation_lab import config_from_spec, run_validation

    toml = config_from_spec({"results": [{"path": "T.npz", "physics": "heat", "units": {"T": "K"}}]})
    (work / "pinneapple-check.toml").write_text(toml)
    report = run_validation(work)
    report.to_dict()                 # JSON: categories, outcomes, provenance, evidence ledger
    render_html(report.to_dict())    # self-contained HTML

The configuration written to the report reproduces the run: ``pinneapple check --config pinneapple-check.toml``.
"""
from __future__ import annotations

import html
import json
import os
import platform
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ledger import EvidenceLedger, sha256_bytes, sha256_file, sha256_json, utc_now

__all__ = ["write_inputs", "tables_to_csv", "CATEGORIES", "DISCLAIMER", "LAB_SCHEMA", "category_of", "to_toml", "config_from_spec",
           "model_ref_target", "assert_no_code", "CodeNotAllowed", "run_validation", "LabReport", "render_html",
           "summarize"]

LAB_SCHEMA = "pinneapple-validation-lab/v1"
CONFIG_NAME = "pinneapple-check.toml"

CATEGORIES: dict[str, dict[str, str]] = {
    "verification": {"title": "Numerical verification",
                     "question": "Were the equations solved right? (residuals, finite values, intact artefacts)"},
    "validation": {"title": "Validation against data",
                   "question": "Does it match trusted reference data, inside the range the data covers?"},
    "physics": {"title": "Physical consistency",
                "question": "Does it conserve what must be conserved, meet its boundary conditions, constraints "
                            "and units?"},
}

DISCLAIMER = ("No single test, nor this set of tests, guarantees that the model represents reality. A pass means "
              "only that the checks listed here ran on these inputs, with these tolerances, and held. Checks that "
              "did not run are listed as not run; they are not passes.")

_BY_RULE = {"PC001": "verification", "PC002": None, "PC403": "verification", "PC901": "verification",
            "PC401": "physics", "PC402": "physics", "PC451": "physics", "PC452": "validation",
            "PC453": "validation"}
_BY_FAMILY = {"1": "physics", "2": "physics", "3": "verification", "5": "physics", "6": "validation",
              "7": "validation", "8": "verification"}


def category_of(rule_id: str) -> str | None:
    """Group of a ``pinneapple check`` rule (None for PC002, which is about coverage, not a check)."""
    if rule_id in _BY_RULE:
        return _BY_RULE[rule_id]
    return _BY_FAMILY.get(rule_id[2:3])


# ---------------------------------------------------------------------------------------------- configuration
_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k: str) -> str:
    return k if _BARE.match(k) else json.dumps(k)


def _val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return "nan" if v != v else ("inf" if v == float("inf") else "-inf" if v == float("-inf") else repr(v))
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(str(k))} = {_val(x)}" for k, x in v.items()) + " }" if v else "{}"
    if v is None:
        raise ValueError("TOML has no null: leave the key out")
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def to_toml(doc: dict[str, Any]) -> str:
    """Write a ``pinneapple-check.toml``: ``check`` as a table, ``result``/``model``/``dataset`` as arrays of
    tables, nested values as inline tables. Only what the configuration needs (no dates, no nulls)."""
    out: list[str] = []
    if doc.get("check"):
        out.append("[check]")
        out += [f"{_key(k)} = {_val(v)}" for k, v in doc["check"].items() if v is not None]
        out.append("")
    for sec in ("result", "model", "dataset"):
        for t in doc.get(sec, []) or []:
            out.append(f"[[{sec}]]")
            out += [f"{_key(k)} = {_val(v)}" for k, v in t.items() if v is not None and v != {} and v != []]
            out.append("")
    if doc.get("integrity"):
        out.append("[integrity]")
        out += [f"{_key(k)} = {_val(v)}" for k, v in doc["integrity"].items() if v is not None]
        out.append("")
    return "\n".join(out)


def model_ref_target(ref: dict[str, Any]) -> dict[str, Any]:
    """A ``[[model]]`` table from a model reference: ``{"zoo": name}``, a ModelRef with ``uri = "zoo:<name>"``, or
    one with ``source == "zoo"`` and ``name``. ``inputs``/``outputs`` may be names or ``{"name": ...}`` variables;
    ``train_range``, ``queries``, ``bounds``, ``units``, ``tolerances`` and ``sha256`` are passed on."""
    uri = str(ref.get("uri") or "")
    name = ref.get("zoo") or (uri[4:] if uri.startswith("zoo:") else None) or \
        (ref.get("name") if str(ref.get("source", "")).lower() in ("zoo", "pp.zoo") else None)
    if not name:
        raise ValueError("only zoo models can be referenced (give {'zoo': name} or uri 'zoo:<name>'); uploaded code "
                         "is not run")
    t: dict[str, Any] = {"name": ref.get("label") or str(name), "zoo": str(name)}
    for k in ("inputs", "outputs"):
        if ref.get(k):
            t[k] = [v.get("name") if isinstance(v, dict) else str(v) for v in ref[k]]
    for k in ("queries", "physics", "units", "bounds", "tolerances"):
        if ref.get(k):
            t[k] = ref[k]
    if ref.get("train_range") or ref.get("train-range"):
        t["train-range"] = ref.get("train_range") or ref.get("train-range")
    if ref.get("sha256"):
        t["sha256"] = ref["sha256"]
    return t


def config_from_spec(spec: dict[str, Any]) -> str:
    """``pinneapple-check.toml`` from a plain description of what to check::

        {"results": [{"path": "pred.npz", "physics": "heat", "parameters": {"k": "10 W/(m*K)"},
                      "units": {"T": "K"}, "bounds": {"T": [250, 400]}, "bc": {"x=max": {"T": 300}},
                      "reference": {"path": "ref.npz", "rel-l2": 0.01}, "tolerances": {"residual": 0.05}}],
         "models": [{"zoo": "heatsink_forced_mlp"}],
         "select": ["ALL"], "fail_on": "error"}

    Keys of each result are those of ``[[result]]`` in ``pinneapple check`` (``_`` or ``-``)."""
    check = {"select": list(spec.get("select") or ["ALL"]), "fail-on": spec.get("fail_on") or spec.get("fail-on")
             or "error"}
    if spec.get("ignore"):
        check["ignore"] = list(spec["ignore"])
    results = []
    for r in spec.get("results", []) or []:
        r = {k.replace("_", "-"): v for k, v in r.items() if v not in (None, "", {}, [])}
        if "path" not in r:
            raise ValueError("every result needs a path")
        if isinstance(r.get("reference"), dict):
            r["reference"] = {k.replace("_", "-"): v for k, v in r["reference"].items() if v not in (None, "")}
        results.append(r)
    models = [model_ref_target(m) for m in spec.get("models", []) or []]
    if not results and not models:
        raise ValueError("nothing to check: give at least one result or one zoo model")
    return to_toml({"check": check, "result": results, "model": models})


class CodeNotAllowed(ValueError):
    """The configuration would import or unpickle user-supplied code."""


def assert_no_code(cfg) -> None:
    """Refuse targets that execute code from the inputs: a ``file.py:function`` or checkpoint model, a Hub
    download. Zoo models (shipped with the library, hashes checked) and data files are allowed."""
    for m in cfg.models:
        if m.zoo:
            continue
        raise CodeNotAllowed(f"model {m.name!r}: only zoo models can be checked here (a {m.path.suffix or 'callable'} "
                             "model would run uploaded code); run `pinneapple check` locally or in CI for it")


# ---------------------------------------------------------------------------------------------- running
def write_inputs(root: str | Path, files: dict[str, bytes]) -> list[str]:
    """Write uploaded files (``{relative path: bytes}``, e.g. from ``pinneapple_data.cae.upload.expand``) under
    ``root``; a name that would land outside ``root`` is refused."""
    base = Path(root).resolve()
    out = []
    for name, data in files.items():
        p = (base / name).resolve()
        if base not in p.parents:
            raise ValueError(f"{name}: path outside the upload")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        out.append(p.relative_to(base).as_posix())
    return out


def tables_to_csv(root: str | Path, config_text: str | None = None) -> tuple[str | None, dict[str, str]]:
    """Parquet files and JSON tables (a list of row objects, or an object of equal-length columns) under ``root``
    become CSV next to them, which ``pinneapple check`` reads as point clouds or quantities of interest. Returns the
    configuration with the references renamed and ``{old: new}``. Other JSON (manifests, reports) is left alone."""
    import pandas as pd

    base = Path(root).resolve()
    done: dict[str, str] = {}
    for f in sorted([*base.rglob("*.parquet"), *base.rglob("*.json")]):
        try:
            if f.suffix == ".parquet":
                df = pd.read_parquet(f)
            else:
                obj = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
                    df = pd.DataFrame(obj)
                elif isinstance(obj, dict) and obj and all(isinstance(v, list) for v in obj.values()) \
                        and len({len(v) for v in obj.values()}) == 1:
                    df = pd.DataFrame(obj)
                else:
                    continue
        except (ValueError, OSError, ImportError):
            continue
        new = f.with_suffix(".csv")
        if new.exists():
            continue
        df.to_csv(new, index=False)
        done[f.relative_to(base).as_posix()] = new.relative_to(base).as_posix()
    if config_text is not None:
        for old, new in done.items():
            config_text = config_text.replace(f'"{old}"', f'"{new}"').replace(f"'{old}'", f"'{new}'")
    return config_text, done


def _hash_path(p: Path) -> str:
    if p.is_file():
        return sha256_file(p)
    if p.is_dir():
        return sha256_json({f.relative_to(p).as_posix(): sha256_file(f) for f in sorted(p.rglob("*")) if f.is_file()})
    return sha256_bytes(str(p).encode())                 # zoo:name, hub ids: the name itself


def _manifest(root: Path) -> list[dict[str, Any]]:
    return [{"path": f.relative_to(root).as_posix(), "sha256": sha256_file(f), "bytes": f.stat().st_size}
            for f in sorted(root.rglob("*")) if f.is_file()]


def _versions() -> dict[str, str]:
    import numpy

    from pinneapple_check import __version__ as v
    return {"pinneapple": v, "python": platform.python_version(), "numpy": numpy.__version__}


@dataclass
class LabReport:
    check_report: Any                                    # pinneapple_check.Report
    ledger: EvidenceLedger
    config_text: str
    inputs: list[dict[str, Any]]
    started_at: str
    finished_at: str
    versions: dict[str, str] = field(default_factory=dict)
    title: str = "Validation"

    def outcomes(self) -> list[dict[str, Any]]:
        out = []
        for o in self.check_report.outcomes:
            d = o.to_dict()
            d["category"] = category_of(o.rule)
            d["check_id"] = f"{o.rule}/{o.target}/{o.subject}"
            out.append(d)
        return out

    def categories(self) -> dict[str, dict[str, Any]]:
        """Per group: counts and, per target, what ran or that nothing did."""
        outs = self.outcomes()
        targets = [t["name"] for t in self.check_report.targets]
        res: dict[str, dict[str, Any]] = {}
        for key, meta in CATEGORIES.items():
            mine = [o for o in outs if o["category"] == key and o["rule"] != "PC001"]
            per = {}
            for t in targets:
                ts = [o for o in mine if o["target"] == t]
                per[t] = ("not run" if not ts else "failed" if any(not o["passed"] and o["severity"] == "error"
                                                                  for o in ts)
                          else "warning" if any(not o["passed"] for o in ts) else "passed")
            res[key] = {**meta, "checks": len(mine), "passed": sum(o["passed"] for o in mine),
                        "errors": sum(not o["passed"] and o["severity"] == "error" for o in mine),
                        "warnings": sum(not o["passed"] and o["severity"] != "error" for o in mine),
                        "targets": per, "not_run": [t for t, s in per.items() if s == "not run"]}
        return res

    @property
    def decision(self) -> str:
        r = self.check_report
        return "fail" if r.errors else "warn" if r.warnings else "pass"

    def to_dict(self) -> dict[str, Any]:
        cr = self.check_report.to_dict()
        body = {
            "schema": LAB_SCHEMA, "title": self.title, "decision": self.decision, "disclaimer": DISCLAIMER,
            "summary": cr["summary"], "categories": self.categories(), "targets": cr["targets"],
            "outcomes": self.outcomes(), "notes": cr["notes"], "trust_card": cr["trust_card"],
            "provenance": {"started_at": self.started_at, "finished_at": self.finished_at, "tool": "pinneapple check",
                           "versions": self.versions, "config_sha256": sha256_bytes(self.config_text.encode()),
                           "inputs_sha256": sha256_json(self.inputs), "inputs": self.inputs},
            "config": self.config_text,
            "reproduce": f"pinneapple check --config {CONFIG_NAME} --json report.json  # in a folder with the inputs",
            "ledger": self.ledger.to_dict(),
        }
        body["report_sha256"] = sha256_json(body)
        return body


_CWD_LOCK = threading.Lock()


def _strip_field(p: Any) -> Path:
    s = str(p)
    return Path(s.rsplit(":", 1)[0]) if ":" in s[2:] else Path(s)


def _assert_confined(cfg: Any, root: Path) -> None:
    """Uploaded configs may only point at uploaded files: a result or reference path outside ``root`` (``/etc/...``,
    ``../``) would read server files into the report."""
    for t in cfg.targets:
        for p in (getattr(t, "path", None), getattr(getattr(t, "reference", None), "path", None)):
            if p is None or getattr(t, "zoo", None) or getattr(t, "hub", None):
                continue
            q = _strip_field(p)
            q = (q if q.is_absolute() else root / q).resolve()
            if not q.is_relative_to(root):
                raise ValueError(f"{t.name}: {p} is outside the uploaded files")


def run_validation(workdir: str | Path, config: str | Path | None = None, *, allow_code: bool = False,
                   title: str = "Validation") -> LabReport:
    """Run ``pinneapple check`` on the inputs under ``workdir`` with ``config`` (default
    ``workdir/pinneapple-check.toml``) and record every outcome in an evidence ledger."""
    from pinneapple_check import load_config, run

    root = Path(workdir).resolve()
    cfile = Path(config) if config else root / CONFIG_NAME
    if not cfile.is_file():
        raise FileNotFoundError(f"no {CONFIG_NAME} in {root}")
    cfg = load_config(cfile, root=root)
    if not allow_code:
        assert_no_code(cfg)
        _assert_confined(cfg, root)
    text = cfile.read_text(encoding="utf-8")
    inputs = _manifest(root)
    csha = sha256_bytes(text.encode())
    t_hash = {}
    for t in cfg.targets:
        parts = {"config": csha, "target": _hash_path(Path(str(t.path).rsplit(":", 1)[0]) if ":" in str(t.path)[2:]
                                                      else Path(t.path))}
        if getattr(t, "reference", None) is not None:
            parts["reference"] = _hash_path(t.reference.path)
        t_hash[t.name] = sha256_json(parts)
    t0 = utc_now()
    with _CWD_LOCK:                                       # the working directory is process-wide
        cwd = os.getcwd()
        try:
            os.chdir(root)                                # relative paths in messages
            report = run(config=cfg, use_baseline=False)
        finally:
            os.chdir(cwd)
    t1 = utc_now()
    versions = _versions()
    ledger = EvidenceLedger()
    for o in report.outcomes:
        ledger.record(f"{o.rule}/{o.target}/{o.subject}", passed=o.passed or o.baselined,
                      inputs_sha256=t_hash.get(o.target, sha256_json({"config": csha, "inputs": inputs})),
                      outputs={"value": o.value, "threshold": o.threshold, "message": o.message,
                               "severity": o.severity},
                      started_at=t0, finished_at=t1, name=o.title, category=category_of(o.rule) or "coverage",
                      tool="pinneapple check", tool_version=versions["pinneapple"])
    return LabReport(report, ledger, text, inputs, t0, t1, versions, title)


def summarize(lab: dict[str, Any]) -> dict[str, Any]:
    """Compact view of a lab report dict: decision, per-group counts and groups that did not run."""
    return {"decision": lab.get("decision"), "summary": lab.get("summary"),
            "categories": {k: {x: v[x] for x in ("title", "checks", "passed", "errors", "warnings", "not_run")}
                           for k, v in (lab.get("categories") or {}).items()}}


# ---------------------------------------------------------------------------------------------- HTML
_CSS = """body{font:14px/1.5 system-ui,sans-serif;color:#172033;max-width:1080px;margin:24px auto;padding:0 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:22px 0 6px;border-bottom:1px solid #dde3ec}
.muted{color:#5d6b82}.note{background:#fff7e6;border:1px solid #f5d38a;border-radius:8px;padding:8px 12px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}th,td{border-bottom:1px solid #e6eaf0;
padding:4px 6px;text-align:left;vertical-align:top}th{background:#f6f8fb}.passed{color:#15803d;font-weight:600}
.failed{color:#b91c1c;font-weight:700}.warning{color:#b45309;font-weight:600}.notrun{color:#64748b;font-style:italic}
pre{background:#f6f8fb;border:1px solid #e6eaf0;border-radius:6px;padding:8px;overflow:auto;font-size:12px}
code{font-size:12px}"""


def _cls(s: str) -> str:
    return {"passed": "passed", "failed": "failed", "warning": "warning"}.get(s, "notrun")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, (list, dict)):
        return html.escape(json.dumps(v))[:120]
    return html.escape(str(v)) if v is not None else ""


def render_html(lab: dict[str, Any]) -> str:
    """Self-contained HTML of a lab report dict (no scripts, no external files)."""
    e = html.escape
    led = EvidenceLedger.from_dict(lab.get("ledger") or {})
    p = lab.get("provenance", {})
    rows = []
    for key, cat in lab.get("categories", {}).items():
        rows.append(f"<h2>{e(cat['title'])}</h2><p class='muted'>{e(cat['question'])}</p>")
        rows.append("<table><tr><th>Target</th><th>Status</th></tr>" + "".join(
            f"<tr><td>{e(t)}</td><td class='{_cls(s)}'>{e(s)}</td></tr>" for t, s in cat["targets"].items())
            + "</table>")
        outs = [o for o in lab.get("outcomes", []) if o.get("category") == key and o["rule"] != "PC001"]
        if outs:
            body = []
            for o in outs:
                st = led.status(o["check_id"])
                shown = st if st == "not run" else ("passed" if o["passed"] else
                                                    "failed" if o["severity"] == "error" else "warning")
                body.append(f"<tr><td>{e(o['rule'])}</td><td>{e(o['target'])}</td><td>{e(o['subject'])}</td>"
                            f"<td class='{_cls(shown)}'>{e(shown)}</td><td>{_fmt(o.get('value'))}</td>"
                            f"<td>{_fmt(o.get('threshold'))}</td><td>{e(o['message'])}</td></tr>")
            rows.append("<table><tr><th>Rule</th><th>Target</th><th>Subject</th><th>Status</th><th>Value</th>"
                        "<th>Tolerance</th><th>Message</th></tr>" + "".join(body) + "</table>")
        else:
            rows.append("<p class='notrun'>No check of this group ran.</p>")
    inputs = "".join(f"<tr><td>{e(f['path'])}</td><td><code>{e(f['sha256'])}</code></td><td>{f['bytes']}</td></tr>"
                     for f in p.get("inputs", []))
    notes = "".join(f"<li>{e(n)}</li>" for n in lab.get("notes", []))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{e(lab.get('title', 'Validation'))}
</title><style>{_CSS}</style></head><body>
<h1>{e(lab.get('title', 'Validation'))} &mdash; decision: <span class="{_cls({'pass': 'passed', 'fail': 'failed',
'warn': 'warning'}.get(lab.get('decision'), ''))}">{e(str(lab.get('decision', '')).upper())}</span></h1>
<p class="muted">{lab['summary']['checks']} checks on {lab['summary']['targets']} target(s): {lab['summary']['passed']}
 passed, {lab['summary']['errors']} errors, {lab['summary']['warnings']} warnings. Ran {e(p.get('started_at', ''))} to
 {e(p.get('finished_at', ''))} with {e(p.get('tool', ''))} {e(str(p.get('versions', {}).get('pinneapple', '')))}.</p>
<p class="note">{e(lab.get('disclaimer', DISCLAIMER))}</p>
{''.join(rows)}
{f'<h2>Notes</h2><ul>{notes}</ul>' if notes else ''}
<h2>Reproduce</h2><p>Inputs SHA-256 (manifest) <code>{e(p.get('inputs_sha256', ''))}</code>, configuration SHA-256
<code>{e(p.get('config_sha256', ''))}</code>, report SHA-256 <code>{e(lab.get('report_sha256', ''))}</code>.</p>
<pre>{e(lab.get('reproduce', ''))}</pre><pre>{e(lab.get('config', ''))}</pre>
<table><tr><th>Input file</th><th>SHA-256</th><th>Bytes</th></tr>{inputs}</table>
</body></html>"""
