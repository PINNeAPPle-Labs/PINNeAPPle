"""Engineering evidence: turn experiments and simulations into a reproducible report and a zip package.

Sources are the files other tools wrote, read as they are and hashed:

=======================  ===================================================================================
``validation-lab``       a Validation Lab report (``pinneapple-validation-lab/v1``): checks and their ledger
``ledger``               an evidence ledger (``pinneapple-evidence-ledger/v1``)
``pinneapple-check``     a ``pinneapple check --json`` report (no ledger: its checks read "not run" here)
``comparator``           a Simulation Comparator result (global and per-field errors)
``contract``             a StudyResult, ModelRef or DatasetSpec (JSON with ``"type"`` or ``"contract"``)
``metrics``              a table (CSV, Parquet, JSON list of rows): one row per run/method, numeric columns
``metadata``             any other JSON object (simulation metadata, configuration)
``file``                 anything else: referenced by name, size and SHA-256
=======================  ===================================================================================

The central rule, enforced by :func:`build_report` and tested: a test is reported as passed or failed only when
an intact :class:`~pinneapple_veriphysics.ledger.EvidenceEntry` records that it ran (inputs hash, outputs, time).
A source that only says "passed" gives "not run", with the claim shown next to it.

::

    srcs = [load_source(name, data) for name, data in files]
    report = build_report({"title": "Heat sink v2", "baseline": "FEM", "candidate": "PINN"}, srcs)
    html = render_html(report, figures=make_figures(report, "out"))
    pdf = render_pdf(report, figures=make_figures(report, "out"))      # needs reportlab
    zip_bytes = build_package(report, srcs, html=html, pdf=pdf, figures=...)
    assert verify_package(zip_bytes) == []
"""
from __future__ import annotations

import base64
import csv
import html as _html
import io
import json
import math
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ledger import NOT_RUN, EvidenceLedger, canonical_json, sha256_bytes, sha256_json, utc_now
from .validation_lab import DISCLAIMER, LAB_SCHEMA

__all__ = ["Source", "load_source", "build_report", "make_figures", "render_html", "render_pdf", "build_package",
           "verify_package", "to_study_result", "REPORT_SCHEMA", "CONTRACT_TYPES"]

REPORT_SCHEMA = "pinneapple-evidence-report/v1"
CONTRACT_TYPES = ("StudyResult", "ModelRef", "DatasetSpec", "ParameterSpace")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class Source:
    name: str
    kind: str
    sha256: str
    bytes: int
    raw: bytes = field(repr=False, default=b"")
    url: str = ""
    data: Any = None                         # parsed JSON (dict or list), or rows of a table
    contract: str = ""                       # StudyResult / ModelRef / DatasetSpec / ParameterSpace
    problems: list[str] = field(default_factory=list)   # schema problems of a contract (pinneapple_data.contracts)

    def ref(self) -> dict[str, Any]:
        out = {"name": self.name, "kind": self.kind + (f":{self.contract}" if self.contract else ""),
               "sha256": self.sha256, "bytes": self.bytes, "url": self.url}
        if self.problems:
            out["contract_problems"] = self.problems
        return out


def _contract_type(d: dict[str, Any]) -> str:
    for k in ("contract", "type", "kind", "$type"):
        v = str(d.get(k, ""))
        for c in CONTRACT_TYPES:
            if v.lower() == c.lower() or v.lower().endswith("." + c.lower()):
                return c
    return ""


def _contract_problems(obj: dict[str, Any]) -> list[str]:
    """Schema problems of a contract under ``pinneapple_data.contracts`` (it is read either way)."""
    try:
        from pinneapple_data.contracts import validate
    except ImportError:                                         # pydantic missing: read it duck-typed
        return []
    if "kind" not in obj:
        return ["no `kind` field: read by its type key, not validated against the contract schema"]
    return list(validate(obj))


def _rows_csv(text: str) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(io.StringIO(text)))
    return [{k: _num(v) for k, v in r.items() if k is not None} for r in rows]


def _num(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return float(v) if v.strip() else v
        except ValueError:
            return v
    return v


def load_source(name: str, data: bytes, url: str = "") -> Source:
    """Read one file and decide what it is (never fails: unknown content is kept as a referenced file)."""
    s = Source(name=name, kind="file", sha256=sha256_bytes(data), bytes=len(data), raw=data, url=url)
    low = name.lower()
    if low.endswith(".parquet"):
        try:
            import pandas as pd
            s.data, s.kind = pd.read_parquet(io.BytesIO(data)).to_dict("records"), "metrics"
        except Exception:                                       # pyarrow missing or not a table: keep as a file
            pass
        return s
    if low.endswith((".csv", ".tsv")):
        try:
            rows = _rows_csv(data.decode("utf-8-sig"))
            if rows and any(isinstance(v, float) for v in rows[0].values()):
                s.data, s.kind = rows, "metrics"
        except (UnicodeDecodeError, csv.Error):
            pass
        return s
    if not low.endswith(".json"):
        return s
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return s
    s.data = obj
    if isinstance(obj, list):
        s.kind = "metrics" if obj and all(isinstance(r, dict) for r in obj) else "metadata"
        return s
    if not isinstance(obj, dict):
        s.kind = "metadata"
        return s
    if obj.get("schema") == LAB_SCHEMA:
        s.kind = "validation-lab"
    elif obj.get("schema") == "pinneapple-evidence-ledger/v1":
        s.kind = "ledger"
    elif obj.get("tool") == "pinneapple check" and "checks" in obj:
        s.kind = "pinneapple-check"
    elif isinstance(obj.get("fields"), list) and isinstance(obj.get("global"), dict):
        s.kind = "comparator"
    elif _contract_type(obj):
        s.kind, s.contract = "contract", _contract_type(obj)
        s.problems = _contract_problems(obj)
    else:
        s.kind = "metadata"
    return s


# ---------------------------------------------------------------------------------------------- report
def _claimed_tests(s: Source) -> list[dict[str, Any]]:
    """What a source *says* about tests: (check_id, claim). Claims are shown, never trusted."""
    d, out = s.data, []
    if s.kind == "validation-lab":
        for o in d.get("outcomes", []):
            out.append({"check_id": o["check_id"], "name": o.get("title") or o["rule"], "category": o.get("category"),
                        "claim": "passed" if o["passed"] else "failed", "detail": o.get("message", "")})
    elif s.kind == "pinneapple-check":
        for o in d.get("checks", []):
            out.append({"check_id": f"{o['rule']}/{o['target']}/{o['subject']}", "name": o.get("title", o["rule"]),
                        "category": None, "claim": "passed" if o.get("passed") else "failed",
                        "detail": o.get("message", "")})
    elif s.kind == "comparator":
        for f in d.get("fields", []):
            if f.get("verdict"):
                out.append({"check_id": f"compare/{f.get('reference')}", "name": f"Field {f.get('reference')} within "
                            f"tolerance", "category": "validation", "claim": "passed" if f["verdict"] == "pass"
                            else "failed", "detail": f"relative L2 {f.get('summary', {}).get('rel_l2')}"})
    elif s.kind == "contract":
        for t in d.get("tests") or d.get("checks") or []:
            if isinstance(t, dict):
                cid = str(t.get("check_id") or t.get("id") or t.get("name"))
                st = t.get("status") if "status" in t else ("passed" if t.get("passed") else "failed")
                st = _CLAIM.get(str(st).upper(), st)
                out.append({"check_id": cid, "name": str(t.get("name", cid)), "category": t.get("category"),
                            "claim": str(st), "detail": str(t.get("detail", ""))})
    return out


# StudyResult.checks[].status of the shared contracts -> the words of a claim
_CLAIM = {"PASS": "passed", "FAIL": "failed", "WARN": "warning", "N/A": "not applicable", "NOT_RUN": "not run"}


def _ledger_of(s: Source) -> EvidenceLedger:
    d = s.data
    if s.kind == "validation-lab":
        return EvidenceLedger.from_dict(d.get("ledger") or {})
    if s.kind == "ledger":
        return EvidenceLedger.from_dict(d)
    if s.kind == "contract":
        for k in ("evidence", "ledger"):
            if isinstance(d.get(k), (dict, list)):
                return EvidenceLedger.from_dict(d[k])
    return EvidenceLedger()


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _metrics(sources: Sequence[Source]) -> list[dict[str, Any]]:
    """Rows {run, metric, value, source}: every numeric value a source reports per run."""
    out = []
    for s in sources:
        if s.kind == "metrics":
            for i, r in enumerate(s.data):
                label = next((str(v) for v in r.values() if isinstance(v, str) and v), f"row {i + 1}")
                out += [{"run": label, "metric": k, "value": float(v), "source": s.name}
                        for k, v in r.items() if _finite(v)]
        elif s.kind == "comparator":
            g = s.data.get("global", {})
            out += [{"run": s.name, "metric": f"global {k}", "value": float(v), "source": s.name}
                    for k, v in g.items() if _finite(v)]
            for f in s.data.get("fields", []):
                v = (f.get("summary") or {}).get("rel_l2")
                if _finite(v):
                    out.append({"run": s.name, "metric": f"rel L2 {f.get('reference')}", "value": float(v),
                                "source": s.name})
        elif s.kind == "contract" and isinstance(s.data.get("metrics"), dict):
            label = str(s.data.get("name") or s.data.get("id") or s.name)
            out += [{"run": label, "metric": k, "value": float(v), "source": s.name}
                    for k, v in s.data["metrics"].items() if _finite(v)]
    return out


def _comparison(metrics: list[dict[str, Any]], baseline: str, candidate: str) -> list[dict[str, Any]]:
    """Rows of metrics found for both runs. A run belongs to the baseline when its label equals the baseline label
    or contains it, and does not also contain the candidate label (a study named "FEM vs PINN" is neither)."""
    bl, cl = baseline.lower(), candidate.lower()

    def pick(mine, other):
        exact = {m["metric"]: m for m in metrics if m["run"].lower() == mine}
        loose = {m["metric"]: m for m in metrics if mine in m["run"].lower() and other not in m["run"].lower()}
        return {**loose, **exact}
    b, c = pick(bl, cl), pick(cl, bl)
    rows = []
    for k in [k for k in c if k in b]:
        bv, cv = b[k]["value"], c[k]["value"]
        rows.append({"metric": k, "baseline": bv, "candidate": cv,
                     "change": (cv - bv) / abs(bv) if bv else None, "baseline_run": b[k]["run"],
                     "candidate_run": c[k]["run"]})
    return rows


def _test_row(cid: str, c: dict[str, Any], ledger: EvidenceLedger, source: str) -> dict[str, Any]:
    e = ledger.latest(cid)
    bad = [x for x in ledger.entries if x.check_id == cid and not x.valid]
    why = "" if e is not None else (("record invalid: " + ", ".join(bad[-1].problems())) if bad
                                     else "no evidence record")
    return {"check_id": cid, "name": c["name"], "category": c.get("category") or (e.category if e else ""),
            "status": ledger.status(cid), "source": source, "source_claim": c["claim"], "why": why,
            "detail": (e.outputs or {}).get("message", "") if e else c.get("detail", ""),
            "started_at": e.started_at if e else "", "finished_at": e.finished_at if e else "",
            "inputs_sha256": e.inputs_sha256 if e else "", "tool": e.tool if e else ""}


def build_report(meta: dict[str, Any], sources: Sequence[Source]) -> dict[str, Any]:
    """The report as data. ``meta``: title, project, author, purpose, methodology, conditions (dict),
    limitations (list), baseline and candidate (labels matched against run names), references (list)."""
    # Evidence is matched per source: a claim is backed by the source's own records or by a standalone ledger
    # file, never by another run that happens to use the same check id.
    shared = EvidenceLedger()
    for s in sources:
        if s.kind == "ledger":
            shared.extend(_ledger_of(s))
    ledger = EvidenceLedger()
    tests: list[dict[str, Any]] = []
    claimed_shared: set[str] = set()
    for s in sources:
        own = _ledger_of(s) if s.kind != "ledger" else EvidenceLedger()
        ledger.extend(own)
        both = EvidenceLedger([*own.entries, *shared.entries])
        claims = {c["check_id"]: c for c in _claimed_tests(s)}
        for cid in own.check_ids():
            if cid not in claims:
                e = next(x for x in reversed(own.entries) if x.check_id == cid)
                claims[cid] = {"check_id": cid, "name": e.name, "category": e.category, "claim": None, "detail": ""}
        for cid, c in claims.items():
            if cid in shared.check_ids():
                claimed_shared.add(cid)
            tests.append(_test_row(cid, c, both, s.name))
    ledger.extend(shared)
    for cid in shared.check_ids():
        if cid not in claimed_shared:
            e = next(x for x in reversed(shared.entries) if x.check_id == cid)
            src = next(s.name for s in sources if s.kind == "ledger" and cid in _ledger_of(s).check_ids())
            tests.append(_test_row(cid, {"name": e.name, "category": e.category, "claim": None, "detail": ""},
                                   shared, src))
    metrics = _metrics(sources)
    baseline, candidate = str(meta.get("baseline") or ""), str(meta.get("candidate") or "")
    comparison = _comparison(metrics, baseline, candidate) if baseline and candidate else []
    conditions = dict(meta.get("conditions") or {})
    configs = []
    for s in sources:
        if s.kind == "validation-lab":
            configs.append({"source": s.name, "config": s.data.get("config", ""),
                            "reproduce": s.data.get("reproduce", "")})
        if s.kind == "contract":
            for k in ("parameters", "conditions", "config"):
                if isinstance(s.data.get(k), dict):
                    conditions.update({f"{s.name}: {kk}": vv for kk, vv in s.data[k].items()})
    contracts = [{"source": s.name, "type": s.contract,
                  "name": s.data.get("name") or s.data.get("title") or s.data.get("id") or "",
                  "summary": {k: v for k, v in s.data.items() if isinstance(v, (str, int, float)) and k not in
                              ("type", "contract", "kind")}} for s in sources if s.kind == "contract"]
    limitations = list(meta.get("limitations") or [])
    for s in sources:                                  # StudyResult.limits: what the result does not show
        if s.kind == "contract":
            limitations += [f"{s.name}: {x}" for x in s.data.get("limits") or [] if isinstance(x, str)]
    n_not = sum(t["status"] == NOT_RUN for t in tests)
    if n_not:
        limitations.append(f"{n_not} test(s) are reported as not run: no intact evidence record shows that they ran "
                           "(a source claiming a pass is not evidence).")
    for s in sources:
        if s.kind == "validation-lab":
            for _key, cat in (s.data.get("categories") or {}).items():
                if cat.get("not_run"):
                    limitations.append(f"{s.name}: no {cat['title'].lower()} check ran for "
                                       f"{', '.join(cat['not_run'][:6])}{' ...' if len(cat['not_run']) > 6 else ''}.")
    if baseline and candidate and not comparison:
        limitations.append(f"No metric was found for both the baseline ({baseline!r}) and the candidate "
                           f"({candidate!r}): there is no baseline comparison.")
    limitations.append(DISCLAIMER)
    counts = {k: sum(t["status"] == k for t in tests) for k in ("passed", "failed", NOT_RUN)}
    body = {
        "schema": REPORT_SCHEMA, "generated_at": utc_now(),
        "title": meta.get("title") or "Engineering evidence report", "project": meta.get("project", ""),
        "author": meta.get("author", ""), "purpose": meta.get("purpose", ""),
        "methodology": meta.get("methodology", ""), "baseline": baseline, "candidate": candidate,
        "conditions": conditions, "configs": configs, "contracts": contracts,
        "tests": tests, "test_counts": counts, "metrics": metrics, "comparison": comparison,
        "limitations": limitations, "references": list(meta.get("references") or []),
        "artifacts": [s.ref() for s in sources], "ledger": ledger.to_dict(),
        "invalid_records": [e.check_id for e in ledger.invalid()],
    }
    body["report_sha256"] = sha256_json(body)
    return body


# ---------------------------------------------------------------------------------------------- figures
def make_figures(report: dict[str, Any], outdir: str | Path) -> dict[str, str]:
    """PNG figures from the report's own numbers: baseline vs candidate metrics, test status per group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pinneapple_tools.reporting.paperkit import CAT, MUTED, savefig, setup_mpl

    setup_mpl()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    figs: dict[str, str] = {}
    comp = [r for r in report.get("comparison", []) if r["baseline"] and r["candidate"]][:12]
    if comp:
        fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(comp) + 1.2))
        y = range(len(comp))
        ax.barh([i + 0.2 for i in y], [r["baseline"] for r in comp], 0.4, color=CAT[0],
                label=f"baseline: {report['baseline']}")
        ax.barh([i - 0.2 for i in y], [r["candidate"] for r in comp], 0.4, color=CAT[1],
                label=f"candidate: {report['candidate']}")
        ax.set_yticks(list(y), [r["metric"] for r in comp])
        ax.invert_yaxis()
        if all(r["baseline"] > 0 and r["candidate"] > 0 for r in comp) and \
                max(max(r["baseline"], r["candidate"]) for r in comp) / \
                min(min(r["baseline"], r["candidate"]) for r in comp) > 100:
            ax.set_xscale("log")
        ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        ax.set_xlabel("value (as reported by the sources)")
        figs["comparison"] = savefig(fig, out / "comparison.png")
        plt.close(fig)
    tests = report.get("tests", [])
    if tests:
        groups = sorted({t["category"] or "other" for t in tests})
        fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(groups) + 1.2))
        left = [0] * len(groups)
        for st, col in (("passed", CAT[2]), ("failed", CAT[7]), (NOT_RUN, MUTED)):
            vals = [sum(1 for t in tests if (t["category"] or "other") == g and t["status"] == st) for g in groups]
            ax.barh(groups, vals, left=left, color=col, label=st)
            left = [a + b for a, b in zip(left, vals, strict=False)]
        ax.invert_yaxis()
        ax.set_xlabel("tests")
        ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        figs["tests"] = savefig(fig, out / "tests.png")
        plt.close(fig)
    return figs


# ---------------------------------------------------------------------------------------------- HTML
_CSS = """body{font:14px/1.55 system-ui,sans-serif;color:#172033;max-width:1040px;margin:24px auto;padding:0 16px}
h1{font-size:24px;margin:0}h2{font-size:17px;margin:24px 0 6px;border-bottom:1px solid #dde3ec;padding-bottom:3px}
.muted{color:#5d6b82}.note{background:#fff7e6;border:1px solid #f5d38a;border-radius:8px;padding:8px 12px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}th,td{border-bottom:1px solid #e6eaf0;
padding:4px 6px;text-align:left;vertical-align:top}th{background:#f6f8fb}code{font-size:11.5px;word-break:break-all}
.passed{color:#15803d;font-weight:600}.failed{color:#b91c1c;font-weight:700}.notrun{color:#64748b;font-style:italic}
img{max-width:100%}pre{background:#f6f8fb;border:1px solid #e6eaf0;border-radius:6px;padding:8px;overflow:auto;
font-size:12px}"""


def _f(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return _html.escape("" if v is None else str(v))


def _pct(v: float | None) -> str:
    return "" if v is None else f"{100 * v:+.1f} %"


def _st(s: str) -> str:
    return {"passed": "passed", "failed": "failed"}.get(s, "notrun")


def render_html(report: dict[str, Any], figures: dict[str, str] | None = None) -> str:
    """Self-contained HTML (figures inlined as PNG data URIs)."""
    e = _html.escape
    figures = figures or {}

    def img(key, cap):
        p = figures.get(key)
        if not p:
            return ""
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        return f"<figure><img alt='{e(cap)}' src='data:image/png;base64,{b64}'><figcaption class='muted'>{e(cap)}" \
               f"</figcaption></figure>"

    c = report["test_counts"]
    tests = "".join(
        f"<tr><td>{e(t['name'])}<br><code>{e(t['check_id'])}</code></td><td>{e(t['category'] or '')}</td>"
        f"<td class='{_st(t['status'])}'>{e(t['status'])}</td><td>{e(t['why'] or t['detail'] or '')}</td>"
        f"<td>{e(t['source'])}: {e(t['source_claim'] or '-')}</td><td>{e(t['started_at'])}</td><td><code>{e(t['inputs_sha256'][:16])}"
        f"</code></td></tr>" for t in report["tests"])
    comp = "".join(f"<tr><td>{e(r['metric'])}</td><td>{_f(r['baseline'])}</td><td>{_f(r['candidate'])}</td>"
                   f"<td>{_pct(r['change'])}</td></tr>"
                   for r in report["comparison"])
    if comp:
        comp_html = (f"<p>Baseline <b>{e(report['baseline'])}</b>, candidate <b>{e(report['candidate'])}</b>.</p>"
                     f"<table><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Change</th></tr>{comp}</table>")
    else:
        comp_html = '<p class="notrun">No baseline comparison: no metric was found for both runs.</p>'
    purpose = f"<p>{e(report['purpose'])}</p>" if report["purpose"] else ""
    cond = "".join(f"<tr><td>{e(str(k))}</td><td>{_f(v) if not isinstance(v, (dict, list)) else e(json.dumps(v))}"
                   f"</td></tr>" for k, v in report["conditions"].items())
    arts = "".join(f"<tr><td>{e(a['name'])}</td><td>{e(a['kind'])}</td><td>{a['bytes']}</td><td><code>"
                   f"{e(a['sha256'])}</code></td><td>{e(a['url'])}</td></tr>" for a in report["artifacts"])
    cons = "".join(f"<li>{e(x['type'])} <b>{e(str(x['name']))}</b> ({e(x['source'])})</li>" for x in report["contracts"])
    cfgs = "".join(f"<p class='muted'>{e(x['source'])}</p><pre>{e(x['config'])}</pre><pre>{e(x['reproduce'])}</pre>"
                   for x in report["configs"])
    lim = "".join(f"<li>{e(x)}</li>" for x in report["limitations"])
    refs = "".join(f"<li>{e(x)}</li>" for x in report["references"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{e(report['title'])}</title>
<style>{_CSS}</style></head><body>
<h1>{e(report['title'])}</h1>
<p class="muted">{e(report['project'])}{' · ' if report['project'] else ''}{e(report['author'])}{' · ' if report['author']
 else ''}generated {e(report['generated_at'])} · report SHA-256 <code>{e(report['report_sha256'])}</code></p>
<h2>1. Summary</h2>
<p>{c['passed']} test(s) passed, {c['failed']} failed and {c['not run']} not run, out of {len(report['tests'])}. A test
 counts as passed or failed only when an evidence record shows it ran (inputs SHA-256, outputs, time); otherwise it is
 printed as <span class="notrun">not run</span>.</p>{purpose}
<h2>2. Methodology</h2><p>{e(report['methodology']) or '<span class="notrun">Not described.</span>'}</p>
{f'<ul>{cons}</ul>' if cons else ''}
<h2>3. Conditions</h2>{f'<table>{cond}</table>' if cond else '<p class="notrun">No conditions recorded.</p>'}{cfgs}
<h2>4. Baseline comparison</h2>
{comp_html}
{img('comparison', 'Baseline and candidate, as reported by the sources.')}
<h2>5. Tests and evidence</h2>{img('tests', 'Tests per group: passed, failed, not run (no evidence record).')}
<table><tr><th>Test</th><th>Group</th><th>Status</th><th>Detail</th><th>Source: its claim</th><th>Ran at (UTC)</th>
<th>Inputs SHA-256</th></tr>{tests}</table>
<h2>6. Limitations</h2><ul>{lim}</ul>
<h2>7. References and artifacts</h2>{f'<ul>{refs}</ul>' if refs else ''}
<table><tr><th>Artifact</th><th>Kind</th><th>Bytes</th><th>SHA-256</th><th>URL</th></tr>{arts}</table>
</body></html>"""


# ---------------------------------------------------------------------------------------------- PDF
def render_pdf(report: dict[str, Any], figures: dict[str, str] | None = None, path: str | Path | None = None) \
        -> bytes:
    """PDF through :class:`pinneapple_tools.reporting.paperkit.Paper` (needs reportlab)."""
    import tempfile

    from pinneapple_tools.reporting.paperkit import Paper, fmt

    figures = figures or {}
    c = report["test_counts"]
    p = Paper(report["title"], subtitle=report.get("project") or "Engineering evidence report",
              authors=report.get("author") or "PINNeAPPle Evidence Report",
              affiliation="Reproducible technical evidence", report_id=report["report_sha256"][:12],
              footer="Engineering Evidence Report · a test passes only with a recorded run")
    p.abstract(f"{c['passed']} test(s) passed, {c['failed']} failed and {c['not run']} not run, out of "
               f"{len(report['tests'])}. A test is reported as passed or failed only when an evidence record shows "
               f"that it ran (inputs SHA-256, outputs and time); otherwise it is printed as not run. "
               + (report.get("purpose") or ""))
    p.section("Methodology")
    p.p(report.get("methodology") or "Not described.")
    if report["contracts"]:
        p.bullets([f"{x['type']} {x['name']} ({x['source']})" for x in report["contracts"]])
    p.section("Conditions")
    if report["conditions"]:
        p.table([["Condition", "Value"]] + [[str(k), str(v)[:160]] for k, v in report["conditions"].items()],
                "Conditions recorded by the author and the sources.", col_widths_cm=[6, 10])
    else:
        p.p("No conditions recorded.")
    if report["configs"]:
        p.p("The check configurations that produced the evidence are in the appendix.")
    p.section("Baseline comparison")
    if report["comparison"]:
        p.table([["Metric", "Baseline", "Candidate", "Change"]] +
                [[r["metric"], fmt(r["baseline"]), fmt(r["candidate"]),
                  _pct(r["change"])] for r in report["comparison"][:40]],
                f"Baseline {report['baseline']} and candidate {report['candidate']}.", col_widths_cm=[7, 3, 3, 3])
        if figures.get("comparison"):
            p.figure(figures["comparison"], "Baseline and candidate, as reported by the sources.", width_cm=13)
    else:
        p.p("No baseline comparison: no metric was found for both runs.")
    p.section("Tests and evidence")
    if figures.get("tests"):
        p.figure(figures["tests"], "Tests per group: passed, failed, not run (no evidence record).", width_cm=13)
    rows = [["Test", "Status", "Ran at (UTC)", "Inputs SHA-256"]]
    for t in report["tests"][:400]:
        label = t["check_id"] if t["name"] in t["check_id"] else f"{t['name']} ({t['check_id']})"
        rows.append([f"{t['source']}: {label}"[:140], t["status"] + (f" ({t['why']})" if t["why"] else ""),
                     t["started_at"][:19].replace("T", " "), t["inputs_sha256"][:12]])
    p.table(rows, "Every test the sources mention; status from the evidence ledger only.",
            col_widths_cm=[8.4, 3.4, 2.6, 1.8])
    p.section("Limitations")
    p.bullets(report["limitations"])
    p.section("Artifacts")
    p.table([["Artifact", "Kind", "SHA-256"]] + [[a["name"], a["kind"], a["sha256"]] for a in report["artifacts"]],
            "Original artifacts, by SHA-256 (in the evidence package under sources/).", col_widths_cm=[5, 2.8, 8.4])
    if report["references"]:
        p.references(report["references"])
    if report["configs"]:
        p.pagebreak()
        p.section("Appendix: configurations")
        for x in report["configs"]:
            p.p(f"Configuration of {x['source']} (reproduce with: {x['reproduce']}):")
            p.code(x["config"][:6000])
    if path is None:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "report.pdf"
            p.build(str(f))
            return f.read_bytes()
    p.build(str(path))
    return Path(path).read_bytes()


# ---------------------------------------------------------------------------------------------- package
def to_study_result(report: dict[str, Any], app: str = "evidence") -> dict[str, Any]:
    """The report as a ``StudyResult`` (``pinneapple_data.contracts``, study type ``evidence``) for the next app:
    one check per test (PASS / FAIL with its record as evidence, NOT_RUN otherwise), the test counts as metrics, the
    artifacts by SHA-256 and the limitations as limits."""
    from pinneapple_data.contracts import CheckRecord, FileRef, Provenance, StudyResult

    status = {"passed": "PASS", "failed": "FAIL"}
    checks = [CheckRecord(name=f"{t['source']}: {t['check_id']}"[:200], status=status.get(t["status"], "NOT_RUN"),
                          detail=(t["why"] or t["detail"] or "")[:500],
                          evidence=(f"ran {t['started_at']} to {t['finished_at']}, inputs sha256 {t['inputs_sha256']}"
                                    if t["status"] in status else ""))
              for t in report["tests"]]
    arts = [FileRef(path=a["url"] or a["name"], sha256=a["sha256"], size=a["bytes"],
                    description=f"{a['name']} ({a['kind']})") for a in report["artifacts"]]
    try:
        from pinneapple_check import __version__ as version
    except ImportError:                                         # pragma: no cover
        version = ""
    sr = StudyResult(id=f"evidence-{report['report_sha256'][:16]}", study_type="evidence", app=app,
                     title=report["title"], metrics={k.replace(" ", "_"): v for k, v in report["test_counts"].items()},
                     checks=checks, artifacts=arts, limits=list(report["limitations"]),
                     provenance=Provenance(app=app, source="evidence report", library_version=version,
                                           notes=f"report sha256 {report['report_sha256']}"))
    return sr.to_dict()


def build_package(report: dict[str, Any], sources: Sequence[Source], *, html: str | None = None,
                  pdf: bytes | None = None, figures: dict[str, str] | None = None) -> bytes:
    """Zip: report.json/html/pdf, evidence-ledger.json, figures/, sources/ (the original bytes) and
    MANIFEST.json with the SHA-256 of every other file."""
    files: dict[str, bytes] = {"report.json": json.dumps(report, indent=1).encode(),
                               "evidence-ledger.json": json.dumps(report["ledger"], indent=1).encode()}
    try:
        files["study_result.json"] = json.dumps(to_study_result(report), indent=1).encode()
    except ImportError:                                         # pydantic missing: no contract export
        pass
    if html is not None:
        files["report.html"] = html.encode()
    if pdf is not None:
        files["report.pdf"] = pdf
    for k, p in (figures or {}).items():
        files[f"figures/{k}.png"] = Path(p).read_bytes()
    used = set()
    for s in sources:
        n = _SAFE.sub("_", Path(s.name).name) or "file"
        while f"sources/{n}" in used:
            n = "_" + n
        used.add(f"sources/{n}")
        files[f"sources/{n}"] = s.raw
    files["README.txt"] = (
        f"{report['title']}\n\nEvidence package generated {report['generated_at']}.\n"
        "MANIFEST.json lists the SHA-256 of every file; sources/ holds the original artifacts unchanged.\n"
        "A test is reported as passed only when evidence-ledger.json holds an intact record of its run.\n").encode()
    manifest = {"schema": "pinneapple-evidence-package/v1", "report_sha256": report["report_sha256"],
                "files": {k: sha256_bytes(v) for k, v in sorted(files.items())}}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in sorted(files.items()):
            z.writestr(k, v)
        z.writestr("MANIFEST.json", canonical_json(manifest))
    return buf.getvalue()


def verify_package(data: bytes) -> list[str]:
    """Problems found in an evidence package (empty list: every file matches the manifest)."""
    probs = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set(z.namelist())
        if "MANIFEST.json" not in names:
            return ["no MANIFEST.json"]
        man = json.loads(z.read("MANIFEST.json"))
        for k, h in man.get("files", {}).items():
            if k not in names:
                probs.append(f"{k}: missing")
            elif sha256_bytes(z.read(k)) != h:
                probs.append(f"{k}: SHA-256 differs from the manifest")
        extra = names - set(man.get("files", {})) - {"MANIFEST.json"}
        probs += [f"{k}: not in the manifest" for k in sorted(extra)]
        if "report.json" in names:
            rep = json.loads(z.read("report.json"))
            body = {k: v for k, v in rep.items() if k != "report_sha256"}
            if sha256_json(body) != rep.get("report_sha256"):
                probs.append("report.json: content does not match its report_sha256")
    return probs
