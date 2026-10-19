"""The evidence of a run, for people and for other tools.

* :func:`write_report`: one self-contained HTML file (images embedded) with the inputs, the tools and versions, a
  timeline of the steps (state, duration, commands, actions, checks, artifacts with SHA-256, logs, screenshots),
  the acceptance criteria and the errors. Print it to PDF from a browser for a PDF.
* :func:`to_study_result` / :func:`write_study_result`: the same facts as a ``pinneapple_data.contracts.StudyResult``
  (the contract the apps share, read by the Evidence Report app): checks, criteria, metrics, files with SHA-256,
  limits and provenance; inputs, decision, step states and images in ``metadata``.

Nothing is inferred: a step is shown as passed only if its record says so, and an image's kind (screen, window,
render, plot) is shown next to it.
"""
from __future__ import annotations

import base64
import html
import json
import time
from pathlib import Path
from typing import Any

from .record import RunRecord, StepRecord

__all__ = ["write_report", "to_study_result", "write_study_result"]
_STATE_COLOR = {"passed": "#15803d", "failed": "#b91c1c", "rejected": "#b45309", "awaiting_confirmation": "#4338ca",
                "blocked": "#6b7280", "declined": "#6b7280", "running": "#0369a1", "pending": "#9ca3af", "skipped": "#9ca3af",
                "paused": "#6b7280"}
_KIND = {"screen": "screenshot of the virtual display", "window": "screenshot of one window",
         "render": "image exported by the application", "plot": "chart drawn from the step's log"}


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _t(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts)) if ts else "-"


def _img(run_dir: Path, rel: str, max_w: int = 1100) -> str:
    p = run_dir / rel
    if not p.is_file():
        return f"<p class=miss>image {_e(rel)} missing</p>"
    data = p.read_bytes()
    if len(data) > 600_000:                                           # keep the report small: re-encode as JPEG
        try:
            import io

            from PIL import Image
            im = Image.open(io.BytesIO(data)).convert("RGB")
            im.thumbnail((max_w, max_w))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=82)
            data, mime = buf.getvalue(), "image/jpeg"
        except Exception:
            mime = "image/png"
    else:
        mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return f'<img src="data:{mime};base64,{base64.b64encode(data).decode()}" alt="{_e(rel)}">'


def _step_html(run_dir: Path, s: StepRecord, n: int) -> str:
    color = _STATE_COLOR.get(s.state, "#6b7280")
    rows = []
    if s.confirmation:
        c = s.confirmation
        who = (f"declined by {_e(c.get('declined_by'))} at {_t(c.get('at'))}"
               + (f": {_e(c.get('reason'))}" if c.get("reason") else "") if c.get("declined_by") else
               f"approved by {_e(c.get('approved_by'))} at {_t(c.get('at'))}" if c.get("approved_by") else
               "<b>waiting for a person to approve</b>")
        rows.append(f"<div class=conf>Confirmation ({_e(', '.join(c.get('categories', [])))}): "
                    f"{_e('; '.join(c.get('reasons', [])))} — {who}</div>")
    if s.error:
        rows.append(f"<pre class=err>{_e(s.error)}</pre>")
    if s.actions:
        rows.append("<h4>Actions</h4><ul>" + "".join(f"<li>{_e(a)}</li>" for a in s.actions) + "</ul>")
    if s.checks:
        rows.append("<h4>Checks</h4><table><tr><th></th><th>check</th><th>result</th><th>source</th></tr>" + "".join(
            f"<tr><td class={'ok' if c.passed else 'ko'}>{'PASS' if c.passed else 'FAIL'}</td><td>{_e(c.name)}</td>"
            f"<td>{_e(c.detail)}</td><td class=src>{_e(c.source)}</td></tr>" for c in s.checks) + "</table>")
    if s.evidence:
        rows.append("<h4>Images</h4>" + "".join(
            f"<figure>{_img(run_dir, ev.path)}<figcaption>{_e(ev.caption)} <span class=kind>"
            f"[{_e(_KIND.get(ev.kind, ev.kind))}; {_e(ev.path)}; sha256 {_e(ev.sha256[:12])}]</span></figcaption>"
            f"</figure>" for ev in s.evidence))
    if s.commands:
        rows.append("<h4>Commands</h4><pre>" + _e("\n".join(s.commands)) + "</pre>")
    if s.artifacts:
        rows.append("<h4>Artifacts</h4><table><tr><th>path</th><th>bytes</th><th>sha256</th></tr>" + "".join(
            f"<tr><td>{_e(a.path)}{'/' if a.kind == 'dir' else ''}</td><td>{a.bytes:,}</td>"
            f"<td class=mono>{_e(a.sha256[:16])}…</td></tr>" for a in s.artifacts) + "</table>")
    if s.logs:
        tails = []
        for rel in s.logs:
            p = run_dir / rel
            if p.is_file():
                lines = p.read_text(errors="replace").splitlines()
                tails.append(f"<details><summary>{_e(rel)} ({len(lines)} lines, last 25 shown)</summary><pre>"
                             + _e("\n".join(lines[-25:])) + "</pre></details>")
        rows.append("<h4>Logs</h4>" + "".join(tails))
    dur = f"{s.duration_s:.1f} s" if s.duration_s is not None else ""
    return (f"<section class=step><div class=head><span class=num>{n}</span><b>{_e(s.title)}</b>"
            f"<code>{_e(s.action)}</code><span class=mode>{_e(s.mode)}</span>"
            f"<span class=state style='background:{color}'>{_e(s.state.replace('_', ' '))}</span>"
            f"<span class=dur>{dur}</span></div>{''.join(rows)}</section>")


_CSS = """body{font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#111827;max-width:1180px;
margin:24px auto;padding:0 16px;background:#fff}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px;
border-bottom:1px solid #e5e7eb;padding-bottom:4px}h4{margin:12px 0 4px;font-size:13px;color:#374151}
table{border-collapse:collapse;width:100%;font-size:12.5px}td,th{border:1px solid #e5e7eb;padding:4px 6px;
text-align:left;vertical-align:top}th{background:#f9fafb}.ok{color:#15803d;font-weight:600}.ko{color:#b91c1c;
font-weight:600}.src,.kind{color:#6b7280;font-size:11.5px}.mono{font-family:ui-monospace,monospace}
pre{background:#f9fafb;border:1px solid #e5e7eb;padding:8px;overflow-x:auto;font-size:12px;white-space:pre-wrap}
.err{background:#fef2f2;border-color:#fecaca;color:#7f1d1d}.conf{background:#eef2ff;border:1px solid #c7d2fe;
padding:6px 8px;margin:6px 0;font-size:13px}.step{border:1px solid #e5e7eb;border-radius:8px;padding:10px 14px;
margin:12px 0}.head{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.num{background:#111827;color:#fff;
border-radius:50%;width:22px;height:22px;display:inline-flex;align-items:center;justify-content:center;font-size:12px}
.state{color:#fff;border-radius:10px;padding:1px 9px;font-size:12px}.mode{color:#6b7280;font-size:12px}
.dur{margin-left:auto;color:#6b7280}figure{margin:8px 0}img{max-width:100%;border:1px solid #e5e7eb}
figcaption{font-size:12.5px;color:#374151}.verdict{font-size:16px;padding:10px 12px;border-radius:8px;margin:12px 0;
color:#fff}.miss{color:#b91c1c}code{font-size:12px}"""


def write_report(run: RunRecord, path: str | Path) -> Path:
    run_dir = Path(run.folder)
    wf = run.workflow
    color = _STATE_COLOR.get(run.state, "#6b7280")
    inputs = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)} {_e((wf.get('inputs', {}).get(k) or {}).get('unit', ''))}"
                     f"</td><td>{_e((wf.get('inputs', {}).get(k) or {}).get('description', ''))}</td></tr>"
                     for k, v in run.inputs.items())
    tools = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v or 'not found')}</td></tr>"
                    for k, v in (run.environment.get("tools") or {}).items())
    crit = "".join(f"<tr><td class={'ok' if c.passed else 'ko'}>{'MET' if c.passed else 'NOT MET'}</td>"
                   f"<td>{_e(c.name)}</td><td>{_e(c.detail)}</td><td class=src>{_e(c.source)}</td></tr>"
                   for c in run.criteria) or "<tr><td colspan=4>not evaluated (the run stopped before)</td></tr>"
    limits = "".join(f"<li>{_e(x)}</li>" for x in wf.get("limits", []))
    body = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport
content="width=device-width,initial-scale=1"><title>Evidence report · {_e(wf.get('title'))}</title>
<style>{_CSS}</style></head><body>
<h1>{_e(wf.get('title'))}</h1>
<div class=src>Workflow <code>{_e(wf.get('id'))}</code> · run <code>{_e(run.run_id)}</code> · started {_t(run.created)}
 · finished {_t(run.finished)} · host {_e(run.environment.get('host'))}</div>
<div class=verdict style="background:{color}">Run {_e(run.state.replace('_', ' ').upper())}: {_e(run.message)}</div>
<p>{_e(wf.get('description'))}</p>
<h2>Inputs</h2><table><tr><th>name</th><th>value</th><th>meaning</th></tr>{inputs}</table>
<h2>Acceptance criteria</h2><table><tr><th></th><th>criterion</th><th>measured</th><th>computed by</th></tr>{crit}</table>
<h2>Steps</h2>{''.join(_step_html(run_dir, s, i + 1) for i, s in enumerate(run.steps))}
<h2>Tools</h2><table><tr><th>tool</th><th>version</th></tr>{tools}</table>
<p class=src>{_e(run.environment.get('platform'))}, Python {_e(run.environment.get('python'))}</p>
<h2>Limits of this workflow</h2><ul>{limits}</ul>
<p class=src>Every number, log and image above was produced by this run; images are labelled with what they are.
A step is marked passed only if it ran and every check on it held. Reproduce: run the same workflow with the same
inputs (run.json lists them) on the same tool versions.</p>
</body></html>"""
    path = Path(path)
    path.write_text(body, encoding="utf-8")
    return path


_STATUS = {"passed": "done", "rejected": "done", "failed": "failed", "blocked": "failed", "declined": "failed",
           "awaiting_confirmation": "partial", "paused": "partial", "running": "running", "pending": "planned"}


def _scalar(v: Any) -> float | str | None:
    if v is None or isinstance(v, bool):
        return None if v is None else str(v)
    if isinstance(v, (int, float)):
        return float(v) if v == v and abs(v) != float("inf") else str(v)
    return json.dumps(v, default=str) if isinstance(v, (list, dict)) else str(v)


def to_study_result(run: RunRecord):
    """The run as a :class:`pinneapple_data.contracts.StudyResult` (``study_type="runs"``, ``app="guiagent"``).

    Checks of every step and the acceptance criteria become ``CheckRecord`` (PASS/FAIL with the measured value, or
    the step record as evidence); produced files become ``FileRef`` with SHA-256; numeric step outputs and criterion
    values become metrics; the workflow's limits are carried over. Inputs, decision, step states and the images go in
    ``metadata``."""
    from pinneapple_data.contracts import CheckRecord, FileRef, Provenance, StudyResult

    def record(name: str, c: Any, where: str) -> CheckRecord:
        value = _scalar(c.value)
        return CheckRecord(name=name, status="PASS" if c.passed else "FAIL", value=value,
                           threshold=_scalar(c.threshold), evidence=where if value is None else "",
                           detail=(c.detail + (f" [{c.source}]" if c.source else ""))[:2000])

    metrics: dict[str, float | int | str | None] = {}
    for s in run.steps:
        for k, v in s.outputs.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics[f"{s.id}.{k}"] = v
    for c in run.criteria:
        if isinstance(c.value, (int, float)) and not isinstance(c.value, bool):
            metrics[f"criterion.{c.name}"] = c.value
    checks = [record(f"{s.id}: {c.name}", c, f"run.json#steps/{s.id}") for s in run.steps for c in s.checks
              if s.action != "verify.criteria"]
    checks += [record(f"criterion: {c.name}", c, "run.json#criteria") for c in run.criteria]
    artifacts = []
    for s in run.steps:
        for a in s.artifacts:
            p = Path(run.folder) / a.path
            is_case = a.kind == "dir" and (p / "system" / "controlDict").is_file()
            fmt = "openfoam" if is_case else "other" if a.kind == "dir" else \
                {".csv": "csv", ".json": "json", ".vtk": "vtk", ".vtu": "vtk"}.get(p.suffix.lower(), "other")
            artifacts.append(FileRef(path=a.path, format=fmt, sha256=a.sha256, size=a.bytes,
                                     description=f"{s.id}{' (folder)' if a.kind == 'dir' else ''}"))
    try:
        from importlib.metadata import version
        lib = version("pinneapple")
    except Exception:
        lib = ""
    decision = {"passed": "ACCEPTED", "rejected": "REJECTED"}.get(run.state, "INCOMPLETE")
    return StudyResult(
        id=run.run_id, study_type="runs", app="guiagent", title=run.workflow.get("title", ""),
        status=_STATUS.get(run.state, "partial"), metrics=metrics, checks=checks, artifacts=artifacts,
        limits=list(run.workflow.get("limits", [])),
        provenance=Provenance(app="guiagent", study_id=run.run_id, source=f"workflow {run.workflow.get('id')}",
                              run_ids=[run.run_id], library_version=lib,
                              notes="pinneapple_tools.automation; report.html and run.json hold the full record"),
        metadata={"decision": decision, "state": run.state, "message": run.message, "inputs": run.inputs,
                  "workflow": run.workflow.get("id"), "tools": run.environment.get("tools"),
                  "steps": [{"id": s.id, "action": s.action, "mode": s.mode, "state": s.state,
                             "duration_s": s.duration_s, "confirmation": s.confirmation} for s in run.steps],
                  "evidence": [{"step": s.id, "path": e.path, "kind": e.kind, "caption": e.caption,
                                "sha256": e.sha256} for s in run.steps for e in s.evidence],
                  "report": "report.html", "record": "run.json"})


def write_study_result(run: RunRecord, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(to_study_result(run).to_json(), encoding="utf-8")
    return path
