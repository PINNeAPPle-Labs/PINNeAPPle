"""Validation dossier: the case, its evidence, its maturity and its decision, as one document an engineer can file.

The dossier is the artefact the loop produces -- the thing a reviewer signs, a colleague reads, or a customer's quality
process accepts as *supporting evidence*. It is deliberately plain and complete rather than persuasive: declared use,
the criteria that were set, the checks that ran (and those that did not), the reference comparisons with their
measured error against the tolerance, the domain of applicability, the uncertainty and its limitations, the expert
decisions with their scope, the maturity on both axes with what is missing, and the provenance needed to reproduce it.

It is **not** a certification: :data:`DISCLAIMER` says so on the document itself. What it establishes is narrower and
more useful -- what was verified, what was validated, for which scope, by whom, with which evidence.

::

    from pinneapple_veriphysics.dossier import ValidationDossier

    d = ValidationDossier.from_case(case)
    d.write("case.html")                      # self-contained HTML
    print(d.render_text())                    # the same content as text
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .findings import FAIL, NOT_RUN, PASS, WARNING
from .maturity import DIMENSIONS
from .validation_case import CaseAssessment, ValidationCase

__all__ = ["DISCLAIMER", "ValidationDossier", "DOSSIER_SCHEMA"]

DOSSIER_SCHEMA = "pinneapple-validation-dossier/v1"

DISCLAIMER = ("Not a certification. This dossier records what was checked, what was compared, what was decided, by "
              "whom, with which evidence and inside which limits. A status of 'validated' is a documented engineering "
              "judgement for a declared scope, not a proof of validity, and it does not extend beyond that scope or "
              "survive a change of model version, inputs or intended use.")

_CSS = """body{font:14px/1.55 system-ui,sans-serif;color:#172033;max-width:1040px;margin:24px auto;padding:0 16px}
h1{font-size:22px;margin:0 0 2px}h2{font-size:16px;margin:24px 0 6px;border-bottom:1px solid #dde3ec;padding-bottom:3px}
.muted{color:#5d6b82}.note{background:#fff7e6;border:1px solid #f5d38a;border-radius:8px;padding:8px 12px;margin:10px 0}
.bad{background:#fef2f2;border-color:#fca5a5}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}th,td{border-bottom:1px solid #e6eaf0;
padding:4px 6px;text-align:left;vertical-align:top}th{background:#f6f8fb}
.pass{color:#15803d;font-weight:600}.fail{color:#b91c1c;font-weight:700}.warn{color:#b45309;font-weight:600}
.notrun{color:#64748b;font-style:italic}.status{font-size:15px;font-weight:700;letter-spacing:.02em}
pre{background:#f6f8fb;border:1px solid #e6eaf0;border-radius:6px;padding:8px;overflow:auto;font-size:12px}
ul,ol{margin:6px 0 6px 18px;padding:0}li{margin:2px 0}code{font-size:12px}"""

_CLS = {PASS: "pass", FAIL: "fail", WARNING: "warn", NOT_RUN: "notrun"}
_STATUS_CLS = {"validated": "pass", "failed": "fail", "review required": "warn", "unvalidated": "notrun"}


@dataclass(frozen=True)
class ValidationDossier:
    """The document of one case at its latest revision."""
    case: ValidationCase
    assessment: CaseAssessment

    @classmethod
    def from_case(cls, case: ValidationCase, assessment: CaseAssessment | None = None) -> ValidationDossier:
        return cls(case=case, assessment=assessment or case.assess())

    # ------------------------------------------------------------------ structure
    def improvement_tasks(self) -> list[dict[str, Any]]:
        """What to do next: the open requirements of both axes, lowest level first."""
        return self.assessment.maturity.improvement_actions()

    def sections(self) -> list[str]:
        return ["Declared use", "Status and findings", "Checks recorded", "Reference evidence", "Applicability",
                "Uncertainty", "Maturity", "Expert decisions", "Limitations", "RunProvenance", "Revision history"]

    def to_dict(self) -> dict[str, Any]:
        c = self.case
        return {
            "schema": DOSSIER_SCHEMA, "disclaimer": DISCLAIMER, "title": c.title, "case_id": c.case_id,
            "requested_use": c.requested_use, "model": c.model.to_dict(), "problem": c.problem.to_dict(),
            "profile": c.profile.to_dict(), "profile_explanation": c.profile.describe(),
            "assessment": self.assessment.to_dict(), "checks": [k.to_dict() for k in c.checks],
            "references": [r.to_dict() for r in c.references],
            "applicability": None if c.applicability is None else c.applicability.to_dict(),
            "uncertainty": None if c.uncertainty is None else c.uncertainty.to_dict(),
            "controls": c.controls.to_dict(), "limitations": list(c.limitations),
            "reviews": [r.to_dict() for r in c.reviews], "provenance": c.provenance.to_dict(),
            "improvement_tasks": self.improvement_tasks(),
            "history": [r.to_dict() for r in c.history if r.snapshot is not None] if c.history else [],
            "history_problems": c.history_problems(),
        }

    # ------------------------------------------------------------------ text
    def render_text(self) -> str:
        c, a = self.case, self.assessment
        lines = [
            f"VALIDATION DOSSIER -- {c.title}",
            f"Case {c.case_id} | model {c.model.name} {c.model.version} | requested use: "
            f"{c.requested_use or 'not stated'}",
            f"Status: {a.status.upper()} (from {a.status_source})",
            f"  {a.reason}",
            "", DISCLAIMER, "",
            "== Declared use ==", c.profile.describe(), "",
            f"Problem: {c.problem.description}"
            + (f" | PDE: {c.problem.pde}" if c.problem.pde else "")
            + (f" | parameters: {dict(c.problem.parameters)}" if c.problem.parameters else ""),
            "", "== Findings ==",
        ]
        lines += [f.render() for f in a.findings] or ["(none)"]
        lines += ["", "== Checks recorded =="]
        for k in c.checks:
            st = "not run" if k.passed is None else ("passed" if k.passed else "FAILED")
            val = "" if k.value is None else f" value {k.value:.6g}"
            lim = "" if k.threshold is None else f" (limit {k.threshold:.6g})"
            lines.append(f"- [{st}] {k.check_id} [{k.category}]{val}{lim}"
                         + (f" -- {k.message}" if k.message else ""))
        if not c.checks:
            lines.append("- (no check recorded)")
        lines += ["", "== Reference evidence =="]
        for r in c.references:
            tol = r.tolerance_pct if r.tolerance_pct is not None else c.profile.target_error_pct
            lines.append(f"- [{r.evidence_id}] {r.kind}: {r.metric} = {r.value_pct:g} % (tolerance {tol:g} %, "
                         f"{'inside' if r.within(c.profile.target_error_pct) else 'ABOVE'}); "
                         f"{'independent' if r.independent else 'not independent'}; "
                         f"source: {r.source or 'not stated'}")
        if not c.references:
            lines.append("- (no comparison against an independent reference)")
        lines += ["", "== Applicability =="]
        if c.applicability is None:
            lines.append("- not assessed")
        else:
            ap = c.applicability
            lines.append(f"- requested condition: {ap.requested_condition or 'not named'}")
            lines.append(f"- verified ranges: "
                         + (", ".join(f"{n} in [{lo:g}, {hi:g}]" for n, lo, hi in ap.verified_ranges) or "none"))
            lines.append("- outside the verified range: "
                         + (", ".join(ap.outside_variables) if ap.outside_variables else "none"))
            if ap.supported_margin is not None:
                lines.append(f"- measured extrapolation margin: +{ap.supported_margin:.0%}")
            lines += [f"- note: {n}" for n in ap.notes]
        lines += ["", "== Uncertainty =="]
        if c.uncertainty is None:
            lines.append("- none recorded: the prediction is a point estimate")
        else:
            u = c.uncertainty
            bits = [f"method: {u.method}",
                    "calibrated against reference data" if u.calibrated else "NOT calibrated (a disagreement bar)"]
            if u.spread_pct is not None:
                bits.append(f"spread {u.spread_pct:g} % of field scale")
            if u.coverage is not None:
                bits.append(f"coverage {u.coverage:.2f}" + ("" if u.target_coverage is None
                                                           else f" vs target {u.target_coverage:.2f}"))
            lines.append("- " + "; ".join(bits))
            if u.limitation:
                lines.append(f"- limitation: {u.limitation}")
        lines += ["", "== Maturity ==", a.maturity.render()]
        lines += ["", "== Expert decisions =="]
        for r in c.reviews:
            lines.append(f"- {r.decided_at} {r.reviewer}"
                         + (f" ({r.role})" if r.role else "") + f": {r.decision} -- {r.justification}"
                         + (f" | scope: {dict(r.scope)}" if r.scope else " | scope: not stated"))
        if not c.reviews:
            lines.append("- none recorded: automated checks alone do not authorize use")
        lines += ["", "== Limitations =="]
        lines += [f"- {x}" for x in c.limitations] or ["- none recorded (that is itself a gap, not a clean bill)"]
        lines += ["", "== RunProvenance =="]
        p = c.provenance
        lines.append(f"- tool: {p.tool or '?'} {p.tool_version}" + (f"; python {p.python}" if p.python else ""))
        lines.append(f"- inputs sha256: {p.data_sha256 or 'not recorded'}")
        lines.append(f"- code sha256: {p.code_sha256 or 'not recorded'}")
        lines.append(f"- seeds: {list(p.seeds) or 'not recorded'}")
        lines.append(f"- tolerances: {dict(p.tolerances) or 'not recorded'}")
        if p.compute:
            lines.append(f"- compute: {dict(p.compute)}")
        problems = c.history_problems()
        lines += ["", "== Revision history =="]
        for r in c.history:
            lines.append(f"- rev {r.revision} {r.created_at} [{r.status}] {r.action}"
                         + (f" ({r.actor})" if r.actor else "") + f" sha256 {r.snapshot_sha256[:12]}")
        lines += [f"- HISTORY PROBLEM: {x}" for x in problems]
        tasks = self.improvement_tasks()
        if tasks:
            lines += ["", "== What to do next =="]
            lines += [f"- ({t['priority']}) [{t['dimension']} L{t['level']}] {t['action']}" for t in tasks]
        return "\n".join(lines)

    # ------------------------------------------------------------------ markdown
    def render_markdown(self) -> str:
        c, a = self.case, self.assessment
        md = [f"# Validation dossier — {c.title}", "",
              f"**Case** `{c.case_id}` · **model** `{c.model.name} {c.model.version}` · "
              f"**requested use** {c.requested_use or 'not stated'}", "",
              f"**Status: {a.status.upper()}** ({a.status_source}) — {a.reason}", "",
              f"> {DISCLAIMER}", "", "## Declared use", "", "```", c.profile.describe(), "```", "",
              "## Findings", ""]
        for f in a.findings:
            md += [f"### {f.finding_id} — {f.title} `{f.status.upper()}`", "",
                   f"- **What happened:** {f.what_happened}", f"- **Why it matters:** {f.why_it_matters}",
                   f"- **What is still unknown:** {f.what_is_unknown}"]
            if f.actions:
                md.append("- **Recommended next steps:**")
                md += [f"  {i}. {x}" for i, x in enumerate(f.actions, 1)]
            if f.evidence:
                md.append(f"- **Evidence:** {', '.join(f'`{e}`' for e in f.evidence)}")
            md.append("")
        md += ["## Checks recorded", "", "| Check | Group | Status | Value | Limit |", "|---|---|---|---|---|"]
        for k in c.checks:
            st = "not run" if k.passed is None else ("passed" if k.passed else "failed")
            md.append(f"| {k.check_id} | {k.category} | {st} | "
                      f"{'' if k.value is None else f'{k.value:.6g}'} | "
                      f"{'' if k.threshold is None else f'{k.threshold:.6g}'} |")
        md += ["", "## Reference evidence", ""]
        for r in c.references:
            tol = r.tolerance_pct if r.tolerance_pct is not None else c.profile.target_error_pct
            md.append(f"- `{r.evidence_id}` {r.kind}: {r.metric} = **{r.value_pct:g} %** against {tol:g} % "
                      f"({'inside' if r.within(c.profile.target_error_pct) else 'above'}); {r.source or 'source not stated'}")
        if not c.references:
            md.append("- no comparison against an independent reference")
        md += ["", "## Maturity", "", "```", a.maturity.render(), "```", "", "## Expert decisions", ""]
        md += [f"- {r.decided_at} **{r.reviewer}**: {r.decision} — {r.justification}"
               + (f" (scope: `{dict(r.scope)}`)" if r.scope else "") for r in c.reviews] or ["- none recorded"]
        md += ["", "## Revision history", ""]
        md += [f"- rev {r.revision} {r.created_at} `{r.status}` {r.action} — sha256 `{r.snapshot_sha256[:12]}`"
               for r in c.history] or ["- no revision recorded"]
        md += [f"- **HISTORY PROBLEM:** {x}" for x in c.history_problems()]
        tasks = self.improvement_tasks()
        if tasks:
            md += ["", "## What to do next", ""]
            md += [f"- ({t['priority']}) [{t['dimension']} L{t['level']}] {t['action']}" for t in tasks]
        return "\n".join(md)

    # ------------------------------------------------------------------ html
    def render_html(self) -> str:
        e = html.escape
        c, a = self.case, self.assessment
        rows = []
        rows.append(f"<h1>Validation dossier &mdash; {e(c.title)}</h1>")
        rows.append(f"<p class='muted'>Case <code>{e(c.case_id)}</code> &middot; model "
                    f"<code>{e(c.model.name)} {e(c.model.version)}</code> &middot; requested use: "
                    f"{e(c.requested_use or 'not stated')}</p>")
        rows.append(f"<p class='status {_STATUS_CLS.get(a.status, '')}'>STATUS: {e(a.status.upper())} "
                    f"<span class='muted'>(from {e(a.status_source)})</span></p><p>{e(a.reason)}</p>")
        rows.append(f"<p class='note{e(' bad' if a.status == 'failed' else '')}'>{e(DISCLAIMER)}</p>")

        rows.append("<h2>Declared use</h2><pre>" + e(c.profile.describe()) + "</pre>")
        rows.append(f"<p class='muted'>Problem: {e(c.problem.description)}"
                    + (f" &middot; PDE: {e(c.problem.pde)}" if c.problem.pde else "")
                    + "</p>")

        rows.append("<h2>Findings</h2>")
        for f in a.findings:
            acts = "".join(f"<li>{e(x)}</li>" for x in f.actions)
            rows.append(
                f"<h3 class='{_CLS.get(f.status, '')}'>{e(f.finding_id)} &mdash; {e(f.title)} "
                f"<span class='muted'>[{e(f.status.upper())}]</span></h3>"
                f"<p><b>What happened:</b> {e(f.what_happened)}<br><b>Why it matters:</b> {e(f.why_it_matters)}<br>"
                f"<b>What is still unknown:</b> {e(f.what_is_unknown)}</p>"
                + (f"<p><b>Recommended next steps:</b></p><ol>{acts}</ol>" if acts else "")
                + (f"<p class='muted'>Evidence: <code>{e(', '.join(f.evidence))}</code></p>" if f.evidence else ""))

        rows.append("<h2>Checks recorded</h2><table><tr><th>Check</th><th>Group</th><th>Status</th><th>Value</th>"
                    "<th>Limit</th><th>Message</th></tr>")
        for k in c.checks:
            st = "not run" if k.passed is None else ("passed" if k.passed else "failed")
            rows.append(f"<tr><td>{e(k.check_id)}</td><td>{e(k.category)}</td>"
                        f"<td class='{_CLS.get(PASS if k.passed else (NOT_RUN if k.passed is None else FAIL), '')}'>"
                        f"{e(st)}</td><td>{'' if k.value is None else f'{k.value:.4g}'}</td>"
                        f"<td>{'' if k.threshold is None else f'{k.threshold:.4g}'}</td>"
                        f"<td>{e(k.message)}</td></tr>")
        rows.append("</table>" + ("" if c.checks else "<p class='notrun'>No check recorded.</p>"))

        rows.append("<h2>Reference evidence</h2><table><tr><th>Evidence</th><th>Kind</th><th>Quantity</th>"
                    "<th>Error</th><th>Tolerance</th><th>Verdict</th><th>Source</th><th>sha256</th></tr>")
        for r in c.references:
            tol = r.tolerance_pct if r.tolerance_pct is not None else c.profile.target_error_pct
            ok = r.within(c.profile.target_error_pct)
            rows.append(f"<tr><td>{e(r.evidence_id)}</td><td>{e(r.kind)}</td><td>{e(r.metric)}</td>"
                        f"<td>{r.value_pct:.4g} %</td><td>{tol:.4g} %</td>"
                        f"<td class='{'pass' if ok else 'fail'}'>{'inside' if ok else 'above'}</td>"
                        f"<td>{e(r.source)}</td><td><code>{e(r.artefact_sha256[:16])}</code></td></tr>")
        rows.append("</table>" + ("" if c.references else "<p class='notrun'>No independent reference recorded.</p>"))

        rows.append("<h2>Applicability</h2>")
        if c.applicability is None:
            rows.append("<p class='notrun'>Not assessed.</p>")
        else:
            ap = c.applicability
            rows.append(f"<p>Requested condition: {e(ap.requested_condition or 'not named')}<br>"
                        f"Verified ranges: {e(', '.join(f'{n} in [{lo:g}, {hi:g}]' for n, lo, hi in ap.verified_ranges) or 'none')}<br>"
                        f"Outside the verified range: <b>{e(', '.join(ap.outside_variables) or 'none')}</b></p>"
                        + ("<ul>" + "".join(f"<li>{e(n)}</li>" for n in ap.notes) + "</ul>" if ap.notes else ""))

        rows.append("<h2>Uncertainty</h2>")
        if c.uncertainty is None:
            rows.append("<p class='notrun'>None recorded: the prediction is a point estimate.</p>")
        else:
            u = c.uncertainty
            rows.append(f"<p>Method: {e(u.method)} &middot; "
                        f"{'calibrated against reference data' if u.calibrated else '<b>not calibrated</b>'}"
                        + (f" &middot; spread {u.spread_pct:g} % of field scale" if u.spread_pct is not None else "")
                        + "</p>" + (f"<p class='muted'>{e(u.limitation)}</p>" if u.limitation else ""))

        rows.append("<h2>Maturity</h2>")
        for dim in DIMENSIONS:
            ax = a.maturity.axes[dim]
            rows.append(f"<p><b>{e(dim.capitalize())} maturity: level {ax.level} &mdash; {e(ax.title)}</b>")
            if ax.next_level is not None:
                rows.append(f"<br><span class='muted'>to reach level {ax.next_level} ({e(ax.next_title)}):</span>")
                rows.append("</p><ul>" + "".join(f"<li>{e(g.statement)} &rarr; {e(g.how_to_meet)}</li>"
                                                 for g in ax.gaps) + "</ul>"
                             if ax.gaps else "</p><p class='muted'>nothing missing at this level</p>")
            else:
                rows.append("</p>")

        rows.append("<h2>Expert decisions</h2>")
        if c.reviews:
            rows.append("<table><tr><th>When</th><th>Reviewer</th><th>Decision</th><th>Justification</th>"
                        "<th>Scope</th></tr>")
            rows += [f"<tr><td>{e(r.decided_at)}</td><td>{e(r.reviewer)}</td><td>{e(r.decision)}</td>"
                     f"<td>{e(r.justification)}</td><td>{e(str(dict(r.scope)))}</td></tr>" for r in c.reviews]
            rows.append("</table>")
        else:
            rows.append("<p class='notrun'>None recorded: automated checks alone do not authorize use.</p>")

        rows.append("<h2>Limitations</h2><ul>"
                    + "".join(f"<li>{e(x)}</li>" for x in c.limitations)
                    + ("" if c.limitations else "<li class='notrun'>none recorded (a gap, not a clean bill)</li>")
                    + "</ul>")

        p = c.provenance
        rows.append("<h2>RunProvenance</h2><table>"
                    + "".join(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in (
                        ("tool", f"{p.tool} {p.tool_version}".strip()), ("python", p.python),
                        ("platform", p.platform), ("inputs sha256", p.data_sha256), ("code sha256", p.code_sha256),
                        ("seeds", str(list(p.seeds))), ("tolerances", str(dict(p.tolerances))),
                        ("compute", str(dict(p.compute))), ("provenance sha256", p.digest())))
                    + "</table>")

        rows.append("<h2>Revision history</h2><table><tr><th>Rev</th><th>When</th><th>Status</th><th>Action</th>"
                    "<th>Actor</th><th>Snapshot sha256</th></tr>")
        rows += [f"<tr><td>{r.revision}</td><td>{e(r.created_at)}</td><td>{e(r.status)}</td><td>{e(r.action)}</td>"
                 f"<td>{e(r.actor)}</td><td><code>{e(r.snapshot_sha256[:16])}</code></td></tr>" for r in c.history]
        rows.append("</table>")
        for x in c.history_problems():
            rows.append(f"<p class='note bad'>HISTORY PROBLEM: {e(x)}</p>")

        tasks = self.improvement_tasks()
        if tasks:
            rows.append("<h2>What to do next</h2><ol>")
            rows += [f"<li><b>[{e(t['priority'])}]</b> {e(t['action'])} <span class='muted'>"
                     f"({e(t['dimension'])} level {t['level']})</span></li>" for t in tasks]
            rows.append("</ol>")

        return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
                f"<title>Validation dossier -- {e(c.title)}</title><style>{_CSS}</style></head><body>"
                + "\n".join(rows) + "</body></html>")

    # ------------------------------------------------------------------ files
    def write(self, path: str | Path, fmt: str = "html") -> Path:
        """Write the dossier (``html``, ``md`` or ``txt``) and return the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = {"html": self.render_html, "md": self.render_markdown, "txt": self.render_text}[fmt]()
        p.write_text(text, encoding="utf-8")
        return p
