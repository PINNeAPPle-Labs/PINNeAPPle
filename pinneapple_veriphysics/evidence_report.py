"""Evidence Report PDF generator.

This is the sprint's "Evidence Report in PDF" deliverable -- see
``helm/docs/PINNeAPPle90/sprint1/04-escopo-minimo-mvps.md``, MVP 1's
minimal flow: Formulario -> Executar -> Capturar -> Validar -> Pontuar ->
**Evidence Report em PDF**.

This module renders EXACTLY the data a (see :class:`~pinneapple_veriphysics
.decision.DecisionRecord` already carries -- the same content
``DecisionRecord.render()`` produces as plain text, plus the per-check
confidence breakdown (``DecisionRecord.confidence_components``) -- into a
PDF. It computes nothing new and imports no verification/scoring code;
every number here was already produced by
the Veriphysics orchestrator's ``analyze()``. This matches the rest of
this repo's "thin layer over already-computed data" pattern, and its
anti-fabrication discipline: a check absent from ``confidence_components``
is rendered as "not run", never a fabricated score (see
``decision.ALL_CONFIDENCE_COMPONENT_NAMES`` and README.md's
"Anti-fabrication design").

Deliberately plain: a single-column, black-on-white document with no
charts, logos, or color -- an evidence document a reviewing engineer can
scan quickly, not a marketing artifact. This matches the sprint's own
scope note ("no polish beyond what credibility requires").

(Extracted from Veriphysics' orchestrator; Apache-2.0 like the rest of PINNeAPPle. Execution queue, API and billing stay in Veriphysics.)
"""
from __future__ import annotations

import io
from typing import Any, Dict, List

try:                                   # reportlab is optional: the Decision Record itself needs only the standard library
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer
    _SEM_REPORTLAB: "ImportError | None" = None
except ImportError as _e:  # pragma: no cover - exercised only where reportlab is missing
    _SEM_REPORTLAB = _e

from .decision import ALL_CONFIDENCE_COMPONENT_NAMES, DecisionRecord

__all__ = ["render_evidence_report_pdf"]


def _escape(text: Any) -> str:
    """Minimal HTML-entity escaping for reportlab's Paragraph mini-markup
    (which interprets '<', '>', '&' as markup) -- every string rendered
    here comes from real pipeline output (a problem description, a
    reasoning string, a source_summary), not from this module, so this is
    display safety, not a validation/fabrication concern."""
    return str(text if text is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="EvidenceTitle", parent=styles["Title"], alignment=TA_LEFT, spaceAfter=10))
    styles.add(ParagraphStyle(name="EvidenceH2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=4))
    styles.add(ParagraphStyle(name="EvidenceBody", parent=styles["Normal"], leading=14, spaceAfter=4))
    return styles


def render_evidence_report_pdf(decision: DecisionRecord) -> bytes:
    """Render *decision* (a real, already-computed ``DecisionRecord``) as
    an Evidence Report PDF and return the raw PDF bytes.

    Sections, in order: problem, recommended strategy + reasoning, trust
    score + coverage, guardrail verdict, the per-check confidence
    breakdown (one line per check, or "not run"), alternatives considered,
    external tools recommended, and the next best action -- the same
    content ``DecisionRecord.render()``'s plain-text report carries, laid
    out as a PDF instead of a terminal-friendly string.
    """
    if _SEM_REPORTLAB is not None:
        raise ImportError("The Evidence Report PDF needs reportlab: pip install reportlab") from _SEM_REPORTLAB
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        title=f"Veriphysics Evidence Report -- {decision.run_id}",
    )
    story: List[Any] = []

    story.append(Paragraph("Veriphysics Evidence Report", styles["EvidenceTitle"]))
    story.append(Paragraph(f"Run: {_escape(decision.run_id)}", styles["EvidenceBody"]))
    story.append(Paragraph(f"Problem: {_escape(decision.problem_description)}", styles["EvidenceBody"]))

    story.append(Paragraph("Recommended strategy", styles["EvidenceH2"]))
    story.append(Paragraph(_escape(decision.recommended_strategy), styles["EvidenceBody"]))
    story.append(Paragraph(_escape(decision.reasoning), styles["EvidenceBody"]))

    story.append(Paragraph("Trust score", styles["EvidenceH2"]))
    if decision.trust_score is None:
        story.append(Paragraph(
            "Not computed (no confidence checks were available for this run).", styles["EvidenceBody"],
        ))
    else:
        story.append(Paragraph(
            f"{decision.trust_score:.0f}/100 (coverage={decision.trust_coverage:.2f} -- fraction of the "
            "4 possible checks this run actually ran; read this alongside the score, never the score "
            "alone -- see the per-check breakdown below).",
            styles["EvidenceBody"],
        ))
    if decision.trustworthy is not None:
        story.append(Paragraph(
            f"Guardrail verdict: {'TRUSTWORTHY' if decision.trustworthy else 'NOT TRUSTWORTHY'}",
            styles["EvidenceBody"],
        ))

    story.append(Paragraph("Per-check confidence breakdown", styles["EvidenceH2"]))
    by_name: Dict[str, Dict[str, Any]] = {c["name"]: c for c in decision.confidence_components}
    rows = []
    for name in ALL_CONFIDENCE_COMPONENT_NAMES:
        comp = by_name.get(name)
        if comp is None:
            rows.append(f"{name}: not run")
        else:
            rows.append(f"{name}: {comp['score']:.2f} -- {comp['source_summary']}")
    story.append(ListFlowable(
        [ListItem(Paragraph(_escape(row), styles["EvidenceBody"])) for row in rows],
        bulletType="bullet", leftIndent=14,
    ))

    if decision.alternatives_considered:
        story.append(Paragraph("Alternatives considered", styles["EvidenceH2"]))
        story.append(ListFlowable(
            [
                ListItem(Paragraph(
                    _escape(f"{a.get('option', '?')}: {a.get('why_not', '')}"), styles["EvidenceBody"],
                ))
                for a in decision.alternatives_considered
            ],
            bulletType="bullet", leftIndent=14,
        ))

    if decision.external_tools_recommended:
        story.append(Paragraph("External tools also worth considering", styles["EvidenceH2"]))
        story.append(Paragraph(_escape(", ".join(decision.external_tools_recommended)), styles["EvidenceBody"]))
        if decision.external_tools_reasoning:
            story.append(Paragraph(_escape(decision.external_tools_reasoning), styles["EvidenceBody"]))

    if decision.applicability:
        a = decision.applicability
        story.append(Paragraph("Applicability map", styles["EvidenceH2"]))
        story.append(ListFlowable(
            [ListItem(Paragraph(_escape(f"[{i['status']}] {i['question']} -- {i['summary']}"), styles["EvidenceBody"]))
             for i in a["checklist"]],
            bulletType="bullet", leftIndent=14,
        ))
        for title, key in (("Directly verified", "verified"),
                           ("Inferred from the model (not directly checked)", "inferred"),
                           ("Unsupported -- do not extrapolate here", "unsupported")):
            story.append(Paragraph(title, styles["EvidenceH2"]))
            rows = a[key] or ["(none)"]
            story.append(ListFlowable(
                [ListItem(Paragraph(_escape(r), styles["EvidenceBody"])) for r in rows],
                bulletType="bullet", leftIndent=14,
            ))

    if decision.next_best_action:
        story.append(Paragraph("Next best action", styles["EvidenceH2"]))
        story.append(Paragraph(_escape(decision.next_best_action), styles["EvidenceBody"]))
    else:
        story.append(Spacer(1, 1))

    doc.build(story)
    return buf.getvalue()
