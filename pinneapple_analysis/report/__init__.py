"""One-page engineering report for an experiment (issue #919).

    from pinneapple_analysis.report import experiment_report
    html, pdf = experiment_report("examples/end_to_end/ahmed_body")     # -> <folder>/report/{index.html,report.pdf}

    python -m pinneapple_analysis.report examples/condensation_falsification --theme dark

The layout is fixed, so two reports can be read side by side: header (title, the question, date, commit, hardware,
wall time, cost), three to six KPI cards with their interval, reference and verdict, the verdict table read from the
run's own checks, the figures the manifest names, uncertainty and validity (the nine-item protocol checklist of
:mod:`pinneapple_lab.quality` and the six-stage trust card of :mod:`pinneapple_lab.curation`), and provenance with
the reproduce command.

It adds no third format: every section is a view of something the repository already computes -- the manifest
(:mod:`pinneapple_lab.manifest`), the curation card, the quality checklist, the run's ``costs.json`` -- and a number
no check names is shown as *no check*, never as a pass. See :mod:`.experiment` for what each section reads.
"""
from __future__ import annotations

from .experiment import (
    SCHEMA,
    Report,
    ReportError,
    experiment_report,
    gather,
    render_html,
    render_pdf,
)

__all__ = ["SCHEMA", "Report", "ReportError", "experiment_report", "gather", "render_html", "render_pdf"]
