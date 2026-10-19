"""pinneapple_veriphysics — the verification and evidence layer of Veriphysics, as part of PINNeAPPle.

What a run produces and how it is reported, built only from objects the analysis modules already computed (no number or verdict
of its own):

=========================  ==============================================================================================
``decision``               ``DecisionRecord``: recommendation, trust score and coverage, per-check evidence, alternatives
``evidence_report``        the Evidence Report as a PDF (needs the optional ``reportlab``)
``applicability``          the applicability map: evidence chain, 8-item checklist and the variable envelope that was tested
``scope``                  the scope of use: which uses a result serves, where it fails and why (beyond approve / reject)
``robustness``             measured robustness studies (does the verdict survive perturbing the problem?)
``recommend``              formulate a problem and recommend a solver family and architecture from the verification catalog
``execution_log``          a collector for what each step did (shown next to the result)
``example_cases``          worked example problems for the recommender
``ledger``                 the evidence ledger: a check reads "passed" only with an intact record of its run
``validation_lab``         ``pinneapple check`` grouped into numerical verification, validation against data and
                           physical consistency, with a ledger entry per outcome (the Validation Lab app)
``evidence_package``       engineering evidence report (HTML, PDF) and zip package from other tools' outputs
=========================  ==============================================================================================

Veriphysics (the product) keeps the job queue, the HTTP API, billing and the web app, and imports this package.
"""
from __future__ import annotations

import importlib as _importlib

__all__ = ["decision", "evidence_report", "applicability", "scope", "robustness", "recommend", "execution_log", "example_cases",
           "ledger", "validation_lab", "evidence_package"]


def __getattr__(name: str):
    if name in __all__:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
