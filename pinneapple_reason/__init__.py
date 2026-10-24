"""pinneapple_reason: the chain of whys of an engineering design, computed.

Why does a roof slope? Because flat tiles let the rain through the laps: capillarity, wind and the ponded storm push
water higher than a flat lap rises. Why a ceiling? Because on a clear night the tile radiates to the sky and its
underside drops below the dew point of the room air. This package makes that reasoning explicit and quantitative for
any project:

    requirement -> naive concept -> physical check (numbers) -> failure mode found (mechanism + the number that
    proves it) -> countermeasure (new element, with cost and expected effect) -> check again -> ... -> all pass

    from pinneapple_reason import reason, cases, write_report
    chain = reason(*cases.roof())
    print(chain.explain())                          # version 0 -> version N, and why
    write_report(chain, "roof_report")              # chain.json, report.md / .html, timeline and margin figures

Parts: :mod:`.chain` (data model), :mod:`.library` and :mod:`.modes` (failure modes and countermeasures by physical
domain), :mod:`.engine` (the loop), :mod:`.trust` (trust report and six-stage card), :mod:`.report`, :mod:`.cases`,
:mod:`.llm` (an LLM may propose candidates; it never supplies a number).
"""
from __future__ import annotations

from . import cases
from .chain import (
    Chain,
    Concept,
    Finding,
    Option,
    Requirement,
    Step,
    Validity,
    Verification,
    text_of,
    tr,
)
from .engine import Advisor, Reasoner, reason
from .library import (
    LIBRARY,
    Countermeasure,
    FailureMode,
    Infeasible,
    Library,
    Need,
    register_countermeasure,
    register_mode,
)

__all__ = ["Chain", "Concept", "Finding", "Option", "Requirement", "Step", "Validity", "Verification", "tr",
           "text_of", "Reasoner", "reason", "Advisor", "LIBRARY", "Library", "FailureMode", "Countermeasure",
           "Infeasible", "Need", "register_mode", "register_countermeasure", "cases", "write_report", "trust_report",
           "run_case", "verify", "describe_library"]


def write_report(chain: Chain, out_dir, lang: str = "pt", **kw):
    """Write chain.json, report.md, report.html, the figures and the trust card to ``out_dir``."""
    from .report import write_report as _w
    return _w(chain, out_dir, lang=lang, **kw)


def trust_report(chain: Chain, **kw):
    from .trust import trust_report as _t
    return _t(chain, **kw)


def run_case(name: str, **kw) -> Chain:
    """Reason through one of :data:`cases.CASES`."""
    return reason(*cases.get(name), **kw)


def verify(concept: Concept, requirements) -> list[Verification]:
    """Run every failure mode of ``requirements`` on ``concept`` (no countermeasures)."""
    return Reasoner().verify(concept, requirements)


def describe_library(lang: str = "pt") -> dict:
    """The library: failure modes by domain and their countermeasures."""
    return LIBRARY.describe(lang)
