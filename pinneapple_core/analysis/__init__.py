"""The analysis contract: what a calculation takes, what it gives back, and the sentence it words.

A calculation becomes an *analysis* when it declares itself, and declaring it buys the four things a report, a form,
an app and an assistant otherwise each re-derive by hand:

* the **inputs** with their unit, their range of validity and their name in a sentence;
* the **outputs** with units, checked after the call;
* a **result envelope** that never raises at the caller — a missing value, a NaN, a value outside the model or a
  bug inside it comes back as a status (``ok``/``warning``/``bad``/``not_applicable``/``insufficient_data``/
  ``error``) with a reason a person can act on;
* one **sentence**, written once next to the code that knows it is true, in English or Portuguese, with the numbers
  in the reader's format.

    from pinneapple_core.analysis import analysis, run_analysis, InputSpec, OutputSpec, tr

    @analysis("pipe.mean_velocity",
              label=tr("velocidade média", "mean velocity"),
              inputs={"Q": InputSpec(unit="m³/s", gt=0, label=tr("vazão", "flow rate")),
                      "D": InputSpec(unit="m", gt=0, label=tr("diâmetro", "diameter"))},
              outputs={"U": OutputSpec(unit="m/s", label=tr("velocidade", "velocity"))},
              describe="U = {U:.3} m/s in a {D:.3} m pipe",
              example={"Q": 0.01, "D": 0.1})
    def mean_velocity(Q, D):
        import math
        return {"U": 4 * Q / (math.pi * D * D), "D": D}   # what the sentence names travels with the result

    >>> run_analysis("pipe.mean_velocity", Q=-1, D=0.1).status.value
    'insufficient_data'

Nothing here is a physics model: this package is the envelope, the registry and the wording. The analyses the
library ships live with the models they wrap (``pinneapple_physics.closed_form.analyses``).
"""
from __future__ import annotations

from .check import check_all, check_analysis, junk_calls
from .contracts import (
    Finding,
    InputSpec,
    Maturity,
    OutputSpec,
    Reference,
    Result,
    Severity,
    Status,
)
from .registry import (
    BUILTIN_MODULES,
    Analysis,
    AnalysisError,
    DuplicateAnalysis,
    InvalidAnalysis,
    NotEnoughData,
    analyses,
    analysis,
    analysis_card,
    get_analysis,
    register_analysis,
    run_analysis,
    unregister_analysis,
)
from .text import (
    counted,
    fmt_number,
    fmt_percent,
    format_template,
    language,
    plural,
    say,
    text_of,
    tr,
    use_language,
)

__all__ = [
    # declaring and running
    "analysis", "analyses", "get_analysis", "run_analysis", "analysis_card", "register_analysis",
    "unregister_analysis", "Analysis", "BUILTIN_MODULES",
    # the envelope
    "Result", "Finding", "Status", "Severity", "Maturity", "Reference", "InputSpec", "OutputSpec", "NotEnoughData",
    # the data-hygiene kit
    "check_analysis", "check_all", "junk_calls",
    # text and numbers in the reader's language
    "tr", "text_of", "say", "language", "use_language", "fmt_number", "fmt_percent", "plural", "counted",
    "format_template",
    # refusals
    "AnalysisError", "DuplicateAnalysis", "InvalidAnalysis",
]
