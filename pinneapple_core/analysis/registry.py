"""The analysis registry: every analysis declares what it takes, what it gives back and how it words the answer.

An *analysis* is one calculation a person asks for by name — the friction factor of a pipe, the base temperature of
a fin array, the head rise of a valve closure. Writing it as an analysis buys four things that a bare function does
not have:

* **it says what it takes** — every input carries its unit, its range of validity and a name in a sentence, so a
  form, a report and an assistant read the same declaration instead of guessing;
* **it says what it gives back** — the outputs, with units, checked after the call (an output the analysis did not
  return is an error, not a silently missing key);
* **it never raises at the caller** — a missing input, a NaN, a value outside the model's range or an exception
  inside the calculation becomes a :class:`~pinneapple_core.analysis.contracts.Result` with a status and a reason a
  person can act on;
* **it words its own finding** — one ``describe`` template, in English and Portuguese, with the numbers in the
  language's format, so the sentence a client reads is written once, next to the code that knows it is true.

    from pinneapple_core.analysis import InputSpec, OutputSpec, analysis, tr

    @analysis(
        "pipe.darcy_pressure_drop",
        label=tr("queda de pressão de Darcy–Weisbach", "Darcy–Weisbach pressure drop"),
        inputs={"f": InputSpec(gt=0, label=tr("fator de atrito", "friction factor")),
                "L": InputSpec(unit="m", gt=0, label=tr("comprimento", "length"))},
        outputs={"dp": OutputSpec(unit="Pa", label=tr("queda de pressão", "pressure drop"))},
        describe="dp = {dp:.0} Pa over {L:.1} m",
    )
    def darcy_pressure_drop(f, L, D, rho, U):
        return {"dp": f * (L / D) * 0.5 * rho * U * U}

    >>> from pinneapple_core.analysis import run_analysis
    >>> run_analysis("pipe.friction_factor", Re=45_000, rel_rough=4.5e-4).status.value
    'ok'

The registry is the same protocol as ``pp.register``: ``pp.register(obj, kind="analysis")``,
``pp.registered("analysis")`` and ``pp.resolve("analysis", name)`` reach these analyses too.
"""
from __future__ import annotations

import inspect
import numbers
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_core.jsonable import to_jsonable

from .contracts import (
    Finding,
    InputSpec,
    Maturity,
    OutputSpec,
    Reference,
    Result,
    Severity,
    Status,
    as_mapping,
    as_severity,
    as_status,
)
from .text import fmt_number, format_template, language, say, text_of, tr

__all__ = ["Analysis", "AnalysisError", "DuplicateAnalysis", "InvalidAnalysis", "NotEnoughData", "analysis",
           "analyses", "get_analysis", "run_analysis", "analysis_card", "register_analysis", "unregister_analysis",
           "BUILTIN_MODULES"]


class AnalysisError(Exception):
    """Anything the analysis registry refuses."""


class DuplicateAnalysis(AnalysisError, ValueError):
    """The id is taken (``replace=True`` on :func:`register_analysis` overwrites)."""


class InvalidAnalysis(AnalysisError, TypeError):
    """The declaration does not match the function it wraps."""


class NotEnoughData(Exception):
    """Raise inside an analysis when the inputs cannot answer the question it asks.

    The registry turns it into ``status="insufficient_data"`` with the message as the reason, which is the
    difference between a caller seeing "the window is 3 samples long, 30 are needed" and a traceback.
    """


#: modules whose analyses are registered the first time the registry is read; a missing one is not an error
#: (``pinneapple_physics`` is an optional part of an installation only in the sense that it is not imported yet)
BUILTIN_MODULES: tuple[str, ...] = ("pinneapple_physics.closed_form.analyses",)

_REGISTRY: dict[str, Analysis] = {}
_LOADED = False
_LOCK = threading.RLock()


@dataclass(frozen=True)
class Analysis:
    """One calculation, declared: what it takes, what it gives back, and the sentence that words its result."""

    id: str
    function: Callable[..., Any]
    label: Any = ""
    category: str = ""
    description: Any = ""
    inputs: Mapping[str, InputSpec] = field(default_factory=dict)
    outputs: Mapping[str, OutputSpec] = field(default_factory=dict)
    describe: Any = None                     # a tr() template, a plain string or callable(metrics, lang)
    maturity: Maturity = Maturity.EXPERIMENTAL
    references: Sequence[Reference] = ()
    limitations: Sequence[Any] = ()
    example: Mapping[str, Any] = field(default_factory=dict)   # one valid call, for the card and the smoke check

    # -- what it says about itself ------------------------------------------
    def label_of(self, lang: str | None = None) -> str:
        return text_of(self.label, lang) or self.id

    def sentence(self, metrics: Mapping[str, Any], lang: str | None = None) -> str:
        """The finding in one line: the ``describe`` template filled with the numbers, in the language asked for."""
        if callable(self.describe):
            try:
                return str(self.describe(dict(metrics), lang or language()))
            except Exception:  # noqa: BLE001 - a finding that cannot be worded must not break the result
                return self.label_of(lang)
        if self.describe:
            return format_template(text_of(self.describe, lang), metrics, lang)
        shown = ", ".join(f"{name} = {_show(value, lang)}" for name, value in metrics.items())
        return f"{self.label_of(lang)}: {shown}"

    def card(self, lang: str | None = None) -> dict[str, Any]:
        """Everything a form, a report or an assistant needs to call this analysis, as JSON-ready data."""
        return {
            "id": self.id,
            "label": self.label_of(lang),
            "category": self.category,
            "description": text_of(self.description, lang),
            "inputs": {name: spec.to_dict(lang) for name, spec in self.inputs.items()},
            "outputs": {name: spec.to_dict(lang) for name, spec in self.outputs.items()},
            "settings": list(self.settings()),
            "example": to_jsonable(dict(self.example)),
            "maturity": self.maturity.value,
            "references": [r.to_dict() for r in self.references],
            "limitations": [text_of(lim, lang) for lim in self.limitations],
        }

    def settings(self) -> list[str]:
        """The function's own optional parameters that are not declared inputs (``tol``, ``max_iter``, ...)."""
        return [name for name, p in inspect.signature(self.function).parameters.items()
                if name not in self.inputs and p.default is not inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]

    # -- running it ----------------------------------------------------------
    def run(self, /, strict: bool = False, **values: Any) -> Result:
        """Call the analysis. Input problems come back as a :class:`Result`, never as an exception.

        ``strict=True`` re-raises an exception from inside the calculation (for its own tests); a caller who is
        showing a person a result leaves it off, because "error: singular matrix" is a result the screen can show.
        """
        lang = language()
        declared = set(self.inputs)
        taken = declared | set(self.settings())
        problems: list[str] = []

        unknown = [name for name in values if name not in taken]
        if unknown:
            problems.append(say(tr(f"entrada desconhecida: {', '.join(sorted(unknown))}",
                                   f"unknown input: {', '.join(sorted(unknown))}"), lang))

        for name, spec in self.inputs.items():
            problems.extend(spec.check(name, values.get(name), lang))

        used = {name: value for name, value in values.items() if name in taken}
        if problems:
            return Result(id=self.id, status=Status.INSUFFICIENT_DATA, severity=Severity.NONE,
                          sentence=problems[0], metrics={}, inputs=_inputs(values), problems=tuple(problems),
                          references=self.references, limitations=self.limitations_of(lang),
                          maturity=self.maturity, language=lang)

        finding = Finding()
        try:
            returned = self.function(**values)
        except NotEnoughData as exc:
            return Result(id=self.id, status=Status.INSUFFICIENT_DATA, severity=Severity.NONE,
                          sentence=str(exc), metrics={}, inputs=_inputs(used), problems=(str(exc),),
                          references=self.references, limitations=self.limitations_of(lang),
                          maturity=self.maturity, language=lang)
        except Exception as exc:  # noqa: BLE001 - the caller gets a result; its own tests pass strict=True
            if strict:
                raise
            reason = f"{type(exc).__name__}: {exc}"
            return Result(id=self.id, status=Status.ERROR, severity=Severity.NONE, sentence=reason, metrics={},
                          inputs=_inputs(used), problems=(reason,), references=self.references,
                          limitations=self.limitations_of(lang), maturity=self.maturity, language=lang)

        if isinstance(returned, Finding):
            finding = returned
            metrics = dict(returned.metrics)
        else:
            metrics = _as_metrics(returned, self.outputs)
            if metrics is None:
                reason = say(tr(f"a análise devolveu {type(returned).__name__}, que não é um resultado",
                                f"the analysis returned {type(returned).__name__}, which is not a result"), lang)
                return Result(id=self.id, status=Status.ERROR, sentence=reason, problems=(reason,),
                              inputs=_inputs(used), references=self.references,
                              limitations=self.limitations_of(lang), maturity=self.maturity, language=lang)

        missing = [name for name in self.outputs if name not in metrics]
        if missing:
            reason = say(tr(f"a análise não devolveu {', '.join(missing)}",
                            f"the analysis did not return {', '.join(missing)}"), lang)
            return Result(id=self.id, status=Status.ERROR, sentence=reason, metrics=metrics,
                          inputs=_inputs(used), problems=(reason,), references=self.references,
                          limitations=self.limitations_of(lang), maturity=self.maturity, language=lang)

        sentence = self.sentence(metrics, lang)
        if finding.message:
            sentence = text_of(finding.message, lang) or sentence
        return Result(id=self.id, status=as_status(finding.status), severity=as_severity(finding.severity),
                      sentence=sentence, metrics=metrics, inputs=_inputs(used),
                      evidence=dict(finding.evidence), references=self.references,
                      limitations=self.limitations_of(lang) + [text_of(lim, lang) for lim in finding.limitations],
                      maturity=self.maturity, language=lang)

    def limitations_of(self, lang: str | None = None) -> list[str]:
        return [text_of(lim, lang) for lim in self.limitations if text_of(lim, lang)]

    def __call__(self, /, **values: Any) -> Result:
        return self.run(**values)

    def __repr__(self) -> str:
        return f"Analysis({self.id!r}, {len(self.inputs)} inputs, {len(self.outputs)} outputs)"


# --------------------------------------------------------------------------- declaring one
def analysis(_id: str | None = None, *, label: Any = "", category: str = "", description: Any = "",
             inputs: Mapping[str, InputSpec] | None = None, outputs: Mapping[str, OutputSpec] | None = None,
             describe: Any = None, maturity: Maturity | str = Maturity.EXPERIMENTAL,
             references: Sequence[Reference] = (), limitations: Sequence[Any] = (),
             example: Mapping[str, Any] | None = None, replace: bool = False):
    """Declare a function as an analysis and register it (usable bare or as ``@analysis(...)``).

    The declaration is checked against the function at import time, not at failure time: an input that is not a
    parameter of the function, a parameter without a default that nobody declared, ``*args`` or ``**kwargs`` all
    raise :class:`InvalidAnalysis` while the module is being imported, which is where a mistyped declaration is
    cheap to fix.
    """
    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        spec = Analysis(
            id=_id or function.__name__,
            function=function,
            label=label or function.__name__.replace("_", " "),
            category=category,
            description=description or (inspect.getdoc(function) or "").split("\n\n")[0].strip(),
            inputs=dict(inputs or {}),
            outputs=dict(outputs or {}),
            describe=describe,
            maturity=Maturity(maturity) if isinstance(maturity, str) else maturity,
            references=tuple(references),
            limitations=tuple(limitations),
            example=dict(example or {}),
        )
        _validate(spec)
        register_analysis(spec, replace=replace)
        function.analysis = spec  # type: ignore[attr-defined] - ``my_function.analysis`` is the declaration
        return function

    return decorate


def _validate(spec: Analysis) -> None:
    if not str(spec.id).strip():
        raise InvalidAnalysis("an analysis needs an id")
    parameters = inspect.signature(spec.function).parameters
    for name, parameter in parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise InvalidAnalysis(f"{spec.id}: *args/**kwargs cannot be declared as an analysis ({name})")
    if "strict" in spec.inputs:
        raise InvalidAnalysis(f"{spec.id}: 'strict' is the registry's own argument; rename the input")
    for name in spec.inputs:
        if name not in parameters:
            raise InvalidAnalysis(f"{spec.id}: input {name!r} is not a parameter of {spec.function.__name__} "
                                  f"(it takes {', '.join(parameters) or 'nothing'})")
        if not isinstance(spec.inputs[name], InputSpec):
            raise InvalidAnalysis(f"{spec.id}: input {name!r} is not an InputSpec")
    for name, parameter in parameters.items():
        if parameter.default is parameter.empty and name not in spec.inputs:
            raise InvalidAnalysis(f"{spec.id}: parameter {name!r} has no default and no InputSpec, so a caller "
                                  f"cannot know what it takes")
    for name, output in spec.outputs.items():
        if not isinstance(output, OutputSpec):
            raise InvalidAnalysis(f"{spec.id}: output {name!r} is not an OutputSpec")
    for name in spec.example:
        if name not in spec.inputs:
            raise InvalidAnalysis(f"{spec.id}: example value {name!r} is not a declared input")


def register_analysis(spec: Analysis, *, replace: bool = False) -> Analysis:
    """Add an :class:`Analysis` to the registry (what ``pp.register(obj, kind="analysis")`` calls)."""
    if not isinstance(spec, Analysis):
        raise InvalidAnalysis(f"expected an Analysis, got {type(spec).__name__}")
    with _LOCK:
        if spec.id in _REGISTRY and not replace:
            raise DuplicateAnalysis(f"analysis {spec.id!r} is already registered; pass replace=True to overwrite")
        _REGISTRY[spec.id] = spec
    return spec


def unregister_analysis(name: str) -> Analysis:
    """Remove an analysis; the ones the library ships stay unless a caller removes them on purpose."""
    with _LOCK:
        try:
            return _REGISTRY.pop(name)
        except KeyError as exc:
            raise AnalysisError(f"analysis {name!r} is not registered") from exc


def get_analysis(name: str) -> Analysis:
    """The analysis registered under ``name`` (the library's own are loaded on first use)."""
    _load_builtins()
    with _LOCK:
        try:
            return _REGISTRY[name]
        except KeyError as exc:
            known = ", ".join(sorted(_REGISTRY)) or "none"
            raise AnalysisError(f"analysis {name!r} is not registered; known: {known}") from exc


def analyses(category: str | None = None) -> list[str]:
    """Every registered analysis id, sorted (optionally only one category)."""
    _load_builtins()
    with _LOCK:
        names = sorted(_REGISTRY)
        if category is None:
            return names
        return [n for n in names if _REGISTRY[n].category == category]


def run_analysis(name: str, /, *, strict: bool = False, **values: Any) -> Result:
    """Run one analysis by id: ``run_analysis("pipe.friction_factor", Re=45_000)``."""
    return get_analysis(name).run(strict=strict, **values)


def analysis_card(name: str, lang: str | None = None) -> dict[str, Any]:
    """What the analysis takes and gives back, as JSON-ready data (the shape a form or an assistant fills)."""
    return get_analysis(name).card(lang)


def _load_builtins() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        _LOADED = True                      # before the imports: a module that imports this one must not loop
        import importlib

        for module in BUILTIN_MODULES:
            try:
                importlib.import_module(module)
            except ImportError:             # pragma: no cover - the physics package is part of the library
                continue


# --------------------------------------------------------------------------- helpers
def _as_metrics(returned: Any, outputs: Mapping[str, OutputSpec]) -> dict[str, Any] | None:
    """A calculation's return value as ``{output: value}``, or None when it cannot be read as one."""
    as_dict = as_mapping(returned)
    if as_dict is not None:
        return as_dict
    if isinstance(returned, numbers.Real) and not isinstance(returned, bool):
        name = next(iter(outputs), "value")
        return {name: float(returned)}
    if isinstance(returned, np.ndarray):
        name = next(iter(outputs), "value")
        return {name: float(returned) if returned.ndim == 0 else returned.tolist()}
    if isinstance(returned, Sequence) and not isinstance(returned, (str, bytes)):
        names = list(outputs) or [f"value{i}" for i in range(len(returned))]
        if len(names) == len(returned):
            return dict(zip(names, returned, strict=False))
    return None


def _inputs(values: Mapping[str, Any]) -> dict[str, Any]:
    """The inputs as a report can print them: an array is summarised, never dumped whole."""
    shown: dict[str, Any] = {}
    for name, value in values.items():
        if isinstance(value, np.ndarray) and value.size > 16:
            finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else value
            shown[name] = {"shape": list(value.shape),
                           "min": float(np.min(finite)) if finite.size else None,
                           "max": float(np.max(finite)) if finite.size else None}
        else:
            shown[name] = to_jsonable(value)
    return shown


def _show(value: Any, lang: str | None = None) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Real):
        return fmt_number(value, 4, lang)
    return "—" if value is None else str(value)
