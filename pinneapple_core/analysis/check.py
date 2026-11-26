"""The kit an analysis has to pass: junk in, a result out, and never an exception.

The rule of this library is that bad data never crashes — a dead sensor, a NaN, a blank field, a value outside the
model. It is easy to state and easy to lose, so it is checked the same way for every analysis instead of being
trusted: :func:`junk_calls` builds the hostile calls from the declaration alone, and :func:`check_analysis` runs
them and reports what went wrong.

    from pinneapple_core.analysis import check_analysis, get_analysis

    assert check_analysis(get_analysis("pipe.friction_factor")) == []

It is a test helper and a gate: a new analysis is not finished until its own module asserts that line, and a
registry-wide run (``python -m pinneapple_core.analysis.check``) says which ones would fail.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import Result
from .registry import Analysis, analyses, get_analysis

__all__ = ["junk_calls", "check_analysis", "check_all"]


def junk_calls(spec: Analysis) -> list[tuple[str, dict[str, Any]]]:
    """The hostile calls for ``spec``, as complete calls: the example with one thing broken, one thing at a time.

    The first one passes nothing at all (when the analysis needs anything): the form that was never filled in is
    the most common hostile input there is.
    """
    example = dict(spec.example)
    cases: list[tuple[str, dict[str, Any]]] = []
    if any(input_spec.required for input_spec in spec.inputs.values()):
        cases.append(("nothing at all", {}))

    def case(label: str, **values: Any) -> None:
        cases.append((label, {**example, **values}))

    for name, input_spec in spec.inputs.items():
        case(f"{name} missing", **{name: None})
        if input_spec.kind in {"number", "integer"}:
            case(f"{name} NaN", **{name: float("nan")})
            case(f"{name} +inf", **{name: float("inf")})
            case(f"{name} -inf", **{name: float("-inf")})
            case(f"{name} not a number", **{name: "twelve"})
            if input_spec.min is not None:
                case(f"{name} below its minimum", **{name: float(input_spec.min) - 1.0})
            if input_spec.gt is not None:
                case(f"{name} below its exclusive minimum", **{name: float(input_spec.gt)})
            if input_spec.max is not None:
                case(f"{name} above its maximum", **{name: float(input_spec.max) + 1.0})
            if input_spec.lt is not None:
                case(f"{name} above its exclusive maximum", **{name: float(input_spec.lt)})
        elif input_spec.kind == "array":
            case(f"{name} empty", **{name: np.asarray([], dtype=float)})
            case(f"{name} with a NaN", **{name: np.asarray([1.0, math.nan, 3.0])})
            case(f"{name} of text", **{name: ["a", "b"]})
        elif input_spec.kind == "text":
            case(f"{name} not text", **{name: 3})
    return cases


def check_analysis(spec: Analysis | str, *, cases: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
                   check_example: bool = True) -> list[str]:
    """Every way ``spec`` fails the kit — empty when it passes.

    Each hostile call must come back as a :class:`~pinneapple_core.analysis.contracts.Result` whose status is not
    ``ok`` (it has to say what it could not do) and whose sentence is not empty (a person has to be able to read
    why). The example must come back ``ok``: a declaration whose own example fails is a declaration that lies.
    """
    analysis = get_analysis(spec) if isinstance(spec, str) else spec
    problems: list[str] = []
    base = dict(analysis.example)

    missing = [name for name, input_spec in analysis.inputs.items()
               if input_spec.required and name not in base]
    if check_example and missing:
        problems.append(f"{analysis.id}: the example does not give {', '.join(missing)}")

    for label, values in (cases if cases is not None else junk_calls(analysis)):
        try:
            result = analysis.run(**dict(values))
        except Exception as exc:  # noqa: BLE001 - that is exactly what this kit is looking for
            problems.append(f"{analysis.id}: raised {type(exc).__name__} with {label}: {exc}")
            continue
        problems.extend(_judge(analysis.id, label, result))

    if check_example and not missing:
        try:
            example = analysis.run(**base)
        except Exception as exc:  # noqa: BLE001 - the kit reports it like any other failure
            problems.append(f"{analysis.id}: raised {type(exc).__name__} on its own example: {exc}")
        else:
            if not example.ok or not example.sentence:
                problems.append(f"{analysis.id}: its own example came back {example.status.value}: {example.sentence}")
    return problems


def check_all(*, category: str | None = None) -> dict[str, list[str]]:
    """``{analysis id: problems}`` for every registered analysis (only the ones that fail)."""
    failing: dict[str, list[str]] = {}
    for name in analyses(category):
        found = check_analysis(get_analysis(name))
        if found:
            failing[name] = found
    return failing


def _judge(name: str, label: str, result: Any) -> list[str]:
    if not isinstance(result, Result):
        return [f"{name}: returned {type(result).__name__} instead of a Result with {label}"]
    problems = []
    if result.status.value == "ok":
        problems.append(f"{name}: answered ok with {label}; it should say what it could not do")
    if not str(result.sentence).strip():
        problems.append(f"{name}: no sentence with {label}, so nobody can read why")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m pinneapple_core.analysis.check [id ...]``: exit 1 when an analysis fails the kit."""
    import sys

    names = list(argv if argv is not None else sys.argv[1:])
    failing = {name: check_analysis(get_analysis(name)) for name in names} if names else check_all()
    failing = {name: found for name, found in failing.items() if found}
    for name, found in failing.items():
        print(f"{name}:")
        for problem in found:
            print(f"  - {problem}")
    print(f"{len(failing)} analysis(es) failing the data-hygiene kit" if failing else "every analysis passes the kit")
    return 1 if failing else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
