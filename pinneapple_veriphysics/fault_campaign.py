"""Benchmark authorization gates separately from numerical fault detection."""
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class FaultCase:
    stage: str
    name: str
    should_permit: bool
    authorize: Callable[[], bool]


@dataclass(frozen=True)
class FaultOutcome:
    stage: str
    name: str
    expected: bool
    actual: bool | None
    error: str = ""


def run_campaign(cases: Iterable[FaultCase]) -> dict:
    """Measure clean acceptance, unsafe acceptance and execution errors by stage.

    Exceptions and nonboolean responses are benchmark errors, never credited
    as successful fault detection. Each stage needs clean and faulty controls.
    """
    outcomes = []
    seen = set()
    for case in cases:
        key = (case.stage, case.name)
        if key in seen or not all(key) or type(case.should_permit) is not bool:
            raise ValueError("Invalid or duplicate case")
        seen.add(key)
        try:
            actual = case.authorize()
            if type(actual) is not bool:
                raise TypeError("Gate must return a boolean")
            outcomes.append(FaultOutcome(*key, case.should_permit, actual))
        except Exception as exc:
            outcomes.append(FaultOutcome(*key, case.should_permit, None,
                                         type(exc).__name__))
    if not outcomes:
        raise ValueError("Campaign must contain cases")
    stages = {}
    for stage in sorted({o.stage for o in outcomes}):
        rows = [o for o in outcomes if o.stage == stage]
        clean = [o for o in rows if o.expected]
        faulty = [o for o in rows if not o.expected]
        detected = sum(o.actual is False for o in faulty)
        accepted = sum(o.actual is True for o in clean)
        errors = sum(o.actual is None for o in rows)
        stages[stage] = {
            "clean_cases": len(clean), "fault_cases": len(faulty),
            "detection_rate": detected / len(faulty) if faulty else None,
            "clean_acceptance_rate": accepted / len(clean) if clean else None,
            "unsafe_acceptances": sum(o.actual is True for o in faulty),
            "false_rejections": sum(o.actual is False for o in clean),
            "errors": errors,
            "passed": bool(clean and faulty and not errors
                           and detected == len(faulty) and accepted == len(clean)),
        }
    return {"passed": all(s["passed"] for s in stages.values()),
            "stages": stages, "outcomes": [vars(o) for o in outcomes]}
