"""Reproducing a recorded run: resolve, verify, run, compare, and say which of three things happened (#1480).

An experiment id from six months ago is not reproducible by re-running it: the package moved, the machine changed,
and nothing in the output says so. ``reproduce(exp_id)`` answers the question the re-run does not:

============  ==========================================================================================
``reproduced``  same versions, same inputs, every metric within the recorded tolerance
``divergent``   it ran, but the environment or the inputs differ, or a metric left its tolerance — and the
                differences are listed, so "why is this number different" has an answer
``refused``     it cannot be run faithfully: a dependency version, an artifact or an input is missing. The
                refusal names it instead of running on whatever happens to be present
============  ==========================================================================================

Refusal is a **result**, not an exception to work around: a caller that gets ``refused`` has learned something
about the record, and a caller that gets ``divergent`` decides whether comparing is still worth it.

Hardware is compared and reported separately from software. A run on another CPU is not the same run — the
metrics are compared within the tolerance instead of to the last bit — but it is not the same failure as a
version bump, and the report says which one happened.

Two kinds of record are reproduced, and more can register their own:

* an **experiment** — a directory with ``experiment.json`` and ``result.json``, what :meth:`Experiment.save` and
  :meth:`ExperimentResult.save` write;
* a **workflow run** — a directory with ``flow.json`` and ``run.json``, registered by :mod:`pinneapple_flow`.

    >>> import pinneapple as pp
    >>> r = pp.experiments.reproduce("runs/burgers_pinn")      # doctest: +SKIP
    >>> r.outcome, r.divergent                                  # doctest: +SKIP
    ('reproduced', False)
    >>> print(r.to_markdown())                                  # doctest: +SKIP
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Difference", "Measurement", "MetricComparison", "Reproduction", "Reproducer", "Refused", "environment",
           "producers", "register_reproducer", "reproduce", "DEFAULT_RTOL", "HARDWARE_KEYS", "SOFTWARE_KEYS"]

#: relative tolerance a metric is compared with when the record declares none
DEFAULT_RTOL = 1e-6

#: environment keys that describe the software that ran: a difference here means a different program
SOFTWARE_KEYS = ("python", "pinneapple", "numpy", "torch", "cuda", "commit")

#: environment keys that describe the machine: a difference here is recorded, and the metrics are compared within
#: tolerance rather than bit for bit
HARDWARE_KEYS = ("platform", "machine", "processor", "cpu_count", "gpu", "blas", "threads")

REFUSED, DIVERGENT, REPRODUCED = "refused", "divergent", "reproduced"


@dataclass(frozen=True)
class Difference:
    """One thing that is not what the record says: which kind, what it is, and both values."""

    kind: str                       # software | hardware | input | metric
    what: str
    recorded: Any
    now: Any
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "what": self.what, "recorded": self.recorded, "now": self.now, "note": self.note}

    def __str__(self) -> str:
        return f"{self.kind}: {self.what} {self.recorded!r} -> {self.now!r}"


@dataclass(frozen=True)
class MetricComparison:
    """One recorded metric against the value this run produced, with the tolerance it was judged by."""

    name: str
    recorded: float | None
    now: float | None
    rtol: float
    note: str = ""

    @property
    def delta(self) -> float | None:
        if self.recorded is None or self.now is None:
            return None
        return float(self.now) - float(self.recorded)

    @property
    def relative(self) -> float | None:
        if self.recorded is None or self.now is None:
            return None
        scale = max(abs(float(self.recorded)), 1e-300)
        return abs(float(self.now) - float(self.recorded)) / scale

    @property
    def ok(self) -> bool:
        """Within tolerance. A metric the record has and this run does not is **not** ok, and neither is the
        reverse: a silently missing metric is how a comparison passes without comparing anything."""
        if self.recorded is None or self.now is None:
            return False
        if self.rtol == 0.0:
            return float(self.now) == float(self.recorded)
        rel = self.relative
        return rel is not None and rel <= self.rtol

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "recorded": self.recorded, "now": self.now, "delta": self.delta,
                "relative": self.relative, "rtol": self.rtol, "ok": self.ok, "note": self.note}

    def __str__(self) -> str:
        if self.recorded is None or self.now is None:
            return f"{self.name}: {'not recorded' if self.recorded is None else 'not produced'}"
        return f"{self.name}: {self.recorded:.6g} -> {self.now:.6g} ({'ok' if self.ok else 'out of tolerance'})"


@dataclass(frozen=True)
class Measurement:
    """What running the record again produced: the metrics to compare and the environment it ran in."""

    metrics: dict[str, float]
    environment: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Reproducer:
    """One kind of record that can be reproduced: how to recognise it and how to run it again.

    ``prepare`` returns ``(record, run)``, or raises :class:`Refused` naming what is missing. Everything after the
    re-run — comparing the environment, the inputs and the metrics, and deciding the outcome — is shared, so a new
    kind cannot invent its own definition of "reproduced".
    """

    kind: str
    detect: Callable[[Path], bool]
    prepare: Callable[[Path], tuple[dict[str, Any], Callable[[], Measurement]]]


class Refused(Exception):
    """The record cannot be run faithfully. Carries the sentence the caller reads."""


@dataclass
class Reproduction:
    """The answer: an outcome, the differences behind it, and the metrics compared."""

    experiment: str
    outcome: str
    reason: str = ""
    kind: str = ""
    differences: list[Difference] = field(default_factory=list)
    metrics: list[MetricComparison] = field(default_factory=list)
    recorded_environment: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = ""
    rerun_at: str = ""

    @property
    def reproduced(self) -> bool:
        return self.outcome == REPRODUCED

    @property
    def divergent(self) -> bool:
        return self.outcome == DIVERGENT

    @property
    def refused(self) -> bool:
        return self.outcome == REFUSED

    @property
    def metrics_out_of_tolerance(self) -> list[str]:
        return [m.name for m in self.metrics if not m.ok]

    @property
    def environment_recorded(self) -> bool:
        """Whether the record says what it ran on. Without it the versions were not compared, and the report says so
        rather than calling the run a reproduction because the numbers happened to match."""
        return bool(self.recorded_environment)

    @property
    def software_changes(self) -> list[Difference]:
        return [d for d in self.differences if d.kind == "software"]

    @property
    def hardware_changes(self) -> list[Difference]:
        return [d for d in self.differences if d.kind == "hardware"]

    def __bool__(self) -> bool:
        """True only when it reproduced: ``if pp.experiments.reproduce(exp):`` reads as the question it is."""
        return self.reproduced

    def to_dict(self) -> dict[str, Any]:
        return {"experiment": self.experiment, "kind": self.kind, "outcome": self.outcome, "reason": self.reason,
                "differences": [d.to_dict() for d in self.differences],
                "metrics": [m.to_dict() for m in self.metrics],
                "recorded_environment": self.recorded_environment, "environment": self.environment,
                "recorded_at": self.recorded_at, "rerun_at": self.rerun_at}

    def to_markdown(self) -> str:
        """The report a person reads: the outcome, why, and the two tables behind it."""
        lines = [f"# Reproduction of `{self.experiment}`", "", f"**Outcome: {self.outcome}.** {self.reason}".strip(),
                 ""]
        if self.metrics:
            lines += ["| Metric | Recorded | Now | Relative | Tolerance | |", "|---|---|---|---|---|---|"]
            for m in self.metrics:
                rec = "—" if m.recorded is None else f"{m.recorded:.6g}"
                now = "—" if m.now is None else f"{m.now:.6g}"
                rel = "—" if m.relative is None else f"{m.relative:.3g}"
                lines.append(f"| `{m.name}` | {rec} | {now} | {rel} | {m.rtol:g} | "
                             f"{'ok' if m.ok else 'out'} |")
        if self.differences:
            lines += ["", "| Kind | What | Recorded | Now |", "|---|---|---|---|"]
            lines += [f"| {d.kind} | `{d.what}` | {d.recorded} | {d.now} |" for d in self.differences]
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        return (f"Reproduction({self.experiment!r}, {self.outcome!r}, {len(self.differences)} differences, "
                f"{len(self.metrics_out_of_tolerance)} metrics out of tolerance)")


def environment() -> dict[str, Any]:
    """The environment fingerprint this library records with a run: python, versions, commit, machine.

    Exposed because a kind of record that is not an :class:`Experiment` still has to record the same keys for its
    reproduction to be comparable with the rest."""
    from .experiment import _environment
    return _environment()


_REGISTRY: dict[str, Reproducer] = {}


def register_reproducer(reproducer: Reproducer) -> Reproducer:
    """Add a kind of record that can be reproduced. Re-registering the same kind with another meaning is refused,
    so two packages cannot silently disagree about what "workflow" means."""
    old = _REGISTRY.get(reproducer.kind)
    if old is not None and old != reproducer:
        raise ValueError(f"a reproducer for {reproducer.kind!r} is already registered")
    _REGISTRY[reproducer.kind] = reproducer
    return reproducer


def producers() -> dict[str, Reproducer]:
    """The kinds this installation can reproduce (the workflow one arrives with ``pinneapple_flow``)."""
    _ensure_builtin()
    return dict(_REGISTRY)


def _ensure_builtin() -> None:
    if "experiment" not in _REGISTRY:
        register_reproducer(experiment_reproducer())
    if "workflow" in _REGISTRY:
        return
    try:                                       # a workflow run is reproduced by the package that runs workflows
        import pinneapple_flow.reproducibility  # noqa: F401
    except Exception:                          # not installed here: the kind is simply absent, not refused
        return


# ---------------------------------------------------------------------------------------- the experiment kind
def _flatten(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    """``{"relative_l2": {"u": 1e-3}}`` -> ``{"relative_l2.u": 1e-3}``: one name per compared number."""
    out: dict[str, float] = {}
    for field_name, values in (metrics or {}).items():
        if isinstance(values, Mapping):
            for metric_name, value in values.items():
                try:
                    out[f"{field_name}.{metric_name}"] = float(value)
                except (TypeError, ValueError):
                    continue
        else:
            try:
                out[str(field_name)] = float(values)
            except (TypeError, ValueError):
                continue
    return out


def experiment_reproducer() -> Reproducer:
    from .experiment import Experiment

    def detect(directory: Path) -> bool:
        return (directory / "experiment.json").is_file()

    def prepare(directory: Path) -> tuple[dict[str, Any], Callable[[], Measurement]]:
        from .solving import MethodNotAvailable

        result_path = directory / "result.json"
        if not result_path.is_file():
            raise Refused(f"no result.json next to experiment.json in {directory}: there is nothing recorded to "
                          "compare a re-run against")
        record = json.loads(result_path.read_text())
        try:
            experiment = Experiment.load(str(directory))
        except (KeyError, TypeError, ValueError) as exc:
            raise Refused(f"the experiment definition cannot be rebuilt: {exc}") from None
        declared = {k: float(v) for k, v in (record.get("tolerances") or {}).items()}

        def run() -> Measurement:
            try:
                res = experiment.run()
            except MethodNotAvailable as exc:
                raise Refused(f"the method is not available in this installation: {exc}") from None
            except (KeyError, ValueError, TypeError, ImportError) as exc:
                raise Refused(f"the run cannot be performed: {type(exc).__name__}: {exc}") from None
            return Measurement(metrics=_flatten(res.metrics), environment=dict(res.environment),
                               extra={"n_eval_points": res.n_eval_points,
                                      "problem_fingerprint": res.problem_fingerprint,
                                      "method": experiment.method})

        inputs = {"problem_fingerprint": record.get("problem_fingerprint"),
                  "n_eval_points": record.get("n_eval_points"),
                  "method": record.get("config", {}).get("method")}
        return {"id": str(record.get("config", {}).get("name") or directory.name),
                "metrics": _flatten(record.get("metrics")),
                "environment": dict(record.get("environment") or {}),
                "tolerances": declared, "inputs": inputs,
                "recorded_at": str(record.get("started_at") or "")}, run

    return Reproducer("experiment", detect, prepare)


# ---------------------------------------------------------------------------------------- the shared comparison
def _resolve(exp_id: str | os.PathLike[str], root: str | os.PathLike[str] | None) -> Path:
    """``exp_id`` is a directory, an ``experiment.json``/``run.json`` file, or a name under ``root``."""
    path = Path(exp_id)
    if path.is_file():
        return path.parent
    if path.is_dir():
        return path
    if root is not None:
        candidate = Path(root) / str(exp_id)
        if candidate.is_dir():
            return candidate
    raise Refused(f"no recorded run at {str(path)!r}" + (f" or under {str(root)!r}" if root else ""))


def reproducer_for(directory: Path) -> Reproducer | None:
    """The registered kind that recognises ``directory``, or ``None``."""
    _ensure_builtin()
    for reproducer in _REGISTRY.values():
        try:
            if reproducer.detect(directory):
                return reproducer
        except Exception:                      # a kind that cannot even look at it does not claim it
            continue
    return None


def _compare_environment(recorded: Mapping[str, Any], now: Mapping[str, Any]) -> list[Difference]:
    out: list[Difference] = []
    for key in sorted(set(recorded) | set(now)):
        if key not in now:
            out.append(Difference("environment", key, recorded[key], None, "recorded but not reported now"))
            continue
        if key not in recorded:
            out.append(Difference("environment", key, None, now[key], "reported now, not recorded"))
            continue
        if recorded[key] == now[key]:
            continue
        kind = "software" if key in SOFTWARE_KEYS else "hardware" if key in HARDWARE_KEYS else "environment"
        out.append(Difference(kind, key, recorded[key], now[key]))
    return out


def _compare_inputs(recorded: Mapping[str, Any], now: Mapping[str, Any]) -> list[Difference]:
    return [Difference("input", key, recorded.get(key), now.get(key))
            for key in sorted(recorded)
            if recorded.get(key) is not None and recorded.get(key) != now.get(key)]


def reproduce(exp_id: str | os.PathLike[str], *, root: str | os.PathLike[str] | None = None,
              rtol: float | None = None, exact: bool = False,
              tolerances: Mapping[str, float] | None = None) -> Reproduction:
    """Re-run the recorded experiment ``exp_id`` and say whether it reproduced.

    ``rtol``: relative tolerance for every metric without its own; ``exact=True`` asks for bit-identical numbers
    (``rtol = 0``) and is only honest for a deterministic method on the same machine. ``tolerances`` overrides per
    metric, and the record's own ``tolerances`` override both — a record that declares how close is close enough is
    the one that decides. Never raises for a record that cannot be run: that is ``outcome == "refused"``.
    """
    try:
        directory = _resolve(exp_id, root)
    except Refused as exc:
        return Reproduction(str(exp_id), REFUSED, str(exc))
    reproducer = reproducer_for(directory)
    if reproducer is None:
        known = ", ".join(sorted(producers())) or "none"
        return Reproduction(str(exp_id), REFUSED,
                            f"{directory} is not a record any registered kind recognises (kinds: {known})")
    try:
        record, run = reproducer.prepare(directory)
    except Refused as exc:
        return Reproduction(str(exp_id), REFUSED, str(exc), kind=reproducer.kind)
    try:
        measured = run()
    except Refused as exc:
        return Reproduction(str(record.get("id") or exp_id), REFUSED, str(exc), kind=reproducer.kind,
                            recorded_environment=dict(record.get("environment") or {}),
                            recorded_at=str(record.get("recorded_at") or ""))
    except (KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
        return Reproduction(str(record.get("id") or exp_id), REFUSED,
                            f"the run failed: {type(exc).__name__}: {exc}", kind=reproducer.kind)

    tolerances = {**(record.get("tolerances") or {}), **(tolerances or {})}
    default_rtol = 0.0 if exact else (DEFAULT_RTOL if rtol is None else float(rtol))
    recorded_metrics = dict(record.get("metrics") or {})
    comparisons = [
        MetricComparison(name, recorded_metrics.get(name), measured.metrics.get(name),
                         float(tolerances.get(name, default_rtol)))
        for name in sorted(set(recorded_metrics) | set(measured.metrics))
    ]
    recorded_environment = dict(record.get("environment") or {})
    if recorded_environment:
        changes = _compare_environment(recorded_environment, measured.environment)
    else:                                       # nothing to compare against: said plainly, not silently agreed
        changes = [Difference("environment", "environment", None, None,
                              "the record does not say which versions or machine it ran on, so that was not compared")]
    changes += _compare_inputs(record.get("inputs") or {}, measured.extra)
    changes += [Difference("metric", m.name, m.recorded, m.now,
                           "outside tolerance" if not m.ok else "") for m in comparisons if not m.ok]

    outcome = DIVERGENT if changes else REPRODUCED
    reason = ("every metric is within the recorded tolerance and nothing in the environment or the inputs moved"
              if outcome == REPRODUCED else _describe(changes))
    environment = dict(measured.environment)
    return Reproduction(str(record.get("id") or directory.name), outcome, reason, kind=reproducer.kind,
                        differences=changes, metrics=comparisons,
                        recorded_environment=recorded_environment, environment=environment,
                        recorded_at=str(record.get("recorded_at") or ""))


def _describe(changes: list[Difference]) -> str:
    """The one-line reason: what changed, with the software ones first because they explain the metrics."""
    software = [d for d in changes if d.kind in ("software", "input")]
    metrics = [d for d in changes if d.kind == "metric"]
    hardware = [d for d in changes if d.kind in ("hardware", "environment")]
    parts = []
    if software:
        parts.append("software or inputs differ: " + ", ".join(f"{d.what} {d.recorded!r} -> {d.now!r}"
                                                               for d in software))
    if metrics:
        parts.append("metrics out of tolerance: " + ", ".join(d.what for d in metrics))
    if hardware:
        parts.append("the machine differs: " + ", ".join(f"{d.what} {d.recorded!r} -> {d.now!r}"
                                                         for d in hardware))
    return "; ".join(parts) or "diverged"
