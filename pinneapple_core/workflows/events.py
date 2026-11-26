"""Structured events for a workflow run: one JSON Lines file next to the run, and the spans it becomes (#1481).

A run directory already holds enough to *continue* a run — steps, states, hashes, a log per step — and not enough to
*explain* one. At 03:00 the operator gets a directory and has to reconstruct which step failed, on which attempt, with
which command, what the solver printed last, how long it took and how much memory it used. This module writes that
down as events while the run happens::

    <root>/runs/<run id>/events.jsonl
    {"event": "run.started",   "ts": 1763999999.1, "run_id": "...", "dag": "...", "host": "...", ...}
    {"event": "step.started",  "ts": ..., "step_id": "solve", "kind": "command", "attempt": 1, ...}
    {"event": "step.finished", "ts": ..., "step_id": "solve", "status": "done", "seconds": 1.83,
                               "exit_code": 0, "peak_rss_bytes": 41295872, "executor": "local"}
    {"event": "step.log",      "ts": ..., "step_id": "solve", "tail": ["...last lines..."], "reason": "failed"}
    {"event": "step.retry",    "ts": ..., "step_id": "solve", "attempt": 1, "why": "StepFailed: ..."}
    {"event": "step.cancelled","ts": ..., "step_id": "solve", "why": "..."}
    {"event": "run.finished",  "ts": ..., "status": "failed", "seconds": 12.4, "counts": {...}}

Three rules this module holds:

* **A measurement the executor cannot make is absent from the event, not zero.** ``peak_rss_bytes`` is written by the
  local executor, which samples the process tree; docker reports the ``docker`` CLI process, whose memory is not the
  step's, so it is dropped with that reason in ``memory_note``; slurm reports nothing.
* **Writing an event never fails the run it describes.** Every write is guarded: an unwritable run directory costs the
  log line, not the step.
* **The tail kept is bounded** (``TAIL_LINES`` lines, ``TAIL_CHARS`` characters) so a step that prints a megabyte does
  not turn the run record into one.

``to_otlp`` turns the same events into OTLP/HTTP JSON spans, which ``pinneapple runs show --otlp ENDPOINT`` posts —
the same structure an existing observability stack already receives. The exporter is exercised against a local HTTP
server in the tests; it has **not** been run against a real collector (Jaeger, Tempo, an OTel sidecar), and says so
when the post fails rather than pretending it exported.
"""
from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "EVENT_KINDS",
    "TAIL_CHARS",
    "TAIL_LINES",
    "EventStream",
    "MeteredExecutor",
    "OtlpExporter",
    "dag_hash",
    "event_file",
    "host_name",
    "read_events",
    "tail_of",
    "to_otlp",
]

#: Every event this module writes. A reader can rely on these names.
EVENT_KINDS = ("run.started", "run.finished", "step.started", "step.finished", "step.log", "step.retry",
               "step.cancelled")

#: How much of a step's output is kept: enough to see the error, small enough to read in a terminal.
TAIL_LINES = 20
TAIL_CHARS = 4000


def host_name() -> str:
    """The machine the run happened on (a run record that cannot say that is not reproducible)."""
    try:
        return socket.gethostname() or "unknown"
    except OSError:                                     # pragma: no cover - a host with no name
        return "unknown"


def dag_hash(doc: Mapping[str, Any] | Sequence[Any] | str) -> str:
    """A stable hash of the workflow document, so two runs of the same DAG are recognisable as such."""
    if isinstance(doc, str):
        payload = doc
    else:
        payload = json.dumps(doc, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def event_file(run_dir: str | Path) -> Path:
    return Path(run_dir) / "events.jsonl"


def tail_of(text: str, *, lines: int = TAIL_LINES, chars: int = TAIL_CHARS) -> list[str]:
    """The last ``lines`` lines of ``text``, each truncated so the whole stays under ``chars``."""
    if not text:
        return []
    kept = text.rstrip("\n").splitlines()[-lines:]
    out: list[str] = []
    total = 0
    for line in reversed(kept):
        if total + len(line) > chars:
            break
        out.append(line)
        total += len(line) + 1
    return list(reversed(out))


@dataclass
class EventStream:
    """Writes one JSON object per line. Thread-safe (the runner runs steps in a pool) and never raises."""

    path: Path | None = None
    enabled: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _fh: Any = field(default=None, repr=False)
    _seq: int = 0
    errors: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.path is not None and self.enabled:
            try:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(self.path, "a", encoding="utf-8")       # noqa: SIM115 - closed in close()
            except OSError as exc:
                self.errors.append(f"cannot write {self.path}: {exc}")
                self._fh = None

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event. Returns what was written, so a caller (or a test) can read it without re-reading.

        The first parameter is ``event`` rather than ``kind`` because the fields of an event routinely include a
        ``kind`` of their own (a step's kind, e.g. ``command`` or ``openfoam``).
        """
        if event not in EVENT_KINDS:
            raise ValueError(f"unknown event {event!r}; known: {list(EVENT_KINDS)}")
        with self._lock:
            self._seq += 1
            record = {"event": event, "ts": round(time.time(), 6), "seq": self._seq,
                      **{k: v for k, v in fields.items() if v is not None}}
            if not self.enabled or self._fh is None:
                return record
            try:
                self._fh.write(json.dumps(record, default=str) + "\n")
                self._fh.flush()
            except (OSError, TypeError, ValueError) as exc:  # a run is not lost for an event
                self.errors.append(f"event {event}: {exc}")
            return record

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

    def __enter__(self) -> EventStream:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def read_events(path_or_dir: str | Path) -> list[dict[str, Any]]:
    """Read the events next to a run. A half-written last line (a crash mid-write) is skipped, not raised."""
    p = Path(path_or_dir)
    if p.is_dir():
        p = event_file(p)
    if not p.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict) and "event" in doc:
            out.append(doc)
    return out


class MeteredExecutor:
    """Wraps a step's executor so the engine can read what the actual command returned.

    The adapter calls ``ctx.execute(...)`` and keeps only what it needs from the :class:`ExecResult`; the exit code,
    the flags and any measurement the executor made would be lost to the run record. This proxy keeps the last result
    per thread (steps run in a pool) and delegates everything else — ``name``, ``available``, ``script``,
    ``command_line`` — to the real executor, so no adapter has to know it is there.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self._local = threading.local()

    # -- delegation ------------------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return getattr(self.inner, "name", "")

    @property
    def options(self) -> dict[str, Any]:
        return getattr(self.inner, "options", {})

    def available(self) -> tuple[bool, str]:
        return self.inner.available()

    def describe(self) -> dict[str, Any]:
        return self.inner.describe()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.inner, attr)

    # -- running ---------------------------------------------------------------------------------------------------
    def run(self, argv: Sequence[str], **kwargs: Any) -> Any:
        result = self.inner.run(argv, **kwargs)
        self._local.result = result
        return result

    def last(self) -> Any:
        """The last :class:`ExecResult` this thread's step produced, or ``None`` when it ran no command."""
        return getattr(self._local, "result", None)

    def forget(self) -> None:
        self._local.result = None


# ----------------------------------------------------------------------------------------------- OTLP
def to_otlp(events: Iterable[Mapping[str, Any]], *, service: str = "pinneapple",
            scope: str = "pinneapple.workflows") -> dict[str, Any]:
    """The events as an OTLP/HTTP JSON ``TracesData`` document: a run span with one child span per step.

    Deterministic identifiers: the trace id is the first 32 hex characters of the run's DAG hash, a step's span id the
    first 16 of ``<run>:<step>``, so the same run exported twice lands on the same trace.
    """
    events = list(events)
    run_id = next((str(e.get("run_id")) for e in events if e.get("run_id")), "")
    start = next((e for e in events if e.get("event") == "run.started"), None)
    end = next((e for e in events if e.get("event") == "run.finished"), None)
    trace = _hex(run_id or "run", 32)
    spans: list[dict[str, Any]] = []
    run_span_id = _hex(f"{run_id}:run", 16)
    if start is not None:
        spans.append(_span(run_span_id, None, f"run {run_id or '?'}", trace, start.get("ts"),
                           (end or {}).get("ts") or start.get("ts"), _attributes(start), (end or {}).get("status")))
    step_times: dict[str, list[float]] = {}
    for e in events:
        if e.get("event") in ("step.started", "step.finished") and e.get("step_id"):
            step_times.setdefault(str(e["step_id"]), []).append(float(e.get("ts") or 0.0))
    for step_id in sorted(step_times):
        first = next(e for e in events if e.get("event") == "step.started" and e.get("step_id") == step_id)
        last = next((e for e in events if e.get("event") == "step.finished" and e.get("step_id") == step_id), None)
        attrs = _attributes(first)
        status = None
        if last is not None:
            attrs.update({k: v for k, v in _attributes(last).items() if k not in attrs})
            status = last.get("status")
        spans.append(_span(_hex(f"{run_id}:{step_id}", 16), run_span_id, f"step {step_id}", trace, first.get("ts"),
                           (last or first).get("ts"), attrs, status))
    return {"resourceSpans": [{"resource": {"attributes": [_attr("service.name", service),
                                                            _attr("host.name", host_name())]},
                               "scopeSpans": [{"scope": {"name": scope}, "spans": spans}]}]}


def _span(span_id: str, parent: str | None, name: str, trace: str, start: Any, end: Any,
          attributes: Mapping[str, Any], status: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {"traceId": trace, "spanId": span_id, "name": name, "kind": 1,
                           "startTimeUnixNano": _nanos(start), "endTimeUnixNano": _nanos(end),
                           "attributes": [_attr(k, v) for k, v in sorted(attributes.items())]}
    if parent:
        doc["parentSpanId"] = parent
    if status is not None:
        ok = str(status).lower() in ("done", "ok", "skipped")
        doc["status"] = {"code": 1 if ok else 2, "message": str(status)}
    return doc


def _attributes(event: Mapping[str, Any]) -> dict[str, Any]:
    skip = {"event", "ts", "seq", "tail"}
    out: dict[str, Any] = {}
    for k, v in event.items():
        if k in skip or v is None:
            continue
        if isinstance(v, (str, bool, int, float)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in v):
            out[k] = ",".join(str(x) for x in v)
    return out


def _attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _hex(text: str, width: int) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:width]


def _nanos(ts: Any) -> int:
    try:
        return int(float(ts) * 1_000_000_000)
    except (TypeError, ValueError):
        return 0


@dataclass
class OtlpExporter:
    """POSTs :func:`to_otlp`'s document to an OTLP/HTTP endpoint (``.../v1/traces`` by default).

    ``export`` returns what happened instead of raising: ``{"ok": False, "error": "..."}`` when the collector is not
    there, which the CLI prints. Nothing is retried and nothing is buffered to disk — an observability export that
    quietly queues is an observability export that lies about what it delivered.
    """

    endpoint: str
    timeout: float = 5.0
    service: str = "pinneapple"

    @property
    def url(self) -> str:
        base = self.endpoint.rstrip("/")
        return base if base.endswith("/v1/traces") else f"{base}/v1/traces"

    def export(self, events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        payload = to_otlp(events, service=self.service)
        spans = len(payload["resourceSpans"][0]["scopeSpans"][0]["spans"])
        body = json.dumps(payload).encode()
        request = urllib.request.Request(self.url, data=body, method="POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - the endpoint is the operator's
                return {"ok": 200 <= int(response.status) < 300, "status": int(response.status), "spans": spans,
                        "url": self.url, "error": "" if 200 <= int(response.status) < 300 else
                        f"HTTP {response.status} from {self.url}"}
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"ok": False, "status": None, "spans": spans, "url": self.url,
                    "error": f"{type(exc).__name__}: {exc}"}
