"""The state a run carries from stage to stage, and what it can be resumed from.

A :class:`RunState` is the single place a stage writes its artefacts and the next one reads them. It is also what
``resume`` restores: ``state.json`` next to the run's folder records which stages finished and where their
artefacts are, so a run interrupted after an expensive reference campaign does not repeat it (``execution.resume``)
and ``status()`` / ``history()`` can report what happened without re-running anything.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .results import STAGE_ORDER, StageRecord

__all__ = ["RunState", "STATE_FILE", "CONFIG_FILE"]


STATE_FILE = "state.json"
CONFIG_FILE = "config.json"


@dataclass
class RunState:
    """Everything a run knows, and where it put it."""

    cfg: Any
    run_id: str
    out_dir: Path
    bundle: Any = None
    design: Any = None
    cases: Any = None                  # DataFrame of the design's cases
    points: Any = None                 # probe points of a field surrogate (n, len(coords))
    dataset: Any = None                # a pinneapple_analysis.surrogate SurrogateData
    reference: dict[str, Any] = field(default_factory=dict)     # case index -> Solution (or its failure note)
    model: Any = None                  # a Solution (physics) or a StudyResult (surrogate)
    model_object: Any = None           # whatever the export needs (Solution, SurrogateModel, torch module)
    metrics: dict[str, Any] = field(default_factory=dict)
    verification: list[Any] = field(default_factory=list)
    validated_domain: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, StageRecord] = field(default_factory=lambda: {s: StageRecord(s) for s in STAGE_ORDER})
    started: float = field(default_factory=time.perf_counter)
    notes: list[str] = field(default_factory=list)

    # -------------------------------------------------------------------------------------------- stages
    def record(self, stage: str) -> StageRecord:
        if stage not in self.stages:
            self.stages[stage] = StageRecord(stage)
        return self.stages[stage]

    def status_of(self, stage: str) -> str:
        return self.stages[stage].status if stage in self.stages else "pending"

    def finished(self, stage: str) -> bool:
        return self.status_of(stage) == "done"

    def ordered_records(self) -> list[StageRecord]:
        return [self.stages[s] for s in STAGE_ORDER if s in self.stages]

    # -------------------------------------------------------------------------------------------- paths
    def path(self, *parts: str) -> Path:
        p = self.out_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def artifacts(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for rec in self.ordered_records():
            for k, v in rec.artifacts.items():
                if isinstance(v, (str, Path)) and str(v):
                    out[k] = str(v)
        return out

    # -------------------------------------------------------------------------------------------- resume
    def save(self) -> Path:
        """Write ``state.json``: the stage table and the artefacts, enough to resume and to report.

        The configuration is written next to it as ``config.json`` (once per run, or again when it changes), so the
        folder is self-describing: ``pp.SurrogatePipeline.from_run(folder)`` reloads the run without the YAML it
        came from.
        """
        from .results import _jsonable
        config_path = self.out_dir / CONFIG_FILE
        try:
            payload = self.cfg.to_dict()
            if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != payload:
                config_path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        except Exception:                                          # a config that cannot be written is not fatal
            pass
        metrics = {}
        for key, value in self.metrics.items():
            if key in ("pinn_report",):                  # a validation report object, rebuilt by re-running
                continue
            try:
                metrics[key] = _jsonable(value)
            except Exception:                             # a metric that cannot be serialised is not recorded
                continue
        payload = {
            "run_id": self.run_id, "name": getattr(self.cfg, "name", self.run_id),
            "out_dir": str(self.out_dir), "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stages": [r.to_dict() for r in self.ordered_records()],
            "artifacts": self.artifacts(),
            "metrics": metrics,
            "validated_domain": _jsonable(self.validated_domain),
            "notes": list(self.notes),
        }
        p = self.out_dir / STATE_FILE
        p.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, out_dir: str | Path) -> dict[str, Any] | None:
        p = Path(out_dir) / STATE_FILE
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):                              # a truncated state file is not fatal
            return None

    def restore_stage_status(self, saved: dict[str, Any] | None) -> None:
        """Mark the stages a previous run finished as done, keeping their artefacts (for ``resume``)."""
        if not saved:
            return
        for rec in saved.get("stages", []):
            name = rec.get("stage")
            if name == "verification":
                continue                       # cheap and must reflect the final artefacts: always re-run
            if name in self.stages and rec.get("status") == "done":
                self.stages[name].status = "done"
                self.stages[name].seconds = float(rec.get("seconds") or 0.0)
                self.stages[name].artifacts = dict(rec.get("artifacts") or {})
                self.stages[name].notes = list(rec.get("notes") or []) + ["restored from a previous run"]
        self.metrics.update({k: v for k, v in (saved.get("metrics") or {}).items() if k not in self.metrics})
        if saved.get("validated_domain") and not self.validated_domain:
            self.validated_domain = dict(saved["validated_domain"])
