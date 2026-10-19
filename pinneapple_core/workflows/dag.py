"""Workflows as a DAG of steps: definition, validation (unknown dependencies, cycles), order, JSON.

    wf = Workflow("beam", [Step("mesh", "python", {"function": "pkg.mod:make_deck"}),
                           Step("solve", "calculix", {"inp_from": "mesh", "job": "job"}, needs=["mesh"])])
    wf.order()            # ["mesh", "solve"]
    wf.save("beam.workflow.json"); Workflow.load("beam.workflow.json")

``kind`` names an adapter (``command``, ``python``, ``openfoam``, ``calculix`` or one registered with
``register_adapter``); ``executor`` names an executor (``local``, ``docker``, ``slurm`` or one registered with
``register_executor``). ``workflow_from_cases`` expands a step template once per case of a table.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")


class WorkflowError(ValueError):
    """The workflow is not a valid DAG (unknown step, duplicate id, cycle)."""


@dataclass
class Step:
    id: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    needs: list[str] = field(default_factory=list)
    executor: str = "local"
    executor_options: dict[str, Any] = field(default_factory=dict)
    retries: int = 0
    timeout: float | None = None
    inputs: list[str] = field(default_factory=list)          # files whose content enters the step's hash
    group: str = ""                                          # e.g. the case id of a per-case step

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Workflow:
    name: str
    steps: list[Step]
    fail_fast: bool = False                    # True: stop starting steps after the first failure
    max_parallel: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.steps = [s if isinstance(s, Step) else Step(**s) for s in self.steps]
        self.validate()

    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.steps]

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def validate(self) -> None:
        ids = self.ids
        bad = [i for i in ids if not _ID.match(i)]
        if bad:
            raise WorkflowError(f"invalid step ids {bad}: letters, digits, _ . - only")
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise WorkflowError(f"duplicate step ids: {dup}")
        known = set(ids)
        for s in self.steps:
            missing = [d for d in s.needs if d not in known]
            if missing:
                raise WorkflowError(f"step {s.id!r} needs unknown steps {missing}")
            if s.retries < 0:
                raise WorkflowError(f"step {s.id!r}: retries must be >= 0")
        self.order()

    def order(self) -> list[str]:
        """Topological order (Kahn), stable in the order steps were given; WorkflowError on a cycle."""
        indeg = {s.id: len(set(s.needs)) for s in self.steps}
        children: dict[str, list[str]] = {s.id: [] for s in self.steps}
        for s in self.steps:
            for d in set(s.needs):
                children[d].append(s.id)
        ready = [s.id for s in self.steps if indeg[s.id] == 0]
        out: list[str] = []
        while ready:
            n = ready.pop(0)
            out.append(n)
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        if len(out) != len(self.steps):
            cyc = sorted(i for i, d in indeg.items() if d > 0)
            raise WorkflowError(f"the workflow has a cycle through {cyc}")
        return out

    def downstream(self, step_id: str) -> set[str]:
        """Every step that depends on ``step_id``, directly or not."""
        out: set[str] = set()
        frontier = [step_id]
        while frontier:
            cur = frontier.pop()
            for s in self.steps:
                if cur in s.needs and s.id not in out:
                    out.add(s.id)
                    frontier.append(s.id)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "fail_fast": self.fail_fast, "max_parallel": self.max_parallel,
                "metadata": self.metadata, "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Workflow:
        known = {"name", "steps", "fail_fast", "max_parallel", "metadata"}
        extra = set(d) - known
        if extra:
            raise WorkflowError(f"unknown workflow keys: {sorted(extra)}")
        steps = []
        for s in d["steps"]:
            extra = set(s) - set(Step.__dataclass_fields__)
            if extra:
                raise WorkflowError(f"step {s.get('id')!r}: unknown keys {sorted(extra)}")
            steps.append(Step(**s))
        return cls(name=d["name"], steps=steps, fail_fast=bool(d.get("fail_fast", False)),
                   max_parallel=int(d.get("max_parallel", 1)), metadata=dict(d.get("metadata") or {}))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Workflow:
        return cls.from_dict(json.loads(Path(path).read_text()))


def fill(obj: Any, values: Mapping[str, Any]) -> Any:
    """Fill ``{name}`` placeholders in strings (recursively in dicts and lists). A string that is exactly one
    placeholder takes the value itself, so numbers stay numbers."""
    if isinstance(obj, str):
        m = re.fullmatch(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", obj)
        if m and m.group(1) in values:
            return values[m.group(1)]
        if "{" in obj:
            try:
                return obj.format_map(_Defaulting(values))
            except (ValueError, IndexError):
                return obj
        return obj
    if isinstance(obj, dict):
        return {k: fill(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill(v, values) for v in obj]
    return obj


class _Defaulting(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def workflow_from_cases(name: str, cases: Sequence[Mapping[str, Any]], template: Sequence[Step], *,
                        id_column: str | None = None, final: Sequence[Step] = (), **kw) -> Workflow:
    """One copy of the ``template`` steps per case, ids ``<case id>.<step id>``, placeholders ``{param}`` filled
    with the case's values; ``final`` steps run after all cases (their ``needs`` may name template step ids, which
    then mean that step of every case)."""
    steps: list[Step] = []
    tmpl_ids = {t.id for t in template}
    case_ids = []
    for i, case in enumerate(cases):
        cid = str(case[id_column]) if id_column else str(case.get("case_id", case.get("case", f"case{i:04d}")))
        cid = re.sub(r"[^A-Za-z0-9_.\-]", "_", cid)
        case_ids.append(cid)
        values = {k: _plain(v) for k, v in case.items()}
        values["case_id"] = cid
        for t in template:
            steps.append(Step(id=f"{cid}.{t.id}", kind=t.kind, params=fill(t.params, values),
                              needs=[f"{cid}.{d}" if d in tmpl_ids else d for d in t.needs],
                              executor=t.executor, executor_options=dict(t.executor_options), retries=t.retries,
                              timeout=t.timeout, inputs=[fill(p, values) for p in t.inputs], group=cid))
    for f in final:
        needs = []
        for d in f.needs:
            needs += [f"{c}.{d}" for c in case_ids] if d in tmpl_ids else [d]
        steps.append(Step(id=f.id, kind=f.kind, params=dict(f.params), needs=needs, executor=f.executor,
                          executor_options=dict(f.executor_options), retries=f.retries, timeout=f.timeout,
                          inputs=list(f.inputs), group=f.group))
    return Workflow(name, steps, **kw)


def _plain(v):
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:  # pragma: no cover
        pass
    return v
