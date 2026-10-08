"""Engineering model lineage: artifacts (CAD, mesh, simulation, dataset, model, prediction, ...) linked by "derived
from" edges into a digital thread, with the checks that make it trustworthy and the questions it answers.

    g = Lineage.from_json(obj)          # full form {"artifacts": [...]} or the simple chain {"geometry": ..., ...}
    g.checks()                          # broken references, cycles, changed files, stale derivations, mixed revisions
    g.upstream("PRED-883")              # how was this produced?
    g.downstream("CAD-001")             # what depends on this (impact of a change)?
    g.to_prov()                         # W3C PROV-JSON
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

KINDS = ["geometry", "mesh", "simulation", "result", "experiment", "post-processing", "dataset", "model", "prediction",
         "report", "code", "file"]
CHAIN_KEYS = [("geometry", "geometry"), ("cad", "geometry"), ("mesh", "mesh"),
              ("simulation", "simulation"), ("postprocessing", "post-processing"), ("post_processing", "post-processing"),
              ("dataset", "dataset"), ("model", "model"), ("prediction", "prediction"), ("result", "prediction"),
              ("report", "report")]
PROVENANCE_FIELDS = ("file", "version", "software", "timestamp", "hash", "owner", "origin")


@dataclass
class Artifact:
    id: str
    kind: str = "file"
    name: str = ""
    file: Optional[str] = None
    version: Optional[str] = None
    software: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None            # ISO 8601
    hash: Optional[str] = None                 # sha256 (hex, full or prefix) of the file as recorded
    owner: Optional[str] = None
    origin: Optional[str] = None               # where it lives / came from (path, URL, system)
    inputs: List[str] = field(default_factory=list)
    item: Optional[str] = None                 # the thing this is a revision of ("bracket CAD"); revisions share it
    notes: Optional[str] = None
    detected: Optional[str] = None             # how it was found (auto-detection rule), if not declared
    actual_hash: Optional[str] = None          # sha256 of the uploaded file, when the file was uploaded

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [], {})}

    def completeness(self) -> float:
        return sum(1 for f in PROVENANCE_FIELDS if getattr(self, f)) / len(PROVENANCE_FIELDS)


def _ts(s: Optional[str]) -> Optional[_dt.datetime]:
    if not s:
        return None
    try:
        t = _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        return None


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Lineage:
    def __init__(self, artifacts: Iterable[Artifact] = (), project: str = ""):
        self.project = project
        self.nodes: Dict[str, Artifact] = {}
        for a in artifacts:
            self.add(a)

    # ------------------------------------------------------------------ construction
    def add(self, a: Artifact) -> Artifact:
        if a.kind not in KINDS:
            a.kind = "file"
        if a.id in self.nodes:                                         # merge: declared fields win over detected
            old = self.nodes[a.id]
            for k, v in a.to_dict().items():
                if k == "kind" and v == "file":
                    continue                                          # the default never overrides a detected kind
                if k == "inputs":
                    old.inputs = list(dict.fromkeys(old.inputs + v))
                elif k == "parameters":
                    old.parameters = {**old.parameters, **v}
                elif getattr(old, k) in (None, "", [], {}) or not a.detected:
                    setattr(old, k, v)
            return old
        self.nodes[a.id] = a
        return a

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "Lineage":
        if "artifacts" in obj:
            arts = []
            for raw in obj["artifacts"]:
                raw = dict(raw)
                if "id" not in raw:
                    raise ValueError(f"artifact without id: {raw}")
                ins = raw.pop("inputs", raw.pop("derived_from", []))
                ins = [ins] if isinstance(ins, str) else list(ins or [])
                known = {k: raw.pop(k) for k in list(raw) if k in Artifact.__dataclass_fields__}
                params = dict(known.pop("parameters", {}) or {})
                params.update(raw)                                    # unknown keys are kept as parameters
                arts.append(Artifact(**known, inputs=[str(i) for i in ins], parameters=params))
            return cls(arts, project=str(obj.get("project", "")))
        # simple chain: {"project": ..., "geometry": ..., "mesh": ..., "solver": ..., ...}
        g = cls(project=str(obj.get("project", "")))
        solver = obj.get("solver")                                    # describes the simulation, not a node
        if isinstance(solver, str):
            parts = solver.split()
            solver = ({"software": " ".join(parts[:-1]), "version": parts[-1]}
                      if len(parts) > 1 and re.match(r"^v?\d", parts[-1]) else {"software": solver})
        prev: Optional[str] = None
        for key, kind in CHAIN_KEYS:
            if key not in obj:
                continue
            v = obj[key]
            spec = v if isinstance(v, dict) else {"name": str(v)}
            aid = str(spec.get("id") or spec.get("name") or key)
            known = {k: spec[k] for k in spec if k in Artifact.__dataclass_fields__ and k not in ("id", "kind", "inputs")}
            params = {k: spec[k] for k in spec if k not in Artifact.__dataclass_fields__}
            if kind == "simulation" and isinstance(solver, dict):
                known = {**{k: v for k, v in solver.items() if k in Artifact.__dataclass_fields__}, **known}
            a = Artifact(id=aid, kind=kind, **{"name": aid, **known}, parameters=params,
                         inputs=[prev] if prev else [])
            g.add(a)
            prev = aid
        if not g.nodes:
            raise ValueError("no artifacts: give {'artifacts': [...]} or the chain keys "
                             + ", ".join(k for k, _ in CHAIN_KEYS))
        return g

    def to_json(self) -> Dict[str, Any]:
        return {"schema": "pinneapple.lineage/1", "project": self.project,
                "artifacts": [a.to_dict() for a in self.nodes.values()]}

    # ------------------------------------------------------------------ structure
    def edges(self) -> List[Tuple[str, str]]:
        return [(i, a.id) for a in self.nodes.values() for i in a.inputs if i in self.nodes]

    def children(self, aid: str) -> List[str]:
        return [a.id for a in self.nodes.values() if aid in a.inputs]

    def _cycles(self) -> List[List[str]]:
        color: Dict[str, int] = {}
        stack: List[str] = []
        found: List[List[str]] = []

        def visit(n: str):
            color[n] = 1
            stack.append(n)
            for c in self.children(n):
                if color.get(c) == 1:
                    found.append(stack[stack.index(c):] + [c])
                elif c not in color:
                    visit(c)
            stack.pop()
            color[n] = 2
        for n in self.nodes:
            if n not in color:
                visit(n)
        return found

    def layers(self) -> Dict[str, int]:
        """Column of each node for a left-to-right layout (longest path from a source); cycles broken."""
        depth: Dict[str, int] = {}

        def d(n: str, seen: Set[str]) -> int:
            if n in depth:
                return depth[n]
            if n in seen:
                return 0
            seen = seen | {n}
            ins = [i for i in self.nodes[n].inputs if i in self.nodes]
            depth[n] = 0 if not ins else 1 + max(d(i, seen) for i in ins)
            return depth[n]
        for n in self.nodes:
            d(n, set())
        return depth

    # ------------------------------------------------------------------ questions
    def upstream(self, aid: str) -> List[str]:
        """Everything this artifact was produced from, nearest first ("how was this produced?")."""
        out, q, seen = [], deque(self.nodes[aid].inputs), {aid}
        while q:
            n = q.popleft()
            if n in seen or n not in self.nodes:
                continue
            seen.add(n)
            out.append(n)
            q.extend(self.nodes[n].inputs)
        return out

    def downstream(self, aid: str) -> List[str]:
        """Everything derived from this artifact ("what does a change here affect?")."""
        out, q, seen = [], deque(self.children(aid)), {aid}
        while q:
            n = q.popleft()
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
            q.extend(self.children(n))
        return out

    def sources_of_kind(self, aid: str, kind: str) -> List[str]:
        """E.g. which simulations a model was trained on: upstream artifacts of a kind."""
        return [n for n in self.upstream(aid) if self.nodes[n].kind == kind]

    # ------------------------------------------------------------------ checks
    def checks(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        add = lambda status, rule, aid, title, detail="", fix="": out.append(  # noqa: E731
            {"status": status, "rule": rule, "artifact": aid, "title": title, "detail": detail, "fix": fix})
        for a in self.nodes.values():
            for i in a.inputs:
                if i not in self.nodes:
                    add("fail", "missing_input", a.id, f"{a.id} is derived from {i}, which is not in the lineage",
                        "The chain is broken here: nobody can tell how this input was produced.",
                        f"Add {i} (or correct the reference).")
            if a.hash and a.actual_hash and not a.actual_hash.startswith(a.hash.lower()[:len(a.actual_hash)]) and \
                    not a.hash.lower().startswith(a.actual_hash[:len(a.hash)]):
                add("fail", "hash_mismatch", a.id, f"{a.file or a.id} changed since it was recorded",
                    f"Recorded sha256 {a.hash[:12]}…, uploaded file {a.actual_hash[:12]}…: the file is not the one "
                    "this lineage describes, so everything derived from it is unverified.",
                    "Re-record the artifact (new version) and re-run what depends on it, or restore the recorded file.")
            ta = _ts(a.timestamp)
            for i in a.inputs:
                b = self.nodes.get(i)
                tb = _ts(b.timestamp) if b else None
                if ta and tb and tb > ta:
                    add("warn", "stale", a.id, f"{a.id} is older than its input {i}",
                        f"{i} was modified at {b.timestamp}, after {a.id} was produced ({a.timestamp}): {a.id} may "
                        "come from a previous version of it.",
                        f"Re-run {a.id} from the current {i}, or record which version of {i} it used.")
            if a.kind in ("simulation", "model", "prediction", "dataset", "mesh") and not a.inputs:
                add("warn", "no_inputs", a.id, f"{a.kind.capitalize()} {a.id} has no recorded inputs",
                    "A derived artifact without a source cannot be traced or reproduced.",
                    "Record what it was produced from.")
            missing = [f for f in ("software", "version", "timestamp", "hash", "owner") if not getattr(a, f)]
            if a.kind not in ("file",) and len(missing) >= 3:
                add("info", "incomplete", a.id, f"{a.id}: provenance incomplete (no {', '.join(missing)})",
                    "Traceability needs who, when, with what and the content hash.", "Fill in the missing fields.")
        for cyc in self._cycles():
            add("fail", "cycle", cyc[0], "Cycle: " + " → ".join(cyc), "An artifact cannot be derived from itself.",
                "Remove the wrong edge.")
        # mixed revisions: one artifact (transitively) built from two revisions of the same item
        for a in self.nodes.values():
            if a.kind not in ("dataset", "model", "prediction"):
                continue                                              # comparisons/reports mix revisions on purpose
            up = self.upstream(a.id)
            by_item: Dict[str, Set[str]] = {}
            for n in up:
                it = self.nodes[n].item
                if it:
                    by_item.setdefault(it, set()).add(n)
            for it, revs in by_item.items():
                if len(revs) > 1:
                    # report once, at the first artifact that mixes them (not again downstream)
                    direct = [i for i in a.inputs if i in self.nodes and self.nodes[i].kind in ("dataset", "model", "prediction")]
                    if any(len({x for x in revs if x in ([d] + self.upstream(d))}) > 1 for d in direct):
                        continue                                      # already reported upstream
                    add("warn", "mixed_revisions", a.id, f"{a.id} combines {len(revs)} revisions of {it}: "
                        f"{', '.join(sorted(revs))}", "Results from different revisions are mixed in one "
                        "artifact (e.g. a model trained on simulations of two geometry revisions).",
                        "Confirm it is intended, or rebuild from one revision.")
        if self.nodes and not [c for c in out if c["status"] in ("fail", "warn")]:
            add("pass", "ok", None, f"Lineage consistent: {len(self.nodes)} artifacts, {len(self.edges())} links")
        return out

    # ------------------------------------------------------------------ exports
    def to_prov(self) -> Dict[str, Any]:
        """W3C PROV-JSON: artifacts as entities, their production as activities, owners as agents."""
        ns = "pinneapple"
        doc: Dict[str, Any] = {"prefix": {ns: "https://pinneapple.org/lineage#"}, "entity": {}, "activity": {},
                               "agent": {}, "wasDerivedFrom": {}, "wasGeneratedBy": {}, "wasAttributedTo": {},
                               "used": {}}
        k = 0
        for a in self.nodes.values():
            e = f"{ns}:{re.sub(r'[^A-Za-z0-9_.-]', '_', a.id)}"
            attrs = {f"{ns}:kind": a.kind, "prov:label": a.name or a.id}
            for f in ("file", "version", "software", "hash", "origin", "item"):
                if getattr(a, f):
                    attrs[f"{ns}:{f}"] = getattr(a, f)
            if a.timestamp:
                attrs["prov:generatedAtTime"] = a.timestamp
            doc["entity"][e] = attrs
            if a.inputs:
                act = f"{ns}:produce_{re.sub(r'[^A-Za-z0-9_.-]', '_', a.id)}"
                doc["activity"][act] = {f"{ns}:software": a.software} if a.software else {}
                doc["wasGeneratedBy"][f"_:g{k}"] = {"prov:entity": e, "prov:activity": act}
                for i in a.inputs:
                    ie = f"{ns}:{re.sub(r'[^A-Za-z0-9_.-]', '_', i)}"
                    k += 1
                    doc["wasDerivedFrom"][f"_:d{k}"] = {"prov:generatedEntity": e, "prov:usedEntity": ie}
                    doc["used"][f"_:u{k}"] = {"prov:activity": act, "prov:entity": ie}
            if a.owner:
                ag = f"{ns}:{re.sub(r'[^A-Za-z0-9_.-]', '_', a.owner)}"
                doc["agent"][ag] = {"prov:label": a.owner}
                k += 1
                doc["wasAttributedTo"][f"_:a{k}"] = {"prov:entity": e, "prov:agent": ag}
            k += 1
        return {kk: v for kk, v in doc.items() if v}

    def to_markdown(self) -> str:
        lay = self.layers()
        lines = [f"# Lineage: {self.project or 'project'}", ""]
        for a in sorted(self.nodes.values(), key=lambda x: (lay.get(x.id, 0), x.id)):
            lines.append(f"## {a.id} ({a.kind})")
            for f in ("name", "file", "version", "software", "timestamp", "hash", "owner", "origin", "item"):
                v = getattr(a, f)
                if v:
                    lines.append(f"- **{f}**: {v}")
            if a.parameters:
                lines.append("- **parameters**: " + ", ".join(f"{k} = {v}" for k, v in a.parameters.items()))
            if a.inputs:
                lines.append("- **derived from**: " + ", ".join(a.inputs))
            lines.append("")
        cs = self.checks()
        lines += ["## Checks", ""] + [f"- {c['status'].upper()}: {c['title']}" for c in cs]
        return "\n".join(lines) + "\n"
