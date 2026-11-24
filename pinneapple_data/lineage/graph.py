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
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

KINDS = ["problem", "geometry", "mesh", "simulation", "result", "experiment", "post-processing", "dataset", "model",
         "prediction", "validation", "report", "code", "file"]
CHAIN_KEYS = [("problem", "problem"), ("geometry", "geometry"), ("cad", "geometry"), ("mesh", "mesh"),
              ("simulation", "simulation"), ("postprocessing", "post-processing"), ("post_processing", "post-processing"),
              ("dataset", "dataset"), ("model", "model"), ("prediction", "prediction"), ("result", "prediction"),
              ("report", "report")]
# the chain form is one straight line and a validation record is a branch off the model, so it has no key here:
# write it as an artifact ({"artifacts": [{"kind": "validation", "inputs": ["MODEL"]}]}) or let
# pinneapple_data.lineage.from_registry build it from the trust report stored with a model version
PROVENANCE_FIELDS = ("file", "version", "software", "timestamp", "hash", "owner", "origin")
# what a prediction is asked to answer for, in the order of the question (issue #156)
TRACE_GROUPS = ("problem", "geometry", "simulations", "datasets", "models", "validation")
# the kinds that answer each group. A stored dataset is one simulation run's arrays (DatasetStore), so it answers for
# the run as well when the lineage has no separate simulation node; a mesh answers for the geometry.
TRACE_KINDS = {"problem": ("problem",), "geometry": ("geometry", "mesh"), "simulations": ("simulation", "dataset"),
               "datasets": ("dataset",), "models": ("model",), "validation": ("validation",)}


@dataclass
class Artifact:
    id: str
    kind: str = "file"
    name: str = ""
    file: str | None = None
    version: str | None = None
    software: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None            # ISO 8601
    hash: str | None = None                 # sha256 (hex, full or prefix) of the file as recorded
    owner: str | None = None
    origin: str | None = None               # where it lives / came from (path, URL, system)
    inputs: list[str] = field(default_factory=list)
    item: str | None = None                 # the thing this is a revision of ("bracket CAD"); revisions share it
    notes: str | None = None
    detected: str | None = None             # how it was found (auto-detection rule), if not declared
    actual_hash: str | None = None          # sha256 of the uploaded file, when the file was uploaded

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [], {})}

    def completeness(self) -> float:
        return sum(1 for f in PROVENANCE_FIELDS if getattr(self, f)) / len(PROVENANCE_FIELDS)

    @property
    def verified(self) -> bool:
        """The recorded hash matches the uploaded file (either may be a prefix of the other, 16 hex digits at
        least)."""
        if not (self.hash and self.actual_hash):
            return False
        rec, act = self.hash.lower(), self.actual_hash
        return act.startswith(rec[:16]) or rec.startswith(act[:16])


def _ts(s: str | None) -> _dt.datetime | None:
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
        self.nodes: dict[str, Artifact] = {}
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
    def from_json(cls, obj: dict[str, Any]) -> Lineage:
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
        prev: str | None = None
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

    def to_json(self) -> dict[str, Any]:
        return {"schema": "pinneapple.lineage/1", "project": self.project,
                "artifacts": [a.to_dict() for a in self.nodes.values()]}

    # ------------------------------------------------------------------ structure
    def edges(self) -> list[tuple[str, str]]:
        return [(i, a.id) for a in self.nodes.values() for i in a.inputs if i in self.nodes]

    def children(self, aid: str) -> list[str]:
        return [a.id for a in self.nodes.values() if aid in a.inputs]

    def _cycles(self) -> list[list[str]]:
        color: dict[str, int] = {}
        stack: list[str] = []
        found: list[list[str]] = []

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

    def layers(self) -> dict[str, int]:
        """Column of each node for a left-to-right layout (longest path from a source); cycles broken."""
        depth: dict[str, int] = {}

        def d(n: str, seen: set[str]) -> int:
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
    def upstream(self, aid: str) -> list[str]:
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

    def downstream(self, aid: str) -> list[str]:
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

    def sources_of_kind(self, aid: str, kind: str) -> list[str]:
        """E.g. which simulations a model was trained on: upstream artifacts of a kind."""
        return [n for n in self.upstream(aid) if self.nodes[n].kind == kind]

    def brief(self, aid: str) -> dict[str, Any]:
        """The short form of an artifact, for an answer: what it is, which version, where it lives."""
        a = self.nodes[aid]
        d: dict[str, Any] = {"id": a.id, "kind": a.kind, "name": a.name or a.id}
        for f in ("version", "file", "software", "timestamp", "hash", "origin", "item"):
            if getattr(a, f):
                d[f] = getattr(a, f)
        if a.inputs:
            d["derived_from"] = list(a.inputs)
        return d

    def trace(self, aid: str) -> dict[str, Any]:
        """Everything this prediction was built from, grouped by the question it answers (issue #156).

        ``{"prediction": ..., "problem": [...], "geometry": [...], "simulations": [...], "datasets": [...],
        "models": [...], "validation": [...], "missing": [...], "complete": bool}``

        Each group holds the upstream artifacts of its kind, nearest first, in the short form of :meth:`brief`, so a
        dataset arrives with its version and a model with the version that produced the prediction. ``missing`` names
        the groups nothing in the lineage answers, and ``complete`` is false when there is at least one: the answer
        says what is not known instead of leaving a hole to be read as "fine".
        """
        if aid not in self.nodes:
            raise KeyError(f"{aid} is not in the lineage")
        upstream = self.upstream(aid)
        # a validation record hangs off the model it validates, so it is downstream of the model and never upstream
        # of the prediction: gather the ones attached to the models this artifact was produced from
        validations = [n for n in upstream if self.nodes[n].kind == "validation"]
        for model in [n for n in upstream if self.nodes[n].kind == "model"]:
            validations += [c for c in self.children(model) if self.nodes[c].kind == "validation"]
        out: dict[str, Any] = {"prediction": self.brief(aid)}
        missing: list[str] = []
        for group in TRACE_GROUPS:
            if group == "validation":
                found = [self.brief(n) for n in dict.fromkeys(validations)]
            else:
                found = [self.brief(n) for n in upstream if self.nodes[n].kind in TRACE_KINDS[group]]
            out[group] = found
            if not found:
                missing.append(group)
        out["missing"] = missing
        out["complete"] = not missing
        return out

    # ------------------------------------------------------------------ checks
    def checks(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
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
            by_item: dict[str, set[str]] = {}
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
    def flags(self, checks: list[dict[str, Any]] | None = None) -> dict[str, str]:
        """Worst check status per artifact: ``{artifact id: "fail" | "warn"}`` (artifacts that pass are absent)."""
        flag: dict[str, str] = {}
        for c in self.checks() if checks is None else checks:
            if c.get("artifact") and c["status"] in ("fail", "warn") and flag.get(c["artifact"]) != "fail":
                flag[c["artifact"]] = c["status"]
        return flag

    def report(self) -> dict[str, Any]:
        """Everything a viewer needs: the nodes (each with its layer, completeness, outputs, flag and hash
        verification), the edges, the checks, the overall status (PASS / WARNING / FAIL) and a summary (artifact and
        link counts, artifacts per kind, verified hashes, mean completeness)."""
        lay, checks = self.layers(), self.checks()
        flag = self.flags(checks)
        nodes = []
        for a in self.nodes.values():
            d = a.to_dict()
            d.update(layer=lay.get(a.id, 0), completeness=round(a.completeness(), 2), outputs=self.children(a.id),
                     flag=flag.get(a.id), verified=a.verified)
            nodes.append(d)
        statuses = {c["status"] for c in checks}
        worst = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
        kinds = [a.kind for a in self.nodes.values()]
        return {"project": self.project, "nodes": nodes, "edges": self.edges(), "checks": checks,
                "status": {"fail": "FAIL", "warn": "WARNING", "pass": "PASS"}[worst],
                "summary": {"artifacts": len(self.nodes), "links": len(self.edges()),
                            "by_kind": {k: kinds.count(k) for k in set(kinds)},
                            "verified": sum(1 for n in nodes if n["verified"]),
                            "completeness": round(sum(a.completeness() for a in self.nodes.values())
                                                  / max(len(self.nodes), 1), 2)}}

    def to_prov(self) -> dict[str, Any]:
        """W3C PROV-JSON: artifacts as entities, their production as activities, owners as agents."""
        ns = "pinneapple"
        doc: dict[str, Any] = {"prefix": {ns: "https://pinneapple.org/lineage#"}, "entity": {}, "activity": {},
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
