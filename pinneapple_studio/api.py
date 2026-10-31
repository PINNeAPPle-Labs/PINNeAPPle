"""The Studio's JSON API as a FastAPI router: the catalog of tools, the gallery and the jobs that run them.

Mount it in any app (the local ``pinneapple studio`` server, the Engineering API):

    app.include_router(pinneapple_studio.api.router(), prefix="/api/studio")

Security belongs to the app that mounts it (login, API keys, rate limits, same-origin writes); the router only reads
the repository and submits jobs to the queue it is given (:func:`pinneapple_studio.runner.queue` by default).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse

from . import catalog as CAT
from . import gallery as GAL
from .search import Index


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.catalog: dict[str, Any] | None = None
        self.index: Index | None = None
        self.gallery: list[dict[str, Any]] | None = None

    def cat(self) -> dict[str, Any]:
        with self.lock:
            if self.catalog is None:
                self.catalog = CAT.load()
                self.index = Index(self.catalog)
            return self.catalog

    def idx(self) -> Index:
        self.cat()
        assert self.index is not None
        return self.index

    def gal(self) -> list[dict[str, Any]]:
        with self.lock:
            if self.gallery is None:
                self.gallery = GAL.load()
            return self.gallery


def _card(t: dict[str, Any]) -> dict[str, Any]:
    return {k: t[k] for k in ("id", "name", "module", "package", "kind", "category", "icon", "status", "doc1",
                              "tasks", "areas", "products")} | {"pt": t["pt"][:400], "n_tests": t["n_tests"],
                                                                 "examples": t["examples"][:5]}


def router(queue=None) -> APIRouter:
    st = _State()
    r = APIRouter(tags=["studio"])
    jobs: dict[str, Any] = {"q": queue}

    def q():
        if jobs["q"] is None:
            from .runner import queue as make
            jobs["q"] = make()
        return jobs["q"]

    @r.get("/summary")
    def summary() -> dict[str, Any]:
        c = st.cat()
        return {"commit": c["commit"], "counts": c["counts"], "status_rules": c["status_rules"],
                "categories": c["categories"], "tasks": c["tasks"], "areas": c["areas"], "products": c["products"],
                "facets": st.idx().facets(), "gallery": GAL.summary(st.gal()),
                "prose_unmatched": {k: len(v) for k, v in c["prose_unmatched"].items()},
                "prose_missing": {k: len(v) for k, v in c["prose_missing"].items()}}

    @r.get("/tools")
    def tools(q_: str = Query("", alias="q"), category: str | None = None, status: str | None = None,
              task: str | None = None, area: str | None = None, product: str | None = None,
              package: str | None = None, kind: str | None = None, limit: int = Query(60, le=500),
              offset: int = 0) -> dict[str, Any]:
        hits = st.idx().search(q_, category=category, status=status, task=task, area=area, product=product,
                               package=package, kind=kind, limit=100_000)
        tools_ = st.cat()["tools"]
        page = hits[offset:offset + limit]
        return {"total": len(hits), "items": [_card(tools_[h["id"]]) | {"score": h["score"]} for h in page],
                "facets": st.idx().facets([h["id"] for h in hits])}

    @r.get("/tools/{tool_id}")
    def tool(tool_id: str) -> dict[str, Any]:
        c = st.cat()
        t = c["tools"].get(tool_id)
        if t is None:
            raise HTTPException(404, f"unknown tool {tool_id}")
        return t | {"package_pt": c["packages"].get(t["package"], {}).get("pt", {}),
                    "module_doc": c["modules"].get(t["module"], {}).get("doc", "")}

    @r.get("/compare")
    def compare(ids: str) -> list[dict[str, Any]]:
        c = st.cat()
        out = []
        for i in ids.split(","):
            if i in c["tools"]:
                t = c["tools"][i]
                out.append(_card(t) | {"signature": t["signature"], "category_reason": t["category_reason"],
                                       "status_reason": t["status_reason"], "form": t.get("form"),
                                       "related": t["related"]})
        return out

    @r.get("/tree")
    def tree(package: str | None = None) -> dict[str, Any]:
        c = st.cat()
        out: dict[str, Any] = {}
        for t in c["tools"].values():
            if package and t["package"] != package:
                continue
            out.setdefault(t["package"], {}).setdefault(t["module"], []).append(
                {"id": t["id"], "name": t["name"], "category": t["category"], "status": t["status"]})
        return {"packages": {p: {"pt": c["packages"].get(p, {}).get("pt", {}).get("objetivo", ""),
                                 "doc1": c["packages"].get(p, {}).get("doc1", ""), "modules": mods}
                             for p, mods in sorted(out.items())}}

    @r.get("/modules/{module}")
    def module(module: str) -> dict[str, Any]:
        c = st.cat()
        if module not in c["modules"]:
            raise HTTPException(404, f"unknown module {module}")
        return c["modules"][module] | {"tools": [_card(t) for t in c["tools"].values() if t["module"] == module]}

    @r.get("/glossary")
    def glossary() -> list[dict[str, str]]:
        return st.cat()["glossary"]

    @r.get("/gallery")
    def gallery(kind: str | None = None, state: str | None = None, q_: str = Query("", alias="q")):
        items = st.gal()
        words = q_.lower().split()
        sel = [e for e in items if (not kind or e["kind"] == kind) and (not state or e["state"] == state)
               and all(w in (e["title"] + " " + e.get("description", "") + " " + e["key"]).lower() for w in words)]
        return {"total": len(sel), "summary": GAL.summary(items),
                "items": [{k: e.get(k) for k in ("id", "kind", "key", "title", "description", "domain", "state",
                                                 "blocked_by", "flows", "measured")} for e in sel]}

    @r.get("/gallery/{entry_id:path}")
    def gallery_entry(entry_id: str) -> dict[str, Any]:
        e = next((e for e in st.gal() if e["id"] == entry_id), None)
        if e is None:
            raise HTTPException(404, f"unknown gallery entry {entry_id}")
        return e

    @r.post("/jobs", status_code=202)
    def submit(body: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        kind, params = body.get("kind"), body.get("params") or {}
        if kind == "tool":
            t = st.cat()["tools"].get(params.get("tool", ""))
            if t is None or t["category"] != "executable":
                raise HTTPException(400, "not an executable tool")
        elif kind == "gallery":
            e = next((e for e in st.gal() if e["id"] == params.get("entry")), None)
            if e is None:
                raise HTTPException(400, "unknown gallery entry")
            if e["blocked_by"]:
                raise HTTPException(409, {"detail": "cannot run here", "blocked_by": e["blocked_by"]})
        else:
            raise HTTPException(400, "kind must be tool or gallery")
        return {"job_id": q().submit(kind, params, owner=str(body.get("owner", "")))}

    @r.get("/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        try:
            return q().get(job_id)
        except KeyError:
            raise HTTPException(404, "unknown job") from None

    @r.get("/jobs/{job_id}/logs")
    def logs(job_id: str, since: int = 0) -> list[dict[str, Any]]:
        return q().logs(job_id, since=since)

    @r.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict[str, Any]:
        return q().cancel(job_id)

    @r.get("/jobs/{job_id}/files/{name:path}")
    def job_file(job_id: str, name: str):
        base = (Path(q().workdir) / job_id).resolve()
        p = (base / name).resolve()
        if base not in p.parents or not p.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(p)

    return r
