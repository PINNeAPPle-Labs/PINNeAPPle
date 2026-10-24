""""My Hub": the tools a person composed and saved, as Hub items in a local folder (one JSON file each).

    from pinneapple_hub.library.myhub import MyHub
    mine = MyHub("~/.pinneapple/hub")                   # the Hub app keeps one folder per caller
    item = mine.save(compose_result, name="Heat rod", run=validation)   # -> {"id": "mine:heat_rod", ...}
    mine.list(); mine.get("mine:heat_rod"); mine.delete("mine:heat_rod")
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .sandbox import code_hash

MAX_ITEMS = 200
MAX_BYTES = 512 * 1024


class MyHubError(ValueError):
    pass


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:48] or "tool"


class MyHub:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, item_id: str) -> Path:
        if not item_id.startswith("mine:") or not re.fullmatch(r"[a-z0-9_]+", item_id[5:]):
            raise KeyError(item_id)
        return self.root / f"{item_id[5:]}.json"

    def save(self, composed: dict[str, Any], *, name: str, description: str = "",
             run: dict[str, Any] | None = None) -> dict[str, Any]:
        """Store a :func:`pinneapple_hub.library.compose.compose` result as a Hub item (``kind`` tool, ``subkind``
        "my tool"); saving the same name again replaces it. ``run``: a sandbox run of its code, kept only when it
        ran this exact code."""
        stem = _slug(name)
        if len(self.list()) >= MAX_ITEMS and not (self.root / f"{stem}.json").exists():
            raise MyHubError(f"My Hub holds at most {MAX_ITEMS} tools")
        code = composed["code"]
        run = run if run and run.get("code_hash") == code_hash(code) else None
        types = [n["type"] for n in composed["flow"].get("nodes", [])]
        item = {"id": f"mine:{stem}", "kind": "tool", "subkind": "my tool", "name": stem, "title": name[:80],
                "summary": description[:200] or f"{len(types)} blocks: " + ", ".join(types[:6]),
                "description": description[:2000], "domain": "general", "tags": sorted(set(types))[:12],
                "quality": None, "code": code, "flow": composed["flow"],
                "run": ({k: run.get(k) for k in ("ok", "stdout", "stderr", "seconds", "code_hash")} |
                        {"status": "ok" if run.get("ok") else "failed"}) if run else
                {"status": "not run", "code_hash": code_hash(code)},
                "results": None, "examples": [f"node:{t}" for t in dict.fromkeys(types)], "files": [], "links": {},
                "extra": {"expose": composed.get("expose", [])},
                "saved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        text = json.dumps(item, indent=1, ensure_ascii=False)
        if len(text.encode()) > MAX_BYTES:
            raise MyHubError("this tool is too large to save")
        (self.root / f"{stem}.json").write_text(text)
        return item

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except ValueError:
                continue
        return sorted(out, key=lambda i: i.get("saved", ""), reverse=True)

    def get(self, item_id: str) -> dict[str, Any]:
        p = self._path(item_id)
        if not p.exists():
            raise KeyError(item_id)
        return json.loads(p.read_text())

    def delete(self, item_id: str) -> None:
        p = self._path(item_id)
        if not p.exists():
            raise KeyError(item_id)
        p.unlink()
