"""A browsable catalogue of the lab: one self-contained HTML page (data and thumbnails embedded).

    from pinneapple_lab import LabStore
    LabStore("lab").catalog_html("lab/index.html")      # or: python -m pinneapple_lab report --html
"""
from __future__ import annotations

import base64
import glob
import io
import json
import os
import time
from typing import Any

_TEMPLATE = os.path.join(os.path.dirname(__file__), "catalog_template.html")


def thumbnail_bytes(path: str, width: int = 360) -> bytes | None:
    """JPEG thumbnail of an image (first frame of a GIF), or None if it cannot be read."""
    try:
        from PIL import Image
        im = Image.open(path)
        im.seek(0)
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, max(1, int(im.height * width / im.width))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=78, optimize=True)
        return buf.getvalue()
    except Exception:
        return None


SHOWCASE_TOKENS = (("interstellar", 9), ("beauty", 8), ("render", 6), ("flight", 6), ("twin", 5), ("streamlines", 5),
                   ("wake", 3), ("cp", 3), ("vorticity", 3), (".gif", 4))


def showcase_score(path: str) -> int:
    n = os.path.basename(path).lower()
    return sum(w for t, w in SHOWCASE_TOKENS if t in n)


def _gif_small(path: str, width: int = 420, max_frames: int = 40) -> str | None:
    """A smaller animated GIF (fewer frames, narrower) as a data URI, for the gallery."""
    try:
        from PIL import Image, ImageSequence
        im = Image.open(path)
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
        step = max(1, len(frames) // max_frames)
        frames = frames[::step][:max_frames]
        h = max(1, int(frames[0].height * width / frames[0].width))
        frames = [f.resize((width, h)).convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
        buf = io.BytesIO()
        frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], loop=0,
                       duration=im.info.get("duration", 80) * step, optimize=True)
        return "data:image/gif;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _thumb(path: str, width: int = 360) -> str | None:
    data = thumbnail_bytes(path, width)
    return None if data is None else "data:image/jpeg;base64," + base64.b64encode(data).decode()


def _load(path: str, default: Any) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def collect(store, *, thumbs_per_run: int = 1, max_runs_per_experiment: int = 400,
            thumbs: str = "embed") -> dict[str, Any]:
    """Everything the catalogue page shows. ``thumbs="embed"`` inlines JPEG thumbnails (a self-contained file);
    ``thumbs="url"`` points them at the server's ``thumb/`` route (``pinneapple_lab.server``)."""
    from .spec import get
    try:
        from .curation import curate
        cur = curate(store)
    except Exception:                                      # noqa: BLE001 - the page works without curation
        cur = {"runs": {}, "experiments": {}}
    experiments = []
    showcase: list = []
    for exp, st in sorted(store.status().items()):
        try:
            cls = get(exp)
            meta = {"description": cls.description, "tags": list(cls.tags), "version": cls.version,
                    "defaults": cls.params}
        except KeyError:
            meta = {"description": "", "tags": [], "version": "", "defaults": {}}
        runs = []
        for r in store.runs(exp)[-max_runs_per_experiment:]:
            d = store.run_dir(exp, r["run_id"])
            rec = _load(os.path.join(d, "run.json"), {})
            figs = sorted(glob.glob(os.path.join(d, "figures", "*.png")) + glob.glob(os.path.join(d, "figures", "*.jpg")))
            gifs = sorted(glob.glob(os.path.join(d, "figures", "*.gif")))
            figs_all = sorted(figs + gifs, key=lambda f: -showcase_score(f))
            shown = figs_all[:thumbs_per_run]
            for f in figs_all[:3]:
                sc_ = showcase_score(f)
                if sc_ >= 5:
                    showcase.append((sc_, exp, r["run_id"], f))
            if thumbs == "url":
                tl = ["thumb/" + os.path.relpath(f, os.path.join(store.root, "runs")) for f in shown]
            else:
                tl = [t for t in (_thumb(f) for f in shown) if t]
            runs.append({
                "id": r["run_id"], "status": r["status"], "seconds": r["seconds"], "params": r["params"],
                "metrics": {k: v for k, v in r["metrics"].items() if isinstance(v, (int, float, str)) or v is None},
                "checks": _load(os.path.join(d, "validation.json"), []),
                "stages": rec.get("stages", []), "code": rec.get("code", {}).get("files", []),
                "git": rec.get("git"), "started": r["started"], "datasets": rec.get("datasets", {}),
                "figures": [os.path.relpath(f, store.root) for f in figs + gifs],
                "dir": os.path.relpath(d, os.path.join(store.root, "runs")), "thumbs": tl,
                "error": (r.get("error") or "")[-1500:],
                "quality": _quality(cur["runs"].get(r["run_id"])),
            })
        ds = {}
        for x in store.datasets(exp):
            ds[x["name"]] = ds.get(x["name"], 0) + x["n_samples"]
        q = cur["experiments"].get(exp)
        experiments.append({"name": exp, **meta, "status": st, "runs": runs, "datasets": ds,
                            "quality": None if q is None else {k: q[k] for k in ("tier", "tier_name", "best_run",
                                                                                 "best_score", "tier_counts",
                                                                                 "usable_fraction", "readiness", "gaps",
                                                                                 "reviewed")}})
    items = []
    for key, it in cur.get("items", {}).items():
        items.append({"key": key, "experiment": it["experiment"], "case": it["case"], "title": it["title"],
                      "tier": it["tier"], "tier_name": it["tier_name"], "score": it["best_score"],
                      "best_run": it["best_run"], "story": it.get("story", ""), "runs": it["runs"],
                      "usable": it["usable_fraction"], "plan": it["plan"], "description": it.get("description", "")})
    gallery, seen = [], set()
    for _, exp, rid, f in sorted(showcase, key=lambda t: -t[0]):
        key = (exp, os.path.basename(f))
        if key in seen or len(gallery) >= 16:
            continue
        seen.add(key)
        rel = os.path.relpath(f, os.path.join(store.root, "runs"))
        if thumbs == "url":
            src = ("files/" if f.endswith(".gif") else "thumb/") + rel
        else:
            src = _gif_small(f) if f.endswith(".gif") else _thumb(f, 640)
        if src:
            gallery.append({"experiment": exp, "run": rid, "src": src, "name": os.path.basename(f),
                            "file": os.path.relpath(f, store.root)})
    return {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "experiments": experiments,
            "portfolio": items, "gallery": gallery}


def _quality(a):
    if not a:
        return None
    return {"tier": a["tier"], "tier_name": a["tier_name"], "score": a["score"], "coverage": a["coverage"],
            "dims": {k: {"score": d["score"], "status": d["status"], "tier": d["tier"], "evidence": d["evidence"][:4]}
                     for k, d in a["dimensions"].items()},
            "readiness": a["readiness"], "gaps": a["gaps"], "reviewed": a["reviewed"]}


def write_html(store, path: str | None = None, *, standalone: bool = True, **kw) -> str:
    """Write the catalogue page. ``standalone`` wraps it in a full HTML document (open it from disk); without it,
    the body-level page is written (for hosts that add their own document skeleton)."""
    path = path or os.path.join(store.root, "index.html")
    with open(path, "w") as f:
        f.write(render_page(collect(store, **kw), standalone=standalone))
    return path


def render_page(data: dict[str, Any], *, standalone: bool = True) -> str:
    """The catalogue page with ``data`` (from ``collect``) embedded."""
    with open(_TEMPLATE) as f:
        html = f.read()
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/null", blob)
    if standalone:
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n'
                + html + "\n</body>\n</html>\n")
    return html
