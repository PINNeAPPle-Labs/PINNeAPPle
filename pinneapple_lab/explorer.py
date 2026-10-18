"""The experiments page: every PINNeAPPle experiment with its A-F seal, trust card, measured metrics, next steps,
figures, movies, viewers and links, served by the Lab server (``/experiments``) or exported as static HTML.

What it lists, from two sources and nothing else:

* every item of the Lab curation (``<lab>/curation.json``): tier and seal reason, the six-stage trust card, the
  best run's measured metrics, its figures (stamped with the seal), movies and 3-D viewer, the failing check for E
  and F, and the path to the next letter (``pinneapple_lab.curation.next_steps``);
* every experiment manifest under ``examples/`` (``pinneapple_lab.manifest``). A folder whose scripts the Lab has
  curated shows those items and the best letter among them; any other shows "not curated" with the command that
  curates it (``python -m pinneapple_lab register <folder>``).

Counts per letter are counted from the list itself. Nothing is invented: an entry without metrics shows none.

    from pinneapple_lab.explorer import collect, coverage, write_static
    data = collect(LabStore("lab"))                       # what the page and /api/experiments show
    write_static(LabStore("lab"), "site/experiments")      # index.html + media/ (stamped figures, seals, movies)
    python -m pinneapple_lab site --out site               # the catalogue and site/experiments/
    python -m pinneapple_lab coverage -o docs/lab/coverage.md
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.parse
from typing import Any

from . import manifest as M
from .curation import TIERS, _docs_mentions
from .seal import SEALABLE, TIER_COLOURS, TIER_INK, TIER_TEXT, load_curation, seal_for

GITHUB = "https://github.com/PINNeAPPle-Labs/PINNeAPPle"
FIG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif")
VID_EXT = (".mp4", ".webm")
STATIC_VIDEO_MAX = 10 * 1024 * 1024          # larger movies stay on the server in a static export
TEMPLATE = os.path.join(os.path.dirname(__file__), "experiments_template.html")
UI = {
    "en": {"title": "PINNeAPPle experiments", "search": "Search title, domain, method…", "all": "All",
           "domain": "Domain", "method": "Method", "ready": "Ready to use", "sort": "Sort", "by_letter": "by letter",
           "by_date": "newest first", "not_curated": "not curated", "back": "← All experiments",
           "trust": "Trust card: six questions", "metrics": "Measured metrics", "next": "What is missing for the "
           "next letter, and how", "gallery": "Figures", "videos": "Movies", "play": "Play", "viewer": "Open the 3-D "
           "viewer", "links": "Links", "guide": "Guide", "script": "Script", "pr": "Pull request", "problem":
           "Main problem", "items": "Lab items", "how_curate": "Curate it with", "none": "None recorded.",
           "count": "experiments", "no_action": "no PINNeAPPle action yet", "why": "why", "generated": "Generated"},
    "pt": {"title": "Experimentos PINNeAPPle", "search": "Buscar título, domínio, método…", "all": "Todos",
           "domain": "Domínio", "method": "Método", "ready": "Pronto para uso", "sort": "Ordenar",
           "by_letter": "por nível", "by_date": "mais recentes", "not_curated": "sem curadoria",
           "back": "← Todos os experimentos", "trust": "Cartão de confiança: seis perguntas",
           "metrics": "Métricas medidas", "next": "O que falta para o próximo nível, e como", "gallery": "Figuras",
           "videos": "Vídeos", "play": "Reproduzir", "viewer": "Abrir o visualizador 3-D", "links": "Links",
           "guide": "Guia", "script": "Script", "pr": "Pull request", "problem": "Problema principal",
           "items": "Itens do Lab", "how_curate": "Faça a curadoria com", "none": "Nada registrado.",
           "count": "experimentos", "no_action": "ainda sem ação PINNeAPPle", "why": "porquê", "generated": "Gerado"},
}
STAGES_PT = {"data_geometry": "Dados e geometria", "model": "Modelo", "physics": "Restrições físicas",
             "benchmark": "Benchmark", "uncertainty": "Incerteza", "decision": "Decisão de engenharia"}


def _q(path: str) -> str:
    return urllib.parse.quote(path)


def _read(path: str, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _class_meta(exp: str) -> dict[str, Any]:
    try:
        from .spec import get
        cls = get(exp)
        mod = cls.__module__.replace(".", "/") + ".py"
        return {"tags": list(cls.tags), "description": cls.description, "module": mod}
    except Exception:  # noqa: BLE001
        return {"tags": [], "description": "", "module": ""}


def _lab_entry(key: str, rec: dict[str, Any], cur: dict[str, Any], root: str) -> dict[str, Any]:
    exp, rid = rec["experiment"], rec.get("best_run", "")
    run_dir = os.path.join(root, "runs", exp, rid)
    run = _read(os.path.join(run_dir, "run.json"), {})
    meta = _class_meta(exp)
    s = seal_for(key, cur)
    hl = rec.get("headline") or {}
    media = []
    fig_dir = os.path.join(run_dir, "figures")
    for f in sorted(os.listdir(fig_dir)) if os.path.isdir(fig_dir) else []:
        ext = os.path.splitext(f)[1].lower()
        if ext in FIG_EXT or ext in VID_EXT:
            media.append({"kind": "video" if ext in VID_EXT else "figure", "run": f"{exp}/{rid}/figures/{f}",
                          "name": f})
    viewers = [f"{exp}/{rid}/viewer/index.html"] if os.path.exists(os.path.join(run_dir, "viewer", "index.html")) \
        else []
    metrics = [{"name": k, "value": v, "unit": "", "how": "recorded by the Lab run (ctx.metric)"}
               for k, v in _read(os.path.join(run_dir, "metrics.json"), {}).items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    from .showcase import TITLES
    title = rec.get("title") or key
    if exp != "example_script" and "/" in (rec.get("case") or ""):
        title = rec["case"]                            # a case name with a slash ("60/40 brass") stays whole
    title = title[:1].upper() + title[1:]
    title, title_pt = TITLES.get(key, (title, title))   # the curated display names, where there are some
    desc = rec.get("description") or meta["description"]
    script = rec.get("case") if exp == "example_script" else meta["module"]
    docs = _docs_mentions(exp)
    stages = {k: {"title": v.get("title"), "title_pt": STAGES_PT.get(k, v.get("title")), "question": v.get("question"),
                  "status": v.get("status"), "evidence": (v.get("evidence") or [])[:3], "next": v.get("next", "")}
              for k, v in (rec.get("stages") or {}).items()}
    return {
        "key": key, "source": "lab", "title": title, "title_pt": title_pt, "description": desc,
        "domain": M.domain_of(" ".join([key, title, rec.get("title") or ""]), meta["tags"]), "methods": M.methods_of(" ".join([title, desc, exp, *meta["tags"]])),
        "status": "curated", "tier": rec["tier"], "tier_name": TIERS[rec["tier"]],
        "seal": {"reason": s.reason, "reason_pt": s.reason_pt, "kind": s.reason_kind, "hint": s.hint,
                 "hint_pt": s.hint_pt},
        "problem": ({"text": hl.get("text", ""), "text_pt": hl.get("text_pt", ""), "hint": hl.get("hint", ""),
                     "hint_pt": hl.get("hint_pt", "")} if rec["tier"] in ("E", "F") else None),
        "stages": stages, "metrics": metrics, "next_steps": rec.get("next_steps", []),
        "ready": [u for u, r in (rec.get("readiness") or {}).items() if r.get("ready")],
        "media": media, "viewers": viewers,
        "links": {"guide": docs[0] if docs else "", "script": script or "", "pr": []},
        "date": run.get("started") or 0, "experiment": exp, "run": rid, "seal_key": key,
    }


def _manifest_entry(path: str, m: M.Manifest, lab_items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = os.path.dirname(path)
    rel = os.path.relpath(base, M.REPO)
    linked = sorted(k for k, it in lab_items.items() if it.get("experiment") == "example_script"
                    and (it.get("case") or "").startswith(rel + "/"))
    own = f"manifest/{m.id}" if f"manifest/{m.id}" in lab_items else None
    keys = sorted(([own] if own else []) + linked, key=lambda k: (k != own, SEALABLE.index(lab_items[k]["tier"]), k))
    tiers = [lab_items[k]["tier"] for k in keys]                     # best letter first
    tier = min(tiers, key=SEALABLE.index) if tiers else None
    media = [{"kind": "figure", "repo": f"{rel}/{f}", "name": os.path.basename(f)} for f in m.files.get("figures", [])
             if f.lower().endswith(FIG_EXT)]
    media += [{"kind": "video" if f.lower().endswith(VID_EXT) else "figure", "repo": f"{rel}/{f}",
               "name": os.path.basename(f)} for f in m.files.get("videos", [])]
    return {
        "key": m.id, "source": "manifest", "title": m.title, "description": m.description, "domain": m.domain,
        "methods": m.methods, "status": "curated" if tier else "not curated", "tier": tier,
        "tier_name": TIERS[tier] if tier else "", "seal": None, "problem": None, "stages": {},
        "metrics": [x for x in m.metrics if isinstance(x.get("value"), (int, float))],
        "next_steps": [], "ready": [], "media": media, "viewers": [f"{rel}/{v}" for v in m.files.get("viewers", [])],
        "links": {"guide": m.links.get("guide", ""), "script": m.links.get("script", ""),
                  "pr": list(m.links.get("pr", []))},
        "items": [{"key": k, "tier": lab_items[k]["tier"], "title": lab_items[k].get("title", k)} for k in keys],
        "curate": None if tier else f"python -m pinneapple_lab register {rel}",
        "date": 0, "modules": m.modules, "scripts": m.scripts, "seal_key": keys[0] if keys else None,
    }


def collect(store, examples: str | None = None) -> dict[str, Any]:
    """Every experiment the page shows (see the module doc), the per-letter counts counted from them, the filter
    facets and the coverage list."""
    root = os.path.abspath(store.root)
    try:
        cur = load_curation(store)
    except FileNotFoundError:
        from .curation import curate
        cur = curate(store)
    items = cur.get("items", {})
    entries = []
    for key, rec in sorted(items.items()):
        if rec.get("experiment") == "example_script":
            continue                                   # shown inside its examples/ folder's entry
        try:
            entries.append(_lab_entry(key, rec, cur, root))
        except Exception:  # noqa: BLE001 - one unreadable run must not hide the rest
            continue
    for path, m in M.scan([examples or os.path.join(M.REPO, "examples")], lab=""):
        e = _manifest_entry(path, m, items)
        if f"manifest/{m.id}" in items:                # the Lab item made from this manifest: keep one entry
            entries = [x for x in entries if x["key"] != f"manifest/{m.id}"]
            lab = _lab_entry(f"manifest/{m.id}", items[f"manifest/{m.id}"], cur, root)
            e.update({k: lab[k] for k in ("seal", "problem", "stages", "next_steps", "ready", "metrics")
                      if lab[k]})
        else:
            for it in e["items"][:1]:                  # a folder curated through its scripts: the best item's evidence
                lab = _lab_entry(it["key"], items[it["key"]], cur, root)
                e.update({k: lab[k] for k in ("seal", "problem", "stages", "next_steps", "metrics") if lab[k]})
                e["date"] = lab["date"]
        entries.append(e)
    counts = {t: sum(e["tier"] == t for e in entries) for t in SEALABLE}
    counts["not curated"] = sum(e["tier"] is None for e in entries)
    return {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "curated": cur.get("generated", ""),
            "entries": entries, "counts": counts,
            "domains": sorted({e["domain"] for e in entries if e["domain"]}),
            "methods": sorted({x for e in entries for x in e["methods"]}),
            "tiers": {t: {"name": TIERS[t], "colour": TIER_COLOURS[t], "ink": TIER_INK[t], "en": TIER_TEXT["en"][t],
                          "pt": TIER_TEXT["pt"][t]} for t in SEALABLE},
            "ui": UI, "github": GITHUB}


def coverage(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Experiment -> curated yes / no (and its letter), one row per entry of the page."""
    return [{"key": e["key"], "title": e["title"], "source": e["source"], "curated": e["tier"] is not None,
             "tier": e["tier"] or "", "action": e.get("curate") or ""} for e in data["entries"]]


def coverage_md(data: dict[str, Any]) -> str:
    rows = coverage(data)
    n = sum(r["curated"] for r in rows)
    out = ["# Experiments coverage", "",
           f"Generated by `python -m pinneapple_lab coverage` from the Lab curation of {data['curated']} and the "
           f"experiment manifests: {len(rows)} experiments, {n} curated, {len(rows) - n} not curated.", "",
           "| experiment | source | curated | letter | how to curate |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (not r["curated"], r["key"])):
        out.append(f"| {r['title']} (`{r['key']}`) | {r['source']} | {'yes' if r['curated'] else 'no'} | "
                   f"{r['tier']} | {('`' + r['action'] + '`') if r['action'] else ''} |")
    return "\n".join(out) + "\n"


def filter_entries(entries: list[dict[str, Any]], *, letter: str | None = None, domain: str | None = None,
                   method: str | None = None, ready: bool = False, q: str | None = None,
                   sort: str = "letter") -> list[dict[str, Any]]:
    """The page's filters, also behind ``/api/experiments``: ``letter`` (A-F, a comma list, or "none" for not
    curated), ``domain``, ``method``, ``ready`` (ready for at least one use), ``q`` (words in the title,
    description, domain or methods), ``sort`` "letter" or "date"."""
    out = entries
    if letter:
        want = {x.strip().upper() if x.strip().lower() != "none" else None for x in letter.split(",")}
        out = [e for e in out if e["tier"] in want]
    if domain:
        out = [e for e in out if e["domain"] == domain]
    if method:
        out = [e for e in out if method in e["methods"]]
    if ready:
        out = [e for e in out if e["ready"]]
    if q:
        words = q.lower().split()
        out = [e for e in out if all(w in " ".join([e["title"], e["description"], e["domain"], *e["methods"],
                                                     e["key"]]).lower() for w in words)]
    if sort == "date":
        return sorted(out, key=lambda e: -(e["date"] or 0))
    order = {t: i for i, t in enumerate(SEALABLE)}
    return sorted(out, key=lambda e: (order.get(e["tier"], 9), e["title"].lower()))


# ---------------------------------------------------------------------------------------------- page and media
def media_urls(data: dict[str, Any], mode: str = "server") -> dict[str, Any]:
    """Fill ``src`` / ``thumb`` / ``seal`` URLs: ``server`` routes, or relative paths of a static export."""
    for e in data["entries"]:
        k = _q(e.get("seal_key") or e["key"])
        for lang in ("en", "pt"):
            e[f"seal_{lang}"] = (f"/seal.png?key={k}&lang={lang}" if mode == "server" else
                                 f"media/seals/{_safe(e['key'])}_{lang}.png") if e["tier"] else ""
        for x in e["media"]:
            if mode == "server":
                if "run" in x:
                    x["thumb"] = f"/thumb/{_q(x['run'])}"
                    x["src"] = (f"/files/{_q(x['run'])}" if x["kind"] == "video" else
                                f"/stamped/{_q(x['run'])}?key={k}")
                else:
                    x["thumb"] = f"/repo-thumb/{_q(x['repo'])}"
                    x["src"] = f"/repo/{_q(x['repo'])}" if x["kind"] == "video" or not e.get("seal_key") else \
                        f"/stamped-repo/{_q(x['repo'])}?key={k}"
            else:
                stem = _safe(x.get("run") or x.get("repo"))
                x["thumb"] = f"media/thumbs/{stem}.jpg"
                x["src"] = f"media/{'videos' if x['kind'] == 'video' else 'figures'}/{stem}" + \
                    (os.path.splitext(x["name"])[1] if x["kind"] == "video" else ".jpg")
        e["viewer_urls"] = [(f"/files/{_q(v)}" if e["source"] == "lab" else f"/repo/{_q(v)}") if mode == "server"
                            else "" for v in e["viewers"]]
    return data


def _safe(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s)


def render(data: dict[str, Any]) -> str:
    """The page: the template with the data embedded (it filters, sorts and opens details in the browser)."""
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    return html.replace("/*__DATA__*/null", blob)


def media_path(lab_root: str, entry: dict[str, Any], item: dict[str, Any]) -> str | None:
    """The file behind a media item (a Lab run file or a repository file listed by a manifest)."""
    if "run" in item:
        return os.path.join(lab_root, "runs", item["run"])
    return os.path.join(M.REPO, item["repo"])


def stamped_jpeg(path: str, key: str, cur, width: int = 1280) -> bytes | None:
    """The figure stamped with the item's seal, as a JPEG at most ``width`` wide (first frame of a GIF)."""
    import io

    from PIL import Image

    from .seal import stamp_image
    try:
        im = Image.open(path)
        im.seek(0)
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, max(1, int(im.height * width / im.width))))
        out = stamp_image(im, seal_for(key, cur), size=0.3).convert("RGB")
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=85, optimize=True)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


def seal_png(key: str, cur, lang: str = "en") -> bytes:
    import io

    from .seal import render_seal
    buf = io.BytesIO()
    render_seal(seal_for(key, cur), height=72, lang=lang, compact=False).save(buf, "PNG")
    return buf.getvalue()


def write_static(store, out_dir: str, *, examples: str | None = None, media: bool = True) -> str:
    """The same page as static files: ``out_dir/index.html`` and ``out_dir/media/`` (stamped figures at most 1280
    wide, thumbnails, seals in EN and PT, movies up to 10 MB; larger movies are listed as server-only)."""
    from .html import thumbnail_bytes
    data = media_urls(collect(store, examples), "static")
    data["static"] = True
    cur = load_curation(store)
    root = os.path.abspath(store.root)
    if media:
        for sub in ("figures", "thumbs", "videos", "seals"):
            os.makedirs(os.path.join(out_dir, "media", sub), exist_ok=True)
        for e in data["entries"]:
            seal_key = e.get("seal_key")
            if e["tier"] and seal_key:
                for lang in ("en", "pt"):
                    with open(os.path.join(out_dir, e[f"seal_{lang}"]), "wb") as f:
                        f.write(seal_png(seal_key, cur, lang))
            keep = []
            for x in e["media"]:
                src = media_path(root, e, x)
                if not src or not os.path.isfile(src):
                    continue
                if x["kind"] == "video":
                    if os.path.getsize(src) > STATIC_VIDEO_MAX:
                        x["server_only"] = True
                        x["src"] = ""
                    else:
                        shutil.copyfile(src, os.path.join(out_dir, x["src"]))
                else:
                    body = stamped_jpeg(src, seal_key, cur) if (e["tier"] and seal_key) else thumbnail_bytes(src, 1280)
                    if body is None:
                        continue
                    with open(os.path.join(out_dir, x["src"]), "wb") as f:
                        f.write(body)
                th = thumbnail_bytes(src)
                if th is not None:
                    with open(os.path.join(out_dir, x["thumb"]), "wb") as f:
                        f.write(th)
                keep.append(x)
            e["media"] = keep
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(data))
    return path
