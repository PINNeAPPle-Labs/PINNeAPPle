"""The atlas of science areas: what :mod:`pinneapple_science` covers, what the rest of the library already covers,
and what is planned.

The atlas is data (``pinneapple_science/data/atlas.json``, schema ``pinneapple.science.atlas/1``, versioned with
the package): areas, subareas, a status per subarea (``available``, ``partial``, ``planned``) and pointers to the
code that covers them -- a ``pp.science`` experiment, a library module, a problem preset
(``pp.list_presets()``) or a Lab experiment (``pinneapple_lab.available()``). The atlas points to existing code and
never copies it. :func:`validate` checks every pointer against the repository (without importing the modules) and
every registered experiment against the atlas, so the page cannot drift from the code.

    import pinneapple as pp
    a = pp.science.atlas()
    print(a)                       # one line per area with its status and counts
    a.rows()                       # one dict per subarea (area, subarea, status, pointers)
    a.to_markdown()                # the table of docs/guides/science.md
    a.to_html("atlas.html")        # a single-file page with search and filters
"""
from __future__ import annotations

import html
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

ATLAS_PATH = os.path.join(os.path.dirname(__file__), "data", "atlas.json")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUSES = ("available", "partial", "planned")
POINTER_KINDS = ("experiment", "module", "preset", "lab")

__all__ = ["Atlas", "Area", "load", "validate", "ATLAS_PATH"]


@dataclass
class Area:
    id: str
    title: str
    branch: str
    status: str
    description: str
    subareas: list[dict[str, Any]] = field(default_factory=list)

    def pointers(self) -> list[dict[str, str]]:
        return [p for s in self.subareas for p in s.get("covered_by", [])]

    def experiments(self) -> list[str]:
        return [p["experiment"] for p in self.pointers() if "experiment" in p]


@dataclass
class Atlas:
    schema: str
    version: str
    statuses: dict[str, str]
    dims: list[str]
    areas: list[Area]

    def area(self, area_id: str) -> Area:
        for a in self.areas:
            if a.id == area_id or a.id.split(".")[-1] == area_id:
                return a
        raise KeyError(f"no area {area_id!r}; areas: {', '.join(a.id for a in self.areas)}")

    def rows(self) -> list[dict[str, Any]]:
        out = []
        for a in self.areas:
            for s in a.subareas:
                out.append({"area": a.id, "area_title": a.title, "branch": a.branch, "subarea": s["id"],
                            "title": s["title"], "status": s["status"], "covered_by": s.get("covered_by", []),
                            "note": s.get("note", "")})
        return out

    def counts(self) -> dict[str, int]:
        c = {k: 0 for k in STATUSES}
        for r in self.rows():
            c[r["status"]] += 1
        return c

    def __repr__(self) -> str:
        lines = [f"PINNeAPPle science atlas {self.version}: {len(self.areas)} areas, "
                 + ", ".join(f"{v} {k}" for k, v in self.counts().items()) + " subareas"]
        for a in self.areas:
            st = [s["status"] for s in a.subareas]
            n_exp = len(a.experiments())
            lines.append(f"  [{a.status:9s}] {a.title} ({a.id}): {len(st)} subareas, "
                         f"{st.count('available')} available, {st.count('partial')} partial, "
                         f"{st.count('planned')} planned" + (f"; {n_exp} pp.science experiments" if n_exp else ""))
        return "\n".join(lines)

    # ------------------------------------------------------------------ renderings
    def to_markdown(self) -> str:
        L = ["| Area | Subarea | Status | Covered by |", "|---|---|---|---|"]
        for r in self.rows():
            cov = "; ".join(_pointer_text(p) for p in r["covered_by"]) or (r["note"] or "-")
            L.append(f"| {r['area_title']} | {r['title']} | {r['status']} | {cov} |")
        return "\n".join(L) + "\n"

    def to_html(self, path: str | None = None, experiments: Iterable[dict[str, Any]] | None = None) -> str:
        """A single-file page (no external assets): areas as collapsible sections with their subareas, status
        chips, pointers and, for pp.science experiments, level / kind / dimensions; a search box and status and
        branch filters."""
        if experiments is None:
            from .registry import experiments as _ex
            experiments = _ex()
        cards = {e["name"]: e for e in experiments}
        body = []
        for a in self.areas:
            subs = []
            for s in a.subareas:
                ptrs = []
                for p in s.get("covered_by", []):
                    if "experiment" in p and p["experiment"] in cards:
                        c = cards[p["experiment"]]
                        ptrs.append(f"<li><code>pp.science.run(\"{html.escape(c['name'])}\")</code> "
                                    f"{html.escape(c['title'])} <span class=meta>{c['level']} · {c['kind']} · "
                                    f"{' / '.join(c['dims'])}</span></li>")
                    else:
                        ptrs.append(f"<li>{html.escape(_pointer_text(p))}</li>")
                note = f"<p class=note>{html.escape(s['note'])}</p>" if s.get("note") else ""
                subs.append(f"<div class='sub' data-status='{s['status']}'><h4><span class='chip {s['status']}'>"
                            f"{s['status']}</span> {html.escape(s['title'])}</h4>{note}<ul>{''.join(ptrs)}</ul></div>")
            body.append(f"<details class='area' data-branch='{a.branch}' data-status='{a.status}' open><summary>"
                        f"<span class='chip {a.status}'>{a.status}</span> <b>{html.escape(a.title)}</b> "
                        f"<span class=meta>{html.escape(a.id)}</span></summary><p>{html.escape(a.description)}</p>"
                        f"{''.join(subs)}</details>")
        c = self.counts()
        page = _HTML.format(version=html.escape(self.version), n_areas=len(self.areas), available=c["available"],
                            partial=c["partial"], planned=c["planned"], body="\n".join(body),
                            n_exp=len(cards))
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(page)
        return page


def _pointer_text(p: dict[str, str]) -> str:
    kind = next(k for k in POINTER_KINDS if k in p)
    label = {"experiment": "pp.science", "module": "module", "preset": "preset", "lab": "Lab experiment"}[kind]
    what = f" ({p['what']})" if p.get("what") else ""
    return f"{label} `{p[kind]}`{what}"


def load(path: str = ATLAS_PATH) -> Atlas:
    """The atlas (``pp.science.atlas()``): every area and subarea with its status and pointers, from the versioned
    data file ``pinneapple_science/data/atlas.json``."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("schema") != "pinneapple.science.atlas/1":
        raise ValueError(f"{path}: unknown atlas schema {d.get('schema')!r}")
    areas = [Area(a["id"], a["title"], a["branch"], a["status"], a["description"], a.get("subareas", []))
             for a in d["areas"]]
    return Atlas(d["schema"], d["version"], d["statuses"], d["dims"], areas)


def _module_exists(dotted: str) -> bool:
    base = os.path.join(REPO, *dotted.split("."))
    return os.path.isfile(base + ".py") or os.path.isfile(os.path.join(base, "__init__.py"))


def validate(atlas: Atlas | None = None, *, check_presets: bool = True, check_lab: bool = True) -> list[str]:
    """Problems with the atlas (empty when it is consistent): unknown statuses, pointers to modules that do not
    exist, presets or Lab experiments that are not registered, ``pp.science`` experiments that are not
    registered or not listed, and ``available`` subareas without an experiment."""
    from .registry import names

    atlas = atlas or load()
    problems = []
    registered = set(names())
    listed = set()
    presets = labs = None
    if check_presets:
        from pinneapple_physics.pde_environment import list_presets
        presets = set(list_presets())
    if check_lab:
        from pinneapple_lab import available
        labs = set(available())
    ids = set()
    for a in atlas.areas:
        if a.id in ids:
            problems.append(f"duplicate area id {a.id}")
        ids.add(a.id)
        if a.status not in STATUSES:
            problems.append(f"{a.id}: status {a.status!r}")
        for s in a.subareas:
            where = f"{a.id}/{s['id']}"
            if s["status"] not in STATUSES:
                problems.append(f"{where}: status {s['status']!r}")
            exps = [p["experiment"] for p in s.get("covered_by", []) if "experiment" in p]
            if s["status"] == "available" and not exps:
                problems.append(f"{where}: 'available' needs a pp.science experiment")
            if s["status"] == "planned" and s.get("covered_by"):
                problems.append(f"{where}: 'planned' but has pointers (use 'partial')")
            for p in s.get("covered_by", []):
                kinds = [k for k in POINTER_KINDS if k in p]
                if len(kinds) != 1:
                    problems.append(f"{where}: pointer {p} needs exactly one of {POINTER_KINDS}")
                    continue
                k = kinds[0]
                v = p[k]
                if k == "experiment":
                    listed.add(v)
                    if v not in registered:
                        problems.append(f"{where}: experiment {v!r} is not registered")
                elif k == "module" and not _module_exists(v):
                    problems.append(f"{where}: module {v!r} does not exist")
                elif k == "preset" and presets is not None and v not in presets:
                    problems.append(f"{where}: preset {v!r} is not registered")
                elif k == "lab" and labs is not None and v not in labs:
                    problems.append(f"{where}: Lab experiment {v!r} is not registered")
        st = {s["status"] for s in a.subareas}
        if a.status == "available" and "available" not in st:
            problems.append(f"{a.id}: 'available' but no subarea is")
        if a.status == "planned" and "available" in st:
            problems.append(f"{a.id}: 'planned' but a subarea is available")
    for n in sorted(registered - listed):
        problems.append(f"experiment {n!r} is registered but not in the atlas")
    from .registry import get
    for n in sorted(registered):
        e = get(n)
        if e.area not in ids:
            problems.append(f"experiment {n!r}: area {e.area!r} is not in the atlas")
    return problems


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Science atlas</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#d9d8d4; --card:#ffffff;
  --available:#008300; --partial:#c98500; --planned:#8a8986; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --line:#3a3936; --card:#232322; --available:#3fae3f; --partial:#e0a43a; --planned:#9d9c97; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#3a3936; --card:#232322;
  --available:#3fae3f; --partial:#e0a43a; --planned:#9d9c97; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:980px;
  padding:16px; }}
h1 {{ font-size:1.5rem; margin:.2rem 0; }} .meta {{ color:var(--ink2); font-size:.85em; }}
.bar {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }} input, select {{ font:inherit; padding:6px 8px;
  border:1px solid var(--line); border-radius:6px; background:var(--card); color:var(--ink); }}
input {{ flex:1 1 220px; }}
details.area {{ background:var(--card); border:1px solid var(--line); border-radius:8px; margin:10px 0;
  padding:8px 12px; }}
summary {{ cursor:pointer; }} .sub {{ border-top:1px solid var(--line); padding:6px 0 2px 0; }}
.sub h4 {{ margin:4px 0; font-size:.95rem; }} ul {{ margin:4px 0 6px 18px; padding:0; }} li {{ margin:2px 0; }}
code {{ font-size:.85em; }} .note {{ color:var(--ink2); margin:0; }}
.chip {{ display:inline-block; font-size:.75em; padding:1px 8px; border-radius:10px; border:1.5px solid;
  font-weight:600; }}
.chip.available {{ color:var(--available); }} .chip.partial {{ color:var(--partial); }}
.chip.planned {{ color:var(--planned); }}
</style></head><body>
<h1>PINNeAPPle science atlas</h1>
<p class=meta>Version {version} · {n_areas} areas · subareas: {available} available, {partial} partial,
{planned} planned · {n_exp} pp.science experiments. <b>available</b>: pp.science experiments checked against their
references; <b>partial</b>: library code covers part of it; <b>planned</b>: mapped, not yet built.</p>
<div class=bar><input id=q placeholder="search areas, topics, modules"><select id=st><option value="">any status</option>
<option>available</option><option>partial</option><option>planned</option></select><select id=br>
<option value="">physics and other sciences</option><option value="physics">physics</option>
<option value="other">other sciences</option></select></div>
{body}
<script>
const q=document.getElementById('q'), st=document.getElementById('st'), br=document.getElementById('br');
function f(){{ const t=q.value.toLowerCase();
 document.querySelectorAll('details.area').forEach(a=>{{ let any=false;
  a.querySelectorAll('.sub').forEach(s=>{{ const ok=(!st.value||s.dataset.status===st.value)&&
    (!t||s.textContent.toLowerCase().includes(t)||a.querySelector('summary').textContent.toLowerCase().includes(t));
   s.style.display=ok?'':'none'; any=any||ok; }});
  a.style.display=(any&&(!br.value||a.dataset.branch===br.value))?'':'none'; }}); }}
[q,st,br].forEach(e=>e.addEventListener('input',f));
</script></body></html>
"""
