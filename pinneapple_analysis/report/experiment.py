"""One-page engineering report for an experiment: ``experiment.json`` plus the results beside it (#919).

A reader of an experiment folder currently has to open JSON, a README and a curation file to learn what was measured
and whether it passed. This module renders one document with the same layout for every experiment -- header, KPI
cards, verdict table, the figures the manifest names, uncertainty and validity, provenance -- as HTML and as PDF:

    from pinneapple_analysis.report import experiment_report
    r = experiment_report("examples/end_to_end/ahmed_body")     # -> report/index.html + report/report.pdf
    html, pdf = r                                               # the two paths, in that order

    python -m pinneapple_analysis.report examples/condensation_falsification

Everything on the page is read, nothing is typed: the numbers come from the manifest's ``metrics``, the verdicts
from its ``checks`` (which the run wrote through ``ctx.check`` from its own results, the same rows #900 and #902
print), the six-stage trust card from :mod:`pinneapple_lab.curation` -- verbatim from ``lab/curation.json`` when the
folder was imported into the Lab, otherwise its first five stages computed here with the same function and the
decision stage reported as *not graded* rather than invented -- the nine-item protocol checklist from
:mod:`pinneapple_lab.quality`, and the hardware, wall time and cost from the folder's own ``costs.json`` or the
manifest's ``provenance``. A number with no check that names it gets a neutral card saying so, never a green one.

A folder with nothing to report raises :class:`ReportError` naming what is missing; a report of an experiment that
was not measured is worse than no report.

The colours, surfaces and colour maps are the house theme's (:mod:`pinneapple_tools.visualization.theme`), in its
light and dark variants, and the pass/fail inks are the Okabe-Ito green and vermillion of the house palette, so the
verdict survives the common colour-vision deficiencies. Every figure carries alt text.
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Report", "ReportError", "experiment_report", "SCHEMA"]

#: what ``report.json`` says it is, next to the two documents
SCHEMA = "pinneapple.report/1"

#: the inks of a verdict: the Okabe-Ito green and vermillion of the house palette (the theme's ``categorical``),
#: so pass and fail are still told apart by a reader with deuteranopia or protanopia
PASS = "#009E73"
FAIL = "#D55E00"
NEUTRAL = "#5d6b82"


class ReportError(ValueError):
    """The folder cannot be reported on: no manifest, or a manifest with no result behind it."""


@dataclass
class Report:
    """What a report is: the two documents written and the data they were rendered from.

    Iterating yields ``html, pdf`` in that order, which is the signature #919 asks for
    (``experiment_report(folder) -> html, pdf``); ``data`` stays available for a caller that wants the numbers.
    """
    html: str
    pdf: str | None
    data: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield self.html
        yield self.pdf


# ---------------------------------------------------------------------- small helpers
def _esc(text: Any) -> str:
    return _html.escape("" if text is None else str(text), quote=True)


def _fmt(v: Any, unit: str = "") -> str:
    """A number as a reader writes it: 3 significant figures, more only when they are information."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")
    if not isinstance(v, float):
        return str(v)
    if v != v:                                        # NaN
        return "NaN"
    a = abs(v)
    if a == 0:
        return "0"
    if 1e-3 <= a < 1e5:
        txt = f"{v:.4g}"
        return f"{float(txt):,}".replace(",", " ") if a >= 1000 else txt
    return f"{v:.3e}"


def _slug(folder: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", os.path.basename(os.path.normpath(folder)).lower()).strip("-") or "experiment"


def _norm(text: str) -> list[str]:
    """The words of a name, with the numbers kept: ``Re_50_CD`` and ``Re_200_CD`` are two different checks."""
    return [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if len(t) > 2 or t.isdigit()]


def _close(a: Any, b: Any) -> bool:
    """Whether two recorded numbers are the same measurement (a check and the metric it reports)."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1.0)


def _match(metric: str, checks: list[dict]) -> list[dict]:
    """The checks that name ``metric``, the closest name first: one name's words all appear in the other's.

    Deliberately conservative. A card is coloured by a check only when that check clearly speaks of the number --
    ``hypotheses_refuted`` is not evidenced by a table of ``T1 not refuted`` rows, and a report that said it was
    would be worse than a grey card saying no check names it. Among the checks that do speak of it, the one that
    shares the most words comes first, because the reference and the tolerance on a card are that check's:
    ``pipe_bend_bend`` takes the tolerance of ``pipe_bend_bend_vs_reference``, not the looser match of its sibling.
    """
    want = set(_norm(metric))
    if not want:
        return []
    scored: list[tuple[float, int, dict]] = []
    seen: set[str] = set()
    for c in checks:
        have = set(_norm(c.get("name", "")))
        if not have or not (want <= have or have <= want):
            continue
        key = str(c.get("name", ""))
        if key in seen:
            continue
        seen.add(key)
        scored.append((len(want & have) / len(want | have), -len(key), c))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [c for _s, _n, c in scored]


def _band(checks: list[dict]) -> tuple[list[float] | None, str]:
    """The interval the check allows around its reference, and the target as a sentence.

    The tolerance is quoted as the check recorded it -- a relative tolerance as a percentage, an absolute one in
    the quantity's own unit -- and a reference with no tolerance says so instead of implying a zero band.
    """
    for c in checks:
        ref, rtol, atol = c.get("reference"), c.get("rtol"), c.get("atol")
        if ref is not None and isinstance(ref, (int, float)):
            tol = (atol or 0.0) + (rtol or 0.0) * abs(float(ref))
            if tol:
                rate = f" ± {100 * float(rtol):.3g} %" if rtol else f" ± {_fmt(tol)}"
                return [float(ref) - tol, float(ref) + tol], f"reference {_fmt(float(ref))}{rate}"
            return None, f"reference {_fmt(float(ref))} (no tolerance recorded)"
        if c.get("max") is not None:
            return [float("-inf"), float(c["max"])], f"≤ {_fmt(float(c['max']))}"
        if c.get("min") is not None:
            return [float(c["min"]), float("inf")], f"≥ {_fmt(float(c['min']))}"
    return None, ""


# ---------------------------------------------------------------------- the data behind the page
def _manifest(folder: str):
    from pinneapple_lab.manifest import load

    if not os.path.isdir(folder):
        raise ReportError(f"{folder} is not a folder: point experiment_report at an experiment (the folder with its "
                          f"experiment.json)")
    if not os.path.exists(os.path.join(folder, "experiment.json")):
        raise ReportError(f"{folder} has no experiment.json: an experiment is a folder with a manifest "
                          f"(`python -m pinneapple_lab register {os.path.basename(os.path.normpath(folder))}` writes "
                          f"one); a report of a folder nobody described would be a page of guesses")
    m = load(folder)
    return m


def _lab_record(folder: str) -> dict[str, Any] | None:
    """The Lab curation record of a manifest folder, when it was imported into the Lab.

    The record is the real six-stage card: :func:`pinneapple_lab.curation.curate` computed it from the run, and this
    report prints it rather than grading the folder again. Found through the run's own ``params.path``, so a folder
    renamed in the checkout simply has no record instead of the wrong one.
    """
    from pinneapple_lab.manifest import REPO

    cur = os.path.join(REPO, "lab", "curation.json")
    try:
        with open(cur, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError):
        return None
    rel = os.path.relpath(os.path.abspath(os.path.join(folder, "experiment.json")), REPO).replace(os.sep, "/")
    runs = record.get("runs", {})
    for path in sorted((os.path.join(REPO, "lab", "runs", name) for name in
                        _subdirs(os.path.join(REPO, "lab", "runs"))), key=lambda p: p):
        for run_dir in _subdirs(path):
            try:
                with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as fh:
                    run = json.load(fh)
            except (OSError, ValueError):
                continue
            if (run.get("params") or {}).get("path") != rel:
                continue
            rec = runs.get(run.get("run_id"))
            if rec:
                return rec
    return None


def _subdirs(path: str) -> list[str]:
    try:
        return sorted(os.path.join(path, n) for n in os.listdir(path) if os.path.isdir(os.path.join(path, n)))
    except OSError:
        return []


def _stages(folder: str, m, checks: list[dict], metrics: dict[str, float], limitations: list[str]) -> dict[str, Any]:
    """The six-stage trust card: the curation record verbatim, else its first five stages computed from the manifest."""
    from pinneapple_lab import curation

    rec = _lab_record(folder)
    if rec and rec.get("stages"):
        return {"source": "lab/curation.json", "tier": rec.get("tier", ""), "tier_name": rec.get("tier_name", ""),
                "generated": "the Lab curation record", "stages": rec["stages"],
                "readiness": rec.get("readiness", {})}
    is_ml = any(re.search(r"pinn|neural|surrogate|\bgp\b|mlp|rom|pod|operator|network", x, re.I)
                for x in list(m.methods) + [m.domain])
    st = curation.evidence_stages(folder, checks, metrics, {}, is_ml, limitations)
    st["decision"] = {"status": "open", "evidence": ["not graded: no curation record for this folder"],
                      "next": "python -m pinneapple_lab curate (the tier is the Lab's, not this report's)"}
    return {"source": "pinneapple_lab.curation.evidence_stages", "tier": "", "tier_name": "",
            "generated": "the manifest's own checks", "readiness": {},
            "stages": {k: {"title": curation.STAGES[k][0], "question": curation.STAGES[k][1], **v}
                       for k, v in st.items()}}


def _checklist(folder: str) -> dict[str, Any]:
    """The nine-item protocol checklist of :mod:`pinneapple_lab.quality`, plus its badge."""
    try:
        from pinneapple_lab import quality
        c = quality.assess(folder)
    except Exception as exc:                                           # a folder the checklist cannot read
        return {"error": f"{type(exc).__name__}: {exc}", "items": [], "badge": ""}
    return {"badge": c.badge(),
            "met": len(c.met), "applicable": c.applicable,
            "items": [{"id": i.id, "title": i.title, "status": i.status, "evidence": list(i.evidence),
                       "fix": i.fix} for i in c.items],
            "missing": list(c.missing)}


def _costs(folder: str) -> dict[str, Any] | None:
    """``costs.json`` as the studies write it (stages with wall and CPU hours, and the machine they ran on)."""
    for depth in (0, 1, 2):
        pattern = os.path.join(folder, *(["*"] * depth), "costs.json")
        import glob as _glob
        for path in sorted(_glob.glob(pattern)):
            try:
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            total = d.get("total") or {}
            if isinstance(total.get("wall_h"), (int, float)):
                return {"path": os.path.relpath(path, folder), "total": total, "environment": d.get("environment") or {},
                        "stages": d.get("stages") or {}}
    return None


def _header(m, folder: str, costs: dict | None, cost_per_hour: float | None) -> dict[str, Any]:
    prov = m.provenance or {}
    env = prov.get("environment") or {}
    hw: dict[str, Any] = {}
    if costs and costs.get("environment"):
        hw = dict(costs["environment"])
    elif env:
        hw = {"platform": env.get("platform", ""), "python": env.get("python", "")}
    wall = None
    if costs:
        wall = float(costs["total"]["wall_h"]) * 3600.0
    elif isinstance(prov.get("seconds"), (int, float)):
        wall = float(prov["seconds"])
    cost = None
    if cost_per_hour is not None and wall is not None:
        cost = {"usd": round(wall / 3600.0 * float(cost_per_hour), 2), "rate": float(cost_per_hour)}
    return {"title": m.title, "question": m.description, "date": prov.get("created") or "",
            "commit": prov.get("git") or "", "hardware": hw, "cpu_hours": (costs or {}).get("total", {}).get("cpu_h"),
            "core_hours": (costs or {}).get("total", {}).get("cpu_h"),
            "runs": (costs or {}).get("total", {}).get("runs"), "wall_seconds": wall, "cost": cost,
            "status": m.status, "domain": m.domain, "methods": list(m.methods), "id": m.id,
            "costs_file": (costs or {}).get("path", "")}


def _kpis(m, checks: list[dict], limit: int = 6) -> list[dict[str, Any]]:
    """Three to six headline numbers: the ones a check names first, then the manifest's own order."""
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for met in m.metrics:
        name = met.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        hit = _match(name, checks)
        same = [c for c in hit if c.get("value") is None or _close(c["value"], met.get("value"))]
        band, target = _band(same)                 # a check that measures the same number lends its interval
        cards.append({"name": name, "value": met.get("value"), "unit": met.get("unit") or "",
                      "how": met.get("how") or "", "status": ("pass" if all(c.get("passed") for c in hit) else "fail")
                      if hit else "no check", "band": band, "target": target,
                      "check": hit[0].get("name", "") if hit else "", "source": "metric"})
    for c in checks:                                   # a check with its own number is a headline too
        if len(cards) >= limit or c.get("value") is None or c.get("name") in seen:
            continue
        band, target = _band([c])
        seen.add(c.get("name"))
        cards.append({"name": c.get("name"), "value": c.get("value"), "unit": "", "how": c.get("detail") or "",
                      "status": "pass" if c.get("passed") else "fail", "band": band, "target": target,
                      "check": c.get("name"), "source": "check"})
    cards.sort(key=lambda k: 0 if k["status"] in ("pass", "fail") else 1)      # evidenced numbers first
    return cards[:limit]


def _verdicts(checks: list[dict]) -> dict[str, Any]:
    rows = []
    for c in checks:
        measured = ""
        if c.get("value") is not None:
            measured = _fmt(c["value"])
            if c.get("reference") is not None:
                measured += f" vs {_fmt(c['reference'])}"
            tol = (c.get("atol") or 0.0) + (c.get("rtol") or 0.0) * abs(float(c.get("reference") or 0.0))
            if tol:
                measured += f" (±{_fmt(tol)})"
        elif c.get("max") is not None:
            measured = f"≤ {_fmt(c['max'])}"
        elif c.get("min") is not None:
            measured = f"≥ {_fmt(c['min'])}"
        rows.append({"name": c.get("name", ""), "kind": c.get("kind", ""),
                     "verdict": "PASS" if c.get("passed") else "FAIL", "measured": measured,
                     "evidence": c.get("detail") or ""})
    passed = sum(r["verdict"] == "PASS" for r in rows)
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return {"rows": rows, "passed": passed, "failed": len(rows) - passed, "kinds": kinds}


def _figures(folder: str, m, out: str, limit: int, chosen: list[str] | None) -> list[dict[str, Any]]:
    """The figures the manifest names, copied next to the report so it stays readable on its own."""
    names = list(chosen) if chosen else list((m.files or {}).get("figures") or [])
    picked = []
    for name in names[:limit]:
        src = os.path.join(folder, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(out, "assets", os.path.basename(name))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        stem = os.path.splitext(os.path.basename(name))[0].replace("_", " ")
        picked.append({"src": name, "path": os.path.relpath(dst, out).replace(os.sep, "/"),
                       "caption": f"{stem} ({name})"})
    return picked


def _hero(out: str, hero: Any) -> dict[str, Any] | None:
    """The hero asset of #920, when the folder (or the caller) has one: a still, its glTF and the turntable."""
    if hero is None:
        for path in (os.path.join(out, "hero", "hero.json"), os.path.join(out, "hero.json")):
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                return None
        return None
    if isinstance(hero, dict):
        return hero
    if isinstance(hero, str) and os.path.isfile(hero):
        dst = os.path.join(out, "assets", os.path.basename(hero))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(hero) != os.path.abspath(dst):
            shutil.copyfile(hero, dst)
        return {"stills": {"1920x1080": os.path.relpath(dst, out).replace(os.sep, "/")}}
    return None


def _reproduce(m, folder: str) -> str:
    from pinneapple_lab.manifest import REPO

    rel = os.path.relpath(os.path.abspath(folder), REPO).replace(os.sep, "/")
    script = (m.scripts or [""])[0]
    if script:
        return f"git checkout {m.provenance.get('git', '<commit>')} && python {rel}/{script}"
    return f"python -m pinneapple_lab register {rel}"


def gather(folder: str, *, out: str | None = None, limit_kpis: int = 6, limit_figures: int = 4,
           figures: list[str] | None = None, hero: Any = None, cost_per_hour: float | None = None) -> dict:
    """Everything the page shows, as data. Raises :class:`ReportError` when there is nothing to report."""
    folder = os.path.abspath(folder)
    m = _manifest(folder)
    checks = [dict(c) for c in (m.checks or [])]
    from pinneapple_lab.curation import infer_check_kind
    for c in checks:
        c.setdefault("kind", infer_check_kind(c))
    metrics = {x["name"]: x["value"] for x in (m.metrics or []) if isinstance(x.get("value"), (int, float))}
    decl = (m.quality or {}).get("not_measured") or []
    limitations = [str(x) for x in (decl if isinstance(decl, list) else [])]

    verdicts = _verdicts(checks)
    costs = _costs(folder)
    if not metrics and not checks and not any((m.files or {}).get(k) for k in ("figures", "videos", "viewers")):
        raise ReportError(
            f"{folder} has no result to report: its experiment.json records no metric, no check and no figure. A "
            f"report of it would be a title page -- run the experiment (`{_reproduce(m, folder)}`) so the manifest "
            f"records what it measured, or point the report at a folder that has run")
    out = os.path.abspath(out or os.path.join(folder, "report"))
    data: dict[str, Any] = {
        "schema": SCHEMA, "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "folder": folder, "out": out,
        "header": _header(m, folder, costs, cost_per_hour),
        "kpis": _kpis(m, checks, limit=limit_kpis),
        "verdicts": verdicts,
        "figures": _figures(folder, m, out, limit_figures, figures),
        "hero": _hero(out, hero),
        "validity": {"stages": _stages(folder, m, checks, metrics, limitations), "checklist": _checklist(folder),
                     "not_measured": limitations,
                     "not_applicable": (m.quality or {}).get("not_applicable") or {}},
        "provenance": {"source": (m.provenance or {}).get("source", ""), "git": (m.provenance or {}).get("git", ""),
                       "created": (m.provenance or {}).get("created", ""),
                       "seconds": (m.provenance or {}).get("seconds"),
                       "environment": (m.provenance or {}).get("environment") or {},
                       "machine": (costs or {}).get("environment") or {},
                       "references": list(m.references or []),
                       "data": list((m.files or {}).get("data") or []),
                       "modules": list(m.modules or []), "lab": m.lab or {}},
        "reproduce": _reproduce(m, folder),
    }
    return data


# ---------------------------------------------------------------------- the page: HTML
_STATUS_INK = {"pass": PASS, "fail": FAIL, "no check": NEUTRAL,
               "answered": PASS, "partial": NEUTRAL, "open": NEUTRAL, "failed": FAIL,
               "met": PASS, "missing": FAIL, "not applicable": NEUTRAL, "not assessed": NEUTRAL}


def _css(theme: str) -> str:
    """The house theme as CSS: surfaces, ink, line and the two verdict inks of the palette."""
    from pinneapple_tools.visualization.theme import palette, surface
    from pinneapple_tools.visualization.theme import theme as _theme

    s = surface(dark=(theme == "dark"))
    p = palette()
    t = _theme()
    return f"""
:root{{--paper:{s['paper']};--plot:{s['plot']};--ink:{s['ink']};--muted:{s['muted']};--line:{s['line']};
--brand:{t['brand']['blue']};--ember:{t['brand']['ember']};--pass:{PASS};--fail:{FAIL};--neutral:{NEUTRAL};
--c2:{p[1]};--c3:{p[2]};--font:{', '.join(t['font']['stack'])}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--plot);color:var(--ink);font:15px/1.5 var(--font);
-webkit-font-smoothing:antialiased}}
.page{{max-width:1180px;margin:0 auto;background:var(--paper);padding:40px 48px 56px}}
.kicker{{margin:0 0 6px;color:var(--brand);font-weight:700;letter-spacing:.09em;text-transform:uppercase;
font-size:12px}}
h1{{margin:0 0 10px;font-size:30px;line-height:1.2;letter-spacing:-.01em}}
.question{{margin:0 0 22px;color:var(--muted);font-size:16px;max-width:78ch}}
h2{{margin:36px 0 12px;font-size:13px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
border-bottom:1px solid var(--line);padding-bottom:6px}}
dl.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:12px 22px;margin:0}}
dl.meta div{{border-left:3px solid var(--line);padding-left:10px}}
dl.meta dt{{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}}
dl.meta dd{{margin:2px 0 0;font-size:14px;font-variant-numeric:tabular-nums}}
.cards{{display:grid;grid-template-columns:repeat(var(--n,3),minmax(0,1fr));gap:14px}}
@media(max-width:820px){{.cards{{grid-template-columns:1fr}}}}
.card{{border:1px solid var(--line);border-top:4px solid var(--neutral);border-radius:8px;padding:12px 14px;
background:var(--plot)}}
.card.pass{{border-top-color:var(--pass)}} .card.fail{{border-top-color:var(--fail)}}
.card .n{{font-size:10.5px;letter-spacing:.03em;text-transform:uppercase;color:var(--muted);margin:0 0 6px;
overflow-wrap:anywhere;hyphens:auto}}
.card .v{{font-size:27px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.1}}
.card .u{{font-size:13px;color:var(--muted);font-weight:400}}
.card .t{{margin:6px 0 0;font-size:12px;color:var(--muted)}}
.card .s{{margin:8px 0 0;font-size:12px;font-weight:700;color:var(--neutral)}}
.card.pass .s{{color:var(--pass)}} .card.fail .s{{color:var(--fail)}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;font-variant-numeric:tabular-nums}}
caption{{text-align:left;color:var(--muted);font-size:12px;padding-bottom:6px}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
border-bottom:1px solid var(--line);padding:8px 10px 6px 0}}
td{{padding:7px 10px 7px 0;border-bottom:1px solid var(--line);vertical-align:top}}
td.v{{white-space:nowrap;font-weight:700}}
td.v.pass{{color:var(--pass)}} td.v.fail{{color:var(--fail)}}
.kind{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--muted)}}
.ev{{color:var(--muted)}}
figure{{margin:0 0 22px}} figure img{{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;
background:var(--plot)}}
figcaption{{color:var(--muted);font-size:12.5px;margin-top:6px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}}
.stage{{border:1px solid var(--line);border-left:4px solid var(--neutral);border-radius:6px;padding:10px 12px}}
.stage.pass{{border-left-color:var(--pass)}} .stage.fail,.stage.failed{{border-left-color:var(--fail)}}
.stage h3{{margin:0;font-size:14px}} .stage .q{{margin:2px 0 6px;color:var(--muted);font-size:12.5px}}
.stage ul{{margin:0;padding-left:18px;font-size:12.5px;color:var(--muted)}}
ul.plain{{list-style:none;margin:0;padding:0}}
ul.plain li{{padding:6px 0;border-bottom:1px solid var(--line);font-size:13.5px;display:flex;gap:10px}}
.pill{{border-radius:999px;padding:1px 9px;font-size:11.5px;font-weight:700;color:#fff;white-space:nowrap}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;background:var(--plot);
padding:2px 6px;border-radius:4px}}
footer{{margin-top:40px;border-top:1px solid var(--line);padding-top:12px;color:var(--muted);font-size:12px}}
.note{{color:var(--muted);font-size:12.5px}}
@media print{{body{{background:var(--paper)}} .page{{padding:0}} h2{{break-after:avoid}}
figure,table{{break-inside:avoid}}}}
"""


def _pill(status: str) -> str:
    ink = _STATUS_INK.get(status, NEUTRAL)
    return f'<span class="pill" style="background:{ink}">{_esc(status)}</span>'


def render_html(data: dict[str, Any], *, theme: str = "light") -> str:
    """The one-page report as a self-contained HTML document (the figures are files beside it)."""
    h = data["header"]
    ttl = _esc(h["title"] or data["header"].get("id") or "Experiment")
    wall = h.get("wall_seconds")
    wall_txt = ("not recorded" if not wall else f"{wall / 3600:.2f} h" if wall >= 3600 else
                f"{wall / 60:.1f} min" if wall >= 60 else f"{wall:.1f} s")
    cost_txt = ("not priced (no rate given)" if not h.get("cost") else
                f"US$ {h['cost']['usd']:.2f} at {h['cost']['rate']:.2f}/h")
    hw = h.get("hardware") or {}
    hw_txt = ", ".join(x for x in (hw.get("cpu"), f"{hw.get('cpu_count')} cores" if hw.get("cpu_count") else "",
                                   f"{hw.get('mem_gb')} GB" if hw.get("mem_gb") else "", hw.get("platform")) if x)
    meta = [("date", h.get("date") or "-"), ("commit", (h.get("commit") or "-")[:12]),
            ("hardware", hw_txt or (data["provenance"].get("environment") or {}).get("platform", "-")),
            ("wall time", wall_txt), ("cost", cost_txt), ("domain", h.get("domain") or "-")]
    parts: list[str] = [
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{ttl} · PINNeAPPle report</title>",
        f"<style>{_css(theme)}</style></head><body><main class=\"page\">",
        '<p class="kicker">PINNeAPPle · experiment report</p>',
        f"<h1>{ttl}</h1>",
        f'<p class="question">{_esc(h.get("question") or "")}</p>',
        '<dl class="meta">',
    ]
    parts += [f"<div><dt>{_esc(k)}</dt><dd>{_esc(v)}</dd></div>" for k, v in meta]
    parts.append("</dl>")

    # -- KPI cards
    if data["kpis"]:
        parts.append(f'<h2 id="kpis">Headline numbers</h2>'
                     f'<div class="cards" style="--n:{min(3, max(1, len(data["kpis"])))}">')
        for k in data["kpis"]:
            band = ""
            if k.get("band"):
                lo, hi = k["band"]
                band = f"{_fmt(lo) if lo != float('-inf') else '−∞'} … {_fmt(hi) if hi != float('inf') else '∞'}"
            label = {"pass": "check passed", "fail": "check failed", "no check": "no check names this number"}[k["status"]]
            parts.append(
                f'<div class="card {k["status"].replace(" ", "-")}">'
                f'<p class="n">{_esc(k["name"])}</p>'
                f'<p class="v">{_esc(_fmt(k["value"]))}'
                + (f' <span class="u">{_esc(k["unit"])}</span>'
                   if str(k.get("unit") or "") not in ("", "-", "1") else "")
                + "</p>"
                + (f'<p class="t">interval {_esc(band)}<br>{_esc(k["target"])}</p>' if k.get("target") else "")
                + (f'<p class="t">{_esc(k["how"])}</p>' if k.get("how") else "")
                + f'<p class="s">{_esc(label)}</p></div>')
        parts.append("</div>")
    else:
        parts.append('<h2 id="kpis">Headline numbers</h2><p class="note">The manifest records no metric.</p>')

    # -- verdicts
    v = data["verdicts"]
    parts.append('<h2 id="verdicts">Verdicts</h2>')
    if v["rows"]:
        parts.append(f'<p class="note">{v["passed"]} passed, {v["failed"]} failed, read from the run\'s own checks '
                     f'({" · ".join(f"{k}: {n}" for k, n in sorted(v["kinds"].items())) or "no kind recorded"}).'
                     f'</p>')
        parts.append('<table><caption>Every check the run recorded, in the order it recorded them.</caption>'
                     "<thead><tr><th>check</th><th>kind</th><th>measured</th><th>verdict</th>"
                     "<th>evidence</th></tr></thead><tbody>")
        for r in v["rows"]:
            cls = "pass" if r["verdict"] == "PASS" else "fail"
            parts.append(f'<tr><td>{_esc(r["name"])}</td><td class="kind">{_esc(r["kind"])}</td>'
                         f'<td>{_esc(r["measured"])}</td><td class="v {cls}">{_esc(r["verdict"])}</td>'
                         f'<td class="ev">{_esc(r["evidence"])}</td></tr>')
        parts.append("</tbody></table>")
    else:
        parts.append('<p class="note">The run recorded no check: nothing here was compared with a reference, a '
                     'baseline or a physical law.</p>')

    # -- figures
    hero = data.get("hero") or {}
    stills = hero.get("stills") or {}
    hero_path = stills.get("1920x1080") or (next(iter(stills.values())) if stills else None)
    if hero_path and not os.path.isabs(hero_path):
        hero_path = os.path.join(data["out"], hero_path)
    parts.append('<h2 id="figures">Figures</h2>')
    if hero_path and os.path.exists(hero_path):
        rel = os.path.relpath(hero_path, data["out"]).replace(os.sep, "/")
        extra = []
        for label, path in (("turntable", (hero.get("turntable") or {}).get("mp4")),
                            ("glTF", hero.get("gltf")), ("viewer", hero.get("viewer"))):
            if not path:
                continue
            full = path if os.path.isabs(path) else os.path.join(data["out"], path)
            if os.path.exists(full):
                if label == "viewer" and os.path.isdir(full):        # web_viewer returns the page itself
                    full = os.path.join(full, "index.html")
                href = os.path.relpath(full, data["out"]).replace(os.sep, "/")
                extra.append(f'<a href="{_esc(href)}">{label}</a>')
        parts.append(f'<figure><img src="{_esc(rel)}" alt="{ttl}: studio hero render of the result, '
                     f'{_esc(hero.get("backend", "unknown"))} backend">'
                     f'<figcaption>Hero render ({_esc(hero.get("backend", "unknown"))}, '
                     f'{_esc(hero.get("field", {}).get("label", "") if isinstance(hero.get("field"), dict) else "")}'
                     f'): {_esc(hero.get("caption", "the same data as the 2-D figures, in three dimensions."))}'
                     + (f'<br>{" · ".join(extra)}.' if extra else "")
                     + "</figcaption></figure>")
    if data["figures"]:
        for f in data["figures"]:
            parts.append(f'<figure><img src="{_esc(f["path"])}" alt="{ttl}: {_esc(f["caption"])}">'
                         f'<figcaption>{_esc(f["caption"])}</figcaption></figure>')
    else:
        parts.append('<p class="note">The manifest names no figure.</p>')

    # -- uncertainty and validity
    val = data["validity"]
    parts.append('<h2 id="validity">Uncertainty and validity</h2>')
    cl = val["checklist"]
    if cl.get("items"):
        parts.append(f'<p class="note">{_esc(cl["badge"])} — nine protocol items, each met or not, each with the '
                     f'file that said so.</p><ul class="plain">')
        for i in cl["items"]:
            parts.append(f'<li><span style="min-width:150px">{_pill(i["status"])}</span>'
                         f'<span><b>{_esc(i["id"])}</b> {_esc(i["title"])}'
                         + (f'<br><span class="ev">{_esc("; ".join(i["evidence"][:2]))}</span>'
                            if i.get("evidence") else "")
                         + (f'<br><span class="ev">fix: {_esc(i["fix"])}</span>' if i.get("fix") else "")
                         + "</span></li>")
        parts.append("</ul>")
    st = val["stages"]
    parts.append(f'<h2>Trust card: six questions</h2><p class="note">Answered, partial, open or failed, from '
                 f'{_esc(st["source"])}'
                 + (f' (tier {_esc(st["tier"])} {_esc(st["tier_name"])})' if st.get("tier") else
                    " — the tier is the Lab curation's; this folder has no record there")
                 + ".</p><div class=\"grid2\">")
    for key, s in st["stages"].items():
        parts.append(f'<div class="stage {_esc(s.get("status", ""))}"><h3>{_esc(s.get("title", key))}</h3>'
                     f'<p class="q">{_esc(s.get("question", ""))}</p>'
                     f'<p class="q">{_pill(s.get("status", ""))}</p>'
                     '<ul>' + "".join(f"<li>{_esc(x)}</li>" for x in (s.get("evidence") or [])[:4]) + "</ul>"
                     + (f'<p class="q">next: {_esc(s["next"])}</p>' if s.get("next") else "") + "</div>")
    parts.append("</div>")
    if val["not_measured"] or val["not_applicable"]:
        parts.append('<h3 style="margin-top:18px">What was not measured</h3><ul class="plain">')
        parts += [f"<li>{_esc(x)}</li>" for x in val["not_measured"]]
        parts += [f'<li><span class="ev">not applicable — {_esc(k)}: {_esc(v)}</span></li>'
                  for k, v in val["not_applicable"].items()]
        parts.append("</ul>")

    # -- provenance
    p = data["provenance"]
    parts.append('<h2 id="provenance">Provenance</h2><table><caption>Where every number came from and how to '
                 "reproduce it.</caption><tbody>")
    rows = [("manifest", f'{os.path.relpath(os.path.join(data["folder"], "experiment.json"), data["folder"])} '
                         f'(written by {p.get("source") or "unknown"})'),
            ("commit", p.get("git") or "-"), ("created", p.get("created") or "-"),
            ("run environment", ", ".join(f"{k} {v}" for k, v in (p.get("environment") or {}).items()) or "-"),
            ("measuring machine", ", ".join(f"{k} {v}" for k, v in (p.get("machine") or {}).items()) or "-"),
            ("costs", h.get("costs_file") or "the manifest's own seconds"),
            ("modules", ", ".join(p.get("modules") or []) or "-"),
            ("references", "; ".join(p.get("references") or []) or "none recorded"),
            ("data files", ", ".join(p.get("data") or []) or "none recorded"),
            ("reproduce", f'<code>{_esc(data["reproduce"])}</code>')]
    for k, val in rows:
        parts.append(f"<tr><td class=\"kind\">{_esc(k)}</td><td>{val if k == 'reproduce' else _esc(val)}</td></tr>")
    parts.append("</tbody></table>")
    parts.append(f'<footer>Generated {_esc(data["generated"])} by pinneapple_analysis.report from '
                 f'<code>{_esc(os.path.basename(data["folder"]))}/experiment.json</code> and the results beside it; '
                 f'nothing on this page was typed by hand. House theme, {_esc(theme)}.</footer>')
    parts.append("</main></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------- the page: PDF
def render_pdf(data: dict[str, Any], path: str, *, theme: str = "light") -> str:
    """The same page as PDF, in the plain single-column style of :mod:`pinneapple_veriphysics.evidence_report`.

    Text, tables and the figures -- on a page an engineer prints and files. It carries the same numbers as the HTML,
    rendered from the same ``data``, so the two cannot disagree. ``reportlab`` is optional; when it is missing the
    caller gets a :class:`RuntimeError` naming the extra instead of an empty file.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            KeepTogether,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover - only where the extra is missing
        raise RuntimeError('the PDF report needs reportlab: pip install "pinneapple[report]"') from exc

    from pinneapple_tools.visualization.theme import surface

    s = surface(dark=(theme == "dark"))
    ink, muted, line = colors.HexColor(s["ink"]), colors.HexColor(s["muted"]), colors.HexColor(s["line"])
    paper = colors.HexColor(s["paper"])
    pass_c, fail_c = colors.HexColor(PASS), colors.HexColor(FAIL)

    base = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23,
                           textColor=ink, alignment=TA_LEFT, spaceAfter=4)
    sub = ParagraphStyle("s", parent=base["BodyText"], fontSize=9.5, leading=13, textColor=muted, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=13,
                       textColor=ink, spaceBefore=12, spaceAfter=5)
    body = ParagraphStyle("b", parent=base["BodyText"], fontSize=8.8, leading=11.6, textColor=ink)
    small = ParagraphStyle("sm", parent=body, fontSize=7.8, leading=10, textColor=muted)

    def _hx(c) -> str:                          # reportlab wants the leading # (or 0x) in a font colour
        return "#" + c.hexval()[2:]

    def P(text: Any, style=body) -> Paragraph:
        return Paragraph(_esc(text).replace("\n", "<br/>"), style)

    h = data["header"]
    story: list = [P(h["title"] or "Experiment", title), P(h.get("question") or "", sub)]
    wall = h.get("wall_seconds")
    wall_txt = ("not recorded" if not wall else f"{wall / 3600:.2f} h" if wall >= 3600 else
                f"{wall / 60:.1f} min" if wall >= 60 else f"{wall:.1f} s")
    hw = h.get("hardware") or {}
    hw_txt = ", ".join(x for x in (hw.get("cpu"), f"{hw.get('cpu_count')} cores" if hw.get("cpu_count") else "",
                                   hw.get("platform")) if x)
    meta = [("date", h.get("date") or "-"), ("commit", (h.get("commit") or "-")[:12]),
            ("hardware", hw_txt or "-"), ("wall time", wall_txt),
            ("cost", f"US$ {h['cost']['usd']:.2f}" if h.get("cost") else "not priced (no rate given)"),
            ("domain", h.get("domain") or "-")]
    mt = Table([[P(k, small), P(v, body)] for k, v in meta], colWidths=[24 * mm, 150 * mm])
    mt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                            ("TOPPADDING", (0, 0), (-1, -1), 1)]))
    story += [mt, Spacer(1, 6)]

    # KPI cards as a grid of two columns
    if data["kpis"]:
        story.append(P("Headline numbers", h2))
        cells = []
        for k in data["kpis"]:
            ink_c = {"pass": pass_c, "fail": fail_c}.get(k["status"], muted)
            label = {"pass": "check passed", "fail": "check failed",
                     "no check": "no check names this number"}[k["status"]]
            band = ""
            if k.get("band"):
                lo, hi = k["band"]
                band = f"{_fmt(lo) if lo != float('-inf') else '-inf'} .. {_fmt(hi) if hi != float('inf') else 'inf'}"
            cells.append([Paragraph(f'<b>{_esc(k["name"])}</b>', small),
                          Paragraph(f'<font size="13"><b>{_esc(_fmt(k["value"]))}</b></font>'
                                    + (f' <font size="8" color="{s["muted"]}">{_esc(k["unit"])}</font>'
                                       if str(k.get("unit") or "") not in ("", "-", "1") else ""), body),
                          P(band + (f" ({k['target']})" if k.get("target") else ""), small),
                          Paragraph(f'<font color="{_hx(ink_c)}"><b>{label}</b></font>', small)])
        half = (len(cells) + 1) // 2
        rows = [[cells[i], cells[i + half] if i + half < len(cells) else [Paragraph("", small)] * 4]
                for i in range(half)]
        t = Table(rows, colWidths=[87 * mm] * 2)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.4, line),
                               ("INNERGRID", (0, 0), (-1, -1), 0.3, line), ("LEFTPADDING", (0, 0), (-1, -1), 5),
                               ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 3),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        story.append(t)

    # verdict table
    v = data["verdicts"]
    story.append(P("Verdicts", h2))
    if v["rows"]:
        story.append(P(f'{v["passed"]} passed, {v["failed"]} failed, read from the run\'s own checks.', small))
        rows = [[P("check", small), P("kind", small), P("measured", small), P("verdict", small), P("evidence", small)]]
        for r in v["rows"]:
            c = pass_c if r["verdict"] == "PASS" else fail_c
            rows.append([P(r["name"], body), P(r["kind"], small), P(r["measured"], body),
                         Paragraph(f'<font color="{_hx(c)}"><b>{r["verdict"]}</b></font>', body),
                         P(r["evidence"], small)])
        t = Table(rows, colWidths=[44 * mm, 18 * mm, 33 * mm, 18 * mm, 61 * mm], repeatRows=1)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.4, line),
                               ("LINEBELOW", (0, 1), (-1, -1), 0.2, line), ("TOPPADDING", (0, 0), (-1, -1), 2),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        story.append(t)
    else:
        story.append(P("The run recorded no check: nothing here was compared with a reference, a baseline or a "
                       "physical law.", small))

    # figures
    figs = []
    hero = data.get("hero") or {}
    stills = hero.get("stills") or {}
    hero_path = stills.get("1920x1080") or (next(iter(stills.values())) if stills else None)
    if hero_path:
        if not os.path.isabs(hero_path):
            hero_path = os.path.join(data["out"], hero_path)
        if os.path.exists(hero_path):
            figs.append((hero_path, f'hero render ({hero.get("backend", "unknown")} backend)'))
    figs += [(os.path.join(data["out"], f["path"]), f["caption"]) for f in data["figures"]]
    if figs:
        story.append(P("Figures", h2))
        for path_i, cap in figs[:5]:
            try:
                im = Image(path_i)
                w = 150 * mm
                im.drawWidth, im.drawHeight = w, w * im.imageHeight / im.imageWidth
                story.append(KeepTogether([im, P(cap, small), Spacer(1, 4)]))
            except Exception:                       # an unreadable figure never takes the report down
                story.append(P(f"[figure not embedded: {os.path.basename(path_i)}]", small))

    # validity
    story.append(P("Uncertainty and validity", h2))
    cl = data["validity"]["checklist"]
    if cl.get("items"):
        story.append(P(cl["badge"], small))
        rows = [[Paragraph(f'<font color="#{({"met": PASS, "missing": FAIL}.get(i["status"], NEUTRAL)).lower().lstrip("#")}">'
                           f'<b>{_esc(i["status"])}</b></font>', small), P(i["id"], body), P(i["title"], small),
                 P("; ".join(i["evidence"][:2]), small)] for i in cl["items"]]
        t = Table(rows, colWidths=[18 * mm, 24 * mm, 33 * mm, 99 * mm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.2, line),
                               ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
        story.append(t)
    st = data["validity"]["stages"]
    story.append(P(f'Trust card (from {st["source"]}'
                   + (f', tier {st["tier"]} {st["tier_name"]})' if st.get("tier") else
                      "; this folder has no Lab record, so it is not graded here)"), h2))
    rows = [[P("question", small), P("status", small), P("evidence", small)]]
    for k, s_ in st["stages"].items():
        c = colors.HexColor(_STATUS_INK.get(s_.get("status", ""), NEUTRAL))
        rows.append([P(f'{s_.get("title", k)}: {s_.get("question", "")}', body),
                     Paragraph(f'<font color="{_hx(c)}"><b>{_esc(s_.get("status", ""))}</b></font>', body),
                     P("; ".join(s_.get("evidence") or [])[:400], small)])
    t = Table(rows, colWidths=[55 * mm, 23 * mm, 96 * mm], repeatRows=1)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.2, line),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    story.append(t)
    if data["validity"]["not_measured"]:
        story.append(P("What was not measured", h2))
        story += [P(f"- {x}", small) for x in data["validity"]["not_measured"]]

    # provenance
    p = data["provenance"]
    story.append(P("Provenance", h2))
    rows = [[P("commit", small), P(p.get("git") or "-", body)],
            [P("created", small), P(p.get("created") or "-", body)],
            [P("run environment", small), P(", ".join(f"{k} {v}" for k, v in (p.get("environment") or {}).items())
                                            or "-", body)],
            [P("measuring machine", small), P(", ".join(f"{k} {v}" for k, v in (p.get("machine") or {}).items())
                                              or "-", body)],
            [P("references", small), P("; ".join(p.get("references") or []) or "none recorded", body)],
            [P("data files", small), P(", ".join(p.get("data") or []) or "none recorded", body)],
            [P("reproduce", small), P(data["reproduce"], body)]]
    t = Table(rows, colWidths=[32 * mm, 142 * mm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.2, line),
                           ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(P(f'Generated {data["generated"]} by pinneapple_analysis.report from experiment.json and the '
                   f'results beside it; nothing on this page was typed by hand.', small))

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def _paint(canvas, doc):                    # the house paper colour, behind every page
        canvas.saveState()
        canvas.setFillColor(paper)
        canvas.rect(0, 0, *A4, stroke=0, fill=1)
        canvas.restoreState()

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                            bottomMargin=15 * mm, title=_esc(h["title"] or "Experiment report"), author="PINNeAPPle",
                            subject="One-page experiment report")
    doc.build(story, onFirstPage=_paint, onLaterPages=_paint)
    return path


# ---------------------------------------------------------------------- the one call
def experiment_report(folder: str, *, out: str | None = None, theme: str = "light", pdf: bool = True,
                      figures: list[str] | None = None, limit_figures: int = 4, limit_kpis: int = 6,
                      hero: Any = None, cost_per_hour: float | None = None) -> Report:
    """Render the one-page report of ``folder`` (a folder with ``experiment.json``); returns its two paths.

    ``out`` defaults to ``<folder>/report``: ``index.html``, ``report.pdf``, ``report.json`` (the data the page was
    rendered from, which is what an index or a test reads) and ``assets/`` with the figures it shows. ``theme`` is
    ``"light"`` or ``"dark"`` and takes the colours from the house theme. ``figures`` overrides which of the
    manifest's figures are shown (default: the first ``limit_figures`` it names). ``hero`` is a :func:`hero` result
    (or a still path) from :mod:`pinneapple_tools.visualization.studio.hero`; without one, a ``hero.json`` left in
    ``out`` by that call is picked up. ``cost_per_hour`` prices the wall time -- without a rate the report says the
    run is not priced rather than inventing a number.
    """
    if theme not in ("light", "dark"):
        raise ValueError(f'theme={theme!r}: "light" or "dark"')
    data = gather(folder, out=out, limit_kpis=limit_kpis, limit_figures=limit_figures, figures=figures, hero=hero,
                  cost_per_hour=cost_per_hour)
    os.makedirs(data["out"], exist_ok=True)
    html_path = os.path.join(data["out"], "index.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render_html(data, theme=theme))
    pdf_path = render_pdf(data, os.path.join(data["out"], "report.pdf"), theme=theme) if pdf else None
    with open(os.path.join(data["out"], "report.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, default=str)
    return Report(html=html_path, pdf=pdf_path, data=data)
