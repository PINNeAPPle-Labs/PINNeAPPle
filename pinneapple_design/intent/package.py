"""DesignPackage: everything needed to build and check the design, and the files it is written to.

``package.write(folder)`` writes

* ``design.json`` - the whole package (requirements with provenance, plan, design, every number with its fidelity,
  margins, robustness, trust report, operating window, bill of materials, build instructions, limits);
* ``report.html`` (self-contained) and ``report.pdf``;
* ``geometry/<name>.stl``, ``.step`` (faceted B-rep), ``.glb``, and rendered views;
* ``check/`` - quantities of interest and a ``pinneapple-check.toml`` so that ``pinneapple check check/`` re-checks
  the margins and the agreement between the reduced model and the full solver.
"""
from __future__ import annotations

import base64
import csv
import html
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .pipeline import (
    Assessment,
    CallLog,
    Candidate,
    OptimisationResult,
    PhysicsPlan,
    Robustness,
    Verification,
)
from .spec import RequirementSpec


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not math.isfinite(f) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def make_geometry(name: str, title: str, bodies: list):
    """A :class:`pinneapple_design.geometry_library.core.Geometry` from bodies (exports STL / STEP / glTF)."""
    from pinneapple_design.geometry_library import meshops
    from pinneapple_design.geometry_library.core import Geometry, GeometrySpec

    for b in bodies:
        if b.closed:
            b.vertices, b.faces = meshops.orient_outward(b.vertices, b.faces)
    spec = GeometrySpec(name=name, title=title, category="design", description=title,
                        builder=lambda p, lod: bodies, params=(), dimensions={}, frame="z up, metres",
                        sources=(), physics=(), camera={"azimuth": 235.0, "elevation": 22.0})
    return Geometry(name, bodies, {}, "light", spec)


@dataclass
class DesignPackage:
    spec: RequirementSpec
    plan: PhysicsPlan
    optimisation: OptimisationResult
    candidate: Candidate
    design: dict[str, Any]
    robustness: Robustness
    alternatives: list[dict[str, Any]]
    verification: Verification
    assessment: Assessment
    operating_window: list[dict[str, Any]]
    bom: list[dict[str, Any]]
    build: list[str]
    limits: list[str]
    numbers: list[dict[str, Any]]           # every reported number with its fidelity
    calls: CallLog
    savings: dict[str, Any]
    geometry: Any = None
    figures: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    seed: int = 0
    rounds: list[dict[str, Any]] = field(default_factory=list)   # calibration rounds against the full solver
    status: str = "designed"

    # ------------------------------------------------------------------ text
    @property
    def verdict(self) -> str:
        return self.assessment.verdict

    def summary(self) -> str:
        a = self.assessment
        lines = [f"{self.plan.title}: {a.verdict} (P = {a.probability:.3f}, 95 % CI {a.ci[0]:.3f}-{a.ci[1]:.3f}; "
                 f"trust {a.trust.decisao})"]
        for v in self.plan.variables:
            val = self.design.get(v.name)
            lines.append(f"  {v.label or v.name}: {val if isinstance(val, str) else f'{val:.4g}'} {v.unit}".rstrip())
        lines.append(f"  model calls: {self.calls.calls}; full solver: "
                     f"{'ran' if self.verification.ran else 'not run (' + self.verification.reason + ')'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return _clean({
            "status": self.status, "seed": self.seed, "verdict": self.verdict,
            "requirements": self.spec.to_dict(), "plan": self.plan.to_dict(),
            "design": self.design, "numbers": self.numbers,
            "optimisation": self.optimisation.to_dict(), "robustness": self.robustness.to_dict(),
            "alternatives": self.alternatives, "verification": self.verification.to_dict(),
            "assessment": self.assessment.to_dict(), "operating_window": self.operating_window,
            "bill_of_materials": self.bom, "build_instructions": self.build, "limits": self.limits,
            "calls": self.calls.to_dict(), "savings": self.savings, "calibration_rounds": self.rounds,
            "files": self.files,
        })

    # ------------------------------------------------------------------ files
    def write(self, out_dir: str, *, pdf: bool = True, renders: bool = True) -> dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        files = dict(self.files)
        if self.geometry is not None:
            gdir = os.path.join(out_dir, "geometry")
            os.makedirs(gdir, exist_ok=True)
            for ext in ("stl", "step", "glb"):
                if ext == "step" and files.get("step_exact"):
                    files["step"] = files["step_exact"]     # the solver's exact B-rep beats a faceted copy
                    continue
                p = os.path.join(gdir, f"{self.geometry.name}.{ext}")
                self.geometry.export(p)
                files[ext] = p
            if renders:
                p = os.path.join(gdir, f"{self.geometry.name}.png")
                self.geometry.render(p, size=(960, 640))
                files["render"] = p
                self.figures.setdefault("render", p)
        files.update(self._write_check(os.path.join(out_dir, "check")))
        self.files = files
        p = os.path.join(out_dir, "design.json")
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        files["json"] = p
        files["html"] = write_html(self, os.path.join(out_dir, "report.html"))
        if pdf:
            try:
                files["pdf"] = write_pdf(self, os.path.join(out_dir, "report.pdf"))
            except ImportError:
                pass
        self.files = files
        with open(os.path.join(out_dir, "design.json"), "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return files

    def _write_check(self, d: str) -> dict[str, str]:
        os.makedirs(os.path.join(d, "qoi"), exist_ok=True)
        rows = [(f.name, self.candidate.outputs.get(f.name)) for f in _modes(self)]
        rows = [(k, v) for k, v in rows if v is not None and math.isfinite(v)]
        q = os.path.join(d, "qoi", "margins.csv")
        with open(q, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "value", "unit"])
            for k, v in rows:
                w.writerow([k, f"{v:.6g}", "1"])
        toml = ["# written by pinneapple_design.intent: run `pinneapple check` on this folder", "",
                "[[result]]", 'path = "qoi/margins.csv"',
                "bounds = { " + ", ".join(f"{k} = [0.0, 1e9]" for k, _ in rows) + " }"]
        out = {"check_margins": q}
        comp = [r for r in self.verification.comparison if r["full"] is not None and math.isfinite(r["reduced"])
                and r.get("kind", "agree") == "agree"]
        if comp:
            os.makedirs(os.path.join(d, "reference"), exist_ok=True)
            pr, pf = os.path.join(d, "qoi", "reduced.csv"), os.path.join(d, "reference", "reduced.csv")
            for path, key in ((pr, "reduced"), (pf, "full")):
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["name", "value", "unit"])
                    for r in comp:
                        w.writerow([r["output"], f"{r[key]:.6g}", "1"])
            toml += ["", "[[result]]", 'path = "qoi/reduced.csv"', "[result.reference]",
                     'path = "reference/reduced.csv"',
                     "qoi = { " + ", ".join(f"{r['output']} = {{ rel = 0.10 }}" for r in comp) + " }"]
            out["check_reduced"] = pr
        t = os.path.join(d, "pinneapple-check.toml")
        with open(t, "w") as f:
            f.write("\n".join(toml) + "\n")
        out["check_toml"] = t
        return out


def _modes(pkg: DesignPackage):
    from .parse import get_template
    return get_template(pkg.plan.template).failure_modes


# ============================================================================================ HTML report
_CSS = """
:root{--fg:#1d232b;--muted:#5b6672;--bg:#ffffff;--card:#f5f7f9;--line:#d9dee4;--ok:#16794c;--bad:#b42318;
--warn:#a15c07;--accent:#2457a6}
@media (prefers-color-scheme: dark){:root{--fg:#e6e9ed;--muted:#a3adb8;--bg:#14181d;--card:#1d232a;--line:#323b45;
--ok:#4cc38a;--bad:#ff6b5e;--warn:#f0a845;--accent:#7aa7ff}}
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg);
margin:0 auto;max-width:1060px;padding:24px 16px}
h1{font-size:1.6rem;margin:.2rem 0}h2{font-size:1.15rem;margin:2rem 0 .6rem;border-bottom:1px solid var(--line);
padding-bottom:.3rem}
.muted{color:var(--muted)}.verdict{display:inline-block;padding:.25rem .7rem;border-radius:6px;font-weight:600;
background:var(--card);border:1px solid var(--line)}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.4rem 0}
th,td{border-bottom:1px solid var(--line);padding:.35rem .5rem;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}td.num{font-variant-numeric:tabular-nums;text-align:right}
.scroll{overflow-x:auto}.figs img{max-width:100%;border:1px solid var(--line);border-radius:6px;margin:.4rem 0;
background:#fff}
code{background:var(--card);padding:0 .25rem;border-radius:3px}
"""


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            return "-"
        return f"{float(v):.{digits}g}"
    return html.escape(str(v))


def _table(rows: list[dict[str, Any]], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p class='muted'>none</p>"
    h = "".join(f"<th>{html.escape(t)}</th>" for _, t in cols)
    body = []
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k)
            cls = " class='num'" if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else ""
            cells.append(f"<td{cls}>{_fmt(v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><tr>{h}</tr>{''.join(body)}</table></div>"


def _img(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "gif": "image/gif", "jpg": "image/jpeg"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return f"<img alt='' src='data:{mime};base64,{base64.b64encode(f.read()).decode()}'>"


def write_html(pkg: DesignPackage, path: str) -> str:
    a = pkg.assessment
    cls = {"will work": "ok", "likely to work": "ok", "marginal": "warn"}.get(a.verdict, "bad")
    t = a.trust.to_dict()
    checks = [{"check": c.get("nome", c.get("name")), "status": c.get("status"), "critical": c.get("critica",
                                                                                                    c.get("critical")),
               "value": c.get("valor", c.get("value")), "limit": c.get("limite", c.get("limit")),
               "source": c.get("fonte", c.get("source")), "reason": c.get("motivo", c.get("reason"))}
              for c in t["checagens"]]
    rob = pkg.robustness
    parts = [
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>{html.escape(pkg.plan.title)} design</title>"
        f"<style>{_CSS}</style></head><body>",
        f"<p class='muted'>Design from intent · template <code>{html.escape(pkg.plan.template)}</code> · seed "
        f"{pkg.seed}</p><h1>{html.escape(pkg.plan.title)}</h1>",
        f"<p>Request: <em>{html.escape(pkg.spec.text)}</em></p>",
        f"<p><span class='verdict {cls}'>{html.escape(a.verdict)}</span> &nbsp; probability that every failure mode "
        f"stays away: <b>{a.probability:.3f}</b> (95 % interval {a.ci[0]:.3f} to {a.ci[1]:.3f}, {rob.n} Monte-Carlo "
        f"samples on the {html.escape(rob.level.replace('_', ' '))} model) · trust decision <b>{t['decisao']}</b></p>",
        "<h2>Requirements</h2>",
        _table(pkg.spec.table(_slots(pkg)), [("label", "requirement"), ("value", "value"), ("provenance", "provenance"),
                                             ("source", "quote / source")]),
    ]
    if pkg.spec.unsupported:
        parts.append("<p class='warn'>Not addressed: " + "; ".join(html.escape(u) for u in pkg.spec.unsupported)
                     + "</p>")
    parts += ["<h2>Physics plan</h2><ul>" + "".join(f"<li>{html.escape(p)}</li>" for p in pkg.plan.physics) + "</ul>",
              _table(pkg.plan.fidelities, [("level", "fidelity"), ("label", "model"), ("method", "method"),
                                           ("available", "available"), ("reason", "why not")]),
              "<h2>Design</h2>",
              _table([{"variable": v.label or v.name, "value": pkg.design.get(v.name), "unit": v.unit,
                       "range searched": f"{v.lower:g} to {v.upper:g}" if not v.choices else ", ".join(v.choices)}
                      for v in pkg.plan.variables], [("variable", "variable"), ("value", "value"), ("unit", "unit"),
                                                     ("range searched", "range searched")]),
              "<h2>Every number and the model it comes from</h2>",
              _table(pkg.numbers, [("quantity", "quantity"), ("value", "value"), ("unit", "unit"),
                                   ("fidelity", "fidelity"), ("note", "note")]),
              "<h2>Verification against the full solver</h2>",
              (f"<p>{html.escape(pkg.verification.label)}: "
               + ("ran" if pkg.verification.ran else "not run: " + html.escape(pkg.verification.reason)) + "</p>"),
              _table(pkg.verification.comparison, [("output", "output"), ("full", "full solver"),
                                                   ("reduced", "reduced model"), ("closed_form", "closed form"),
                                                   ("rel_err", "|full - reduced| / full"), ("kind", "test")]),
              ]
    if pkg.verification.result is not None:
        parts.append(_table(pkg.verification.result.checks, [("name", "solver check"), ("passed", "passed"),
                                                             ("value", "value"), ("limit", "limit"),
                                                             ("detail", "detail")]))
        parts.append("<ul>" + "".join(f"<li class='muted'>{html.escape(n)}</li>" for n in
                                      pkg.verification.result.notes) + "</ul>")
    parts += ["<h2>Robustness</h2>",
              _table(rob.inputs, [("name", "uncertain input"), ("nominal", "nominal"), ("spread", "spread"),
                                  ("dist", "distribution"), ("source", "why this range")]),
              _table([{"failure mode": k, "P(margin >= 0)": v, "5th percentile of margin": rob.margin_p05.get(k)}
                      for k, v in rob.per_mode.items()],
                     [("failure mode", "failure mode"), ("P(margin >= 0)", "P(margin >= 0)"),
                      ("5th percentile of margin", "5th percentile of margin")]),
              _table([{"input": k, "rank correlation with the smallest margin": v} for k, v in
                      rob.sensitivity.items()], [("input", "input"),
                                                 ("rank correlation with the smallest margin", "rank correlation "
                                                                                              "with the smallest "
                                                                                              "margin")]),
              "<h2>Trust report</h2>",
              _table(checks, [("check", "check"), ("status", "status"), ("critical", "critical"), ("value", "value"),
                              ("limit", "limit"), ("source", "source"), ("reason", "reason")]),
              "<h2>Operating window</h2>",
              _table(pkg.operating_window, [("quantity", "quantity"), ("value", "design"), ("min", "min"),
                                            ("max", "max"), ("unit", "unit"), ("fidelity", "fidelity"),
                                            ("note", "note")]),
              "<h2>Bill of materials</h2>",
              _table([{**b, "sources": "; ".join(sorted({p['source'] for p in b.get('properties', {}).values()}))}
                      for b in pkg.bom], [("item", "item"), ("material", "material"), ("quantity", "qty"),
                                          ("size", "size"), ("mass_g", "mass [g]"), ("sources", "property sources")]),
              "<h2>Build instructions</h2><ol>" + "".join(f"<li>{html.escape(s)}</li>" for s in pkg.build) + "</ol>",
              "<h2>Model calls</h2>",
              _table([{"fidelity": k, "designs evaluated": pkg.calls.calls[k], "wall [s]": pkg.calls.wall[k]}
                      for k in pkg.calls.calls], [("fidelity", "fidelity"), ("designs evaluated", "designs evaluated"),
                                                  ("wall [s]", "wall [s]")]),
              f"<p>{html.escape(pkg.savings.get('text', ''))}</p>",
              "<h2>Limits and assumptions</h2><ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in
                                                             pkg.limits + a.assumptions) + "</ul>"]
    figs = [p for p in pkg.figures.values() if os.path.exists(p)]
    if figs:
        parts.append("<h2>Figures</h2><div class='figs'>" + "".join(_img(p) for p in figs) + "</div>")
    if pkg.files:
        parts.append("<h2>Files</h2><ul>" + "".join(f"<li><code>{html.escape(k)}</code>: "
                                                    f"{html.escape(os.path.basename(v))}</li>"
                                                    for k, v in pkg.files.items()) + "</ul>")
    parts.append("</body></html>")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    return path


def _slots(pkg: DesignPackage):
    from .parse import get_template
    return get_template(pkg.plan.template).slot_map()


def write_pdf(pkg: DesignPackage, path: str) -> str:
    """A short PDF: the verdict, the design, the verification and the figures (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    a = pkg.assessment
    lines = [pkg.plan.title, "", f"Request: {pkg.spec.text}", "",
             f"Verdict: {a.verdict}   P = {a.probability:.3f} (95 % CI {a.ci[0]:.3f}-{a.ci[1]:.3f})   "
             f"trust: {a.trust.decisao}", "", "Design:"]
    for v in pkg.plan.variables:
        val = pkg.design.get(v.name)
        lines.append(f"   {v.label or v.name}: {val if isinstance(val, str) else f'{val:.4g}'} {v.unit}")
    lines += ["", "Numbers (fidelity):"]
    lines += [f"   {r['quantity']}: {_fmt(r['value'])} {r.get('unit', '')}  [{r['fidelity']}]" for r in pkg.numbers]
    lines += ["", "Verification:"]
    lines += [f"   {r['output']}: full {_fmt(r['full'])}, reduced {_fmt(r['reduced'])}, rel. err {_fmt(r['rel_err'])}"
              for r in pkg.verification.comparison] or [f"   not run: {pkg.verification.reason}"]
    lines += ["", "Limits:"] + [f"   - {s}" for s in pkg.limits]
    with PdfPages(path) as pdf:
        for k in range(0, len(lines), 48):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.06, 0.96, "\n".join(_wrap(lines[k:k + 48])), va="top", family="monospace", fontsize=7.2)
            pdf.savefig(fig)
            plt.close(fig)
        for p in pkg.figures.values():
            if os.path.exists(p) and p.lower().endswith(".png"):
                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
                ax.imshow(plt.imread(p))
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
    return path


def _wrap(lines: list[str], width: int = 118) -> list[str]:
    import textwrap
    out = []
    for s in lines:
        out += textwrap.wrap(s, width, subsequent_indent="      ") or [""]
    return out
