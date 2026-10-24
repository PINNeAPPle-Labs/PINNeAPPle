"""Committed results of :mod:`pinneapple_science`: one folder per experiment with every dimension it runs in,
its Lab manifest (``experiment.json``), its Lab seal and a README; and the catalogue with the atlas.

    python -m pinneapple_science report all          # every experiment (full size), manifests, seals, catalogue
    python -m pinneapple_science report em.gauss_law

Layout: ``examples/science/<area>/<experiment>/<dim>/`` holds ``result.json`` (values, comparisons, checks,
trust report, data), the figures and the movies; the experiment folder holds ``experiment.json`` (registered in
the Lab, so it gets an A-F grade) and copies of the figures stamped with that seal (``*.sealed.png``);
``examples/science/`` holds the catalogue (``README.md``), the atlas (``atlas.html``, ``ATLAS.md``) and its own
manifest.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .atlas import load as load_atlas
from .core import Result
from .registry import get, names

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, "examples", "science")

__all__ = ["write", "write_all", "write_catalogue", "folder_of", "table", "RESULTS_DIR"]


def folder_of(name: str, root: str | None = None) -> str:
    e = get(name)
    area = e.area.split(".")[-1]
    return os.path.join(root or RESULTS_DIR, area, name.split(".", 1)[-1])


def _dim_dir(dim: str) -> str:
    return dim.replace("+", "_plus_")


def write(name: str, *, root: str | None = None, quick: bool = False, register: bool = True,
          seal: bool = True) -> dict[str, Any]:
    """Run ``name`` in every dimension, save the results, write the README and the Lab manifest, register it in
    the Lab and stamp the figures with its seal. Returns ``{"folder", "results", "item", "sealed"}``."""
    e = get(name)
    folder = folder_of(name, root)
    results: list[Result] = []
    files: list[tuple[str, str]] = []
    for dim in e.dims:
        r = e.run(dim, quick=quick)
        paths = r.save(os.path.join(folder, _dim_dir(dim)))
        for k, pth in paths.items():
            kind = "data" if k == "result" or pth.endswith(".npz") else None
            files.append((pth, kind))
        results.append(r)
    from pinneapple_lab import experiment

    rel = os.path.relpath(folder, REPO).replace(os.sep, "/")
    with experiment(rel, title=f"{e.title} ({', '.join(e.dims)})", description=e.description,
                    domain=f"science / {e.area}", methods=[f"pinneapple_science: {name}"],
                    modules=["pinneapple_science", "pinneapple_simulation.electromagnetics"],
                    out_dir=folder, register=register, references=list(e.references),
                    links={"guide": "docs/guides/science.md"}) as ex:
        ex.param(level=e.level, kind=e.kind, dims=list(e.dims), quick=quick)
        for r in results:
            for c in r.comparisons:
                ex.log_metric(f"rel_error[{r.dim}: {c.quantity}]", c.rel_error, "-",
                              f"|computed - reference| / |reference| ({c.kind}: {c.source})")
                ex.check(f"[{r.dim}] {c.quantity}", c.passed, value=c.computed, reference=c.reference,
                         rtol=c.tol, kind="reference", label=f"{c.quantity} vs {c.kind.replace('_', ' ')}",
                         detail=c.source)
            for k in r.checks:
                if k["passed"] is not None:
                    ex.check(f"[{r.dim}] {k['name']}", k["passed"], value=k["value"], detail=k["detail"],
                             kind="physics")
            ex.check(f"[{r.dim}] trust report", r.trust.decisao == "APPROVED", detail=f"decision {r.trust.decisao}",
                     kind="verification")
            for name_, v in r.values.items():
                ex.log_metric(f"{r.dim}: {name_}", v["value"], v["unit"] or "-", v["how"])
        for pth, kind in files:
            ex.add_file(pth, kind)
    from . import plots
    for r in results:
        for fg in r.figures.values():
            plots.close(fg)
    sealed = []
    item = ex.item
    if register and seal and item:
        from pinneapple_lab.seal import seal_for, stamp_file

        s = seal_for(item)
        for r in results:
            d = os.path.join(folder, _dim_dir(r.dim))
            for fig in r.figures:
                src = os.path.join(d, f"{fig}.png")
                sealed.append(stamp_file(src, s, os.path.join(d, f"{fig}.sealed.png")))
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme(name, results, sealed, item))
    return {"folder": folder, "results": results, "item": item, "sealed": sealed}


def readme(name: str, results: list[Result], sealed: list[str], item: str | None) -> str:
    e = get(name)
    L = [f"# {e.title}", "", e.description, "",
         f"**Area:** `{e.area}` ({e.topic}) · **level:** {e.level} · **kind:** {e.kind} · **dimensions:** "
         f"{', '.join(e.dims)}", "", "```python", "import pinneapple as pp", "",
         f'r = pp.science.run("{name}", dim="{e.dims[0]}")    # also: ' + ", ".join(f'"{d}"' for d in e.dims),
         "print(r)                     # comparisons with their references, checks, trust decision", "```", ""]
    for r in results:
        d = _dim_dir(r.dim)
        L += [f"## {r.dim}", "", f"Trust report: **{r.trust.decisao}** (score {r.trust.score:.2f}, coverage "
              f"{r.trust.cobertura:.0%}); run time {r.wall_time_s:.1f} s.", "",
              "| quantity | computed | reference | rel. error | tolerance | reference kind | source |",
              "|---|---|---|---|---|---|---|"]
        for c in r.comparisons:
            err = f"{c.rel_error:.2e}" if c.tol > 0 else f"{c.abs_error:.2e} (abs.)"
            tol = f"{c.tol:.2g}" if c.tol > 0 else f"{c.atol:.2g} (abs.)"
            L.append(f"| {c.quantity} | {c.computed:.6g} {c.unit} | {c.reference:.6g} {c.unit} | {err} | "
                     f"{tol} | {c.kind} | {c.source} |")
        if r.checks:
            L += ["", "| physics check | result | value | limit |", "|---|---|---|---|"]
            for k in r.checks:
                res = "n/a" if k["passed"] is None else ("pass" if k["passed"] else "FAIL")
                val = "" if k["value"] is None else f"{k['value']:.3g}"
                lim = "" if k["limit"] is None else f"{k['limit']:.3g}"
                L.append(f"| {k['name']} | {res} | {val} | {lim} |")
        figs = [f"{d}/{f}.sealed.png" if any(s.endswith(os.path.join(d, f"{f}.sealed.png")) for s in sealed)
                else f"{d}/{f}.png" for f in r.figures]
        L += [""] + [f"![{os.path.basename(f)}]({f})" for f in figs]
        for mv in r.movies:
            L += ["", f"Animation (3-D field in time): [{mv}.mp4]({d}/{mv}.mp4), ![{mv}]({d}/{mv}.gif)"]
        L.append("")
    L += ["## References", ""] + [f"- {x}" for x in e.references]
    if item:
        L += ["", f"Lab item `{item}` (manifest `experiment.json`); figures marked `.sealed` carry its seal."]
    L += ["", f"Regenerate: `python -m pinneapple_science report {name}`.", ""]
    return "\n".join(L)


def table(root: str | None = None) -> list[dict[str, Any]]:
    """Experiment x dimension x quantity x reference x measured error, from the committed ``result.json`` files."""
    rows = []
    for n in names():
        f = folder_of(n, root)
        for dim in get(n).dims:
            p = os.path.join(f, _dim_dir(dim), "result.json")
            if not os.path.isfile(p):
                continue
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            for c in d["comparisons"]:
                rows.append({"experiment": n, "dim": dim, "level": d["level"], "exp_kind": d["kind"], **c,
                             "decision": d["trust"]["decision"]})
    return rows


def write_catalogue(root: str | None = None, register: bool = True) -> str:
    """``examples/science/README.md`` (one row per comparison from the committed results), ``ATLAS.md``,
    ``atlas.html`` and the folder's manifest."""
    root = root or RESULTS_DIR
    atlas = load_atlas()
    os.makedirs(root, exist_ok=True)
    atlas.to_html(os.path.join(root, "atlas.html"))
    with open(os.path.join(root, "ATLAS.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Science atlas {atlas.version}\n\n" + atlas.to_markdown())
    rows = table(root)
    L = ["# PINNeAPPle science: experiments", "",
         "Every row is a quantity computed by an experiment of `pinneapple_science` (`pp.science`) and compared with "
         "its reference; the values come from the committed `result.json` files. Regenerate with "
         "`python -m pinneapple_science report all`. The map of every area (available, partial, planned) is "
         "[ATLAS.md](ATLAS.md) and [atlas.html](atlas.html).", "",
         "| experiment | dim | level | kind | quantity | computed | reference | error | tol | reference kind | trust |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        link = os.path.relpath(folder_of(r["experiment"], root), root).replace(os.sep, "/")
        rel = r["tol"] > 0
        tol = f"{r['tol']:.2g}" if rel else f"{r['atol']:.2g} (abs.)"
        err = f"{r['rel_error']:.2e}" if rel else f"{r['abs_error']:.2e} (abs.)"
        L.append(f"| [{r['experiment']}]({link}/README.md) | {r['dim']} | {r['level']} | {r['exp_kind']} | "
                 f"{r['quantity']} | {r['computed']:.6g} | {r['reference']:.6g} | {err} | {tol} | {r['kind']} | "
                 f"{r['decision']} |")
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    from pinneapple_lab import experiment

    with experiment("examples/science", title="Science experiments and atlas",
                    description="pp.science: experiments by area, level, kind and dimension, each checked against "
                                "its reference; the atlas of every area", domain="science",
                    modules=["pinneapple_science"], out_dir=root, register=register,
                    links={"guide": "docs/guides/science.md"}) as ex:
        for n in names():
            oks = [r["passed"] for r in rows if r["experiment"] == n]
            if oks:
                ex.check(f"{n}: every comparison passes", all(oks), detail=f"{len(oks)} comparisons",
                         kind="reference")
        ex.log_metric("comparisons", len(rows), "", "rows of the catalogue (committed results)")
        ex.log_metric("comparisons_passed", sum(r["passed"] for r in rows), "", "rows within tolerance")
        ex.add_file(os.path.join(root, "atlas.html"), "viewers")
    return os.path.join(root, "README.md")


def write_all(*, root: str | None = None, quick: bool = False, register: bool = True) -> list[dict[str, Any]]:
    out = [write(n, root=root, quick=quick, register=register) for n in names()]
    write_catalogue(root, register=register)
    return out
