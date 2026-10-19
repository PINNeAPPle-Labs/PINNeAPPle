"""``python -m pinneapple_lab ...`` -- run experiments and inspect the lab database.

    python -m pinneapple_lab list
    python -m pinneapple_lab run oscillator -p zeta=0.3 -p method=rk4
    python -m pinneapple_lab sweep cylinder_lbm -g Re=40,80,120,160 -j 4
    python -m pinneapple_lab sweep oscillator -n 50                     # Latin hypercube over the experiment's space
    python -m pinneapple_lab status
    python -m pinneapple_lab report --html                              # lab/CATALOG.md, DATASETS.md, index.html
    python -m pinneapple_lab export cylinder_lbm vorticity out/cylinder_vorticity.npz
    python -m pinneapple_lab serve --host 0.0.0.0 --port 8093                # the catalogue as a web app
    python -m pinneapple_lab examples --run --match use_cases                # examples and use cases into the lab
    python -m pinneapple_lab curate                                          # A flagship .. F not usable, per run
    python -m pinneapple_lab brief --tier A,B --use paper -o paper_candidates.pdf     # reports, one or many items
    python -m pinneapple_lab review kepler_law --novelty 4 --story "..." --approve paper,marketing
    python -m pinneapple_lab seal repo_results/ahmed_body -o seal.svg                # the item's quality seal
    python -m pinneapple_lab stamp repo_results/ahmed_body render.jpg -o out.jpg     # seal a figure / GIF / MP4
    python -m pinneapple_lab stamp --all --out export/sealed                         # every figure of every item, A-F
    python -m pinneapple_lab stamp --all --tiers A,B --out export/validated          # only the validated items
    python -m pinneapple_lab legend --lang pt -o seal_legend_pt.png                  # the six seals, A to F
    python -m pinneapple_lab seal --explain em_pinn/em_tm_cavity --lang pt          # what it lacks, and how
    python -m pinneapple_lab improve em_pinn/em_tm_cavity [--run]                   # dry run, or run and re-curate
    python -m pinneapple_lab register examples/my_study                             # manifest -> Lab item, curated
    python -m pinneapple_lab scaffold my_study                                      # a new example, wired
    python -m pinneapple_lab manifests --check                                      # the CI gate
    python -m pinneapple_lab site --out site                                        # catalogue + site/experiments
    python -m pinneapple_lab coverage -o docs/lab/coverage.md                       # experiment -> curated yes/no
    python -m pinneapple_lab showcase --lang en,pt --out docs/assets/showcase         # the animated panel
Use ``--root`` (or ``$PINNEAPPLE_LAB``) to choose the database folder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _val(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def _kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = _val(v)
    return out


TIER_NAMES_PT = {"A": "referência independente", "B": "referência pendente", "C": "verificação parcial",
                 "D": "exploratório", "E": "problemas conhecidos", "F": "não utilizável"}


def main(argv=None) -> int:
    from . import LabStore, available, get, run, sweep

    ap = argparse.ArgumentParser(prog="python -m pinneapple_lab")
    ap.add_argument("--root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("-p", "--param", action="append")
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("sweep")
    s.add_argument("name")
    s.add_argument("-g", "--grid", action="append", help="key=v1,v2,...")
    s.add_argument("-n", "--samples", type=int, default=0)
    s.add_argument("-p", "--param", action="append", help="fixed key=value for every run")
    s.add_argument("-j", "--jobs", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    sub.add_parser("status")
    sub.add_parser("reindex")
    rep = sub.add_parser("report")
    rep.add_argument("--out", default=None)
    rep.add_argument("--html", action="store_true", help="also write the browsable index.html")
    sv = sub.add_parser("serve", help="serve the catalogue, run files, JSON API and dataset downloads over HTTP")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8093)
    sv.add_argument("--refresh", type=float, default=30.0, help="seconds between re-indexing the run folders")
    cu = sub.add_parser("curate", help="grade every run and experiment (A flagship .. F not usable)")
    cu.add_argument("--json", action="store_true", help="print the experiment table as JSON")
    rv = sub.add_parser("review", help="record a human review of an experiment (qualitative curation)")
    rv.add_argument("name")
    rv.add_argument("--reviewer", default=None)
    rv.add_argument("--story", default=None, help="the one-line story / headline")
    rv.add_argument("--novelty", type=int, default=None, help="1-5")
    rv.add_argument("--clarity", type=int, default=None, help="1-5")
    rv.add_argument("--visual-appeal", type=int, default=None, help="1-5")
    rv.add_argument("--approve", default=None, help="comma list of product,paper,marketing,training_data")
    rv.add_argument("--limitations", default=None)
    br = sub.add_parser("brief", help="a report (html / md / pdf) for one item or a filtered set")
    br.add_argument("-o", "--out", required=True, help="report.html, report.md or report.pdf")
    br.add_argument("--item", action="append", help="item key (experiment or experiment/case); repeatable")
    br.add_argument("--experiment", default=None, help="comma list")
    br.add_argument("--tier", default=None, help="comma list, e.g. A,B")
    br.add_argument("--use", default=None, choices=["product", "paper", "marketing", "training_data"],
                    help="order by distance to this use")
    br.add_argument("--ready", default=None, choices=["product", "paper", "marketing", "training_data"],
                    help="only items ready or approved for this use")
    br.add_argument("--tag", default=None, help="comma list")
    br.add_argument("--limit", type=int, default=None)
    br.add_argument("--title", default=None)
    ex = sub.add_parser("examples", help="list the repository's examples and use cases, or run them into the lab")
    ex.add_argument("--match", default="", help="only scripts whose path contains this text")
    ex.add_argument("--run", action="store_true", help="run them (one at a time: scripts write into the checkout)")
    ex.add_argument("--timeout", type=float, default=900)
    ex.add_argument("--force", action="store_true")
    st = sub.add_parser("site", help="write the self-contained public catalogue for a static host (GitHub Pages)")
    st.add_argument("out", nargs="?", default="docs/lab")
    st.add_argument("--out", dest="out_opt", default=None, help="output folder (same as the positional argument)")
    st.add_argument("--no-media", action="store_true", help="experiments page without copied figures and movies")
    cv = sub.add_parser("coverage", help="experiment -> curated yes / no, as Markdown")
    cv.add_argument("-o", "--out", default=None)
    se = sub.add_parser("seal", help="write the quality seal of a curated item (tier from the curation)")
    se.add_argument("item")
    se.add_argument("-o", "--out", default=None, help="seal.svg or seal.png")
    se.add_argument("--explain", action="store_true",
                    help="print what the item lacks for the next letter and the PINNeAPPle actions for each gap")
    se.add_argument("--no-next", action="store_true", help="leave out the next-letter line (A-D only)")
    se.add_argument("--height", type=int, default=96)
    se.add_argument("--no-reason", action="store_true", help="drop the evidence line (E and F always keep theirs)")
    se.add_argument("--lang", default="en", choices=["en", "pt"])
    se.add_argument("--tiers", default=None, help="refuse items outside these tiers, e.g. A,B (default: all, A-F)")
    sp = sub.add_parser("stamp", help="stamp a figure, render, GIF or video with an item's seal")
    sp.add_argument("item", nargs="?", default=None)
    sp.add_argument("file", nargs="?", default=None)
    sp.add_argument("-o", "--out", default=None, help="output file (default name.sealed.ext); with --all a folder")
    sp.add_argument("--all", action="store_true", help="every figure of every item at --tiers into --out")
    sp.add_argument("--tiers", "--tier", dest="tiers", default=None,
                    help="only items at these tiers, e.g. A,B for the validated ones (default: all, A-F)")
    sp.add_argument("--lang", default="en", choices=["en", "pt"])
    sp.add_argument("--corner", default="auto", choices=["auto", "br", "tr", "bl", "tl"])
    sp.add_argument("--size", type=float, default=0.24, help="seal width / image width")
    sp.add_argument("--opacity", type=float, default=0.95)
    sp.add_argument("--no-reason", action="store_true", help="drop the evidence line (E and F always keep theirs)")
    sp.add_argument("--no-next", action="store_true", help="leave out the next-letter line (A-D only)")
    im = sub.add_parser("improve", help="the suggested actions for an item's next letter; --run runs the registered "
                                        "lab commands among them and re-curates")
    im.add_argument("item")
    im.add_argument("--run", action="store_true", help="run them (default: dry run, print only)")
    im.add_argument("--dry-run", action="store_true", help="print only (the default)")
    im.add_argument("--lang", default="en", choices=["en", "pt"])
    rg = sub.add_parser("register", help="write (if missing) and import an experiment manifest, then curate")
    rg.add_argument("path", help="a folder (examples/<name>, a results folder) or an experiment.json")
    rg.add_argument("--no-lab", action="store_true", help="only write / validate the manifest")
    rg.add_argument("--overwrite", action="store_true", help="regenerate an existing experiment.json from the folder")
    sc = sub.add_parser("scaffold", help="create examples/<name> wired with pinneapple_lab.experiment(...)")
    sc.add_argument("name")
    sc.add_argument("--title", default="")
    sc.add_argument("--domain", default="")
    sc.add_argument("--dir", default=None, help="parent folder (default: the repository's examples/)")
    mf = sub.add_parser("manifests", help="list the experiment manifests; --check is the CI gate")
    mf.add_argument("--check", action="store_true", help="exit 1 if a folder lacks a valid manifest")
    mf.add_argument("roots", nargs="*")
    lg = sub.add_parser("legend", help="the legend of the six seals, A to F (PNG)")
    lg.add_argument("-o", "--out", required=True)
    lg.add_argument("--lang", default="en", choices=["en", "pt"])
    sh = sub.add_parser("showcase", help="the animated showcase panel (MP4, GIF, PNG) from the curation")
    sh.add_argument("--lang", default="en", help="en, pt or en,pt")
    sh.add_argument("--out", default="docs/assets/showcase")
    sh.add_argument("--seconds", type=float, default=8.0)
    sh.add_argument("--fps", type=int, default=15)
    e = sub.add_parser("export")
    e.add_argument("name")
    e.add_argument("dataset")
    e.add_argument("out")
    a = ap.parse_args(argv)

    if a.cmd == "seal":
        from .seal import SealError, save_seal, seal_for
        try:
            s = seal_for(a.item, LabStore(a.root), a.tiers)
        except SealError as e:
            print(e, file=sys.stderr)
            return 2
        if a.explain:
            from .curation import next_steps
            from .next_steps import explain
            name = TIER_NAMES_PT if a.lang == "pt" else None
            print(explain(next_steps(a.item, LabStore(a.root)), a.lang,
                          f"{a.item}: {'nível' if a.lang == 'pt' else 'tier'} {s.tier}"
                          + (f" ({name[s.tier]})" if name else "")
                          + (f"\n{'problema' if a.lang == 'pt' else 'problem'}: {s.text(a.lang)[0]}"
                             if s.tier in ("E", "F") else "")))
            if not a.out:
                return 0
        if not a.out:
            ap.error("seal needs -o/--out (or --explain)")
        print(save_seal(s, a.out, height=a.height, reason=not a.no_reason, lang=a.lang, next_step=not a.no_next))
        return 0
    if a.cmd == "register":
        from . import manifest as mfs
        from .capture import register_manifest
        p = a.path
        if os.path.isdir(p) and (a.overwrite or not os.path.exists(os.path.join(p, mfs.FILENAME))):
            from .seal import load_curation
            try:
                cur = load_curation(LabStore(a.root))
            except FileNotFoundError:
                cur = None
            print(mfs.save(mfs.manifest_from_dir(p, cur), p))
        m = mfs.load(p)
        base = p if os.path.isdir(p) else os.path.dirname(p)
        errors = mfs.validate(m, base)
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        if errors:
            return 1
        if not a.no_lab:
            print(register_manifest(p, LabStore(a.root)))
        return 0
    if a.cmd == "scaffold":
        from .manifest import scaffold
        print(scaffold(a.name, a.dir, title=a.title, domain=a.domain))
        return 0
    if a.cmd == "manifests":
        from . import manifest as mfs
        if a.check:
            problems = mfs.check_repo(a.roots[0] if a.roots else None)
            for x in problems:
                print(x)
            print(f"{len(problems)} problem(s)" if problems else "every experiment has a valid manifest")
            return 1 if problems else 0
        for _path, m in mfs.scan(a.roots or None, lab=LabStore(a.root).root):
            print(f"{m.status:12s} {m.domain[:22]:22s} {m.id}")
        return 0
    if a.cmd == "improve":
        from .improve import improve
        return improve(a.item, LabStore(a.root), run=a.run and not a.dry_run, lang=a.lang)
    if a.cmd == "legend":
        from .seal import render_legend
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        render_legend(a.lang).save(a.out)
        print(a.out)
        return 0
    if a.cmd == "stamp":
        from .seal import SealError, seal_for, stamp_all, stamp_file
        kw = {"corner": a.corner, "size": a.size, "opacity": a.opacity, "reason": not a.no_reason, "lang": a.lang,
              "next_step": not a.no_next}
        if a.all:
            man = stamp_all(LabStore(a.root), a.out or "export/sealed", tiers=a.tiers, **kw)
            for k, v in man["items"].items():
                print(f"{v['tier']}  {len(v['files']):3d} files  {k}")
            return 0
        if not (a.item and a.file):
            ap.error("stamp needs ITEM FILE, or --all")
        try:
            s = seal_for(a.item, LabStore(a.root), a.tiers)
        except SealError as e:
            print(e, file=sys.stderr)
            return 2
        print(stamp_file(a.file, s, a.out, **kw))
        return 0
    if a.cmd == "showcase":
        from .showcase import panel
        for lang in a.lang.split(","):
            meta = panel(LabStore(a.root), lang=lang.strip(), out_dir=a.out, seconds=a.seconds, fps=a.fps)
            print(json.dumps({k: meta[k] for k in ("files", "sizes_bytes")}, indent=1))
        return 0
    if a.cmd == "site":
        from .explorer import write_static
        from .html import write_site
        out = a.out_opt or a.out
        print(write_site(LabStore(a.root), out))
        print(write_static(LabStore(a.root), os.path.join(out, "experiments"), media=not a.no_media))
        return 0
    if a.cmd == "coverage":
        from .explorer import collect, coverage_md
        text = coverage_md(collect(LabStore(a.root)))
        if a.out:
            with open(a.out, "w") as f:
                f.write(text)
            print(a.out)
        else:
            print(text)
        return 0
    if a.cmd == "list":
        for n in available():
            cls = get(n)
            print(f"{n:24s} v{cls.version}  {cls.description}")
            print(f"{'':24s} params: {cls.params}")
        return 0
    if a.cmd == "run":
        res = run(a.name, _kv(a.param), root=a.root, force=a.force)
        print(json.dumps({"run_id": res.run_id, "status": res.status, "seconds": res.seconds, "metrics": res.metrics,
                          "dir": res.dir}, indent=1, default=str))
        return 0 if res.ok else 1
    if a.cmd == "sweep":
        grid = {}
        for g in a.grid or []:
            k, _, v = g.partition("=")
            grid[k] = [_val(x) for x in v.split(",")]
        res = sweep(a.name, grid=grid or None, samples=a.samples or None, fixed=_kv(a.param), n_jobs=a.jobs,
                    seed=a.seed, root=a.root, force=a.force)
        for x in res:
            print(f"{x.run_id}  {x.status:24s} {x.seconds:8.1f} s")
        return 0 if all(x.ok for x in res) else 1
    if a.cmd == "brief":
        from .reports import write_report
        split = lambda x: [v.strip() for v in x.split(",")] if x else None  # noqa: E731
        filt = ", ".join(f"{k}={v}" for k, v in (("experiment", a.experiment), ("tier", a.tier), ("use", a.use),
                                                ("ready", a.ready), ("tag", a.tag)) if v)
        print(write_report(LabStore(a.root), a.out, items=a.item, experiment=split(a.experiment), tier=split(a.tier),
                           use=a.use, ready=a.ready, tag=split(a.tag), limit=a.limit, title=a.title, filters=filt))
        return 0
    if a.cmd == "curate":
        from .curation import curate
        out = curate(LabStore(a.root))
        if a.json:
            print(json.dumps(out["experiments"], indent=1, default=str))
        else:
            for exp, e in sorted(out["experiments"].items(), key=lambda kv: kv[1]["tier"]):
                print(f"{e['tier']} {e['tier_name']:12s} {str(e['best_score']):>6s}  {exp}")
        return 0
    if a.cmd == "review":
        from .curation import save_review
        rec = save_review(LabStore(a.root).root, a.name, reviewer=a.reviewer, story=a.story, novelty=a.novelty,
                          clarity=a.clarity, visual_appeal=a.visual_appeal, limitations=a.limitations,
                          approved_for=[x.strip() for x in a.approve.split(",")] if a.approve else None)
        print(json.dumps(rec, indent=1))
        return 0
    if a.cmd == "examples":
        from .experiments.examples import discover_examples
        items = [x for x in discover_examples() if a.match in x["script"]]
        if not a.run:
            for x in items:
                print(f"{x['group']:36s} {x['script']}")
            print(f"{len(items)} scripts")
            return 0
        failed = 0
        for i, x in enumerate(items, 1):
            res = run("example_script", {"script": x["script"], "timeout": a.timeout}, root=a.root, force=a.force)
            failed += not res.ok
            print(f"[{i}/{len(items)}] {res.status:26s} {res.seconds:7.1f} s  {x['script']}", flush=True)
        print(f"{len(items) - failed} of {len(items)} examples ran cleanly")
        return 0 if not failed else 1
    if a.cmd == "serve":
        from .server import serve
        serve(a.root, host=a.host, port=a.port, refresh=a.refresh)
        return 0
    store = LabStore(a.root)
    if a.cmd == "status":
        print(json.dumps(store.status(), indent=1))
    elif a.cmd == "reindex":
        print(store.reindex(), "runs indexed")
    elif a.cmd == "report":
        print(store.catalog(a.out))
        print(store.datasets_catalog())
        from .curation import curate
        curate(store)
        print(os.path.join(store.root, "CURATION.md"))
        if a.html:
            print(store.catalog_html())
    elif a.cmd == "export":
        print(json.dumps(store.export_dataset(a.name, a.dataset, a.out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
