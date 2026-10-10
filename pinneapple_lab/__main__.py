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
    python -m pinneapple_lab curate                                          # A flagship .. D not usable, per run
    python -m pinneapple_lab review kepler_law --novelty 4 --story "..." --approve paper,marketing
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
    cu = sub.add_parser("curate", help="grade every run and experiment (A flagship .. D not usable)")
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
    ex = sub.add_parser("examples", help="list the repository's examples and use cases, or run them into the lab")
    ex.add_argument("--match", default="", help="only scripts whose path contains this text")
    ex.add_argument("--run", action="store_true", help="run them (one at a time: scripts write into the checkout)")
    ex.add_argument("--timeout", type=float, default=900)
    ex.add_argument("--force", action="store_true")
    e = sub.add_parser("export")
    e.add_argument("name")
    e.add_argument("dataset")
    e.add_argument("out")
    a = ap.parse_args(argv)

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
