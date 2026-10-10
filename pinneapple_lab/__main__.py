"""``python -m pinneapple_lab ...`` -- run experiments and inspect the lab database.

    python -m pinneapple_lab list
    python -m pinneapple_lab run oscillator -p zeta=0.3 -p method=rk4
    python -m pinneapple_lab sweep cylinder_lbm -g Re=40,80,120,160 -j 4
    python -m pinneapple_lab sweep oscillator -n 50                     # Latin hypercube over the experiment's space
    python -m pinneapple_lab status
    python -m pinneapple_lab report --html                              # lab/CATALOG.md and lab/index.html
    python -m pinneapple_lab export cylinder_lbm vorticity out/cylinder_vorticity.npz
Use ``--root`` (or ``$PINNEAPPLE_LAB``) to choose the database folder.
"""
from __future__ import annotations

import argparse
import json
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
    store = LabStore(a.root)
    if a.cmd == "status":
        print(json.dumps(store.status(), indent=1))
    elif a.cmd == "reindex":
        print(store.reindex(), "runs indexed")
    elif a.cmd == "report":
        print(store.catalog(a.out))
        if a.html:
            print(store.catalog_html())
    elif a.cmd == "export":
        print(json.dumps(store.export_dataset(a.name, a.dataset, a.out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
