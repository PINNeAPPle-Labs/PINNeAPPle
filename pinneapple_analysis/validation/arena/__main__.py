"""``python -m pinneapple_analysis.validation.arena``: run the arena on Darcy 2-D and Burgers 1-D.

Writes ``arena_<problem>.md`` and ``arena_<problem>.json`` into ``--out`` and the regression cases into
``--cases`` (``<problem>.json``; the files shipped with the library live in ``arena/cases/``). Exit 1 when the
acceptance of #707 fails after the last blue round.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .loop import run_arena


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_analysis.validation.arena",
                                 description="Red team / blue team arena for the validation checks.")
    ap.add_argument("--problems", default="darcy_2d,burgers_1d")
    ap.add_argument("--small", action="store_true", help="CI size (and alpha = 0.1 unless --alpha is given)")
    ap.add_argument("--alpha", type=float, help="family-wise false-alarm rate (default 0.01; 0.1 with --small)")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--method", default="de", choices=["de", "cmaes", "lbfgs"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="folder for the reports")
    ap.add_argument("--cases", help="folder for the regression cases (<problem>.json)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    alpha = a.alpha if a.alpha is not None else (0.1 if a.small else 0.01)
    fails = []
    for name in [p.strip() for p in a.problems.split(",") if p.strip()]:
        rep = run_arena(name, budget="small" if a.small else None, alpha=alpha, rounds=a.rounds, method=a.method,
                        seed=a.seed, verbose=not a.quiet)
        if a.out:
            rep.save(Path(a.out) / f"arena_{name}.md")
            rep.save(Path(a.out) / f"arena_{name}.json")
        else:
            print(rep.to_markdown())
        if a.cases:
            rep.save_cases(Path(a.cases) / f"{name}.json")
        fails += [f"{name}: {f}" for f in rep.failures()]
    for f in fails:
        print("FAIL:", f, file=sys.stderr)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
