"""``python -m pinneapple_simulation.vortex_domains`` -- print what this checkout reproduces.

The cheap checks always run; ``--full`` adds the two cylinder runs (the drag at Re = 40 against Dennis & Chang and
the shedding at Re = 200 against Williamson & Brown), which take minutes.
"""
from __future__ import annotations

import argparse

from .verification import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pinneapple_simulation.vortex_domains")
    parser.add_argument("--full", action="store_true", help="also run the cylinder drag and Strouhal checks")
    parser.add_argument("--panels", type=int, default=96, help="panels on the cylinder")
    parser.add_argument("--t-end", type=float, default=10.0, help="end time of the cylinder run")
    args = parser.parse_args(argv)

    checks = run(quick=not args.full, panels=args.panels, t_end=args.t_end)
    width = max(len(c["name"]) for c in checks)
    for check in checks:
        print(f"{check['name']:<{width}}  {'OK  ' if check['ok'] else 'FAIL'}  {check['detail']}")
    failed = [c["name"] for c in checks if not c["ok"]]
    print()
    print("all checks passed" if not failed else "failed: " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
