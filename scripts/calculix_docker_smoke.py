"""The CalculiX image gate, run locally: build ``pinneapple_simulation/external_solvers/calculix/Dockerfile`` for each
platform and run a real ``ccx`` case inside each image.

    python scripts/calculix_docker_smoke.py                       # the host's platform (arm64 on this Mac)
    python scripts/calculix_docker_smoke.py --platforms linux/amd64 linux/arm64
    python scripts/calculix_docker_smoke.py --platforms linux/amd64 --json

Exit code 0 when every platform built, started and reproduced the cantilever's tip deflection within 2 % of
Timoshenko's theory with the pinned ``ccx`` version; 1 otherwise, with the container's log for what failed.

This is the same work ``.github/workflows/calculix-image.yml`` does on a runner. It lives in a script as well because the
repository's Actions runners are currently unavailable: the gate has to run somewhere it can be seen, and a test that
skips without Docker is not the gate on its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    from pinneapple_simulation.external_solvers.calculix.runner import (
        PLATFORMS,
        docker_available,
        host_platform,
        image_smoke,
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--platforms", nargs="+", default=None, help=f"Docker platform strings (default: {host_platform()};"
                                                                 f" all: {' '.join(PLATFORMS)})")
    ap.add_argument("--json", action="store_true", help="print the measurements as JSON, one object per platform")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds for one ccx run in an image")
    args = ap.parse_args(argv)
    if not docker_available():
        print("docker is not available: this gate builds an image and runs ccx inside it", file=sys.stderr)
        return 2
    rows, ok = [], True
    for platform in args.platforms or [host_platform()]:
        row = image_smoke(platform, timeout=args.timeout)
        rows.append(row)
        ok = ok and row["ok"]
        if not args.json:
            defl = row["tip_deflection_m"]
            err = row["rel_err"]
            got = f"{defl:.6f} m" if defl is not None else "no result"
            rel = f"{100 * err:+.2f} %" if err is not None else "-"
            print(f"{platform:13s} {row['image']:34s} ccx {row['version'] or '?':5s} tip {got:>12s} "
                  f"vs {row['timoshenko_m']:.6f} m  {rel:>9s}  {'ok' if row['ok'] else 'FAILED'}  "
                  f"({row['seconds']:g} s)")
            if not row["ok"] and row["log"]:
                print(row["log"][-1500:], file=sys.stderr)
    if args.json:
        print(json.dumps(rows, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
