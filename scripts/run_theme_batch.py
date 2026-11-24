"""Run a batch of examples twice -- as they are on the branch and as they are on ``origin/main`` -- and build the
before/after sheet that reviews a theme-migration batch (#918).

The manual part of every batch so far was this loop: take the example, run it on the branch, take the same file out
of ``origin/main``, run it again, find which image is the figure, and put the two side by side. That is what this
script does, so a batch is one command plus the reading of the sheet:

    python scripts/run_theme_batch.py examples/control/cartpole_swing_up.py examples/geometry_library/02_wind_turbine_bem.py \\
        --sheet docs/assets/house_theme/batch3_antes_depois.png --title "Batch 3"

Each example is copied into two scratch trees — ``<out>/antes/examples/…`` and ``<out>/depois/examples/…`` — so a
script that writes next to itself writes there and not into the repository. The run reproduces the figure twice from
the same data with one difference, the theme, which is what makes the sheet evidence: same curves, same axes, one look.

A script that fails on either side is reported and left out of the sheet; the batch is not silently shorter. A figure
is picked by name, preferring the largest image the run produced, because a migrated example often saves a multi-panel
figure next to a thumbnail.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_theme_migration import build  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")
#: which side of the sheet a tree is
SIDES = ("antes", "depois")


def _rel(path: Path) -> str:
    """The example path relative to the repository, which is also where it goes under the scratch tree."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        if path.parts and path.parts[0] == "examples":
            return path.as_posix()
        raise SystemExit(f"{path} is neither inside the repository nor written as examples/...") from None


def _images(root: Path) -> dict[str, Path]:
    """Image files under a scratch tree, keyed by path relative to the tree."""
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}


def _source(rel: str, ref: str) -> str:
    """The example as it is in ``ref`` (``origin/main``), so the two runs differ only in the migration."""
    try:
        return subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout
    except subprocess.CalledProcessError:
        raise SystemExit(f"{rel} does not exist in {ref}: cannot run the before side") from None


def _run(script: Path, env: dict[str, str], timeout: float) -> tuple[int, str]:
    """Run one example with its backend forced to Agg; returns the exit code and the tail of its output."""
    try:
        done = subprocess.run([sys.executable, str(script)], cwd=script.parent, env=env, timeout=timeout,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"
    tail = (done.stdout + done.stderr).strip().splitlines()[-6:]
    return done.returncode, "\n".join(tail)


def run_one(rel: str, out: Path, *, ref: str, timeout: float) -> tuple[dict[str, Path], list[str]]:
    """Run both versions of one example and return the images each side produced, plus what went wrong."""
    env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}
    sources = {"antes": _source(rel, ref), "depois": (ROOT / rel).read_text(encoding="utf-8")}
    produced: dict[str, Path] = {}
    problems: list[str] = []
    for side in SIDES:
        tree = out / side
        script = tree / rel
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(sources[side], encoding="utf-8")
        before = set(_images(tree))
        code, tail = _run(script, env, timeout)
        if code != 0:
            problems.append(f"{rel} ({side}): exit {code}\n{tail}")
            continue
        new = {k: v for k, v in _images(tree).items() if k not in before}
        # a script that overwrites a committed figure produces nothing new: fall back to everything it wrote
        pool = new or _images(tree)
        if not pool:
            problems.append(f"{rel} ({side}): ran but wrote no image")
            continue
        produced[side] = max(pool.values(), key=lambda p: p.stat().st_size)
    return produced, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_theme_batch", description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path, help="example scripts of the batch (examples/...)")
    ap.add_argument("--ref", default="origin/main", help="the ref the before side comes from (default origin/main)")
    ap.add_argument("--out", type=Path, default=Path("/tmp/pinneapple-theme-batch"), help="scratch trees")
    ap.add_argument("--sheet", type=Path, help="where to write the before/after sheet")
    ap.add_argument("--title", default="House theme, before and after")
    ap.add_argument("--note", default="")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per run (default 900)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch trees of a previous run")
    a = ap.parse_args(argv)

    if not a.keep and a.out.exists():
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True, exist_ok=True)

    pairs, problems = [], []
    for path in a.paths:
        rel = _rel(path)
        produced, failed = run_one(rel, a.out, ref=a.ref, timeout=a.timeout)
        problems.extend(failed)
        if len(produced) == len(SIDES):
            pairs.append((rel.removeprefix("examples/"), produced["antes"], produced["depois"]))
            print(f"ok      {rel}")
        else:
            print(f"skipped {rel}")

    for problem in problems:
        print(f"\n{problem}", file=sys.stderr)

    if a.sheet and pairs:
        print(build(pairs, a.sheet, title=a.title, note=a.note or f"{len(pairs)} examples, same data, one theme"))
    print(f"\n{len(pairs)} of {len(a.paths)} examples on the sheet, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
