"""Check the built distributions: every wheel package is in both the sdist and the wheel, and imports when the
sdist is installed in a clean virtual environment (roadmap I5, #92).

    python -m build                                   # dist/pinneapple-X.tar.gz and .whl
    python scripts/check_dist.py dist                 # archive contents only (seconds)
    python scripts/check_dist.py dist --install       # + install the sdist in a fresh venv and import every package

``--install`` reuses already-installed dependencies with ``--system-site-packages`` when ``--no-deps`` is given
(CI after ``pip install -e .``), otherwise pip resolves them from PyPI.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# non-Python files the installed library reads at run time (viewers, vendored three.js)
DATA_FILES = [
    "pinneapple_tools/visualization/studio/web/index.html",
    "pinneapple_tools/visualization/studio/web/viewer.js",
    "pinneapple_tools/visualization/studio/web/studio-core.js",
    "pinneapple_tools/visualization/studio/web/vendor/three/three.module.min.js",
    "pinneapple_twin3d/viewer/index.html",
    "pinneapple_twin3d/viewer/viewer.js",
]


def wheel_packages() -> list[str]:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        import tomli as tomllib
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return list(tomllib.load(f)["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])


def _one(dist: str, pattern: str) -> str:
    found = sorted(glob.glob(os.path.join(dist, pattern)))
    if not found:
        sys.exit(f"no {pattern} in {dist}: run `python -m build` first")
    return found[-1]


def check_contents(dist: str, packages: list[str]) -> list[str]:
    sdist, wheel = _one(dist, "*.tar.gz"), _one(dist, "*.whl")
    with tarfile.open(sdist) as t:
        s_names = {"/".join(n.split("/")[1:]) for n in t.getnames()}       # drop the "pinneapple-X/" prefix
    with zipfile.ZipFile(wheel) as z:
        w_names = set(z.namelist())
    problems = []
    for pkg in packages:
        init = f"{pkg}/__init__.py"
        if init not in s_names:
            problems.append(f"sdist {os.path.basename(sdist)} has no {init}")
        if init not in w_names:
            problems.append(f"wheel {os.path.basename(wheel)} has no {init}")
    for f in DATA_FILES:
        if os.path.exists(os.path.join(ROOT, f)) and f not in w_names:
            problems.append(f"wheel {os.path.basename(wheel)} has no {f}")
    return problems


def check_install(dist: str, packages: list[str], no_deps: bool) -> list[str]:
    sdist = _one(dist, "*.tar.gz")
    venv = tempfile.mkdtemp(prefix="pp_sdist_")
    args = [sys.executable, "-m", "venv", venv] + (["--system-site-packages"] if no_deps else [])
    subprocess.run(args, check=True)
    py = os.path.join(venv, "bin", "python")
    pip = [py, "-m", "pip", "install", "-q"] + (["--no-deps"] if no_deps else []) + [sdist]
    subprocess.run(pip, check=True)
    code = (
        "import importlib, json, os, sys\n"
        f"pkgs = {packages!r}\n"
        "bad = {}\n"
        "for p in pkgs:\n"
        "    try:\n"
        "        m = importlib.import_module(p)\n"
        "        if 'site-packages' not in (getattr(m, '__file__', '') or ''):\n"
        "            bad[p] = 'imported from ' + str(getattr(m, '__file__', None)) + ', not the installed sdist'\n"
        "    except Exception as e:\n"
        "        bad[p] = type(e).__name__ + ': ' + str(e)\n"
        "print(json.dumps(bad))\n"
    )
    out = subprocess.run([py, "-c", code], cwd=venv, capture_output=True, text=True)   # cwd: not the repo
    if out.returncode != 0:
        return [f"import run failed: {out.stderr[-2000:]}"]
    import json
    return [f"{p}: {err}" for p, err in json.loads(out.stdout.strip().splitlines()[-1]).items()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dist", nargs="?", default="dist")
    ap.add_argument("--install", action="store_true", help="install the sdist in a fresh venv and import everything")
    ap.add_argument("--no-deps", action="store_true", help="with --install: reuse the current environment's deps")
    a = ap.parse_args(argv)
    packages = wheel_packages()
    problems = check_contents(a.dist, packages)
    if a.install and not problems:
        problems += check_install(a.dist, packages, a.no_deps)
    for p in problems:
        print("FAIL", p)
    print(f"{len(packages)} packages and {len(DATA_FILES)} data files checked{' (installed from the sdist)' if a.install else ''}: "
          f"{'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
