"""Execute the Colab notebooks headless, top to bottom, and check that the committed copies carry no outputs.

    python scripts/run_colab_notebooks.py --smoke            # every notebook with tiny sizes (what CI runs)
    python scripts/run_colab_notebooks.py                    # full sizes (CPU: minutes per notebook)
    python scripts/run_colab_notebooks.py --smoke 01 05      # only the notebooks whose name contains 01 or 05
    python scripts/run_colab_notebooks.py --check-clean      # fail if a committed notebook has outputs
    python scripts/run_colab_notebooks.py --clear            # strip outputs and execution counts in place

Each notebook runs in its own scratch directory with ``PINNEAPPLE_REPO`` pointing at this checkout (so the files the
notebooks read come from the checkout, not from GitHub) and ``PINNEAPPLE_NB_SMOKE=1`` with ``--smoke``. Executed
copies go to ``--out`` (default: a temporary directory), never over the committed notebooks.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks", "colab")


def notebooks(filters):
    paths = sorted(glob.glob(os.path.join(NB_DIR, "*.ipynb")))
    return [p for p in paths if not filters or any(f in os.path.basename(p) for f in filters)]


def keeps_outputs(nb) -> bool:
    """A notebook may opt in to carrying its outputs, with ``metadata.pinneapple.keep_outputs``.

    The default is no outputs: a committed output makes every other open pull request
    conflict on the next merge, so ``--check-clean`` fails on one. A notebook that *is* the
    deliverable -- the quickstart tour, where the printed numbers are the point -- sets the
    flag and is executed before every commit instead of being cleared.
    """
    return bool((nb.get("metadata") or {}).get("pinneapple", {}).get("keep_outputs"))


def has_outputs(nb) -> bool:
    return any(c.get("outputs") or c.get("execution_count") for c in nb.cells if c.cell_type == "code")


def clear(nb):
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
    return nb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filters", nargs="*", help="substrings of notebook names to run")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes (PINNEAPPLE_NB_SMOKE=1)")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per cell")
    ap.add_argument("--out", help="directory for the executed copies")
    ap.add_argument("--check-clean", action="store_true", help="only check that committed notebooks have no outputs")
    ap.add_argument("--clear", action="store_true", help="strip outputs in place")
    a = ap.parse_args(argv)

    import nbformat

    paths = notebooks(a.filters)
    if not paths:
        print("no notebooks matched", file=sys.stderr)
        return 1
    if a.check_clean or a.clear:
        dirty = []
        kept = []
        for p in paths:
            nb = nbformat.read(p, as_version=4)
            if not has_outputs(nb):
                continue
            if keeps_outputs(nb):
                kept.append(os.path.relpath(p, ROOT))
                continue
            dirty.append(os.path.relpath(p, ROOT))
            if a.clear:
                nbformat.write(clear(nb), p)
        if dirty and not a.clear:
            print("notebooks with outputs (run with --clear):\n  " + "\n  ".join(dirty), file=sys.stderr)
            return 1
        if kept:
            print("outputs kept by request (metadata.pinneapple.keep_outputs): " + ", ".join(kept))
        print(("cleared: " + ", ".join(dirty)) if dirty and a.clear else "all notebooks are clean")
        return 0

    from nbclient import NotebookClient

    out = a.out or tempfile.mkdtemp(prefix="colab_nb_")
    os.makedirs(out, exist_ok=True)
    os.environ["PINNEAPPLE_REPO"] = ROOT
    os.environ["PINNEAPPLE_NB_SMOKE"] = "1" if a.smoke else "0"
    os.environ.setdefault("MPLBACKEND", "Agg")
    failed = []
    for p in paths:
        name = os.path.basename(p)
        work = os.path.join(out, name[:-6])
        os.makedirs(work, exist_ok=True)
        nb = nbformat.read(p, as_version=4)
        t0 = time.time()
        try:
            NotebookClient(nb, timeout=a.timeout, kernel_name="python3", resources={"metadata": {"path": work}}).execute()
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - report every notebook, then fail
            status = f"FAILED: {type(exc).__name__}: {str(exc)[-2000:]}"
            failed.append(name)
        nbformat.write(nb, os.path.join(out, name))
        print(f"{name}: {status} ({time.time() - t0:.0f} s)", flush=True)
    print(f"executed copies in {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
