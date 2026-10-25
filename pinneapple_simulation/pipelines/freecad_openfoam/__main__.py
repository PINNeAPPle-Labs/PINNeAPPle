"""``python -m pinneapple_simulation.pipelines.freecad_openfoam``:

* ``run <pipeline.json> <folder> [--inputs JSON]``: run (or resume) one case; the exit code is 0 when the run passed
  or was rejected by its criteria (it finished), 1 otherwise.
* ``sweep <folder> [--parallel N]``: run (or resume) the pending cases of a sweep saved by ``Sweep.save``.
* ``build <template> <output.FCStd> [--backend freecad|fake|auto]``: write a template's parametric document.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_simulation.pipelines.freecad_openfoam")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("spec")
    r.add_argument("folder")
    r.add_argument("--inputs", default="{}")
    s = sub.add_parser("sweep")
    s.add_argument("folder")
    s.add_argument("--parallel", type=int, default=1)
    b = sub.add_parser("build")
    b.add_argument("template")
    b.add_argument("output")
    b.add_argument("--backend", default="auto")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        from .pipeline import PipelineSpec, run_pipeline
        rec = run_pipeline(PipelineSpec.load(a.spec), a.folder, json.loads(a.inputs) or None)
        print(rec.state, rec.message)
        return 0 if rec.state in ("passed", "rejected") else 1
    if a.cmd == "sweep":
        from .sweep import Sweep
        rows = Sweep.load(a.folder).run(parallel=a.parallel)
        return 0 if all(r["state"] in ("passed", "rejected") for r in rows) else 1
    from .models import build_document
    print(json.dumps(build_document(a.template, a.output, backend=a.backend), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
