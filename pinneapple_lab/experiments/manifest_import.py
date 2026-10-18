"""``manifest``: an experiment manifest (``experiment.json``) brought into the Lab as a run, so the curation grades
it A-F with its next steps like any other item. ``pinneapple_lab.experiment(...)`` and ``python -m pinneapple_lab
register`` use it; the run records the manifest's parameters, its measured metrics, its checks (exactly as recorded)
and copies of its figures and movies. A manifest written after a failure (``provenance.error``) makes the run fail."""
from __future__ import annotations

import os

from ..manifest import load
from ..spec import Experiment, register


@register
class ManifestImport(Experiment):
    name = "manifest"
    version = "1"
    description = "An experiment described by its manifest (experiment.json), imported for curation."
    tags = ["manifest"]
    params = {"id": "", "path": ""}
    case_param = "id"

    def run(self, ctx):
        path = ctx.params["path"]
        m = load(path)
        base = os.path.dirname(path)
        ctx.input("manifest", m.to_dict())
        ctx.input("params", m.params)
        for x in m.metrics:
            ctx.metric(x["name"], x["value"])
        for c in m.checks:
            ctx.check(c["name"], c["passed"], value=c.get("value"), reference=c.get("reference"), rtol=c.get("rtol"),
                      atol=c.get("atol"), max=c.get("max"), min=c.get("min"), detail=c.get("detail", ""),
                      kind=c.get("kind"), label=c.get("label", ""))
        for kind in ("figures", "videos"):
            for f in m.files.get(kind, []):
                src = os.path.join(base, f)
                if os.path.isfile(src):
                    ctx.figure_file(src, f.replace("/", "__"))
        err = (m.provenance or {}).get("error")
        if err:
            raise RuntimeError(f"the experiment failed: {err}")
