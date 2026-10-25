"""Physics query node: ask quantities of a chip package at physical references, each answered by the cheapest
valid method with its uncertainty and provenance (library: :mod:`pinneapple_query`, package methods in
:mod:`pinneapple_condense.query`)."""
from __future__ import annotations

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node

_TAGS = ("section:solvers", "thermal", "semiconductor packaging", "query", "provenance", "uncertainty")


@node("query.package", "Physics queries on a chip package", category="Solvers",
      outputs=[Port("metrics", "metrics")],
      params=[Param("blocks", "int", 32, minimum=2, maximum=256, help="blocks per side (one microbump + TSV each)"),
              Param("queries", "list", ["Tj_max region:die", "T die@0.5,0.5", "q interface:die/tim",
                                        "q ubump[3,4]", "T die@0.25,0.75,0.5"],
                    help="'<quantity> <where>' (T, Tj_max, q, R_ja; die@u,v[,w], region:die[...], ubump[I,J], "
                         "interface:die/tim, lid:top)"),
              Param("accuracy", "float", 0.1, help="absolute target (K for temperatures; 0: most accurate)"),
              Param("rigorous", "bool", False, help="only guaranteed intervals"),
              Param("seed", "int", 0)],
      calls="pinneapple_condense.query.package_engine / pinneapple_query.Engine.query", tags=_TAGS)
def package_queries(ctx, inputs, params):
    """Build the condensed package once (a calibration solve of its own power map), then answer each query by the
    cheapest method that meets the target -- a read of a solved state, a rigorous monotone bound, superposition of
    solved states, a new condensed solve -- and say which, with the uncertainty and whether it was read or
    predicted."""
    from pinneapple_condense.package import uniform_package
    from pinneapple_condense.query import package_engine
    nb = int(params["blocks"])
    eng, model = package_engine(uniform_package((nb, nb), seed=int(params["seed"])), reference_cells=0)
    rows = []
    acc = float(params["accuracy"]) or None
    for q in params["queries"] or []:
        try:
            qty, where = str(q).split(None, 1)
            a = eng.query(qty, where.strip(), accuracy=acc, rigorous=bool(params["rigorous"]))
        except (ValueError, LookupError) as e:
            raise StepFailed(f"{q!r}: {e}") from e
        rows.append([q, a.value, a.unit, a.uncertainty, a.kind, a.method, a.seconds])
    ctx.table("Answers", ["query", "value", "unit", "uncertainty", "kind", "method", "seconds"], rows)
    ctx.summary(f"{len(rows)} queries on {nb * nb} blocks; build {model.timings.get('calibrate', 0):.1f} s")
    return {"metrics": {"answers": [dict(zip(["query", "value", "unit", "uncertainty", "kind", "method", "seconds"],
                                             r, strict=False)) for r in rows], "build_s": model.timings.get("calibrate")}}
