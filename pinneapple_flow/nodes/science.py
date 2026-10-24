"""Science nodes: run an experiment of :mod:`pinneapple_science` (area, level, kind, dimension) and draw the atlas
of the areas it covers."""
from __future__ import annotations

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node

#: kept literal so registering the nodes imports nothing heavy; tests/test_science.py checks it equals
#: pinneapple_science.names()
EXPERIMENTS = ["em.ampere_law", "em.bench_mu0_solenoid", "em.bench_rc_discharge", "em.biot_savart",
               "em.cavity_resonance", "em.circuits_rc_rl_rlc", "em.coulomb_superposition", "em.dipole_radiation",
               "em.faraday_induction", "em.fresnel_snell", "em.gauss_law", "em.hall_effect", "em.lossy_plane_wave",
               "em.mie_scattering", "em.parallel_plate_capacitor", "em.research_fdtd_3d",
               "em.research_inverse_permittivity", "em.research_surrogate_hall", "em.research_uq_waveguide",
               "em.waveguide_modes"]
DIMS = ["first", "0D", "1D", "2D", "3D", "3D+t"]
_TAGS = ("section:validation", "science", "electromagnetism", "academic", "laboratory")


@node("science.run", "Science experiment", category="Validate",
      outputs=[Port("metrics", "metrics"), Port("report", "report")],
      params=[Param("experiment", "enum", "em.faraday_induction", choices=EXPERIMENTS,
                    help="the experiment (pp.science.experiments() lists area, level, kind and dimensions)"),
              Param("dim", "enum", "first", choices=DIMS, help="the dimension to run in (first = its first)"),
              Param("quick", "bool", True, help="reduced sizes (seconds); off for the committed resolution")],
      calls="pinneapple_science.registry.run(name, dim) over the pinneapple_science.em experiments "
            "(pinneapple_science.core.Result, pinneapple_science.instruments, pinneapple_science.numerics, "
            "pinneapple_science.plots)", tags=_TAGS)
def run_experiment(ctx, inputs, params):
    """Run a science experiment: every computed quantity against its cited reference, physics checks, trust report."""
    import pinneapple_science as S
    e = S.get(params["experiment"])
    dim = None if params["dim"] == "first" else params["dim"]
    if dim is not None and dim not in e.dims:
        raise StepFailed(f"{e.name} runs in {', '.join(e.dims)}, not {dim}")
    r = e.run(dim, quick=bool(params["quick"]))
    for c in r.comparisons:
        ctx.metric(f"rel_error[{c.quantity}]", c.rel_error, f"vs {c.kind.replace('_', ' ')}: {c.source}")
    ctx.table("Comparisons", ["quantity", "computed", "reference", "unit", "rel. error", "tolerance", "source"],
              [[c.quantity, c.computed, c.reference, c.unit, c.rel_error, c.tol, c.source] for c in r.comparisons])
    ctx.table("Trust report", ["check", "status", "value", "limit"],
              [[c.name, c.status, c.value, c.limit] for c in r.trust.checks])
    ctx.summary(f"{r.title} [{r.dim}, {r.level}, {r.kind}]: {'PASS' if r.passed else 'FAIL'}, trust {r.trust.decisao}")
    paths = r.save(str(ctx.path(params["experiment"].replace(".", "_"))))
    return {"metrics": r.to_dict(), "report": {"path": paths["result"], "kind": "json"}}


@node("science.atlas", "Science atlas", category="Analysis", outputs=[Port("report", "report")],
      params=[Param("status", "enum", "any", choices=["any", "available", "partial", "planned"],
                    help="which subareas to list in the table")],
      calls="pinneapple_science.atlas.load().to_html(path)", tags=("section:validation", "science", "atlas"))
def atlas(ctx, inputs, params):
    """The map of the science areas: available (pp.science experiments), partial (library code), planned."""
    import pinneapple_science as S
    a = S.atlas()
    rows = [r for r in a.rows() if params["status"] in ("any", r["status"])]
    ctx.table("Atlas", ["area", "subarea", "status", "covered by"],
              [[r["area_title"], r["title"], r["status"], len(r["covered_by"])] for r in rows])
    for k, v in a.counts().items():
        ctx.metric(f"subareas_{k}", v)
    path = a.to_html(str(ctx.path("atlas.html")))
    ctx.summary(f"{len(a.areas)} areas; " + ", ".join(f"{v} {k}" for k, v in a.counts().items()) + " subareas")
    return {"report": {"path": str(ctx.path("atlas.html")), "kind": "html", "bytes": len(path)}}
