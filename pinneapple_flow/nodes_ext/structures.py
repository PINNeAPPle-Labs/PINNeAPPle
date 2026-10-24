"""Structures nodes: CalculiX (``examples/calculix_cantilever``) through
:mod:`pinneapple_simulation.external_solvers.calculix`: a deck (the example's committed one or a cantilever written
by the library), ``ccx`` (native or its Docker image), and the results of an ``.frd`` file (the run's or the
example's committed one) with the tip deflection, the peak bending stress and the Euler-Bernoulli value."""
from __future__ import annotations

import shutil

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact
from ..types import register_port_type
from .examples import example_dir

register_port_type("ccx_deck", "a CalculiX job folder (<job>.inp, and after a run <job>.frd / .dat / .sta)")


def _ccx_available():
    try:
        from pinneapple_simulation.external_solvers.calculix import ccx_backend
    except Exception as exc:  # noqa: BLE001
        return False, f"needs pinneapple_simulation.external_solvers.calculix ({exc})"
    return (True, "") if ccx_backend() else (False, "needs CalculiX: ccx on the PATH (apt install calculix-ccx) "
                                                     "or Docker for its image")


@node("calculix.deck", "CalculiX deck", category="Structures",
      outputs=[Port("deck", "ccx_deck"), Port("metrics", "metrics")],
      params=[Param("source", "enum", "example", choices=["example", "cantilever"],
                    help="example: the committed deck of examples/calculix_cantilever (beam.inp); cantilever: "
                         "written by the library (cantilever_hex_mesh + write_inp)"),
              Param("length", "float", 1.0), Param("b", "float", 0.1), Param("h", "float", 0.1),
              Param("nx", "int", 20, minimum=1, maximum=400),
              Param("element_type", "enum", "C3D20R", choices=["C3D20R", "C3D8", "C3D8R"])],
      calls="pinneapple_simulation.external_solvers.calculix.cantilever_hex_mesh + write_inp / read_inp")
def deck(ctx, inputs, params):
    """The input deck of the cantilever (mesh, material, fixed root, tip load)."""
    from pinneapple_simulation.external_solvers import calculix as cx
    d = ctx.path("job")
    d.mkdir(parents=True, exist_ok=True)
    if params["source"] == "example":
        shutil.copy(example_dir("calculix_cantilever") / "beam.inp", d / "beam.inp")
    else:
        model = cx.cantilever_hex_mesh(L=float(params["length"]), b=float(params["b"]), h=float(params["h"]),
                                       nx=int(params["nx"]), element_type=params["element_type"])
        cx.write_inp(model, str(d / "beam.inp"))
    m = cx.read_inp(str(d / "beam.inp"))
    ctx.metric("nodes", int(len(m.nodes)))
    ctx.metric("elements", int(sum(len(v) for v in m.elements.values()) if isinstance(m.elements, dict)
                               else len(m.elements)))
    a = artifact(d, "dir")
    a["job"] = "beam"
    return {"deck": a, "metrics": ctx.report["metrics"]}


@node("calculix.solve", "CalculiX solve (ccx)", category="Structures",
      inputs=[Port("deck", "ccx_deck")], outputs=[Port("deck", "ccx_deck"), Port("metrics", "metrics")],
      params=[Param("threads", "int", 1, minimum=1, maximum=64)],
      calls="pinneapple_simulation.external_solvers.calculix.run_ccx (native ccx or the Docker image)",
      available=_ccx_available)
def solve(ctx, inputs, params):
    from pinneapple_simulation.external_solvers.calculix import ccx_backend, run_ccx
    src = inputs["deck"]
    d = ctx.path("job")
    shutil.copytree(src["$artifact"], d, dirs_exist_ok=True)
    job = src.get("job", "beam")
    r = run_ccx(str(d), job, threads=int(params["threads"]))
    (d / "ccx.log").write_text(r.stdout)
    if r.returncode != 0:
        raise StepFailed(f"ccx failed ({r.returncode}): {r.stdout[-800:]}")
    ctx.metric("backend", ccx_backend())
    a = artifact(d, "dir")
    a["job"] = job
    return {"deck": a, "metrics": ctx.report["metrics"]}


@node("calculix.results", "CalculiX results (.frd)", category="Structures",
      inputs=[Port("deck", "ccx_deck", required=False, help="a solved job; else the example's committed .frd")],
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("example_frd", "str", "calculix_cantilever/beam.frd",
                    help="<example>/<file>.frd when no job is connected (committed results)"),
              Param("load_N", "float", 1000.0, help="total tip load (for the beam-theory value)"),
              Param("E", "float", 2.1e11), Param("length", "float", 1.0), Param("b", "float", 0.1),
              Param("h", "float", 0.1)],
      calls="pinneapple_simulation.external_solvers.calculix.read_frd")
def results(ctx, inputs, params):
    """Tip deflection (the most negative u_z at the free end), the largest |sigma_xx|, and the Euler-Bernoulli tip
    deflection P L^3 / (3 E I) of the same beam."""
    from pinneapple_simulation.external_solvers.calculix import read_frd
    if inputs.get("deck"):
        path = f"{inputs['deck']['$artifact']}/{inputs['deck'].get('job', 'beam')}.frd"
    else:
        ex, _, f = params["example_frd"].partition("/")
        p = (example_dir(ex) / f).resolve()
        if example_dir(ex) not in p.parents or not p.is_file():
            raise StepFailed(f"{params['example_frd']!r} is not a file of the example")
        path = str(p)
    fr = read_frd(path)
    U = fr.field("DISP")
    S = fr.field("STRESS")
    x = fr.coords[:, 0]
    tip = np.isclose(x, x.max())
    k = int(np.argmin(np.where(tip, U[:, 2], np.inf)))
    ctx.metric("nodes", int(len(fr.node_ids)))
    ctx.metric("tip_uz_m", float(U[k, 2]), f"most negative u_z at the free end (node {int(fr.node_ids[k])})")
    ctx.metric("tip_uz_mean_m", float(U[tip, 2].mean()), "mean u_z over the free-end nodes")
    ctx.metric("max_abs_sxx_Pa", float(np.nanmax(np.abs(S[:, 0]))))
    I = float(params["b"]) * float(params["h"]) ** 3 / 12      # noqa: E741
    eb = -float(params["load_N"]) * float(params["length"]) ** 3 / (3 * float(params["E"]) * I)
    ctx.metric("euler_bernoulli_tip_uz_m", eb, "P L^3 / (3 E I), no shear deformation")
    ctx.metric("tip_uz_over_euler_bernoulli", float(U[k, 2] / eb))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    sel = np.isclose(fr.coords[:, 1], np.median(fr.coords[:, 1])) & np.isclose(fr.coords[:, 2], np.median(fr.coords[:, 2]))
    o = np.argsort(x[sel])
    ax.plot(x[sel][o], U[sel, 2][o] * 1e3, "o-", ms=3, label="CalculiX (centre line)")
    xs = np.linspace(0, float(params["length"]), 50)
    L = float(params["length"])
    ax.plot(xs, eb * 1e3 * xs ** 2 * (3 * L - xs) / (2 * L ** 3), "k--", label="Euler-Bernoulli")
    ax.set(xlabel="x [m]", ylabel="u_z [mm]", title="cantilever deflection")
    ax.legend()
    f = ctx.figure(fig, "deflection", "deflection along the beam against beam theory")
    ctx.summary(f"tip {U[k, 2] * 1e3:.4f} mm (beam theory {eb * 1e3:.4f} mm), max |sxx| {np.nanmax(np.abs(S[:, 0])):.4g} Pa")
    return {"metrics": ctx.report["metrics"], "figure": f}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="calculix.cantilever", title="Cantilever beam with CalculiX against beam theory", domain="structures",
        example="calculix_cantilever",
        keywords={"calculix": 3, "ccx": 3, "cantilever": 2, "viga em balanço": 3, "beam": 1, "tip deflection": 2,
                  "bending stress": 1.5, "finite element": 1}, min_score=3,
        nodes=("calculix.deck", "calculix.solve", "calculix.results", "examples.compare"),
        inputs=("the beam dimensions, material and tip load (the example's deck by default)",),
        references=("brenner2008fem",), template="calculix_cantilever/beam",
        description="CalculiX on the committed deck, results against the committed .frd and Euler-Bernoulli"))


_register_capabilities()
