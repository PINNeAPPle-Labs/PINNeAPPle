"""Non-Hermitian physics nodes (``examples/non_hermitian/ep_tuning.py``): the no-go theorem for exceptional-point
tuning in PT-symmetric dimers (Zenodo 23287622) checked symbolically
(:func:`pinneapple_physics.non_hermitian.analyze_ep_tuning` on :func:`~pinneapple_physics.non_hermitian.coupling_classes`)
and numerically (:class:`~pinneapple_physics.non_hermitian.PTDimer`), the hypothesis map as a physics audit
(:func:`pinneapple_physics.physics_audit.audit_assumptions` against
:func:`~pinneapple_physics.non_hermitian.theorem_profile`), and the figure
(:func:`pinneapple_physics.non_hermitian.viz.ep_tuning_figure`)."""
from __future__ import annotations

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact

_KEYS = ["reciprocal_linear", "phase_only", "route_a", "route_c"]


@node("nonhermitian.ep_laws", "EP tuning laws checked against the no-go theorem (symbolic)",
      category="Non-Hermitian physics", outputs=[Port("metrics", "metrics")],
      params=[Param("couplings", "list", list(_KEYS),
                    help="coupling classes of the paper: " + ", ".join(_KEYS))],
      calls="pinneapple_physics.non_hermitian.coupling_classes + analyze_ep_tuning (sympy)")
def ep_laws(ctx, inputs, params):
    """Per coupling class: the EP law gamma_EP^2(lam), whether it equals the linear target J^2 + lam J G exactly, its
    degree in lam, the four assumptions (A PT symmetry, B dimension 2, C polynomial coupling, D reciprocity) and
    whether the theorem applies."""
    from pinneapple_physics.non_hermitian import analyze_ep_tuning, coupling_classes
    classes = {k: (t, a, b) for k, t, a, b in coupling_classes()}
    unknown = [k for k in params["couplings"] if k not in classes]
    if unknown:
        raise StepFailed(f"unknown coupling classes {unknown} (known: {list(classes)})")
    rows = []
    for k in params["couplings"]:
        title, b_ab, b_ba = classes[k]
        rep = analyze_ep_tuning(b_ab, b_ba)
        ctx.log(f"== {title}\n{rep}")
        ctx.metric(f"{k}.exact", int(rep.exact), "1: gamma_EP^2(lam) equals J^2 + lam J G exactly")
        if rep.degree is not None:
            ctx.metric(f"{k}.degree", int(rep.degree), "degree of gamma_EP^2 in lam")
        ctx.metric(f"{k}.theorem_applies", int(rep.theorem_applies))
        for a, ok in rep.assumptions.items():
            ctx.metric(f"{k}.assumption_{a}", int(ok))
        rows.append([title, str(rep.ep_law), rep.perturbation_class, "YES" if rep.exact else "NO",
                     rep.degree if rep.degree is not None else "-",
                     "".join(a for a, ok in rep.assumptions.items() if not ok) or "none",
                     "applies" if rep.theorem_applies else "does not apply"])
    ctx.table("EP laws by coupling class", ["coupling", "gamma_EP^2(lam)", "class", "exact linear", "degree",
                                            "failed assumptions", "no-go theorem"], rows)
    n_exact = sum(int(ctx.report["metrics"][f"{k}.exact"]["value"]) for k in params["couplings"])
    ctx.summary(f"{n_exact} of {len(params['couplings'])} coupling classes tune the EP linearly")
    return {"metrics": ctx.report["metrics"]}


@node("nonhermitian.ep_numeric", "EP location from the matrix: linear coupling, routes A, B and C",
      category="Non-Hermitian physics", outputs=[Port("metrics", "metrics")],
      params=[Param("J", "float", 1.0, minimum=1e-9), Param("G", "float", 0.5),
              Param("linear_c", "float", 0.25, help="reciprocal linear coupling b = J + c lam"),
              Param("lams", "list", [-1.0, 0.0, 1.0], help="control values"),
              Param("feedback_lam", "float", 0.8, help="lam of the route-B feedback fixed point")],
      calls="pinneapple_physics.non_hermitian.PTDimer (ep_gamma_numeric, ep_gamma) + route_a_sqrt_coupling, "
            "route_c_asymmetric, feedback_coupling")
def ep_numeric(ctx, inputs, params):
    """gamma_EP^2 found from the Hamiltonian alone (route C: closed form, the matrix is not PT symmetric) against the
    target J^2 + lam J G, and the route-B feedback fixed point."""
    from pinneapple_physics.non_hermitian import (
        PTDimer,
        feedback_coupling,
        route_a_sqrt_coupling,
        route_c_asymmetric,
    )
    J, G, c = float(params["J"]), float(params["G"]), float(params["linear_c"])
    d_a = PTDimer(route_a_sqrt_coupling(J, G))
    d_c = PTDimer(*route_c_asymmetric(J, G))
    d_l = PTDimer(lambda x: J + c * x)
    rows, err_a, err_c = [], 0.0, 0.0
    for x in map(float, params["lams"]):
        tgt = J ** 2 + x * J * G
        lin, ra, rc = d_l.ep_gamma_numeric(x) ** 2, d_a.ep_gamma_numeric(x) ** 2, d_c.ep_gamma(x) ** 2
        ctx.metric(f"lam{x:+g}.target", tgt)
        ctx.metric(f"lam{x:+g}.linear", lin)
        ctx.metric(f"lam{x:+g}.route_a", ra)
        ctx.metric(f"lam{x:+g}.route_c", rc)
        err_a, err_c = max(err_a, abs(ra - tgt)), max(err_c, abs(rc - tgt))
        rows.append([x, round(tgt, 4), round(lin, 4), round(ra, 4), round(rc, 4)])
    ctx.metric("route_a.max_abs_error", err_a, "max |gamma_EP^2 - target| over lams")
    ctx.metric("route_c.max_abs_error", err_c, "max |gamma_EP^2 - target| over lams")
    b, it = feedback_coupling(J, G, float(params["feedback_lam"]))
    ctx.metric("route_b.fixed_point", b, f"feedback coupling at lam = {params['feedback_lam']}")
    ctx.metric("route_b.iterations", int(it))
    ctx.metric("route_b.fixed_point_error", abs(b ** 2 - (J ** 2 + float(params["feedback_lam"]) * J * G)),
               "|b*^2 - (J^2 + lam J G)|")
    ctx.table(f"gamma_EP^2 (J = {J:g}, G = {G:g})", ["lam", "target", "linear coupling", "route A", "route C"], rows)
    ctx.summary(f"routes A and C on target within {max(err_a, err_c):.1e}; route B fixed point {b:.6f} in {it} steps")
    return {"metrics": ctx.report["metrics"]}


@node("nonhermitian.hypothesis_map", "Does the no-go theorem constrain a design? (physics audit)",
      category="Non-Hermitian physics", outputs=[Port("metrics", "metrics")],
      params=[Param("dimer_b", "list", [1.0, 0.3], help="complex coupling of the PT dimer, [re, im]"),
              Param("gamma", "float", 0.5, help="gain/loss of the operating point"),
              Param("J", "float", 1.0), Param("G", "float", 0.5),
              Param("route_c_lam", "float", 1.0, help="control value of the non-reciprocal design"),
              Param("chain_n", "int", 3, minimum=2, maximum=20, help="sites of the PT chain"),
              Param("chain_b", "float", 1.0)],
      calls="pinneapple_physics.physics_audit.audit_assumptions + non_hermitian.theorem_profile, design_features, "
            "PTDimer, PTChain, route_c_asymmetric")
def hypothesis_map(ctx, inputs, params):
    """Three designs audited against the theorem's assumptions: all VALID means linear tuning is out of reach; a
    VIOLATED assumption names the route that opened up."""
    from pinneapple_physics.non_hermitian import (
        PTChain,
        PTDimer,
        design_features,
        route_c_asymmetric,
        theorem_profile,
    )
    from pinneapple_physics.physics_audit import audit_assumptions
    re_b, im_b = map(float, params["dimer_b"])
    g = float(params["gamma"])
    designs = [("pt_dimer", "PT dimer, polynomial control", PTDimer(complex(re_b, im_b)).hamiltonian(g)),
               ("route_c", "route C, non-reciprocal",
                PTDimer(*route_c_asymmetric(float(params["J"]), float(params["G"]))).hamiltonian(
                    g, float(params["route_c_lam"]))),
               ("chain", f"PT chain, N = {int(params['chain_n'])}",
                PTChain(int(params["chain_n"]), float(params["chain_b"])).hamiltonian(g))]
    rows = []
    for key, name, H in designs:
        a = audit_assumptions(theorem_profile(), design_features(H, True))
        ctx.metric(f"{key}.constrained", int(a.status == "VALID"), "1: every assumption VALID (theorem applies)")
        ctx.metric(f"{key}.n_violated", len(a.violated))
        rows.append([name, a.status, ", ".join(c.name for c in a.violated) or "none"])
    ctx.table("hypothesis map", ["design", "audit", "violated assumptions"], rows)
    return {"metrics": ctx.report["metrics"]}


@node("nonhermitian.ep_figure", "EP tuning figure and PT-chain thresholds", category="Non-Hermitian physics",
      outputs=[Port("metrics", "metrics"), Port("figure", "figure")],
      params=[Param("J", "float", 1.0, minimum=1e-9), Param("G", "float", 0.5),
              Param("chain_sizes", "list", [2, 3, 4, 5], help="uniform chains whose threshold kappa_N^2 is reported")],
      calls="pinneapple_physics.non_hermitian.viz.ep_tuning_figure + PTChain (ep_gamma_numeric, ep_order)")
def ep_figure(ctx, inputs, params):
    """The four-panel figure (spectrum across the EP, EP laws by coupling class, coupling plane, chains) and, for
    uniform chains with |b| = 1, the PT-breaking threshold kappa_N^2 = gamma_EP^2 and the order of the EP there."""
    from pinneapple_physics.non_hermitian import PTChain
    from pinneapple_physics.non_hermitian.viz import ep_tuning_figure
    rows = []
    for n in map(int, params["chain_sizes"]):
        ch = PTChain(n, 1.0)
        g = ch.ep_gamma_numeric()
        ctx.metric(f"chain{n}.kappa2", g ** 2, "gamma_EP^2 of the uniform chain with |b| = 1 (bisection)")
        ctx.metric(f"chain{n}.ep_order", int(ch.ep_order(g)))
        rows.append([n, round(g ** 2, 6), int(ch.ep_order(g))])
    ctx.table("uniform PT chains", ["N", "kappa_N^2", "EP order"], rows)
    p = ctx.path("ep_tuning.png")
    ep_tuning_figure(str(p), float(params["J"]), float(params["G"]))
    ctx.report["figures"].append({"path": str(p), "name": "ep_tuning",
                                  "caption": "spectrum, EP laws, coupling plane, chain thresholds"})
    return {"metrics": ctx.report["metrics"], "figure": artifact(p, "png")}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="nonhermitian.ep_tuning", title="Exceptional-point tuning in PT-symmetric dimers (no-go theorem)",
        domain="non-Hermitian physics", example="non_hermitian",
        keywords={"exceptional point": 3, "exceptional points": 3, "pt-symmetric": 3, "pt symmetric": 3,
                  "pt symmetry": 2.5, "non-hermitian": 3, "non hermitian": 3, "dimer": 1.5, "gain and loss": 2,
                  "no-go theorem": 2},
        min_score=3, nodes=("nonhermitian.ep_laws", "nonhermitian.ep_numeric", "nonhermitian.hypothesis_map",
                            "nonhermitian.ep_figure", "examples.compare"),
        uncited="the theorem the example checks (Zenodo record 23287622) is not in references.yaml",
        template="non_hermitian/ep_tuning",
        description="examples/non_hermitian/ep_tuning.py"))


_register_capabilities()
