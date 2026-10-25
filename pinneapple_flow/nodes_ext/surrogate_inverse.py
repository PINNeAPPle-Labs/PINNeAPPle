"""Surrogate-driven inverse design (:mod:`pinneapple_design.surrogate_inverse`, the five design tools of
``apps/*_ai``): sample the design space, label it with a closed form, train an MLP, descend through the frozen
surrogate and re-check the final design against the closed form (verification card and trust score)."""
from __future__ import annotations

from ..registry import Param, Port, node

#: case -> (module, its two optimisation inputs with the defaults of the app's own tests)
CASES = {
    "heatsink_fins": ("heatsink_fins", {"power_w": 65.0, "cooling_profile": "standard_fan"}),
    "brake_rotor": ("brake_rotor", {"power_w": 800.0, "cooling_profile": "highway"}),
    "transdermal_patch": ("transdermal_patch", {"target_rate_mg_day": 100.0, "target_wear_days": 2.0}),
    "suspension_arm": ("suspension_arm", {"load_n": 1200.0, "target_life_cycles": 1e6}),
    "reflex_port": ("reflex_port", {"cabinet_volume_l": 20.0, "target_freq_hz": 40.0}),
}


@node("design.surrogate_inverse", "Inverse design through a frozen surrogate, re-checked", category="Design",
      outputs=[Port("metrics", "metrics")],
      params=[Param("case", "enum", "heatsink_fins", choices=list(CASES),
                    help="heat sink, brake rotor, transdermal patch, suspension arm, bass-reflex port"),
              Param("inputs", "dict", {}, help="the case's two inputs (empty: the defaults of the app's tests)"),
              Param("n_samples", "int", 6000, minimum=100), Param("epochs", "int", 1500, minimum=10),
              Param("seed", "int", 0)],
      calls="pinneapple_design.surrogate_inverse.<case>.train_surrogate / optimize_design / verify_design "
            "(core.fit_surrogate, core.descend, core.VerificationCard)")
def surrogate_inverse(ctx, inputs, params):
    """Train the case's surrogate, optimise the design through it, verify it against the closed form."""
    import importlib

    from pinneapple_core.workflows import StepFailed
    mod_name, defaults = CASES[str(params["case"])]
    m = importlib.import_module(f"pinneapple_design.surrogate_inverse.{mod_name}")
    args = {**defaults, **dict(params["inputs"] or {})}
    if set(args) != set(defaults):
        raise StepFailed(f"{params['case']} takes {', '.join(defaults)}")
    a, b = (args[k] for k in defaults)
    if "cooling_profile" in args:                    # the profile names a convection coefficient of the case
        if b not in m.COOLING_PROFILES:
            raise StepFailed(f"cooling_profile: one of {', '.join(m.COOLING_PROFILES)}")
        b = m.COOLING_PROFILES[b]
    fit = m.train_surrogate(n_samples=int(params["n_samples"]), epochs=int(params["epochs"]), seed=int(params["seed"]))
    for k, v in fit.to_dict().items():
        ctx.metric(f"surrogate.{k}", v, "held-out validation of the MLP against the closed form")
    r = m.optimize_design(fit.model, float(a), float(b))
    pred_key = next(k for k in vars(r) if k.startswith("surrogate_predicted"))
    card = m.verify_design(r.design, float(a), float(b), getattr(r, pred_key))
    for k, v in m.design_to_dict(r.design).items():
        if isinstance(v, (int, float)):
            ctx.metric(f"design.{k}", v)
    for k, v in card.values.items():
        if isinstance(v, (int, float)):
            ctx.metric(f"verify.{k}", v, "closed form at the final design (the surrogate does not grade itself)")
    ctx.metric("verdict", card.overall_verdict)
    ctx.metric("trust_score", card.trust_score)
    ctx.table("verification card", ["check", "verdict", "detail"], [[c.name, c.verdict, c.detail] for c in card.checks])
    ctx.summary(f"{params['case']}: {card.overall_verdict}, trust {card.trust_score}")
    return {"metrics": ctx.report["metrics"]}
