"""The heuristic rules behind every automatic decision of a flow, with the references they rest on.

A rule cites keys of the curated registry (:mod:`pinneapple_flow.references`), or says plainly why it has none
(``uncited``): reading the user's own words, a project convention, a rule of thumb. ``evidence`` points at what in
this repository backs the rule (a test, an example result, a library constant). ``tests/test_flow_decisions.py``
checks that every rule cites a registry key or is flagged uncited, and that every evidence path exists.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    cites: tuple[str, ...] = ()
    uncited: str = ""                                  # why there is no literature reference
    evidence: tuple[tuple[str, str, str], ...] = ()    # (kind: test | example | library | benchmark, path, detail)
    topic: str = ""


_T_EQ = ("test", "tests/test_flow_equations.py", "FD and PINN solutions of parsed systems against exact solutions")
_EX_HEAT = ("example", "examples/flow_studio/results/heat_1d/summary.json", "heat 1D: FD and PINN vs exact, Lab tier")
_EX_POI = ("example", "examples/flow_studio/results/poisson_2d/summary.json", "Poisson 2D: FD / FEM / PINN vs exact")
_EX_BUR = ("example", "examples/flow_studio/results/burgers_1d/summary.json", "Burgers: FD and PINN vs Cole-Hopf")
_EX_OP = ("example", "examples/flow_studio/results/operator_burgers/summary.json", "DeepONet and FNO on Burgers")
_EX_INV = ("example", "examples/flow_studio/results/inverse_heat/summary.json", "diffusivity identified two ways")
_EX_TWIN = ("example", "examples/flow_studio/results/planned_digital_twin/summary.json", "tank twin, leak detected")

RULES: dict[str, Rule] = {r.id: r for r in [
    # ---- physics identification
    Rule("family.keywords", "PDE family from keywords of the description (heat, wave, Poisson, ...)",
         cites=("evans2010pde",), topic="equations",
         evidence=(("test", "tests/test_flow_planner.py", "detection of the family from descriptions"),)),
    Rule("family.heat", "heat / diffusion: u_t = alpha lap(u)", cites=("carslaw1959heat", "evans2010pde"),
         topic="equations", evidence=(_EX_HEAT,)),
    Rule("family.wave", "waves / vibration: u_tt = c^2 lap(u)", cites=("evans2010pde", "whitham1974waves"),
         topic="equations"),
    Rule("family.poisson", "steady field with a source: lap(u) = f", cites=("evans2010pde",), topic="equations",
         evidence=(_EX_POI,)),
    Rule("family.laplace", "steady field without sources: lap(u) = 0", cites=("evans2010pde",), topic="equations"),
    Rule("family.helmholtz", "time-harmonic waves: lap(u) + k^2 u = f", cites=("evans2010pde",), topic="equations"),
    Rule("family.burgers", "viscous Burgers: u_t + u u_x = nu u_xx",
         cites=("whitham1974waves", "hopf1950burgers", "cole1951burgers"), topic="equations", evidence=(_EX_BUR,)),
    Rule("family.advection_diffusion", "transport by a flow plus diffusion", cites=("leveque2007fd",),
         topic="equations"),
    Rule("family.reaction_diffusion", "diffusion with logistic growth (Fisher-KPP)", cites=("fisher1937wave",),
         topic="equations"),
    Rule("family.steady_heat", "steady heat: Laplace without sources, Poisson with internal generation",
         cites=("carslaw1959heat", "evans2010pde"), topic="equations"),
    Rule("family.default", "no physics recognised: start from the heat equation for the user to edit",
         uncited="a starting point, not an identification", topic="equations"),
    Rule("discovery.sparse_regression", "governing terms from data by sparse regression on candidate derivatives",
         cites=("brunton2016sindy", "rudy2017pdefind"), topic="equations",
         evidence=(("test", "tests/test_flow_generic.py", "recovers the heat equation from FD data"),)),
    # ---- domain
    Rule("dimension.keywords", "dimension from the description's words (rod -> 1-D, plate -> 2-D, block -> 3-D)",
         uncited="a reading of the user's words; confirm it in the domain node", topic="dimension",
         evidence=(("test", "tests/test_flow_planner.py", "dimension detection"),)),
    Rule("dimension.default", "no dimension stated: 2-D (1-D for Burgers, the classic benchmark)",
         uncited="a default, not a reading of the problem", topic="dimension"),
    Rule("dimension.lap_switch", "lap() keeps one equation text valid in 1-D, 2-D and 3-D, so the domain switch "
         "carries the problem", cites=("evans2010pde",), topic="dimension",
         evidence=(("test", "tests/test_flow_equations.py", "lap() expands to the spatial coordinates"),)),
    Rule("domain.unit_box", "unit domain (or the stated length) with time [0, T] from the template",
         uncited="illustrative extent; replace with the real geometry", topic="dimension"),
    # ---- conditions and coefficients
    Rule("bc.dirichlet_template", "fixed values on the boundary (Dirichlet), the template's well-posed setting",
         cites=("evans2010pde",), topic="bcs"),
    Rule("bc.insulated", "insulated / adiabatic faces: zero normal derivative (Neumann)",
         cites=("carslaw1959heat",), topic="bcs"),
    Rule("ic.template", "initial condition of the template: a mode with a known exact solution",
         cites=("carslaw1959heat", "evans2010pde"), topic="bcs"),
    Rule("coef.from_text", "coefficient value written in the description", uncited="the user's own value",
         topic="coefficients"),
    Rule("coef.template_default", "illustrative coefficient value of the template",
         uncited="placeholder: replace with the material's or the fluid's value", topic="coefficients"),
    Rule("exact.template", "exact solution of the template problem (for validation)",
         cites=("carslaw1959heat", "evans2010pde"), topic="validation", evidence=(_T_EQ,)),
    Rule("exact.cole_hopf", "Cole-Hopf exact solution of viscous Burgers", cites=("hopf1950burgers",
                                                                                 "cole1951burgers"),
         topic="validation", evidence=(_EX_BUR,)),
    # ---- methods
    Rule("method.fd", "finite differences / method of lines with BDF for the edited system",
         cites=("leveque2007fd", "hairer1996odes2"), topic="method", evidence=(_T_EQ, _EX_HEAT)),
    Rule("method.fem", "P1 finite elements for steady Poisson / Laplace", cites=("brenner2008fem",),
         topic="method", evidence=(_EX_POI,)),
    Rule("method.pinn", "physics-informed network on the same equations", cites=("raissi2019pinn",),
         topic="method", evidence=(_EX_HEAT, _EX_POI)),
    Rule("method.pinn_training", "Adam then L-BFGS, validated at fresh points (residual, conditions, causality)",
         cites=("kingma2015adam", "liu1989lbfgs", "wang2022causality"), topic="hyperparameters",
         evidence=(("library", "pinneapple_analysis/validation/pinn.py", "train_pinn / validate_pinn"),)),
    Rule("method.openfoam", "finite volumes with OpenFOAM for CFD cases", cites=("moukalled2016fvm",),
         topic="method"),
    Rule("route.alternatives", "offer the classical + PINN route, a surrogate route and an inverse route when "
         "they apply", uncited="a planning heuristic of this library", topic="route"),
    # ---- surrogates
    Rule("surrogate.deeponet", "DeepONet for maps from parameters / input functions to fields",
         cites=("lu2021deeponet",), topic="architecture", evidence=(_EX_OP,)),
    Rule("surrogate.fno", "Fourier neural operator for functions on a regular grid", cites=("li2021fno",),
         topic="architecture", evidence=(_EX_OP,)),
    Rule("surrogate.pod_mlp", "POD basis + network for the coefficients (ROM + NN hybrid)",
         cites=("berkooz1993pod", "benner2015pmor"), topic="architecture"),
    Rule("surrogate.mlp", "plain network baseline", uncited="a generic regression baseline", topic="architecture"),
    Rule("surrogate.gnn", "graph network for mesh-based fields", cites=("pfaff2021meshgraphnet",
                                                                        "battaglia2018graphnets"),
         topic="architecture"),
    Rule("architecture.default_sizes", "suggested widths, depths and modes", topic="architecture",
         uncited="rule-of-thumb sizes; the HPO node or a benchmark should confirm them", evidence=(_EX_OP,)),
    Rule("sampling.lhs", "Latin hypercube samples of the varied coefficients", cites=("mckay1979lhs",),
         topic="hyperparameters"),
    Rule("hpo.search", "TPE (Optuna) when installed, else random search, on the validation split only",
         cites=("akiba2019optuna", "bergstra2012random"), topic="hpo"),
    Rule("hpo.ranges", "default search ranges (learning rate 3e-4 to 1e-2, log; width 16-128; depth 1-4)",
         uncited="rule-of-thumb ranges", topic="hpo"),
    Rule("uq.sobol", "variance-based (Sobol) sensitivity indices", cites=("sobol2001indices",
                                                                          "saltelli2002sensitivity"),
         topic="method"),
    # ---- inverse
    Rule("inverse.pinn", "unknown coefficients trained with the network against observations",
         cites=("raissi2019pinn",), topic="method", evidence=(_EX_INV,)),
    Rule("inverse.calibration", "least-squares calibration with the classical solver, identifiability",
         cites=("ljung1999sysid", "raue2009identifiability"), topic="method", evidence=(_EX_INV,)),
    # ---- validation
    Rule("verification.richardson", "grid convergence: three nested grids (ratio 2), observed order, Richardson "
         "extrapolation and the grid convergence index", cites=("roache1998verification", "oberkampf2010vv"),
         topic="validation"),
    Rule("assimilation.kalman", "sequential state estimation with a Kalman filter (EKF) or its ensemble form (EnKF)",
         cites=("kalman1960filter", "evensen1994enkf"), topic="twin"),
    Rule("uq.propagation", "forward propagation of input uncertainty: Latin hypercube Monte Carlo or polynomial chaos",
         cites=("mckay1979lhs",), topic="hyperparameters"),
    Rule("rom.methods", "reduced-order models compared at equal budget: POD (+ interpolation), DMD, operator "
         "inference", cites=("berkooz1993pod", "schmid2010dmd", "peherstorfer2016opinf"), topic="method"),
    Rule("gnn.meshgraphnet", "a MeshGraphNet (encode-process-decode message passing) for fields on meshes",
         cites=("pfaff2021meshgraphnet", "battaglia2018graphnets"), topic="architecture"),
    Rule("meta.reptile", "meta-learned initialisation over a family of problems (Reptile or MAML), compared with a "
         "random initialisation after the same adaptation steps", cites=("nichol2018reptile", "finn2017maml"),
         topic="method"),
    Rule("transfer.finetune", "transfer learning: start the target problem from a trained network and fine-tune, "
         "compared with training from scratch", uncited="a standard practice; the comparison in the node is the "
         "evidence for this problem", topic="method",
         evidence=(("test", "tests/test_flow_methods.py", "transfer node against training from scratch"),)),
    Rule("optimization.nsga2", "multi-objective search with NSGA-II (non-dominated sorting, crowding distance)",
         cites=("deb2002nsga2",), topic="method"),
    Rule("control.mpc", "model predictive control compared with PID and rule-based baselines on the same plant",
         cites=("rawlings2017mpc",), topic="method"),
    Rule("benchmark.arena", "the same problems, budget and metric for every model (PINN arena)",
         uncited="a project convention: equal budget per model; no single literature source", topic="method",
         evidence=(("test", "tests/test_breadth_six_packages.py", "PINNArenaBenchmark run"),)),
    Rule("validation.reference", "measure every candidate against an exact solution or a reference solution",
         cites=("oberkampf2010vv",), topic="validation", evidence=(_EX_HEAT, _EX_POI)),
    Rule("validation.threshold", "pass at relative L2 <= 2 % (the PINN validation default)",
         uncited="project convention: TOLERANCES['exact'] in pinneapple_analysis.validation.pinn",
         evidence=(("library", "pinneapple_analysis/validation/pinn.py", "TOLERANCES"),), topic="validation"),
    Rule("validation.lab_tiers", "PINNeAPPle Lab seal letters A-F from the curation",
         uncited="the library's own curation rules", topic="validation",
         evidence=(("library", "pinneapple_lab/curation.py", "tier gates"),)),
    # ---- twin
    Rule("twin.enkf", "ensemble Kalman filter for state estimation", cites=("evensen1994enkf",
                                                                            "kalman1960filter"),
         topic="twin", evidence=(_EX_TWIN,)),
    Rule("twin.design", "digital twin = model + live data + assimilation + calibration + anomaly alerts",
         cites=("kapteyn2021twin",), topic="twin", evidence=(_EX_TWIN,)),
    Rule("twin.parameter_state", "coefficients carried as random-walk states so the filter calibrates them",
         uncited="standard practice; the cited EnKF paper covers the filter, not this augmentation",
         topic="twin", evidence=(_EX_TWIN,)),
    Rule("twin.coarse_model", "the identified PDE on a coarse grid as the twin's fast model",
         cites=("leveque2007fd",), topic="twin"),
    Rule("twin.simulated_fault", "the simulated asset changes a coefficient mid-run so the alerts can be seen",
         uncited="a demonstration setting; with a real stream there is no injected change", topic="twin"),
    Rule("twin.sensor_layout", "the number of sensors read from the description, placed evenly inside the domain",
         uncited="a reading of the user's words and a neutral placement; edit the positions in the live-data node",
         topic="twin", evidence=(("test", "tests/test_flow_planner.py", "sensor count read from the text"),)),
    Rule("twin.alert_thresholds", "alerts at chi-square 0.999 consistency and 3-sigma parameter limits",
         uncited="defaults of pinneapple_systems.operations.twin.AlertRules", topic="twin",
         evidence=(("library", "pinneapple_systems/operations/twin.py", "AlertRules"),)),
    # ---- forecasting, flight
    Rule("forecast.predictability_first", "test predictability with surrogate data before modelling",
         cites=("theiler1992surrogate",), topic="method"),
    Rule("forecast.baselines", "compare with persistence and climatology baselines, DM tests",
         cites=("hyndman2021fpp", "diebold1995dm"), topic="validation"),
    Rule("flight.raymer", "take-off distance against Raymer's closed form", cites=("raymer2018aircraft",),
         topic="validation"),
]}


def register_rule(r: Rule) -> Rule:
    """Add a heuristic rule (another package's decisions). It must cite registered references
    (:func:`pinneapple_flow.references.register_references` first) or say why it has none; an existing id is
    refused unless the rule is the same."""
    from .references import registry
    missing = [k for k in r.cites if k not in registry()]
    if missing:
        raise ValueError(f"rule {r.id!r} cites unknown references {missing}")
    if not r.cites and not r.uncited:
        raise ValueError(f"rule {r.id!r} needs cites or an uncited reason")
    if r.id in RULES and RULES[r.id] != r:
        raise ValueError(f"rule {r.id!r} is already registered differently")
    RULES[r.id] = r
    return r


def rule(rule_id: str) -> Rule:
    if rule_id not in RULES:
        raise KeyError(f"unknown rule {rule_id!r}")
    return RULES[rule_id]
