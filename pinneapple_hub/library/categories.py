"""The 20 task categories of the Hub's tools. Every tool gets exactly one, by the longest rule prefix that matches its
key; the key names where the tool comes from (``node:<flow node type>``, ``lib:<module>:<function>`` for a generated
library call, ``fn:<dotted name>``, ``mcp:<agent tool>``). Recipes (worked snippets from the READMEs and AGENTS.md)
are listed one by one in ``RECIPES``: their ids follow the order of the README, so a prefix would misfile them.
``tests/test_hub_library.py`` fails when a tool matches nothing, a category is empty or a rule matches no tool."""
from __future__ import annotations

from typing import Any

#: (slug, title, one-line description, key prefixes), in display order; the longest matching prefix wins, so a
#: specific rule (``node:lib.solver.terramechanics``) beats a general one (``node:lib.solver.``)
CATEGORIES: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("problem", "Problem setup", "Describe the physical problem: presets, generic problems, requirements",
     ("node:problem.", "node:scenario.", "fn:pp.PhysicalProblem", "mcp:list_presets",
      "lib:pinneapple_data.formfill")),
    ("geometry", "Geometry and meshing", "Domains, meshes, shapes from parameters, geometry exports",
     ("node:geometry.", "node:lib.export.geometry", "node:lib.export.usd_validate", "lib:pinneapple_design.geometry",
      "lib:pinneapple_design.fitmesh", "lib:pinneapple_design.bricks")),
    ("equations", "Equations and PDE environments", "PDE systems, boundary and initial conditions, test environments",
     ("node:equations.", "lib:pinneapple_physics.pde_environment", "lib:pinneapple_physics.non_hermitian")),
    ("classical", "Classical solvers", "Finite differences, FEM, LBM, meshless, OpenFOAM, CalculiX, FMU",
     ("node:solver.", "node:lib.solver.", "node:lib.solve.", "node:lib.robot.", "fn:pp.solve", "mcp:solve",
      "lib:pinneapple_simulation.", "lib:pinneapple_physics.solid")),
    ("pinn", "PINNs and neural training", "Train, fine-tune and transfer physics-informed networks",
     ("node:train.", "node:transfer.", "node:meta.", "node:gnn.", "node:lib.model.", "lib:pinneapple_physics.pinn_solver",
      "lib:pinneapple_physics.deepxde_bridge", "lib:pinneapple_neural.", "lib:pinneapple_core.loss",
      "lib:pinneapple_core.optim", "fn:pinneapple_neural.")),
    ("surrogates", "Surrogates and reduced models", "Surrogates, ROM / POD, hyper-parameter search",
     ("node:surrogate.", "node:rom.", "node:lib.rom.", "lib:pinneapple_analysis.surrogate")),
    ("zoo", "Model zoo", "Pre-trained, verified models and their cards",
     ("mcp:zoo_", "fn:pp.zoo", "node:terramechanics.zoo_surrogate")),
    ("checks", "Physics checks and trust cards", "Residuals, conservation, boundary checks, trust cards",
     ("lib:pinneapple_physics.physics_audit", "lib:pinneapple_physics.darcy_checks", "fn:pp.trust_card",
      "fn:pp.validate", "mcp:trust_card", "fn:pinneapple_analysis.validation", "lib:pinneapple.check")),
    ("verification", "Verification and convergence", "Convergence studies, reference comparisons, Lab seals",
     ("node:verify.", "node:validate.", "node:forecast.lab", "lib:pinneapple_analysis.verification",
      "lib:pinneapple_analysis.inspection", "fn:pinneapple_catalog.", "mcp:describe_method",
      "mcp:list_methods")),
    ("benchmarks", "Benchmarks", "Equal-budget comparisons of methods",
     ("node:benchmark.", "node:lib.compare.", "fn:pp.benchmark", "fn:pp.compare", "mcp:benchmark",
      "lib:pinneapple_analysis.benchmark", "lib:pinneapple_tools.benchmark_suite")),
    ("uq", "Uncertainty and sensitivity", "Propagation, Sobol / Morris sensitivity, sampling",
     ("node:uq.", "node:lib.uq.", "node:lib.doe.", "lib:pinneapple_analysis.uncertainty",
      "lib:pinneapple_data.parameter_sampling", "lib:pinneapple_data.grf_sampler")),
    ("inverse", "Inverse problems and assimilation", "Calibration, equation discovery, data assimilation",
     ("node:inverse.", "node:assimilate.", "lib:pinneapple_analysis.data_assimilation")),
    ("design", "Design and optimisation", "Design from intent, optimisers, topology, the chain of whys",
     ("node:optimize.", "node:lib.design.", "node:reason.", "node:autolab.", "fn:pp.design", "mcp:design", "mcp:reason",
      "fn:pinneapple_design.thermal.board_cooling",
      "lib:pinneapple_design.", "lib:pinneapple_data.process_optimizer", "lib:pinneapple_analysis.cost")),
    ("systems", "Process and system components", "Pumps, valves, fatigue, excitation, controllers",
     ("node:control.", "lib:pinneapple_systems.")),
    ("vehicles", "Vehicles, contact and wear", "Rovers, soils, wheels, take-off, contact and wear",
     ("node:vehicle.", "node:terramechanics.", "node:flight.", "lib:pinneapple_physics.tribology",
      "node:lib.solver.terramechanics", "lib:pinneapple_simulation.vehicle")),
    ("electromagnetics", "Electromagnetics and antennas", "Antennas, PCBs, Maxwell checks",
     ("node:lib.antenna.", "lib:pinneapple_physics.pcb", "fn:pp.maxwell", "lib:pinneapple_simulation.electromagnetics")),
    ("time-series", "Time series and digital twins", "Forecasts, predictability, live twins",
     ("node:forecast.", "node:twin.", "node:lib.ts.", "node:data.stream", "fn:pinneapple_systems.time_series")),
    ("data", "Data and datasets", "Datasets, ingestion, observation, adapters, units",
     ("node:data.", "lib:pinneapple_data.", "lib:pinneapple_tools.dataset_quality", "lib:pinneapple_analysis.retrieval",
      "lib:pinneapple_physics.astro")),
    ("reports", "Visualisation and reports", "Plots, 3-D fields, renders, dashboards, reports",
     ("node:viz.", "node:export.", "fn:pp.viz", "fn:pp.dashboard",
      "lib:pinneapple_tools.reporting")),
    ("automation", "Automation, agents and compute", "Agent tools, jobs, compute backends, security, the library itself",
     ("mcp:", "fn:pinneapple_flow.", "fn:pinneapple_hub.", "node:lib.tools.", "lib:pinneapple_tools.",
      "lib:pinneapple_core.", "lib:pinneapple_analysis.study_models")),
]

SLUGS = [c[0] for c in CATEGORIES]

#: recipe id -> category; a new recipe in a README fails the test until it is added here
RECIPES: dict[str, str] = {
    "recipe:pinneapple__0": "automation",                       # what the front door exposes
    "recipe:pinneapple_adaptation__0": "pinn",                  # freeze all but the last layer
    "recipe:pinneapple_analysis__0": "uq",                      # conformal prediction intervals
    "recipe:pinneapple_analysis__1": "inverse",                 # SINDy recovers a missing term
    "recipe:pinneapple_analysis__2": "inverse",                 # extended Kalman filter, pendulum
    "recipe:pinneapple_arena__0": "benchmarks",
    "recipe:pinneapple_blender__0": "reports",                  # coloured point cloud
    "recipe:pinneapple_catalog__0": "data",                     # datasets that allow commercial use
    "recipe:pinneapple_core__0": "verification",                # gradient and integral against calculus
    "recipe:pinneapple_data__0": "uq",                          # Latin hypercube versus random
    "recipe:pinneapple_decision__0": "design",                  # which model for an airfoil study
    "recipe:pinneapple_design__0": "geometry",                  # bracket from signed distance functions
    "recipe:pinneapple_design__1": "geometry",                  # NACA four-digit airfoils
    "recipe:pinneapple_hub__0": "zoo",                          # a model card must make a checkable claim
    "recipe:pinneapple_lab__0": "verification",
    "recipe:pinneapple_lab__1": "verification",                 # grid convergence index
    "recipe:pinneapple_llm__0": "checks",                       # guardrail on a harmonic function
    "recipe:pinneapple_neural__0": "surrogates",                # POD + DMD
    "recipe:pinneapple_neural__1": "surrogates",                # parametric POD with GP uncertainty
    "recipe:pinneapple_neural__2": "pinn",                      # DeepONet
    "recipe:pinneapple_orchestration__0": "automation",
    "recipe:pinneapple_pdb__0": "benchmarks",
    "recipe:pinneapple_perception__0": "data",                  # PIV
    "recipe:pinneapple_physics__0": "pinn",                     # Burgers PINN against Cole-Hopf
    "recipe:pinneapple_physics__1": "checks",                   # closed-form engineering checks
    "recipe:pinneapple_physics__2": "vehicles",                 # Archard sliding wear
    "recipe:pinneapple_problemdesign__0": "automation",         # consequential actions need approval
    "recipe:pinneapple_quantum__0": "verification",             # exact oscillator states as reference
    "recipe:pinneapple_registry__0": "zoo",
    "recipe:pinneapple_security__0": "automation",
    "recipe:pinneapple_simulation__0": "classical",             # C3D8I cantilever against beam theory
    "recipe:pinneapple_simulation__1": "checks",                # pipe-flow correlations
    "recipe:pinneapple_systems__0": "time-series",              # baselines every forecaster must beat
    "recipe:pinneapple_tools__0": "reports",                    # vorticity and Q-criterion
    "recipe:pinneapple_tools__1": "automation",                 # TorchScript export
    "recipe:pinneapple_twin3d__0": "time-series",               # pipe bend with five years of wear
    "recipe:pinneapple_veriphysics__0": "verification",
    "recipe:pinneapple_worldmodel__0": "geometry",              # built-in domains and collocation sets
    "recipe:agents-dashboard-of-a-time-dependent-result": "reports",
    "recipe:agents-equal-budget-comparison": "benchmarks",
    "recipe:agents-pre-trained-models-and-datasets": "zoo",
    "recipe:agents-solve-and-compare": "classical",
    "recipe:agents-validate-any-result-never-report-a-field-without-this": "checks",
    "recipe:agents-validation-status-of-a-method": "verification",
    "recipe:agents-why-each-element-of-a-design-exists": "design",
}


def key(it: dict[str, Any]) -> str:
    """Where a tool comes from, as the prefix rules read it."""
    sub, name = it.get("subkind"), it.get("name") or it["id"].split(":", 1)[1]
    extra = it.get("extra") or {}
    if sub == "library call":
        return "lib:" + (extra.get("calls") or name)
    if sub == "flow node":
        return "node:" + name
    if sub == "function":
        return "fn:" + name
    if sub == "agent tool (MCP)":
        return "mcp:" + name
    return "recipe:" + it["id"].split(":", 1)[1]


def category_of(it: dict[str, Any]) -> str | None:
    """The slug of a tool's category, or None when no rule matches (a test forbids that)."""
    k = key(it)
    if k.startswith("recipe:"):
        return RECIPES.get(k)
    hits = [(len(p), slug) for slug, _title, _desc, prefixes in CATEGORIES for p in prefixes if k.startswith(p)]
    return max(hits)[1] if hits else None


def table() -> list[dict[str, str]]:
    return [{"slug": s, "title": t, "description": d} for s, t, d, _p in CATEGORIES]
