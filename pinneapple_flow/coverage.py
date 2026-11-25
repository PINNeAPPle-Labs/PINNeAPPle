"""Which PINNeAPPle capabilities are flow nodes, and why the others are not.

:func:`coverage` walks every ``pinneapple*`` package and its first-level modules, maps the registered node types onto
them (by the library calls each node declares) and attaches a curated reason (:data:`REASONS`) to whatever has no
node. ``python -m pinneapple_flow coverage -o docs/flow_studio/COVERAGE.md`` writes the table (git-ignored: the docs
build generates it, ``tools/gen_flow_docs.py``) and exits 1 when a module has neither a node nor a reason.
"""
from __future__ import annotations

import pkgutil
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# package or package.module -> why there is no node (or "covered by <node types>" when the coverage is indirect)
REASONS: dict[str, str] = {
    # compatibility shims and infrastructure
    "pinneapple_models": "shim: re-exports pinneapple_neural.architectures (covered there)",
    "pinneapple_train": "shim: re-exports pinneapple_neural.trainer (covered there)",
    "pinneapple_solvers": "shim: re-exports pinneapple_simulation.numerical_solvers (covered there)",
    "pinneapple_orchestration": "pipeline wiring for Prefect; flows are orchestrated by pinneapple_flow itself",
    "pinneapple_plugins": "the plugin contract and its loader: what an installed package may contribute and whether it "
                          "works on this machine. It decides which steps exist and reports on them; it is not a step",
    "pinneapple_plugins.contract": "the declarations a plugin makes (name, contract version, requirements, "
                                   "capabilities) and the version check: a type a loader validates, not a computation",
    "pinneapple_plugins.discovery": "reading the 'pinneapple.plugins' entry-point group from distribution metadata: it "
                                    "finds what is installed, without importing it",
    "pinneapple_plugins.manager": "loading, recording and diagnosing plugins (list, inspect, doctor, enable, disable): "
                                  "the state of the platform, which 'pinneapple plugins' reports",
    "pinneapple_plugins.cli": "the 'pinneapple plugins' command line: a front end over the manager, like Flow Studio",
    "pinneapple_neural.model_provider": "one interface over the doors a trained model arrives through (PyTorch, "
                                        "TorchScript, ONNX, JAX, PhysicsNeMo, Hugging Face): it opens a saved file "
                                        "and hands back a model with its provenance, which is what a comparison reads "
                                        "rather than a computation step",
    "pinneapple_mcp": "the MCP server for coding agents (a protocol front end, like Flow Studio)",
    "pinneapple_flow": "this package",
    "pinneapple_stability": "what the library promises (core vs experimental); metadata, not a computation step",
    "pinneapple_registry": "artifact store and model gateway services (deployment, not a computation step)",
    "pinneapple_security": "integrity, signing, web security and audit (infrastructure used by the apps)",
    "pinneapple_core.service": "job queue / API keys of the app servers (infrastructure)",
    "pinneapple_core.errors": "what a failure tells you to do about it: the action is carried by the exception, not "
                              "computed by a step of a flow",
    "pinneapple_core.run_record": "the record of one run and the command that redoes it: it is what a run leaves "
                                  "behind, not a step inside one",
    "pinneapple_flow.reproducibility": "the record of a finished flow run and its reproducer: it re-runs a saved "
                                       "document to compare it with what was recorded, which is a check on a run, "
                                       "not a step of one",
    "pinneapple_hub.experiment_card": "a model card generated from an experiment record and validated: it is the "
                                      "document a model is published with, built from the record of a run rather "
                                      "than being a step of one",
    "pinneapple_analysis.verification.asme_vv20": "the ASME V&V 20 validation of one number against one datum, and "
                                                  "the V&V 40 credibility block: it is a report a reviewer reads, "
                                                  "built from a mesh study and a data point, not a step of a flow",
    "pinneapple_analysis.bayesopt": "the loop that decides which simulation to run next (Bayesian optimization with "
                                    "constraints): it chooses the calls a campaign makes, it is not one of them",
    "pinneapple_data.regimes": "which bands of Reynolds, Mach and the other dimensionless numbers a dataset reaches: "
                               "it is part of the dataset audit, a check on the data before a flow runs on it",
    "pinneapple_simulation.cfd_validation": "the classical validation cases, their published tables and the solver "
                                            "runs measured against them (#272, #265, #294, #494): verification "
                                            "evidence a reviewer reads, and a study of hours of OpenFOAM per case, "
                                            "not a step of a flow",
    "pinneapple_data.benchmark_suites": "the published freestream conditions of a public validation suite (NASA "
                                        "TMR), read as the input of that same audit: a reference table, not a step "
                                        "of a flow",
    "pinneapple_analysis.trust.corrections": "a reviewer's correction stored as a rule and run as a check of the "
                                             "trust report: it is the record of what an engineer decided, cited by "
                                             "the report, not a computation step of a flow",
    "pinneapple_analysis.trust.flow_card": "mass and energy conservation and the boundary-condition violation "
                                           "measured on a predicted field: it is a check of a result against the "
                                           "physics its spec states, like the scalar card next to it",
    "pinneapple_analysis.inverse_problems.recovery": "a recovered parameter with an interval, a warning when the "
                                                     "data cannot separate it from another, and the coverage that "
                                                     "interval was measured to have: it is the evidence a calibration "
                                                     "is reported with, not a step of a flow",
    "pinneapple_analysis.condition": "what the scales and the loss weights should be before a run starts: it "
                                     "answers a question about the setup of a problem, and the training node "
                                     "consumes the numbers -- the run it changes is the node's, not this",
    "pinneapple_analysis.linalg": "the linear-algebra layer the steps above are built from: an operator, a POD basis, "
                                  "a spectrum, a condition estimate and an adjoint derivative are what a node calls "
                                  "inside its own computation, not a computation a flow chooses to run -- the "
                                  "reduced-order and sensitivity nodes are the steps that use them",
    "pinneapple_physics.dimensions": "a dimensional check of one equation before training: it reads a statement and "
                                     "reports which term broke the units, which is a review of a problem rather "
                                     "than a step of a flow",
    "pinneapple_physics.experiments": "reproducing a recorded run (reproduced / divergent / refused): the definition "
                                      "of the three outcomes, read by pp.experiments rather than placed in a flow",
    # modules the public-API scan finds nothing to wrap in, with the reason read off what they are for
    "pinneapple_data.multifidelity": "the two-fidelity models -- pairing the cheap and expensive values of the same "
                                     "design, the kernel-ridge correction, and the learning curves over scalar targets "
                                     "and over a whole field; they take the corpus's arrays and fitted pairs as "
                                     "objects, so a flow form cannot hold their inputs",
    "pinneapple_data.graph": "the module builds the Graph objects the graph networks train on, from arrays and from "
                             "simulation records; a Graph is not something a flow's parameter form holds, so the "
                             "node that trains a graph model builds it inside the node",
    "pinneapple_analysis.fidelity": "the choice between the surrogate and the solver for each request (#161), with "
                                    "the reason kept in the request's record: it is the router a flow calls, not a "
                                    "computation a node performs",
    "pinneapple_analysis.multifidelity_workflow": "the two data levels of #35 as one workflow -- a nested design, the "
                                                  "co-kriging model against a high-fidelity-only one at equal cost, "
                                                  "and the break-even that decides whether the cheap level pays: it is "
                                                  "the study a campaign runs before it buys solver time, so it chooses "
                                                  "the calls rather than being one of them",
    "pinneapple_data.adaptive_sampling": "the loop that decides which simulation to run next (multi-signal adaptive "
                                         "sampling of #209): it chooses the calls a campaign makes, it is not one of "
                                         "them, in the same way as pinneapple_analysis.bayesopt",
    "pinneapple_physics.active_design": "the same loop behind the option it is reached by (Experiment(active=...) in "
                                        "#34 and #153): it spends a budget of solver runs over the problem's design "
                                        "box and records why, which chooses the calls a flow makes instead of being "
                                        "one of them -- the reason pinneapple_data.adaptive_sampling already gives",
    "pinneapple_data.schema": "the data contract a bundle carries -- manifest, fields, locations, units, gates; it "
                              "is a type the adapters fill and the gates read, not a computation step of its own",
    "pinneapple_data.astro": "the yt reader (#515): one Enzo/FLASH/RAMSES/GADGET/Athena++ output as a "
                             "PhysicalSample, which is the data step the flows already have a node for; a node of its "
                             "own would carry the dataset path and the yt install as parameters",
    "pinneapple_simulation.events": "an event-driven integrator whose inputs are Python callables (the right-hand "
                                    "side and the actions): a flow's parameter form cannot hold them, so a node that "
                                    "runs a case defines its events inside the node",
    "pinneapple_simulation.launch": "ascent trajectories for the launch vertical: the flight profile, its atmosphere "
                                    "and its visuals are arranged by examples/rocket_launch, which no node wraps",
    "pinneapple_core.ontology": "the formal types a problem, a dataset manifest and a model card are all filled into, "
                                "and the JSON schema they are published with (#204): it is the vocabulary the steps "
                                "speak, not a step of its own",
    "pinneapple_neural.hybrid_closure": "the closure arrangement of the #211: a base model plus a learned residual. "
                                        "A flow node would carry a torch training run whose base is a callable, "
                                        "which is what the pinn / train nodes already do",
    "pinneapple_core.series": "downsampling of long series for charts and JSON payloads (presentation helper the "
                              "apps and data reports use, not a computation step)",
    "pinneapple_core.onboarding": "the first hour: doctor / recommend / card / new run before a problem "
                                  "exists, and the solves they print are solve nodes",
    "pinneapple_core.workflows": "the execution engine every flow runs on",
    "pinneapple_core.distributed": "parallel execution around the steps that already have nodes: a sweep runs many "
                                   "solve / simulation nodes, and a flow step is one process by design",
    "pinneapple_core.artifact": "the container a result travels in, not a computation step: the solve / train nodes "
                                "produce what pp.save writes, and a flow step returns its result instead of a file",
    "pinneapple_core.registry": "the name -> object map a plugin registers into, not a step: a registered solver is "
                                "run by the solve nodes and a registered preset by the problem nodes, exactly as before",
    "pinneapple_problemdesign": "conversational problem design with an external LLM agent; the planner and "
                                "problem.generic cover the same ground offline",
    "pinneapple_quantum": "needs a quantum SDK (Qiskit / PennyLane), not installed by the flow extra",
    "pinneapple_worldmodel": "foundation-model training at GPU scale; no CPU-sized entry point",
    "pinneapple_perception": "image / video / audio inputs: no media port type in the flow schema yet",
    "pinneapple_blender": "needs Blender's bpy; viz.render uses it when installed",
    "pinneapple_twin3d": "3-D twin scene server; viz.field3d gives the in-studio 3-D view",
    "pinneapple_hub": "Hugging Face Hub transfers (network access and credentials, not a computation step)",
    "pinneapple_hub.verified_card": "trains or loads a model and issues its model card (a publication step, served "
                                    "by apps/governed_card); the guardrail it runs is reached by the check nodes",
    "pinneapple_hub.library": "the Hub's catalogue of ready-made items (index, snippets, composer); it lists these "
                              "nodes rather than adding steps",
    "pinneapple_catalog": "method / resource catalogue with license gates (read by the planner and docs)",
    "pinneapple_decision": "experiment-selection loop driven by an LLM backend; the planner covers routing",
    "pinneapple_arena": "YAML benchmark campaigns (long runs); single benchmark runs are benchmark.arena nodes",
    "pinneapple_arena.showcase": "one fixed four-architecture Burgers comparison (minutes of training) served by "
                                 "apps/model_benchmarking; single benchmark runs are benchmark.arena nodes",
    "pinneapple_arena.pointwise": "the PINNeAPPle Arena app's training protocol (minutes-long multi-architecture experiments run by apps/pinn_arena); single trainings are pinn / solve nodes",
    "pinneapple_arena.external": "references to third-party libraries run through their own packages (mostly not installed); not a computation step of a flow",
    "pinneapple_pdb": "dataset builder for benchmark publication (sharding, upload)",
    "pinneapple_veriphysics": "report layer over results; decision records + export.report cover it here",
    "pinneapple_design.surrogate_inverse": "inverse design through a surrogate trained per case at start-up "
                                           "(seconds to minutes of MLP training); served by the apps/*_ai apps",
    "pinneapple_design.design_gate": "the staged gate a generated design passes through (geometry, mesh, "
                                     "convergence, physics, objectives) and the record of which stage rejected it: it "
                                     "wraps the steps of a design campaign rather than computing one",
    "pinneapple_check": "the CI physics linter (runs on files in a repository, not in a flow)",
    "pinneapple_lab": "covered by validate.lab (run, curation tiers, seal)",
    "pinneapple_autolab": "mission campaigns of a reference solver (hours of OpenFOAM runs with a budget); run by "
                          "examples/autolab_aero, not as one flow step",
    "pinneapple_llm": "LLM drafting: used by the LLM planner, problem.generic and twin.auto",
    "pinneapple_app": "a separate web application (not a library package)",
    "pinneapple_studio": "the Studio itself: it catalogs every library symbol as a tool and runs tools and examples "
                         "as jobs; a flow calls the tools it lists, not the catalog",
    "pinneapple_autolab.core": "the agent loop (toolbox, planners, execute) run inside autolab.* nodes",
    "pinneapple_reason.llm": "optional LLM advisor of the reasoning engine; flows reason without an LLM",
    "pinneapple_reason.narrative": "the text of Chain.explain, shown in the summary of reason.chain",
    "pinneapple_reason.props": "physical constants and properties read by the failure modes",
    "pinneapple_presets.cases": "the preset classes, run by preset.run / preset.validate",
    "pinneapple_presets.core": "the Preset base class behind preset.run",
    "pinneapple_presets.registry": "the catalogue preset.run chooses from (and the Hub index)",
    "pinneapple_presets.report": "writes the committed preset folders (python -m pinneapple_presets report)",
    "pinneapple_presets.surrogate": "the surrogate behind preset.run with fidelity = surrogate",
    "pinneapple_presets.viz3d": "3-D stills of the committed preset folders",
    "pinneapple_condense.benchmark": "the measured studies of the condensation report (subprocesses, up to an hour; "
                                     "run by examples/physics_condensation)",
    "pinneapple_condense.figures": "figures of the condensation report, drawn from the benchmark JSON",
    "pinneapple_condense.homogenize": "the Keff baseline, compared in the condensation benchmark",
    "pinneapple_condense.learned": "learned block operators (training is a campaign of block condensations); "
                                   "measured in the condensation benchmark",
    "pinneapple_condense.package": "the package description behind condense.package_thermal",
    "pinneapple_condense.substructure": "block condensation behind condense.package_thermal (via the hierarchy)",
    "pinneapple_condense.transient": "Craig-Bampton transient, measured in the condensation benchmark",
    "pinneapple_condense.meshless": "meshless (RBF-FD / GFDM) solution of the package, measured in the condensation "
                                    "benchmark (minutes per point cloud)",
    "pinneapple_condense.exceptions": "prior + exception patches, measured in the condensation benchmark",
    "pinneapple_condense.query": "the package query methods behind query.package",
    "pinneapple_condense.query_benchmark": "the measured query demo of the condensation report (minutes; run by "
                                           "examples/physics_condensation)",
    "pinneapple_query.core": "the query engine and planner behind query.package",
    "pinneapple_query.refs": "physical references parsed by query.package",
    "pinneapple_condense.algebraic": "algebraic condensation of any sparse operator, measured by the condensation "
                                     "falsification campaign (examples/condensation_falsification)",
    "pinneapple_condense.falsify": "the pre-registered falsification studies (subprocesses, minutes to an hour)",
    "pinneapple_condense.flow": "steady MAC-grid flow and heat assembler behind the CFD falsification tests",
    "pinneapple_condense.meshless2d": "2-D meshless discretisations of the CFD falsification tests",
    "pinneapple_analysis.dynamics": "dynamics toolkit (first-passage, Buckingham pi, Fisher, early warning, "
                                    "Koopman) used as a Python API; its flow nodes come with the toolkit's "
                                    "integration pull request",
    "pinneapple_physics.biology": "SBML import and biological ODE / excitable-media models, used through pinneapple_analysis.inverse_problems and examples/biology (a model library, not one flow step)",
    "pinneapple_simulation.granular": "a discrete element solver stepped in a python loop (thousands of steps per "
                                      "case): examples/granular_dem runs the mill, the wear calibration and the "
                                      "breakage test, and tests/test_granular_*.py check each layer",
    "pinneapple_simulation.hemodynamics": "Windkessel outlets and the Windkessel-coupled aorta OpenFOAM run (hours of pimpleFoam, examples/biology/aorta_windkessel.py), not a flow step",
    "pinneapple_analysis.reproduction": "metrics and status rule of examples/paper_reproductions (offline "
                                        "scoring of committed seed results, not a flow step)",
    "pinneapple_physics.paper_problems": "test problems of published papers (grids, references) read by "
                                         "examples/paper_reproductions runners, not a flow step",
    "pinneapple_physics.paper_problems_xtfc": "the X-TFC paper's problems, read by the "
                                              "examples/reproductions/xtfc_schiassi2021 runner, not a flow step",
    "pinneapple_physics.paper_problems_epi": "the compartmental models of the X-TFC parameter-discovery paper, read "
                                             "by the examples/reproductions/schiassi_epi2020 runner, not a flow step",
    "pinneapple_science.report": "writes the committed examples/science folders, manifests and seals "
                                 "(python -m pinneapple_science report)",
    # Evidence, types and checks that cannot be a parameter form. The first four were already unreached on main when
    # the last two arrived (2026-11-22).
    "pinneapple_core.physics_tensor": "the tensor that knows its name, unit, domain and components (#186): it is the "
                                      "type the inputs and outputs of a node are, not a computation of its own",
    "pinneapple_core.catalog": "the read-only catalogue of #1478: it lists what the registry holds and reports where "
                               "the architecture views disagree; it is a query over other modules, not a step",
    "pinneapple_core.reproducibility": "the seed, the environment fingerprint and the twice-run check (#59): it is "
                                       "the evidence a run attaches, and its inputs are a callable and the process, "
                                       "which a flow form cannot hold",
    "pinneapple_data.provenance": "the minimum record a dataset, run or model must carry, and its validator (#39): "
                                  "evidence about a result, the role pinneapple_core.artifact also plays",
    "pinneapple_data.readiness": "the gate that composes schema, units, CAE quality, preflight and VeriPhysics to "
                                 "answer whether a dataset may start a training run (#1491): it runs before the "
                                 "node that trains, which reads its verdict",
    "pinneapple_data.review": "the engineering review record -- a named person approving a specific set of evidence "
                              "(#1493): a judgement and an audit trail, not a computation step",
    "pinneapple_data.registry": "one list of the public datasets and the pre-trained models with the licence, the "
                                "citation and the checksums, and the load of one of them (#528): what it returns is "
                                "the arrays a data node already takes (auto.data.datasets.*), and the choice -- which "
                                "dataset, under which licence -- is made before a flow is drawn and is written into "
                                "the run's record, not recomputed inside it",
    "pinneapple_analysis.decision": "decision-level surrogate validation (#1871): flip rate, ranking preservation, "
                                    "regret and the failure map take the truth and prediction arrays of a whole "
                                    "design sweep, which a flow form cannot hold (the reason "
                                    "pinneapple_data.multifidelity already has)",
    # The five modules the merges of 2026-11-24 brought in without a node and without a reason, which turned
    # tests/test_flow_library_nodes.py red on main for every frente. Each one is a shape a node cannot hold, and the
    # reason says which shape -- the debt that is left is the node, not the sentence.
    "pinneapple_analysis.design_study": "the study specification and run_study of #1036: it takes a whole study "
                                        "(objective, variables with bounds, a runner per point) and a folder, and "
                                        "the app's wizard (#31) is the caller -- a node of it would be the study "
                                        "itself, not a step inside one",
    "pinneapple_analysis.numerics": "the numerical lab of #1048/#1056-#1062: a catalogue of (problem, method) pairs "
                                    "with the order measured against step size and against cost, which the app reads "
                                    "as a table (#29); choosing a method happens before the run, not inside it",
    "pinneapple_analysis.posterior": "posterior sampling with the diagnostics that say whether the posterior is "
                                     "there: HMC and adaptive chains take a log-posterior callable and return chains "
                                     "plus R-hat and effective sample size, which is a study of its own; the node "
                                     "belongs with the inverse-problems one that owns the likelihood",
    "pinneapple_data.co_kriging": "the Kennedy-O'Hagan co-kriging of #103: it takes the arrays of two fidelity levels "
                                  "and returns a fitted surrogate with the discrepancy learned rather than assumed, "
                                  "which is the whole multi-fidelity study -- the sampling step is the one "
                                  "pinneapple_data.multifidelity already covers",
    "pinneapple_neural.physics_model": "the common PhysicsModel interface of #155: it is the type the five model "
                                       "families are, so a node uses it the way a node uses a dataclass, not as a "
                                       "computation of its own",
    "pinneapple_neural.problem_backends": "a problem trained on PyTorch, JAX or PhysicsNeMo (#75): the training "
                                          "itself is the 'pinn' method node (and pp.solve), which is what this "
                                          "backend calls; what it adds is the choice of engine, the artifact and "
                                          "the provenance, which the node already carries",
}


def packages() -> list[str]:
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("pinneapple")
                  and (p / "__init__.py").is_file())


def modules(pkg: str) -> list[str]:
    path = ROOT / pkg
    return sorted(f"{pkg}.{m.name}" for m in pkgutil.iter_modules([str(path)]) if not m.name.startswith("_"))


_MOD = re.compile(r"\bpp\.[a-z_]+|\bpinneapple[a-z_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def node_targets(auto: bool = True) -> dict[str, set[str]]:
    """module path -> node types that call into it (``auto``: with the nodes generated from the public API)."""
    from .registry import REGISTRY
    out: dict[str, set[str]] = {}
    pp = {"pp.solve": "pinneapple_physics.solving", "pp.trust_card": "pinneapple_analysis.trust",
          "pp.sobol_indices": "pinneapple_analysis.sensitivity", "pp.viz": "pinneapple_tools.visualization"}
    for s in REGISTRY.specs():
        if s.type.startswith("custom.") or (not auto and s.type.startswith("auto.")):
            continue                              # user-defined types reuse allowlisted calls; not library coverage
        for m in _MOD.findall(s.calls or ""):
            if m.startswith("pp."):
                out.setdefault("pinneapple", set()).add(s.type)       # the pp namespace itself
            m = pp.get(m, m)
            out.setdefault(m, set()).add(s.type)
    return out


def coverage(auto: bool = True) -> list[dict[str, str]]:
    """One row per package and first-level module: the node types that reach it, or why there are none (a curated
    reason, or the one the public-API scan of :mod:`pinneapple_flow.autocalls` recorded)."""
    targets = node_targets(auto)
    reasons = dict(REASONS)
    if auto:
        from .autocalls import load
        reasons.update(load()["reasons"])
    rows = []
    for pkg in packages():
        mods = modules(pkg) or [pkg]
        for m in [pkg] + [x for x in mods if x != pkg]:
            nodes = sorted({t for k, v in targets.items() for t in v if k == m or k.startswith(m + ".")})
            reason = reasons.get(m) or (reasons.get(pkg) if m != pkg else "") or ""
            rows.append({"module": m, "nodes": ", ".join(nodes), "reason": reason if not nodes else ""})
    return rows


def counts(rows: list[dict[str, str]] | None = None) -> dict[str, int]:
    rows = coverage() if rows is None else rows
    return {"modules": len(rows), "with_nodes": sum(1 for r in rows if r["nodes"]),
            "with_reason": sum(1 for r in rows if not r["nodes"] and r["reason"]),
            "unreached": sum(1 for r in rows if not r["nodes"] and not r["reason"])}


def markdown() -> str:
    rows = coverage()
    c = counts(rows)
    from .autocalls import load
    auto = load()["calls"]
    ok = sum(a.get("check", "").startswith("runs") for a in auto)
    out = ["# Flow Studio coverage of the library", "",
           f"**{c['modules']} packages / modules: {c['with_nodes']} with nodes, {c['with_reason']} with a stated "
           f"reason, {c['unreached']} unreached.**", "",
           "Generated by `python -m pinneapple_flow coverage` from the node registry (each node declares the library "
           "calls it wraps), the curated reasons in `pinneapple_flow/coverage.py` and the public-API scan of "
           "`pinneapple_flow/autocalls.py` (`python -m pinneapple_flow autocalls`). Curated nodes come first in each "
           f"row; `auto.*` nodes are generated from type-hinted public functions ({len(auto)} nodes, {ok} of them run "
           "with their default form values, the rest need input: each node shows its check). A module without a node "
           "says why: a curated reason, or what the scan found (only classes, or functions that take objects, files "
           "or code). `tests/test_flow_library_nodes.py` fails when a public module is neither.", "",
           "| package / module | node types | why not a node |", "|---|---|---|"]
    for r in rows:
        nodes = r["nodes"].split(", ") if r["nodes"] else []
        cur = [n for n in nodes if not n.startswith("auto.")]
        au = [n for n in nodes if n.startswith("auto.")]
        cell = ", ".join(cur + ([f"{len(au)} auto (" + ", ".join(a.rsplit(".", 1)[1] for a in au[:6])
                                 + (", ..." if len(au) > 6 else "") + ")"] if au else []))
        out.append(f"| `{r['module']}` | {cell} | {r['reason']} |")
    return "\n".join(out) + "\n"
