"""The composer: a brief in, a study pipeline out, with every branch decided by a PINNeAPPle recommendation API and
nothing decided by a language model.

The pipeline answers the ten questions in a fixed order, and each answer is either "you already have it" or "this is
how you build it, with these calls, at this estimated cost, checked this way":

1. **physics** -- always first: the equations, the initial and boundary conditions are confirmed (or composed when
   no family covers them) before a single solve.
2. **geometry** -- only when the domain is a real shape. An analytic box or cylinder is not a CAD problem, and
   saying otherwise is how a plan becomes busywork.
3. **mesh** -- a real shape needs one; 3-D needs one even for a canonical domain (the mesher builds it from its
   parameters); a 1-D/2-D analytic domain does not (the solver samples it).
4. **have or generate** -- geometry and mesh are imported when the user has them and generated when they do not.
5. **dataset** -- imported when the user has it, generated when they do not.
6. **solver** -- :func:`pinneapple_analysis.verification.solver_orchestration.select_solver_family` names the family
   that is installed here and documents this physics and dimension, and the cost model
   (:mod:`pinneapple_advisor.estimate`) puts a time and a price per solve on it.
7. **solver or surrogate** -- decided by the numbers: solves x evaluations against the direct budget and against the
   samples a surrogate would need. Cheap -> sweep the solver. Expensive -> train, and the decision string comes from
   :func:`...architecture_recommendation.recommend_architecture`, which can also answer "none".
8. **architecture** -- the same API ranks PINN, XPINN, DeepONet, FNO, MeshGraphNet, PINO and inverse PINN from the
   data on hand, whether parameters/geometry vary, and what the model must return.
9. **dataset recipe** -- a design of experiments with its coverage metric, a test region instead of random rows, and
   the cost of the campaign.
10. **checks and views** -- every stage carries what it must pass and what to look at, so the pipeline is a study
    plan and not a wish list.

Inverse problems add the identifiability check before the fit and the inverse solve itself; optimisation adds the
search on the surrogate with the Pareto front and the verification of the chosen design. Both are added only when
the brief asks for them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import capabilities as cap
from .brief import Brief
from .estimate import DEFAULT as DEFAULT_COST
from .estimate import CostModel
from .questions import next_questions
from .schema import Decision, Estimate, Stage, StudyPlan, Tool, bi

# A study that needs no more than this much solver time is done by the solver itself (10 minutes: the same default
# architecture_recommendation uses when the caller gives no budget).
DIRECT_BUDGET_S = 600.0

# Probing what is installed (OpenFOAM, FEniCS, the internal solvers) runs subprocesses: 17 s cold, 2 s warm on the
# machine this was written on, so the answer is kept for the process and refreshed on request.
FAMILIES_TTL_S = 600.0
_FAMILIES: dict[str, Any] = {"at": 0.0, "value": None}

# Samples a surrogate needs, when the brief does not say: about ten per input for a scalar target (Loeppky, Sacks &
# Welch 2009, "Choosing the sample size of a computer experiment"), and the operator-learning threshold
# architecture_recommendation itself uses for a field.
SAMPLES_PER_INPUT_SCALAR = 10
SAMPLES_FIELD = 50
GEOMETRY_VARIATION_FACTOR = 2.0

# Default evaluations per goal, when the brief has no number: a single forward prediction is one solve; an
# optimisation searches; everything else runs the dataset the surrogate needs.
DEFAULT_OPTIMIZATION_EVALUATIONS = 200

# The inverse scheme runs a solve per outer iteration (field, then parameter): 20 is a first fit, not a promise.
INVERSE_SOLVES = 20

# One surrogate inference, in seconds: a forward pass of a small operator on one case.
SURROGATE_EVALUATION_S = 1e-3


@dataclass
class Composition:
    """The plan plus the raw recommendations it was built from, so a caller can inspect what the plan compressed."""
    plan: StudyPlan
    solver: Any | None = None
    architecture: Any | None = None


def build(brief: Brief, *, cost: CostModel | None = None, available_families: dict[str, Any] | None = None,
          solve_seconds_measured: float | None = None, language: str = "pt") -> Composition:
    """Compose the study pipeline for ``brief``.

    ``available_families`` reuses a previous ``list_available_solver_families()`` (it probes binaries on PATH, so an
    API can cache it); ``solve_seconds_measured`` overrides the cost model with a real measurement.
    """
    cost = cost or DEFAULT_COST
    family = brief.values.get("physics_family") or "custom"
    dim = brief.values.get("dimension") or _dimension_hint(family) or 2
    time_dependent = bool(brief.get("time_dependent"))
    nonlinear = bool(brief.get("nonlinear"))
    turbulent = bool(brief.get("turbulent"))
    goal = brief.get("goal") or "forward"
    cores = int(brief.get("cores") or 1)
    cells = brief.values.get("approx_cells")

    stages: list[Stage] = []
    decisions: list[Decision] = []
    warnings: list[dict[str, str]] = []

    # ---- 1. the physical model ----------------------------------------------------------------------------------
    stages.append(_physics_stage(brief, family, dim, time_dependent, decisions))

    # ---- 2/3/4. geometry and mesh ------------------------------------------------------------------------------
    real_shape = bool(brief.values.get("geometry_present")) or bool(brief.values.get("geometry_varies"))
    mesh_present = bool(brief.values.get("mesh_present"))
    geometry_stage, geometry_needed = _geometry_stage(brief, dim, real_shape, decisions)
    if geometry_stage is not None:
        stages.append(geometry_stage)
    mesh_stage = _mesh_stage(brief, dim, real_shape, mesh_present, geometry_needed, cells, decisions)
    if mesh_stage is not None:
        stages.append(mesh_stage)

    # ---- 6. the solver, and what one solve costs ---------------------------------------------------------------
    solver_rec, solver_warning = _solver_recommendation(brief, family, dim, time_dependent,
                                                        available_families=available_families)
    if solver_warning:
        warnings.append(solver_warning)
    has_solver = solver_rec is not None
    method = getattr(solver_rec, "recommended_family", "") if solver_rec else ""

    # ---- 5/7. dataset or not, and how many evaluations ---------------------------------------------------------
    dataset_present = bool(brief.values.get("dataset_present"))
    n_samples = _dataset_size(brief, dim)
    n_evaluations, evaluations_source = _evaluations(brief, goal, n_samples)
    measured = solve_seconds_measured or brief.values.get("solve_seconds_measured")
    solve_est = cost.solve(family=family, dim=dim, time_dependent=time_dependent, nonlinear=nonlinear,
                           turbulent=turbulent, cells=cells, cores=cores, n_evaluations=1,
                           measured_solve_seconds=measured,
                           budget_hours=brief.values.get("budget_hours"))
    budget_s = _direct_budget(brief)

    # Two calls, on purpose: the first answers "sweep the solver or train at all" with the data on hand; the second,
    # only when training wins, answers "which architecture" with the dataset the plan is about to build.
    first, arch_warning = _architecture_recommendation(brief, has_solver=has_solver,
                                                      solve_seconds=solve_est.seconds_per_evaluation,
                                                      n_evaluations=n_evaluations, direct_budget_s=budget_s,
                                                      goal=goal, n_simulations=None)
    if arch_warning:
        warnings.append(arch_warning)
    direct_only = bool(first is not None and getattr(first, "recommended", None) and
                       first.recommended[0] == "none")
    if direct_only and dataset_present:
        direct_only = False
        decisions.append(Decision(
            topic="direct_or_surrogate", choice="surrogate", source="brief",
            why=bi("Você já tem o dataset: a comparação entre rodar o solver e treinar aponta o solver só porque "
                   "gerar dados novos sairia caro, e isso não se aplica quando os dados já existem -- o passo de "
                   "treino fica.",
                   "You already have the dataset: the solver-versus-training comparison points at the solver only "
                   "because generating new data would be expensive, which does not apply when the data exists -- "
                   "the training step stays."),
            alternatives=["descartar o dataset e rodar o solver em cada caso"]))
    architecture = first
    if not direct_only:
        # The second call asks only the architecture axis: with ``solve_seconds=None`` the "call the solver" answer
        # is out of the question, and the dataset the plan will build counts as data on hand.
        architecture, second_warning = _architecture_recommendation(
            brief, has_solver=has_solver, solve_seconds=None,
            n_evaluations=n_evaluations, direct_budget_s=budget_s, goal=goal,
            n_simulations=max(int(brief.values.get("n_simulations_present") or 0), n_samples))
        if second_warning:
            warnings.append(second_warning)
    arch_choice = (architecture.recommended[0] if getattr(architecture, "recommended", None) else "pinn")

    # One solver run when a campaign follows (it calibrates the cost and gives the reference field); every
    # evaluation when the solver is the answer, because then the campaign is the solver.
    solver_evaluations = n_evaluations if direct_only else 1
    solver_stage = _solver_stage(brief, family, dim, method, solver_rec, solve_est, solver_evaluations,
                                 evaluations_source, direct_only, n_evaluations, decisions)
    stages.append(solver_stage)
    dataset_stage = _dataset_stage(brief, real_shape, n_samples, solve_est, direct_only, dataset_present,
                                   decisions)
    if dataset_stage is not None:
        stages.append(dataset_stage)

    training_stage = _training_stage(brief, arch_choice, n_samples, cost, direct_only, decisions,
                                     dataset_present, dataset_stage is not None)
    if training_stage is not None:
        stages.append(training_stage)

    # ---- inverse / optimisation, when the brief asks for them ---------------------------------------------------
    if goal == "inverse":
        # 20 alternating field/parameter solves: the outer loop of the inverse scheme, not 20 different cases.
        inverse_est = Estimate(
            seconds_per_evaluation=solve_est.seconds_per_evaluation * INVERSE_SOLVES,
            n_evaluations=1, total_seconds=solve_est.seconds_per_evaluation * INVERSE_SOLVES,
            cost_usd=(solve_est.cost_usd * INVERSE_SOLVES) if solve_est.cost_usd else None, cells=solve_est.cells,
            basis=f"{INVERSE_SOLVES} solver runs x {solve_est.seconds_per_evaluation:.4g} s "
                  f"({solve_est.basis})",
            confidence=solve_est.confidence,
            assumptions=list(solve_est.assumptions) + [f"{INVERSE_SOLVES} alternating field/parameter solves "
                                                       f"(the outer loop of the scheme; more iterations = more runs)"])
        stages.append(_inverse_stage(solver_stage.id, dim, inverse_est))
        decisions.append(Decision(
            topic="goal", choice="inverse", source="brief",
            why=bi("O objetivo é recuperar um parâmetro a partir de medições, então a identificabilidade é checada "
                   "antes do ajuste: um parâmetro que não é identificável com estas medições não é recuperado por "
                   "nenhum otimizador.",
                   "The goal is to recover a parameter from measurements, so identifiability is checked before the "
                   "fit: a parameter that these measurements cannot identify is not recovered by any optimiser."),
            alternatives=["forward (train on simulations)", "optimization (search a design)"]))
    if goal == "optimization":
        parent = "training" if training_stage is not None else solver_stage.id
        # A search on the surrogate is thousands of cheap evaluations; a direct search pays the solver for each one.
        if training_stage is not None:
            optimization_est = Estimate(
                seconds_per_evaluation=SURROGATE_EVALUATION_S, n_evaluations=DEFAULT_OPTIMIZATION_EVALUATIONS,
                total_seconds=SURROGATE_EVALUATION_S * DEFAULT_OPTIMIZATION_EVALUATIONS, cost_usd=None, cells=None,
                basis=f"{DEFAULT_OPTIMIZATION_EVALUATIONS} evaluations x {SURROGATE_EVALUATION_S * 1e3:g} ms "
                      f"(surrogate inference, not a solve)",
                confidence="order of magnitude",
                assumptions=["the search runs entirely on the trained surrogate; the chosen optimum is re-run on "
                             "the solver afterwards"])
        else:
            optimization_est = solve_est
        stages.append(_optimization_stage(parent, optimization_est))
        decisions.append(Decision(
            topic="goal", choice="optimization", source="brief",
            why=bi("O objetivo é otimizar um projeto: a busca roda sobre o surrogate (ou sobre o solver, se a "
                   "avaliação direta couber no orçamento) e o ótimo é re-verificado com o solver.",
                   "The goal is to optimise a design: the search runs on the surrogate (or on the solver, when "
                   "direct evaluation fits the budget) and the optimum is re-verified with the solver."),
            alternatives=["forward", "screening"]))

    # ---- 10. what to check, and what to look at -----------------------------------------------------------------
    last_compute = stages[-1].id
    stages.append(_verification_stage(brief, last_compute, dim))
    stages.append(_visualization_stage(brief, "verification", dim))

    # ---- warnings -----------------------------------------------------------------------------------------------
    if family == "custom" or not brief.values.get("physics_family"):
        warnings.append(bi("A física não foi identificada com confiança: o primeiro passo compõe o modelo a partir "
                           "das leis e o confirma antes de qualquer simulação.",
                           "The physics was not identified with confidence: the first step composes the model from "
                           "laws and confirms it before any simulation."))
    if measured is None:
        warnings.append(bi("Tempo e custo são estimativas de ordem de grandeza (célula x iteração x fator de "
                           "paralelismo), não medidas nesta máquina: rode um caso e informe `solve_seconds_measured` "
                           "para o plano passar a usar o seu número.",
                           "Time and cost are order-of-magnitude estimates (cells x iterations x parallel factor), "
                           "not measured on this machine: run one case and pass `solve_seconds_measured` so the "
                           "plan uses your number."))
    budget_hours = brief.values.get("budget_hours")
    campaign = dataset_stage.estimate if dataset_stage is not None and dataset_stage.estimate else None
    if budget_hours and campaign and campaign.total_seconds > float(budget_hours) * 3600:
        warnings.append(bi(f"A campanha estimada ({campaign.total_seconds / 3600:.1f} h) passa do orçamento de "
                           f"{float(budget_hours):g} h: reduza o número de amostras, ou use o solver só nos casos "
                           f"que o surrogate não cobrir.",
                           f"The estimated campaign ({campaign.total_seconds / 3600:.1f} h) exceeds the "
                           f"{float(budget_hours):g} h budget: cut the number of samples, or use the solver only "
                           f"on the cases the surrogate does not cover."))
    if brief.values.get("geometry_varies") and not dataset_present and not direct_only:
        warnings.append(bi("A geometria varia entre os casos: cada amostra do dataset paga uma geometria e uma "
                           "malha novas, e o custo por amostra é o do item 3 somado ao do item 6.",
                           "The geometry varies between cases: every dataset sample pays for a new geometry and a "
                           "new mesh, so the cost per sample is item 3 plus item 6."))

    plan = StudyPlan(
        problem=bi(brief.description or "", brief.description or ""),
        stages=stages, decisions=decisions, open_questions=next_questions(brief), warnings=warnings,
        brief=brief.to_dict(),
        provenance={"schema": "pinneapple.advisor/1", "language": language,
                    "cost_model": "order of magnitude" if measured is None else "measured",
                    "solver_family": method or None,
                    "architecture": list(getattr(architecture, "recommended", []) or []),
                    "n_samples": n_samples, "n_evaluations": n_evaluations,
                    "evaluations_source": evaluations_source,
                    "brief_sources": dict(brief.sources)})
    return Composition(plan=plan, solver=solver_rec, architecture=architecture)


# ---------------------------------------------------------------------------------------------------- stages

def _physics_stage(brief: Brief, family: str, dim: int, time_dependent: bool,
                   decisions: list[Decision]) -> Stage:
    fam = _family(family)
    title_pt = "Confirmar o modelo físico (equações, condições iniciais e de contorno)"
    if fam is not None:
        equations = fam.description
        why = bi(f"A descrição aponta para `{fam.kind}` ({equations}). Antes de simular, confirme a forma das "
                 f"equações, os coeficientes e uma condição em cada fronteira"
                 + (" e a condição inicial no tempo" if time_dependent else "") +
                 ". Se não for esta a física, o passo sugere a correta a partir das leis.",
                 f"The description points to `{fam.kind}` ({equations}). Before simulating, confirm the form of the "
                 f"equations, the coefficients and one condition on every boundary"
                 + (" and the initial condition in time" if time_dependent else "") +
                 ". If this is not the physics, the step suggests the right one from the laws.")
    else:
        why = bi("A física não foi identificada com confiança. O passo compõe o sistema a partir das leis de "
                 "conservação (equações, fechamentos, unidades, uma condição por fronteira) e o submete às "
                 "checagens -- um modelo mal posto é o erro mais caro de todos.",
                 "The physics was not identified with confidence. The step composes the system from conservation "
                 "laws (equations, closures, units, one condition per boundary) and runs the checks -- a "
                 "badly-posed model is the most expensive mistake of all.")
    decisions.append(Decision(
        topic="physics_family", choice=family,
        why=bi(f"Modelo físico assumido: `{family}` em {dim}-D"
               + (", transiente" if time_dependent else ", estacionário") + ".",
               f"Physical model assumed: `{family}` in {dim}-D"
               + (", transient" if time_dependent else ", steady") + "."),
        alternatives=[f.kind for f in _all_families() if f.kind != family][:4],
        values={"dimension": dim, "time_dependent": time_dependent,
                "source": brief.source("physics_family")},
        source="pinneapple_physics.pde_environment.capabilities.identify_pde"))
    return Stage(
        id="physics", kind="physics", title=bi(title_pt, "Confirm the physical model (equations, initial and "
                                                          "boundary conditions)"),
        why=why, status="confirm", answers=[1],
        tools=list(cap.PHYSICS_CONFIRM), inputs=["the problem description"],
        outputs=["equações, coeficientes, condições de contorno e iniciais confirmados",
                 "o ProblemSpec que todas as etapas seguintes usam"],
        checks=[{"what": "cada termo de cada equação e condição tem uma dimensão (unidades)",
                 "tool": "pinneapple_problemdesign.formulate:ModelDraft.checks"},
                {"what": "o sistema é fechado: campos contra equações, todo coeficiente definido",
                 "tool": "pinneapple_problemdesign.formulate:ModelDraft.checks"},
                {"what": "segunda ordem com condição em cada face; campo transiente com condição inicial",
                 "tool": "pinneapple_problemdesign.formulate:ModelDraft.checks"},
                {"what": "solução manufaturada: o resíduo compilado concorda com o SymPy",
                 "tool": "pinneapple_problemdesign.formulate:ModelDraft.checks"}],
        views=[{"what": "o sistema em LaTeX, para conferir com quem conhece a física",
                "tool": "pinneapple_problemdesign.formulate:ModelDraft"}],
        question="" if brief.values.get("physics_family") else "physics_family")


def _geometry_stage(brief: Brief, dim: int, real_shape: bool,
                    decisions: list[Decision]) -> tuple[Stage | None, bool]:
    """The geometry step, and whether a mesh stage may depend on it."""
    present = bool(brief.values.get("geometry_present"))
    varies = bool(brief.values.get("geometry_varies"))
    formats = list(brief.values.get("geometry_formats") or [])
    if dim == 1:
        decisions.append(Decision(topic="geometry", choice="none", values={"dimension": dim},
                                  why=bi("1-D: o domínio é um intervalo. Não há geometria a importar nem a gerar.",
                                         "1-D: the domain is an interval. There is no geometry to import or build."),
                                  source="brief"))
        return None, False
    if present:
        status = "have"
        tools = list(cap.GEOMETRY_IMPORT)
        title = bi("Importar e reparar a sua geometria", "Import and repair your geometry")
        why = bi(f"Você já tem a geometria ({', '.join(formats) if formats else 'formato a confirmar'}): o passo "
                 f"lê, converte e verifica se ela é fechada e malhável, em vez de gerar outra.",
                 f"You already have the geometry ({', '.join(formats) if formats else 'format to confirm'}): the "
                 f"step reads, converts and checks that it is closed and meshable instead of building another one.")
    elif varies:
        status = "generate"
        tools = list(cap.GEOMETRY_GENERATE) + list(cap.GEOMETRY_IMPORT)
        title = bi("Gerar a família de geometrias", "Generate the geometry family")
        why = bi("A geometria varia entre os casos: ela é paramétrica (uma descrição, muitas formas), e o "
                 "parâmetro de forma entra nas amostras do dataset.",
                 "The geometry varies between cases: it is parametric (one description, many shapes), and the shape "
                 "parameter enters the dataset samples.")
    elif dim == 3:
        decisions.append(Decision(topic="geometry", choice="analytic", values={"dimension": dim},
                                  why=bi("3-D, mas sem geometria própria: conta como domínio canônico (caixa, "
                                         "cilindro, canal). Só entra CAD se a forma não for uma primitiva.",
                                         "3-D, but no geometry of its own: it counts as a canonical domain (box, "
                                         "cylinder, channel). CAD only enters if the shape is not a primitive."),
                                  source="brief"))
        return None, False
    else:
        decisions.append(Decision(topic="geometry", choice="analytic", values={"dimension": dim},
                                  why=bi("2-D com domínio analítico (placa, canal, seção): o solver amostra o "
                                         "domínio, sem CAD.",
                                         "2-D with an analytic domain (plate, channel, section): the solver samples "
                                         "the domain, no CAD."), source="brief"))
        return None, False
    decisions.append(Decision(
        topic="geometry", choice=status, values={"dimension": dim, "geometry_varies": varies,
                                                 "formats": formats},
        why=bi(f"Geometria: {'importada' if present else 'gerada'}"
               + (", variando entre os casos" if varies else ", uma forma fixa") + ".",
               f"Geometry: {'imported' if present else 'generated'}"
               + (", varying between cases" if varies else ", one fixed shape") + "."),
        alternatives=["domínio analítico (sem CAD)", "versão paramétrica da mesma forma"],
        source="brief"))
    return Stage(
        id="geometry", kind="geometry", title=title, why=why, status=status, answers=[4, 2],
        tools=tools, inputs=["a descrição da forma" if not present else "os arquivos CAD/STL/STEP"],
        outputs=["superfície fechada e classificada, num formato que o gerador de malha lê"],
        checks=[{"what": "a superfície é fechada (sem bordas abertas) e cada região foi identificada",
                 "tool": "pinneapple_analysis.verification.geometry_intelligence:classify_geometry"},
                {"what": "as dimensões conferem com as do desenho ou da peça real",
                 "tool": "pinneapple_design.intent:design"}],
        views=[{"what": "a geometria e as regiões, antes de malhar",
                "tool": "pinneapple_design.geometry_library:export_web"}],
        question="" if present or varies else "geometry_present"), True


def _mesh_stage(brief: Brief, dim: int, real_shape: bool, mesh_present: bool, geometry_needed: bool,
                cells: int | None, decisions: list[Decision]) -> Stage | None:
    turbulent = bool(brief.get("turbulent"))
    if mesh_present:
        decisions.append(Decision(
            topic="mesh", choice="have", values={"dimension": dim},
            why=bi("Você já tem a malha: o passo lê a malha existente e mede a qualidade dela antes de rodar "
                   "qualquer coisa nela.",
                   "You already have the mesh: the step reads the existing mesh and measures its quality before "
                   "anything runs on it."), source="brief"))
        return Stage(
            id="mesh", kind="mesh", title=bi("Importar e avaliar a sua malha", "Import and assess your mesh"),
            why=bi("A malha está pronta; o que falta é saber se ela serve: elemento invertido, razão de aresta, "
                   "ortogonalidade e, se houver turbulência, y+ na parede.",
                   "The mesh is ready; what is missing is knowing whether it serves: inverted elements, edge ratio, "
                   "orthogonality and, with turbulence, y+ at the wall."),
            status="have", answers=[4, 3], tools=list(cap.MESH_IMPORT) + list(cap.MESH_CHECK),
            inputs=["a malha existente"], outputs=["relatório de qualidade da malha"],
            checks=_mesh_checks(turbulent), views=[{"what": "a malha colorida pela qualidade do elemento",
                                                    "tool": "pinneapple_design.geometry_library:export_web"}])
    if real_shape or dim == 3:
        status = "generate"
        why = bi(
            "Sem malha o solver não vê a forma: o passo gera a malha a partir da geometria"
            + (" (3-D: tetraedros, com camada limite nas paredes)" if dim == 3 else " (2-D: triângulos)")
            + (" e a refina até o y+ exigido pela turbulência." if turbulent else "."),
            "Without a mesh the solver cannot see the shape: the step generates it from the geometry"
            + (" (3-D: tetrahedra, with a boundary layer at the walls)" if dim == 3 else " (2-D: triangles)")
            + (" and refines it to the y+ the turbulence model needs." if turbulent else "."))
        if not real_shape:
            why = bi("3-D sem CAD: o domínio canônico ainda precisa de uma malha volumétrica, e o gerador a "
                     "constrói a partir dos parâmetros da forma.",
                     "3-D without CAD: the canonical domain still needs a volumetric mesh, and the mesher builds it "
                     "from the shape's parameters.")
        decisions.append(Decision(
            topic="mesh", choice="generate", values={"dimension": dim, "cells": cells, "turbulent": turbulent},
            why=bi("Malha gerada"
                   + (f" com cerca de {cells:,} células" if cells else " na resolução padrão (informe a sua)")
                   + ".",
                   "Mesh generated" + (f" with about {cells:,} cells" if cells else " at the default resolution "
                                        "(give yours)") + "."),
            alternatives=["malha sem conformidade (células cortadas) para geometria suja",
                          "malha do próprio usuário"],
            source="brief"))
        return Stage(
            id="mesh", kind="mesh", title=bi("Gerar a malha", "Generate the mesh"), why=why, status=status,
            answers=[3], tools=list(cap.MESH_GENERATE) + list(cap.MESH_CHECK),
            inputs=["a geometria (importada ou gerada)"],
            outputs=["malha não estruturada com a camada limite", "relatório de qualidade e y+"],
            depends_on=["geometry"] if geometry_needed else [], checks=_mesh_checks(turbulent),
            views=[{"what": "a malha colorida pela qualidade do elemento",
                    "tool": "pinneapple_design.geometry_library:export_web"}],
            question="" if cells else "approx_cells")
    decisions.append(Decision(
        topic="mesh", choice="none", values={"dimension": dim},
        why=bi("O domínio é analítico e o solver o amostra direto (grade estruturada ou pontos de colocação): "
               "nenhuma malha a gerar.",
               "The domain is analytic and the solver samples it directly (a structured grid or collocation "
               "points): no mesh to generate."), source="brief"))
    return None


def _mesh_checks(turbulent: bool) -> list[dict[str, str]]:
    checks = [
        {"what": "qualidade de cada elemento: SICN, gama, razão de aresta", "tool": "mesh_intelligence"},
        {"what": "convergência de malha (GCI) em pelo menos dois níveis de refino",
         "tool": "pinneapple_autolab.study.mesh_study:grid_convergence"},
    ]
    if turbulent:
        checks.insert(1, {"what": "primeira célula na altura que o y+ alvo exige",
                          "tool": "pinneapple_analysis.verification.mesh_intelligence:"
                                  "estimate_required_first_cell_height"})
    return checks


def _solver_stage(brief: Brief, family: str, dim: int, method: str, solver_rec: Any, solve_est: Estimate,
                  n_evaluations: int, evaluations_source: str, direct_only: bool, study_evaluations: int,
                  decisions: list[Decision]) -> Stage:
    rec_family = method or "pinn"
    reasoning = getattr(solver_rec, "reasoning", "") if solver_rec is not None else ""
    caveats = list(getattr(solver_rec, "caveats", []) or [])
    decisions.append(Decision(
        topic="solver", choice=rec_family, source="solver_orchestration.select_solver_family",
        why=bi(reasoning or "Nenhuma família clássica instalada cobre esta física: o PINN é o caminho, treinando "
                            "sobre o resíduo da EDP.",
               reasoning or "No installed classical family covers this physics: the PINN path is the one left, "
                            "training on the PDE residual."),
        alternatives=list(getattr(solver_rec, "fallback_families", []) or []),
        values={"dimension": dim, "family": family, "caveats": caveats}))
    total = solve_est
    count_note = (f"x {n_evaluations} evaluations ({evaluations_source})" if direct_only
                  else f"one run to calibrate the cost and give the reference field; the campaign runs "
                       f"{study_evaluations} solves in the dataset stage ({evaluations_source})")
    est = Estimate(seconds_per_evaluation=total.seconds_per_evaluation, n_evaluations=n_evaluations,
                   total_seconds=total.seconds_per_evaluation * n_evaluations,
                   cost_usd=(total.cost_usd or 0.0) * n_evaluations if total.cost_usd is not None else None,
                   cells=total.cells, basis=f"{total.basis}; {count_note}",
                   confidence=total.confidence, assumptions=list(total.assumptions))
    return Stage(
        id="solver", kind="solver",
        title=(bi("Rodar o solver escolhido em cada caso", "Run the chosen solver on every case") if direct_only else
               bi("Rodar o solver escolhido e medir o custo real de uma execução",
                  "Run the chosen solver and measure what one run really costs")),
        why=(bi(f"A família recomendada é `{rec_family}`. O custo direto cabe no orçamento, então o solver "
                f"responde cada caso: são {study_evaluations} execuções, sem surrogate.",
                f"The recommended family is `{rec_family}`. The direct cost fits the budget, so the solver answers "
                f"every case: {study_evaluations} runs, no surrogate.")
             if direct_only else
             bi(f"A família recomendada é `{rec_family}`. Uma execução resolve o caso, dá o campo de referência e "
                f"calibra o custo: a partir dela o tempo estimado deixa de ser ordem de grandeza.",
                f"The recommended family is `{rec_family}`. One run solves the case, gives the reference field and "
                f"calibrates the cost: from it on the estimate stops being an order of magnitude.")),
        status="generate", answers=[6], tools=list(cap.SOLVER_SELECT) + list(cap.SOLVER_RUN),
        inputs=["o ProblemSpec confirmado", "a malha (ou o domínio analítico)"],
        outputs=["o campo (ou o KPI) de cada caso", "o custo medido por execução"],
        depends_on=["physics"] + (["mesh"] if brief.values.get("mesh_present") or _has_mesh(brief) else []),
        estimate=est, checks=list(_solver_checks()), views=list(cap.VIEWS["field"])[:1],
        question="" if brief.values.get("solve_seconds_measured") else "solve_seconds_measured")


def _dataset_stage(brief: Brief, real_shape: bool, n_samples: int, solve_est: Estimate, direct_only: bool,
                   dataset_present: bool, decisions: list[Decision]) -> Stage | None:
    if direct_only:
        decisions.append(Decision(
            topic="dataset", choice="none", source="architecture_recommendation.recommend_architecture",
            why=bi("O custo direto cabe no orçamento: não há dataset a gerar, o próprio solver responde cada caso.",
                   "The direct cost fits the budget: there is no dataset to generate, the solver answers each case."),
            values={"samples": 0}))
        return None
    if dataset_present:
        n_have = brief.values.get("n_simulations_present")
        decisions.append(Decision(
            topic="dataset", choice="have", values={"n_simulations_present": n_have},
            why=bi(f"Você já tem o dataset{f' ({n_have} simulações)' if n_have else ''}: ele entra no schema único "
                   f"da biblioteca e passa pelos mesmos portões, em vez de gerar mais simulações.",
                   f"You already have the dataset{f' ({n_have} simulations)' if n_have else ''}: it goes into the "
                   f"library's one schema and through the same gates, instead of running more simulations."),
            source="brief"))
        return Stage(
            id="dataset", kind="dataset", title=bi("Importar o seu dataset", "Import your dataset"),
            why=decisions[-1].why, status="have", answers=[5, 9], tools=list(cap.DATASET_HAVE),
            inputs=["o dataset existente"], outputs=["PhysicsDataBundle com unidades e proveniência"],
            checks=[{"what": "portões de dados: unidades, nó/célula, licença, SHA-256",
                     "tool": "pinneapple_data.schema:run_gates"},
                    {"what": "a divisão de teste é uma região do espaço de entrada, não linhas sorteadas",
                     "tool": "pinneapple_autolab.study.dataset:region_split"}],
            views=list(cap.VIEWS["study"]))
    per_case = solve_est.seconds_per_evaluation * (2.0 if real_shape and brief.values.get("geometry_varies") else 1.0)
    total = per_case * n_samples
    estimate = Estimate(
        seconds_per_evaluation=per_case, n_evaluations=n_samples, total_seconds=total,
        cost_usd=(solve_est.cost_usd / max(solve_est.seconds_per_evaluation, 1e-9) * total) if solve_est.cost_usd
        else None,
        cells=solve_est.cells,
        basis=(f"{n_samples} simulações x {per_case:.4g} s cada ({solve_est.basis})"
               + (" + geometria e malha por amostra" if real_shape and brief.values.get("geometry_varies") else "")),
        confidence=solve_est.confidence, assumptions=list(solve_est.assumptions))
    decisions.append(Decision(
        topic="dataset", choice="generate", values={"samples": n_samples, "doe": "maximin"},
        why=bi(f"{n_samples} simulações, num desenho maximin sobre o espaço dos parâmetros, com uma região do "
               f"espaço reservada para teste em vez de linhas sorteadas.",
               f"{n_samples} simulations in a maximin design over the parameter space, with a region of the space "
               f"kept for testing instead of random rows."),
        alternatives=["amostragem aleatória (pior cobertura para o mesmo custo)", "grade regular (pior em dimensão "
                      "alta)"],
        source="pinneapple_autolab.study.campaign.plan_doe"))
    return Stage(
        id="dataset", kind="dataset",
        title=bi(f"Gerar o dataset de treino ({n_samples} simulações)",
                 f"Generate the training dataset ({n_samples} simulations)"),
        why=bi("O surrogate só aprende o que o dataset cobre: o desenho de experimentos decide quais casos rodar, a "
               "cobertura (discrepância, distância mínima) diz se o espaço foi bem varrido, e a região de teste "
               "mede extrapolação, não interpolação.",
               "A surrogate only learns what the dataset covers: the design of experiments decides which cases to "
               "run, the coverage (discrepancy, minimum distance) says whether the space was swept, and the test "
               "region measures extrapolation, not interpolation."),
        status="generate", answers=[5, 6, 9], tools=list(cap.DATASET_PLAN) + list(cap.DATASET_BUILD),
        inputs=["o espaço dos parâmetros (faixas)", "o solver e o seu custo por caso"],
        outputs=["dataset com um caso por amostra", "métricas de cobertura", "região de teste reservada"],
        depends_on=["solver"], estimate=estimate,
        checks=[{"what": "cobertura do desenho contra um sorteio aleatório do mesmo tamanho",
                 "tool": "pinneapple_autolab.study.campaign:plan_doe"},
                {"what": "cada campo com unidades, localização e referência (portões de dados)",
                 "tool": "pinneapple_data.schema:run_gates"},
                {"what": "nenhuma amostra falhou em silêncio: cada falha com o seu motivo",
                 "tool": "pinneapple_autolab.study.campaign:run_campaign"}],
        views=list(cap.VIEWS["study"]),
        question="" if brief.values.get("evaluations_needed") else "evaluations_needed")


def _training_stage(brief: Brief, arch: str, n_samples: int, cost: CostModel, direct_only: bool,
                    decisions: list[Decision], dataset_present: bool = False,
                    dataset_built: bool = False) -> Stage | None:
    if direct_only:
        decisions.append(Decision(
            topic="architecture", choice="none", source="architecture_recommendation.recommend_architecture",
            why=bi("Nenhum modelo a treinar: o solver responde cada caso dentro do orçamento.",
                   "No model to train: the solver answers each case within the budget."), values={}))
        return None
    physics_only = arch in ("pinn", "xpinn", "inverse_pinn")
    est = cost.training(arch, n_samples=n_samples, gpu=bool(brief.get("gpu")))
    decisions.append(Decision(
        topic="architecture", choice=arch, source="architecture_recommendation.recommend_architecture",
        why=bi(f"Arquitetura recomendada: `{arch}`"
               + (" (física pura, sem dataset de simulações)" if physics_only else
                  f", treinada sobre {n_samples} simulações") + ".",
               f"Recommended architecture: `{arch}`"
               + (" (physics-only, no simulation dataset)" if physics_only else
                  f", trained on {n_samples} simulations") + "."),
        alternatives=[a for a in ("deeponet", "fno", "mesh_graph_net", "pino", "pinn") if a != arch][:4],
        values={"architecture": arch, "physics_only": physics_only, "samples": 0 if physics_only else n_samples,
                "gpu": bool(brief.get("gpu"))}))
    tools = list(cap.TRAIN_PHYSICS) if physics_only else list(cap.TRAIN_OPERATOR)
    title = (bi("Treinar o PINN e passar as checagens de física", "Train the PINN and pass the physics checks")
             if physics_only else
             bi(f"Treinar e comparar as famílias de surrogate (`{arch}`)", f"Train and compare the surrogate "
                                                                            f"families (`{arch}`)"))
    if physics_only and (dataset_present or dataset_built):
        source = "o seu dataset" if dataset_present else f"as {n_samples} simulações que o plano gera"
        source_en = "your dataset" if dataset_present else f"the {n_samples} simulations the plan builds"
        why = bi(f"O sinal principal é o resíduo da EDP, e {source} entram como supervisão extra: é assim que um "
                 f"PINN usa dados quando eles existem, em vez de jogá-los fora.",
                 f"The main signal is the PDE residual and {source_en} come in as extra supervision: that is how a "
                 f"PINN uses data when there is data, instead of throwing it away.")
    else:
        why = (bi("Sem dados, o sinal de treino é o resíduo da EDP e as condições: um modelo por instância, sem "
                  "generalização entre casos -- o que é correto quando o caso é único.",
                  "With no data the training signal is the PDE residual and the conditions: one model per "
                  "instance, no generalisation across cases -- which is correct when the case is unique.")
               if physics_only else
               bi("Com dados, cada família é treinada no mesmo orçamento e no mesmo teste, com intervalos "
                  "conformais e as checagens de física nas previsões; a regra de escolha é escrita antes de olhar "
                  "o resultado.",
                  "With data, every family trains at the same budget and on the same test, with conformal intervals "
                  "and the physics checks on predictions; the selection rule is written before looking at the "
                  "result."))
    return Stage(
        id="training", kind="training", title=title, why=why, status="generate", answers=[7, 8],
        tools=tools + list(cap.TRAIN_CHECK),
        inputs=["o ProblemSpec" if physics_only else "o dataset"],
        outputs=["o modelo treinado", "o erro em dados não vistos", "a decisão de uso"],
        depends_on=["dataset"] if (dataset_present or dataset_built or not physics_only) else ["physics"],
        estimate=est,
        checks=[{"what": f"erro relativo de {brief.get('accuracy_target'):.3g} em dados não vistos",
                 "tool": "pinneapple_autolab.study.benchmark:benchmark"},
                {"what": "nenhuma previsão fora das faixas amostradas",
                 "tool": "pinneapple_analysis.verification.geometry_ood_guardrail"},
                {"what": "o modelo aprendeu a física e não só o ajuste (resíduo, fluxo, energia, intervenções)",
                 "tool": "pinneapple_analysis.validation.physics_learning:verify"}],
        views=list(cap.VIEWS["field"]) + list(cap.VIEWS["chart"]),
        question="" if brief.values.get("output_kind") else "output_kind")


def _inverse_stage(parent: str, dim: int, estimate: Estimate | None = None) -> Stage:
    return Stage(
        id="inverse", kind="inverse",
        title=bi("Calibrar o parâmetro contra as medições", "Calibrate the parameter against the measurements"),
        why=bi("O desconhecido é um parâmetro físico, não o campo. A identificabilidade é checada antes do ajuste; "
               "depois o campo e o parâmetro são ajustados alternadamente, e o que foi recuperado vem com a "
               "incerteza.",
               "The unknown is a physical parameter, not the field. Identifiability is checked before the fit; then "
               "the field and the parameter are fitted alternately, and what came back comes with its uncertainty."),
        status="generate", answers=[7, 8], tools=list(cap.INVERSE) + list(cap.VERIFY)[:1],
        inputs=["medições (campos, séries ou pontos)", "o modelo físico"],
        outputs=["o parâmetro calibrado", "a incerteza e o intervalo de confiança"],
        depends_on=[parent],
        checks=[{"what": "o parâmetro é identificável com estas medições (posto da matriz de sensibilidade)",
                 "tool": "pinneapple_analysis.inverse_problems.identifiability:analyze_identifiability"},
                {"what": "o campo ajustado respeita a EDP e as condições (carta de confiança)",
                 "tool": "pinneapple_analysis.trust.field_card:trust_card"}],
        views=list(cap.VIEWS["field"]) + list(cap.VIEWS["chart"]), estimate=estimate,
        question="accuracy_target")


def _optimization_stage(parent: str, estimate: Estimate | None = None) -> Stage:
    return Stage(
        id="optimization", kind="optimization",
        title=bi("Otimizar o projeto sobre o surrogate e re-verificar", "Optimise the design on the surrogate and "
                                                                        "re-verify"),
        why=bi("A busca roda no surrogate (muitas avaliações baratas); a frente de Pareto mostra o compromisso entre "
               "os objetivos e o ótimo escolhido é re-rodado no solver antes de virar recomendação.",
               "The search runs on the surrogate (many cheap evaluations); the Pareto front shows the trade-off "
               "between the objectives and the chosen optimum is re-run on the solver before it becomes a "
               "recommendation."),
        status="generate", answers=[7], tools=list(cap.OPTIMIZE) + list(cap.VERIFY)[:1],
        inputs=["o surrogate validado", "as faixas das variáveis e as restrições"],
        outputs=["a frente de Pareto", "o projeto escolhido, re-verificado"],
        depends_on=[parent],
        checks=[{"what": "as restrições valem no ponto ótimo, medidas no solver",
                 "tool": "pinneapple_design.design_optimizer.workflow:design_optimization"},
                {"what": "o ótimo está dentro da região amostrada (não é extrapolação)",
                 "tool": "pinneapple_analysis.trust.field_card:trust_card"}],
        views=list(cap.VIEWS["study"]), estimate=estimate,
        question="evaluations_needed")


def _verification_stage(brief: Brief, parent: str, dim: int) -> Stage:
    return Stage(
        id="verification", kind="verification",
        title=bi("Verificar e registrar o resultado", "Verify and record the result"),
        why=bi("Nenhum campo é reportado sem as checagens: resíduo da EDP, condições de contorno, conservação, "
               "comparação com referência e aplicabilidade fora do domínio treinado -- com uma decisão "
               "APPROVED / REVIEW / REJECT.",
               "No field is reported without its checks: PDE residual, boundary conditions, conservation, reference "
               "comparison and applicability outside the trained domain -- with an APPROVED / REVIEW / REJECT "
               "decision."),
        status="confirm", answers=[10], tools=list(cap.VERIFY),
        inputs=["o resultado (campo, KPI ou modelo)"], outputs=["a carta de confiança", "o registro de proveniência"],
        depends_on=[parent],
        checks=[{"what": "resíduo da EDP nos pontos de avaliação", "tool": "pinneapple_analysis.trust.field_card"},
                {"what": "condições de contorno e conservação", "tool": "pinneapple_analysis.trust.field_card"},
                {"what": "comparação com referência (analítica, literatura ou o solver)", "tool": "pp.compare"},
                {"what": "aplicabilidade: nada fora das faixas treinadas",
                 "tool": "pinneapple_analysis.trust.TrustGate"}],
        views=list(cap.VIEWS["field"]))


def _visualization_stage(brief: Brief, parent: str, dim: int) -> Stage:
    kind = brief.values.get("output_kind") or ("scalar" if brief.get("goal") == "screening" else "grid_field")
    views: list[Tool] = []
    if kind in ("grid_field", "mesh_field"):
        views += cap.VIEWS["field"]
    if brief.values.get("time_dependent") or kind == "time_series":
        views += cap.VIEWS["chart"]
    views += cap.VIEWS["study"]
    unique: list[Tool] = []
    seen: set = set()
    for t in views:
        if t.call not in seen:
            seen.add(t.call)
            unique.append(t)
    return Stage(
        id="visualization", kind="visualization",
        title=bi("Visualizar: o campo, o erro e o estudo", "Visualise: the field, the error and the study"),
        why=bi("O que se olha em cada etapa: o campo em 3-D sobre a malha, o mapa de erro por região (onde erra, "
               "não só quanto), as séries no tempo e o relatório do estudo ao lado dos exemplos medidos da "
               "biblioteca.",
               "What to look at at every stage: the field in 3-D on the mesh, the error map per region (where it "
               "errs, not only how much), the time series and the study report next to the library's measured "
               "examples."),
        status="have", answers=[10], tools=unique,
        inputs=["os resultados de cada etapa"], outputs=["figuras, cenas 3-D e o relatório do estudo"],
        depends_on=[parent],
        checks=[{"what": "cada figura e cena vem do resultado verificado, com a carta de confiança anexada",
                 "tool": "pinneapple_analysis.trust.field_card:trust_card"}],
        views=[{"what": "a galeria do Studio com o estudo e os exemplos comparáveis",
                "tool": "pinneapple_studio.catalog:load"}])


# ---------------------------------------------------------------------------------------------------- the APIs

def solver_families(refresh: bool = False) -> dict[str, Any]:
    """What is installed here, cached for :data:`FAMILIES_TTL_S`: the probe starts subprocesses and is the slowest
    part of a plan (17 s cold, 2 s warm on the reference machine, 0 s from this cache). ``refresh=True`` probes
    again."""
    import time

    if not refresh and _FAMILIES["value"] is not None and time.time() - _FAMILIES["at"] < FAMILIES_TTL_S:
        return _FAMILIES["value"]
    from pinneapple_analysis.verification.solver_orchestration import list_available_solver_families
    _FAMILIES["value"] = list_available_solver_families()
    _FAMILIES["at"] = time.time()
    return _FAMILIES["value"]


def _solver_recommendation(brief: Brief, family: str, dim: int, time_dependent: bool,
                           available_families: dict[str, Any] | None = None) -> tuple[Any | None, Any]:
    try:
        from pinneapple_analysis.verification.solver_orchestration import select_solver_family
        catalog = available_families if available_families is not None else solver_families()
        spec = _spec(family, dim, time_dependent)
        return select_solver_family(spec, catalog=catalog), None
    except Exception as exc:                    # noqa: BLE001 - a missing optional solver stack is a warning
        return None, bi(f"A recomendação de solver não pôde ser calculada ({type(exc).__name__}: "
                        f"{str(exc)[:160]}): o plano assume o PINN, que não depende de um solver externo.",
                        f"The solver recommendation could not be computed ({type(exc).__name__}: "
                        f"{str(exc)[:160]}): the plan assumes the PINN, which needs no external solver.")


def _architecture_recommendation(brief: Brief, *, has_solver: bool, solve_seconds: float | None,
                                 n_evaluations: int,
                                 direct_budget_s: float, goal: str,
                                 n_simulations: int | None = None) -> tuple[Any | None, Any]:
    try:
        from pinneapple_analysis.verification.architecture_recommendation import (
            recommend_architecture,
        )
    except Exception as exc:                    # noqa: BLE001
        return None, bi(f"A recomendação de arquitetura não pôde ser calculada ({type(exc).__name__}): o plano "
                        f"assume o PINN.",
                        f"The architecture recommendation could not be computed ({type(exc).__name__}): the plan "
                        f"assumes the PINN.")
    n_parameters = brief.values.get("n_parameters") or (1 if brief.values.get("parameter_varies") else 0)
    rec = recommend_architecture(
        n_high_fidelity_simulations=(int(n_simulations) if n_simulations is not None
                                     else int(brief.values.get("n_simulations_present") or 0)),
        has_analytical_solver=bool(has_solver),
        needs_parameter_generalization=bool(brief.values.get("parameter_varies")),
        geometry_varies=bool(brief.values.get("geometry_varies")),
        is_inverse_problem=(goal == "inverse"),
        output_kind=brief.values.get("output_kind"),
        n_inputs=n_parameters or None,
        accuracy_target=float(brief.get("accuracy_target") or 0.05),
        solve_seconds=solve_seconds, n_evaluations=n_evaluations, direct_budget_s=direct_budget_s)
    return rec, None


def _spec(family: str, dim: int, time_dependent: bool) -> Any:
    """A minimal ProblemSpec for the solver matcher, which reads only ``pde.kind`` and ``dim``."""
    from pinneapple_physics.pde_environment.capabilities import suggest_problem_spec
    from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec

    hints = suggest_problem_spec(family.replace("_", " "))
    coords: tuple[str, ...] = tuple(["x", "y", "z"][:dim]) + (("t",) if time_dependent else ())
    fields: tuple[str, ...] = tuple(hints.get("fields") or ["u"])
    return ProblemSpec(name="advisor", dim=int(dim), coords=coords, fields=fields,
                       pde=PDETermSpec(kind=family, fields=fields, coords=coords,
                                       params=dict(hints.get("default_params") or {})))


def _family(kind: str) -> Any:
    try:
        from pinneapple_physics.pde_environment.capabilities import get_pde_family
        return get_pde_family(kind)
    except Exception:                           # noqa: BLE001
        return None


def _all_families() -> list[Any]:
    try:
        from pinneapple_physics.pde_environment.capabilities import (
            get_pde_family,
            list_pde_families,
        )
        return [get_pde_family(k) for k in list_pde_families() if get_pde_family(k) is not None]
    except Exception:                           # noqa: BLE001
        return []


def _dimension_hint(family: str) -> int | None:
    fam = _family(family)
    coords = list(getattr(fam, "coords_hint", []) or []) if fam is not None else []
    coords = [c for c in coords if c != "t"]
    return len(coords) if coords else None


def _has_mesh(brief: Brief) -> bool:
    return bool(brief.values.get("mesh_present")) or bool(brief.values.get("geometry_present")) or (
        (brief.values.get("dimension") or 2) == 3)


def _evaluations(brief: Brief, goal: str, n_samples: int) -> tuple[int, str]:
    """How many solver evaluations the study needs, and where the number came from."""
    explicit = brief.values.get("evaluations_needed")
    if explicit:
        return int(explicit), "given in the brief"
    if goal == "forward" and not brief.values.get("parameter_varies"):
        return 1, "one forward prediction: a single solve answers it"
    if goal == "inverse":
        return 1, "the solves come from the fit, not from a dataset"
    if goal == "optimization":
        return DEFAULT_OPTIMIZATION_EVALUATIONS, (f"default for a design search ({DEFAULT_OPTIMIZATION_EVALUATIONS} "
                                                  f"evaluations); give evaluations_needed to replace it")
    return int(n_samples), "the samples the surrogate needs"


def _dataset_size(brief: Brief, dim: int) -> int:
    """Samples for the training set: about ten per input for a scalar, the operator threshold for a field."""
    n_parameters = int(brief.values.get("n_parameters") or (1 if brief.values.get("parameter_varies") else 1))
    output_kind = brief.values.get("output_kind")
    if output_kind == "scalar":
        base = SAMPLES_PER_INPUT_SCALAR * max(n_parameters, 1)
    elif output_kind in ("grid_field", "mesh_field", "time_series"):
        base = SAMPLES_FIELD
    else:
        base = max(SAMPLES_PER_INPUT_SCALAR * max(n_parameters, 1), SAMPLES_FIELD // 2)
    if brief.values.get("geometry_varies"):
        base = int(round(base * GEOMETRY_VARIATION_FACTOR))
    if brief.values.get("dataset_present") and brief.values.get("n_simulations_present"):
        base = max(base, int(brief.values["n_simulations_present"]))
    return int(base)


def _direct_budget(brief: Brief) -> float:
    hours = brief.values.get("budget_hours")
    return float(hours) * 3600.0 if hours else DIRECT_BUDGET_S


def _solver_checks() -> list[dict[str, str]]:
    return [{"what": "uma execução convergiu (resíduo abaixo da tolerância em todas as equações)",
             "tool": "pinneapple_analysis.trust.field_card:trust_card"},
            {"what": "o resultado bate com uma referência (solução exata, literatura ou outro solver)",
             "tool": "pinneapple_analysis.verification.solver_orchestration:compare_solvers"},
            {"what": "conservação: o fluxo líquido na fronteira é a integral da fonte",
             "tool": "pp.trust_card"}]


__all__ = ["build", "Composition", "solver_families", "DIRECT_BUDGET_S", "SAMPLES_PER_INPUT_SCALAR",
           "SAMPLES_FIELD", "FAMILIES_TTL_S"]
