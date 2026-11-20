"""The PINNeAPPle calls each stage can be carried out with, named exactly -- and a check that they exist.

A suggestion that names a function the library does not have is worse than no suggestion, so every :class:`Tool`
written here is a ``"module.path:attribute"`` string that :func:`verify_tools` imports and looks up. The test suite
runs that check over a plan for every shape of problem, which is what keeps this catalog honest as the library
moves.

The catalog is deliberately small: one way to do each thing, the one this library documents in its guides. Where
several exist (a PINN through ``pp.solve``, an operator through the study benchmark), the note says which and why.
"""
from __future__ import annotations

from collections.abc import Iterable

from .schema import Tool, bi

DOCS = {
    "intent": "docs/guides/design_from_intent.md",
    "mesh": "docs/guides/mesh_generation.md",
    "immersed": "docs/guides/immersed_analysis.md",
    "study": "docs/guides/ahmed_body_end_to_end.md",
    "lab": "docs/guides/autonomous_lab.md",
    "validation": "docs/guides/validation_gallery.md",
    "pinn": "docs/guides/pinn_validation.md",
    "llm": "docs/guides/pinneapple_llm_quality_pipeline.md",
    "studio": "docs/guides/studio.md",
    "checks": "docs/guides/physics_checks_in_ci.md",
    "cfd": "docs/guides/freecad_openfoam_pipelines.md",
}


def tool(call: str, pt: str, en: str = "", docs: str = "") -> Tool:
    """One named library call; ``call`` is ``"module:attribute"`` so :func:`verify_tools` can import it."""
    return Tool(call=call, note=bi(pt, en or pt), docs=DOCS.get(docs, docs))


# ---- the physical model (question 1) ----------------------------------------------------------------------------
PHYSICS_CONFIRM = [
    tool("pinneapple_physics.pde_environment.capabilities:identify_pde",
         "pontua as famílias de EDP contra a sua descrição e devolve a mais provável, com o score",
         "scores the PDE families against your description and returns the likeliest, with its score", "intent"),
    tool("pinneapple_physics.pde_environment.capabilities:suggest_problem_spec",
         "sugere campos, coordenadas e parâmetros para a família identificada",
         "suggests fields, coordinates and parameters for the identified family"),
    tool("pinneapple_problemdesign.formulate:formulate",
         "monta o sistema a partir das leis, com unidades, e roda as checagens (dimensão, fechamento, condições, "
         "solução manufaturada)",
         "composes the system from laws, with units, and runs its checks (dimensions, closure, conditions, "
         "manufactured solution)", "intent"),
]
PHYSICS_IMPORT = [
    tool("pinneapple_physics.pde_environment.builder:ProblemBuilder",
         "o problema como campos, EDP e condições -- o objeto que todas as etapas seguintes usam",
         "the problem as fields, PDE and conditions -- the object every later stage uses"),
]

# ---- geometry (questions 2 and 4) -------------------------------------------------------------------------------
GEOMETRY_GENERATE = [
    tool("pinneapple_design.intent:design",
         "um objetivo em texto vira um projeto verificável, ou as perguntas que faltam, ou uma recusa",
         "a goal in words becomes a verified design, or the questions it still needs, or a refusal", "intent"),
    tool("pinneapple_llm:draft_part",
         "uma peça descrita em texto vira uma receita CSG/CadQuery, construída e medida (modelo local)",
         "a part described in words becomes a CSG/CadQuery recipe, built and measured (local model)", "llm"),
    tool("pinneapple_design.geometry_library:list",
         "primitivas paramétricas (caixa, cilindro, esfera, canal) para os casos canônicos",
         "parametric primitives (box, cylinder, sphere, channel) for the canonical cases"),
]
GEOMETRY_IMPORT = [
    tool("pinneapple_simulation.external_solvers.cfd_formats.convert:detect_format",
         "reconhece o formato pelo arquivo (STEP, STL, CGNS, Exodus, Fluent, Abaqus)",
         "recognises the format from the file (STEP, STL, CGNS, Exodus, Fluent, Abaqus)", "cfd"),
    tool("pinneapple_simulation.external_solvers.cfd_formats.convert:convert_surface",
         "converte e repara a superfície para um formato que o gerador de malha lê",
         "converts and repairs the surface into a format the mesher reads", "cfd"),
    tool("pinneapple_analysis.verification.geometry_intelligence:classify_geometry",
         "classifica a superfície (lados, regiões, bordas abertas) -- o que diz se ela é fechada e malhável",
         "classifies the surface (facets, regions, open boundary loops) -- what says whether it is closed and "
         "meshable"),
]

# ---- mesh (question 3) ------------------------------------------------------------------------------------------
MESH_GENERATE = [
    tool("pinneapple_design.geometry.mesh.generate:generate_mesh",
         "malha não estruturada de uma geometria: triângulos em 2-D, tetraedros em 3-D, camada limite nas paredes",
         "an unstructured mesh of a geometry: triangles in 2-D, tetrahedra in 3-D, a boundary layer at the walls",
         "mesh"),
    tool("pinneapple_simulation.immersed:Discretization",
         "alternativa sem malha conforme: grade de fundo com células cortadas, para geometria suja ou que muda "
         "muito entre casos",
         "the no-conforming-mesh alternative: a background grid with cut cells, for a dirty geometry or one that "
         "changes a lot between cases", "immersed"),
]
MESH_AUTOMATIC = [
    tool("pinneapple_physics.pde_environment:ProblemSpec",
         "o domínio analítico (caixa, cilindro) é amostrado pelo próprio solver: sem CAD e sem malha",
         "the analytic domain (box, cylinder) is sampled by the solver itself: no CAD and no mesh"),
]
MESH_IMPORT = [
    tool("pinneapple_simulation.external_solvers.cfd_formats.convert:read_surface_mesh",
         "lê a malha de superfície que você já tem (STL, CGNS, Exodus, Fluent, Abaqus)",
         "reads the surface mesh you already have (STL, CGNS, Exodus, Fluent, Abaqus)", "cfd"),
]
MESH_CHECK = [
    tool("pinneapple_analysis.verification.mesh_intelligence:assess_mesh_quality",
         "qualidade de cada elemento (SICN, gama, razão de aresta) e o relatório de ortogonalidade",
         "quality of every element (SICN, gamma, edge ratio) and the orthogonality report"),
    tool("pinneapple_analysis.verification.mesh_intelligence:estimate_required_first_cell_height",
         "a altura da primeira célula para o y+ que o modelo de turbulência exige",
         "the first-cell height for the y+ the turbulence model needs"),
    tool("pinneapple_autolab.study.mesh_study:grid_convergence",
         "refino em níveis e o índice de convergência de malha (GCI) com a ordem observada",
         "refinement levels and the grid convergence index (GCI) with the observed order", "study"),
]

# ---- the solver (question 6) ------------------------------------------------------------------------------------
SOLVER_SELECT = [
    tool("pinneapple_analysis.verification.solver_orchestration:select_solver_family",
         "qual família de solver disponível cobre esta EDP e esta dimensão, e a citação de escopo que sustenta",
         "which available solver family covers this PDE and dimension, with the scope citation that backs it"),
    tool("pinneapple_analysis.verification.solver_orchestration:list_available_solver_families",
         "o que está de fato instalado nesta máquina agora (OpenFOAM, FEniCS, os solvers internos)",
         "what is actually installed on this machine right now (OpenFOAM, FEniCS, the internal solvers)"),
]
SOLVER_RUN = [
    tool("pinneapple:solve",
         "uma chamada resolve o problema com o método escolhido e registra o custo de cada etapa",
         "one call solves the problem with the chosen method and records the cost of each stage"),
    tool("pinneapple_simulation.numerical_solvers.symbolic_fd:solve_fd",
         "um sistema de equações em texto resolvido por diferenças finitas, sem montar um preset",
         "an equation system written as text, solved by finite differences without building a preset"),
    tool("pinneapple_simulation.external_solvers.cfd_formats.convert:convert_surface",
         "caso o solver seja externo (OpenFOAM), a geometria convertida entra no caso",
         "when the solver is external (OpenFOAM), the converted geometry goes into the case", "cfd"),
]
SOLVER_CHECK = [
    tool("pinneapple_analysis.verification.solver_orchestration:compare_solvers",
         "dois solvers no mesmo problema e nos mesmos pontos, com a diferença entre eles",
         "two solvers on the same problem and the same points, with the difference between them"),
    tool("pinneapple_analysis.trust.field_card:trust_card",
         "resíduo da EDP, condições de contorno, conservação e referência, com decisão APPROVED/REVIEW/REJECT",
         "PDE residual, boundary conditions, conservation and reference, with an APPROVED/REVIEW/REJECT decision",
         "checks"),
]

# ---- the dataset (questions 5, 7 and 9) -------------------------------------------------------------------------
DATASET_PLAN = [
    tool("pinneapple_autolab.study.campaign:plan_doe",
         "desenho de experimentos com maximin e as métricas de cobertura (discrepância, distância mínima) contra "
         "um sorteio aleatório do mesmo tamanho",
         "a design of experiments with maximin and its coverage metrics (discrepancy, minimum distance) against a "
         "random draw of the same size", "study"),
    tool("pinneapple_data.doe:design",
         "as amostragens disponíveis (latin hypercube, maximin, Sobol) sobre o espaço dos parâmetros",
         "the available samplings (Latin hypercube, maximin, Sobol) over the parameter space"),
    tool("pinneapple_autolab.study.campaign:run_campaign",
         "roda a campanha de forma retomável, com o custo por caso e o motivo de cada falha",
         "runs the campaign resumably, with the cost per case and the reason of every failure", "study"),
]
DATASET_BUILD = [
    tool("pinneapple_autolab.study.dataset:build_dataset",
         "as execuções viram um dataset com unidades, proveniência e a referência de cada campo",
         "the runs become a dataset with units, provenance and each field's reference", "study"),
    tool("pinneapple_autolab.study.dataset:region_split",
         "o teste fica numa região do espaço de entrada, não em amostras sorteadas: mede extrapolação",
         "the test set is a region of the input space, not random rows: it measures extrapolation", "study"),
    tool("pinneapple_data.schema:run_gates",
         "os portões de dados: unidades, localização nó/célula, licença e SHA-256 de cada entrada",
         "the data gates: units, node/cell location, licence and the SHA-256 of every input"),
]
DATASET_HAVE = [
    tool("pinneapple_data.schema:PhysicsDataBundle",
         "o seu dataset no schema único da biblioteca, para os mesmos portões e o mesmo benchmark",
         "your dataset in the library's one schema, so it gets the same gates and the same benchmark"),
]

# ---- training (question 8) --------------------------------------------------------------------------------------
TRAIN_PHYSICS = [
    tool("pinneapple:solve",
         "treina o PINN no problema, com a arquitetura e o orçamento registrados",
         "trains the PINN on the problem, with the architecture and the budget recorded", "pinn"),
    tool("pinneapple_analysis.validation.pinn:validate_pinn",
         "resíduo em pontos novos, modos de falha e soluções exatas -- o que um PINN tem de passar",
         "residual on fresh points, failure modes and exact solutions -- what a PINN has to pass", "pinn"),
]
TRAIN_OPERATOR = [
    tool("pinneapple_autolab.study.benchmark:benchmark",
         "as famílias de surrogate no mesmo orçamento e no mesmo conjunto de teste, com intervalos conformais e a "
         "regra escrita de escolha",
         "the surrogate families at the same budget and the same test set, with conformal intervals and a written "
         "selection rule", "study"),
    tool("pinneapple_arena.pointwise:train_torch_model",
         "o treino por arquitetura do registro, com as métricas de campo",
         "the registry's per-architecture training, with the field metrics"),
]
TRAIN_CHECK = [
    tool("pinneapple_analysis.validation.physics_learning:verify",
         "o modelo aprendeu a física ou só o ajuste: resíduo discreto, fluxo na interface, energia, intervenções",
         "whether the model learned the physics or only the fit: discrete residual, interface flux, energy, "
         "interventions"),
    tool("pinneapple_veriphysics.surrogate_gates:--",
         "os portões de um surrogate antes de qualquer uso fora do conjunto de teste",
         "the gates a surrogate passes before any use outside the test set"),
    tool("pinneapple_analysis.validation.validate:pp_validate",
         "um relatório de credibilidade por fator, com a decisão pass / warn / fail",
         "a credibility report per factor, with a pass / warn / fail decision", "validation"),
]

# ---- inverse and optimisation -----------------------------------------------------------------------------------
INVERSE = [
    tool("pinneapple_analysis.inverse_problems.identifiability:analyze_identifiability",
         "antes de ajustar: o parâmetro é identificável com estas medições?",
         "before fitting: is the parameter identifiable from these measurements?"),
    tool("pinneapple_analysis.verification.uncertainty_inverse:solve_inverse_problem",
         "o campo e o parâmetro ajustados alternadamente, com a incerteza do que foi recuperado",
         "the field and the parameter fitted alternately, with the uncertainty of what was recovered"),
]
OPTIMIZE = [
    tool("pinneapple_autolab.study.extras:pareto_on_surrogate",
         "a frente de Pareto sobre o surrogate, com as restrições checadas",
         "the Pareto front on the surrogate, with the constraints checked", "study"),
    tool("pinneapple_design.design_optimizer.workflow:design_optimization",
         "geometria e parâmetros otimizados e re-verificados em malhas refinadas",
         "geometry and parameters optimised and re-verified on refined meshes"),
]

# ---- verification and views -------------------------------------------------------------------------------------
VERIFY = [
    tool("pinneapple_analysis.trust.field_card:trust_card",
         "a carta de confiança de qualquer campo: resíduo, contorno, conservação, referência, aplicabilidade",
         "the trust card of any field: residual, boundary, conservation, reference, applicability", "checks"),
    tool("pinneapple_analysis.verification.provenance:ProvenanceRecord",
         "o registro do que foi rodado, com que entradas, para o resultado poder ser repetido",
         "the record of what was run, with which inputs, so the result can be repeated"),
    tool("pinneapple_check:run", "o linter de física em CI sobre os resultados e os modelos",
         "the physics linter in CI over the results and the models", "checks"),
]
VIEWS: dict[str, list[Tool]] = {
    "field": [
        tool("pinneapple_tools.visualization.studio.fieldview:field_view",
             "o campo em 3-D sobre a malha, com a cor pela grandeza e o mapa de erro contra a referência",
             "the field in 3-D on the mesh, coloured by the quantity, with the error map against the reference",
             "studio"),
        tool("pinneapple_analysis.validation.error_map:error_map",
             "o erro por região, para ver onde o modelo erra e não só quanto",
             "the error per region, to see where the model errs and not only how much"),
    ],
    "chart": [
        tool("pinneapple_tools.visualization.dashboard:Dashboard",
             "séries no tempo com quadros sincronizados, PNG, HTML ou vídeo",
             "time series with synced frames, PNG, HTML or video", "studio"),
    ],
    "study": [
        tool("pinneapple_autolab.study.report:write_report",
             "o relatório do estudo com as métricas, a cobertura do DOE e o erro de cada família",
             "the study report with the metrics, the DOE coverage and each family's error", "lab"),
        tool("pinneapple_studio.catalog:load",
             "a galeria do Studio: o estudo ao lado dos exemplos medidos da biblioteca",
             "the Studio gallery: the study next to the library's measured examples", "studio"),
    ],
    "geometry": [
        tool("pinneapple_design.geometry_library:export_web",
             "a geometria e a malha no visualizador do Studio, coloridas pela qualidade do elemento",
             "the geometry and the mesh in the Studio viewer, coloured by element quality", "studio"),
    ],
}


def verify_tools(tools: Iterable[Tool]) -> list[str]:
    """Import every ``"module:attribute"`` a tool names; return the problems (empty means all of them exist).

    A tool whose attribute is ``"--"`` is a command-line tool: only its module is imported. An import failure is
    reported, never raised, so this can run in a test that also covers a machine without the optional extras.
    """
    import importlib

    problems: list[str] = []
    for t in tools:
        module, _, attr = t.call.partition(":")
        try:
            mod = importlib.import_module(module)
        except Exception as exc:                # noqa: BLE001 - an optional extra that is not installed
            problems.append(f"{t.call}: importing {module} failed ({type(exc).__name__}: {str(exc)[:120]})")
            continue
        if attr and attr != "--" and not hasattr(mod, attr):
            problems.append(f"{t.call}: {module} has no attribute {attr}")
    return problems


__all__ = ["DOCS", "tool", "PHYSICS_CONFIRM", "PHYSICS_IMPORT", "GEOMETRY_GENERATE", "GEOMETRY_IMPORT",
           "MESH_GENERATE", "MESH_AUTOMATIC", "MESH_IMPORT", "MESH_CHECK", "SOLVER_SELECT", "SOLVER_RUN",
           "SOLVER_CHECK", "DATASET_PLAN", "DATASET_BUILD", "DATASET_HAVE", "TRAIN_PHYSICS", "TRAIN_OPERATOR",
           "TRAIN_CHECK", "INVERSE", "OPTIMIZE", "VERIFY", "VIEWS", "verify_tools"]
