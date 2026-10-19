"""Physics-guided design optimization pipeline.

Provides a closed-loop optimizer that:
1. Parametrizes design space (via ParamSpace)
2. Evaluates physics with a surrogate model (FNO / MLP / any nn.Module)
3. Optimizes design via gradient / Bayesian / evolutionary methods
4. Optionally refines with PINNs
5. Handles multi-objective optimization with Pareto fronts
6. Runs design studies (:mod:`.study`, :mod:`.methods`): black-box and gradient methods on one problem
   description and one evaluation budget, high-fidelity verification and active learning (:mod:`.verify`),
   method comparison (:mod:`.compare`), presets built from library physics (:mod:`.presets`)

Quick start::

    from pinneapple_design.design_optimizer import DesignOptLoop, DesignOptConfig, PhysicsSurrogate
    from pinneapple_design.design_optimizer import DragObjective, BoxConstraint, ConstraintSet
    from pinneapple_design.geometry.optimize.loop import ParamSpace

    space = ParamSpace(
        bounds={"chord": (0.5, 2.0), "camber": (0.0, 0.3), "thickness": (0.05, 0.3)},
        x0={"chord": 1.0, "camber": 0.1, "thickness": 0.1},
    )

    surrogate = PhysicsSurrogate.build_mlp(in_dim=3, out_dim=32)
    objective = DragObjective()
    constraints = ConstraintSet()
    constraints.add(BoxConstraint(
        theta_min=torch.tensor([0.5, 0.0, 0.05]),
        theta_max=torch.tensor([2.0, 0.3, 0.3]),
    ))

    loop = DesignOptLoop(space, surrogate, objective, constraints=constraints)
    result = loop.run()
    print(result.summary())
    result.plot_convergence()
"""
from .adjoint import (
    ShapeParametrization,
    ContinuousAdjointSolver,
    DragAdjointObjective,
    naca_parametric,
)
from .pareto import ParetoFront, pareto_dominates, compute_pareto_front, representatives
from .objective import (
    ObjectiveBase,
    DragObjective,
    ThermalEfficiencyObjective,
    StructuralObjective,
    WeightMinimizationObjective,
    CompositeObjective,
)
from .constraints import (
    ConstraintBase,
    BoxConstraint,
    MassConservationConstraint,
    GeometricConstraint,
    ManufacturabilityConstraint,
    ConstraintSet,
)
from .surrogate import SurrogateConfig, PhysicsSurrogate
from .optimizer import (
    ParamSpace,
    OptState,
    DesignOptimizerConfig,
    GradientDesignOptimizer,
    BayesianDesignOptimizer,
    EvolutionaryDesignOptimizer,
)
from .refinement import RefinementConfig, PINNRefinement, RefinementResult
from .pipeline import DesignOptConfig, DesignOptResult, DesignOptLoop
from .study import (
    Variable, DesignSpace, Objective, Constraint, Evaluator, FunctionEvaluator, DesignProblem, Evaluations,
    feasibility_report, hypervolume, constants_of,
)
from .methods import METHODS, OptimizationResult, available_methods, run
from .verify import (
    select_candidates, verify_candidates, VerificationReport, GPSurrogate, surrogate_assisted, ActiveLearningResult,
)
from .compare import MethodComparison, compare_methods
from .evaluators import ZooEvaluator, TorchModuleEvaluator, CommandEvaluator, from_model_ref
from .spec import SpecError, problem_from_spec, reference_for_spec, zoo_problem
from .report import (
    to_parameter_space, designs_frame, study_result, surrogate_from_table, model_ref_for_surrogate, table_evaluator,
    verify_front, write_design_files, chart_points, verify_uploaded_results, DESIGN_FILES,
)

__all__ = [
    # Adjoint shape optimisation (Feature 16)
    "ShapeParametrization",
    "ContinuousAdjointSolver",
    "DragAdjointObjective",
    "naca_parametric",
    # Pareto / multi-objective
    "ParetoFront", "pareto_dominates", "compute_pareto_front", "representatives",
    "ObjectiveBase", "DragObjective", "ThermalEfficiencyObjective",
    "StructuralObjective", "WeightMinimizationObjective", "CompositeObjective",
    "ConstraintBase", "BoxConstraint", "MassConservationConstraint",
    "GeometricConstraint", "ManufacturabilityConstraint", "ConstraintSet",
    "SurrogateConfig", "PhysicsSurrogate",
    "ParamSpace", "OptState",
    "DesignOptimizerConfig", "GradientDesignOptimizer",
    "BayesianDesignOptimizer", "EvolutionaryDesignOptimizer",
    "RefinementConfig", "PINNRefinement", "RefinementResult",
    "DesignOptConfig", "DesignOptResult", "DesignOptLoop",
    # Design studies: one problem description, many methods, high-fidelity verification
    "Variable", "DesignSpace", "Objective", "Constraint", "Evaluator", "FunctionEvaluator", "DesignProblem",
    "Evaluations", "feasibility_report", "hypervolume", "constants_of",
    "METHODS", "OptimizationResult", "available_methods", "run",
    "select_candidates", "verify_candidates", "VerificationReport", "GPSurrogate", "surrogate_assisted",
    "ActiveLearningResult", "MethodComparison", "compare_methods",
    "ZooEvaluator", "TorchModuleEvaluator", "CommandEvaluator", "from_model_ref",
    "to_parameter_space", "designs_frame", "study_result", "surrogate_from_table", "model_ref_for_surrogate",
    "table_evaluator", "verify_front", "write_design_files", "chart_points", "verify_uploaded_results", "DESIGN_FILES",
    "SpecError", "problem_from_spec", "reference_for_spec", "zoo_problem",
]
