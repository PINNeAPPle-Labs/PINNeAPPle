"""The configuration objects of :class:`~pinneapple_pipeline.SurrogatePipeline`.

One dataclass per block of the pipeline, each with ``to_dict()`` and ``from_dict()`` so a pipeline can be written
as YAML, checked in, and re-run elsewhere. Nothing here builds or trains anything: a configuration says *what* is
to be done, and :mod:`pinneapple_pipeline.adapters` maps it onto the modules that already know how to do it
(``pinneapple_physics.pde_environment`` presets, ``pinneapple_data.doe`` designs, ``pinneapple_physics.solving``
reference solvers, ``pinneapple_neural`` architectures and trainer, ``pinneapple_check``, ``pinneapple_tools``
model export).

Only the blocks the pipeline actually runs are here. ``GeometryConfig``, ``MaterialConfig``, ``OptimizationConfig``
(HPO), ``AdaptiveLearningConfig`` and ``DeploymentConfig`` are not: each needs a backend the library does not have
for this path yet, and a configuration field that silently does nothing is worse than an absent one (see
``docs/guides/surrogate_pipeline.md``).
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DimensionConfig", "ProblemConfig", "PhysicsConfig", "BoundaryConditionConfig", "ParameterSpaceConfig",
    "SamplingConfig", "SolverConfig", "DataConfig", "ArchitectureConfig", "TrainingConfig", "LossConfig",
    "ExecutionConfig", "VerificationConfig", "UncertaintyConfig", "ExportConfig", "VisualizationConfig",
    "SurrogateConfig", "PipelineConfig", "DIMENSION_KEYS",
]

DIMENSION_KEYS = ("spatial", "temporal", "coordinate_system", "axisymmetric", "time_domain")
_COORDINATE_SYSTEMS = ("cartesian", "cylindrical", "spherical", "custom")


# --------------------------------------------------------------------------------------------------- errors
class ConfigError(ValueError):
    """A configuration that cannot be run as written. The message names the field and what is accepted."""


def _as_tuple(v: Any) -> tuple:
    if v is None:
        return ()
    if isinstance(v, str):
        return (v,)
    return tuple(v)


def _mapping(v: Any) -> dict:
    return {} if v is None else dict(v)


@dataclass
class _Block:
    """Shared plumbing: every block is a plain dataclass that round-trips through a mapping."""

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if dataclasses.is_dataclass(v):
                v = v.to_dict()
            out[f.name] = v
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None):
        """Build the block from a mapping; unknown keys are an error (a mistyped key is never ignored)."""
        data = dict(data or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"{cls.__name__} has no field(s) {unknown}; it takes {sorted(known)}")
        obj = cls(**data)
        obj.validate()
        return obj

    def validate(self) -> None:                    # pragma: no cover - overridden where there is something to check
        """Raise :class:`ConfigError` when the block cannot be run as written."""


# --------------------------------------------------------------------------------------------------- dimension
@dataclass
class DimensionConfig(_Block):
    """Spatial dimension, time and coordinate system.

    ``spatial=0`` and ``temporal=None`` (the defaults) mean *take it from the problem*: the number of coordinates
    that are not the time coordinate, and whether it has one. Setting either explicitly is a declaration the
    pipeline checks against the problem, so a 2-D config pointed at a 3-D preset fails at the ``problem`` gate
    instead of training silently.
    """

    spatial: int = 0
    temporal: bool | None = None
    coordinate_system: str = "cartesian"
    axisymmetric: bool = False
    time_domain: tuple[float, float] | None = None

    def validate(self) -> None:
        if self.spatial not in (0, 1, 2, 3):
            raise ConfigError(f"dimension.spatial must be 0 (infer), 1, 2 or 3, got {self.spatial}")
        if self.coordinate_system not in _COORDINATE_SYSTEMS:
            raise ConfigError(f"dimension.coordinate_system must be one of {_COORDINATE_SYSTEMS}, "
                              f"got {self.coordinate_system!r}")
        if self.time_domain is not None:
            lo, hi = (float(x) for x in self.time_domain)
            if not lo < hi:
                raise ConfigError(f"dimension.time_domain must be increasing, got {(lo, hi)}")
            self.time_domain = (lo, hi)

    def check_against(self, coords: Sequence[str]) -> list[str]:
        """Disagreements between what was declared and the problem's own coordinates."""
        out = []
        spatial = [c for c in coords if c != "t"]
        if self.spatial and self.spatial != len(spatial):
            out.append(f"dimension.spatial={self.spatial} but the problem has {len(spatial)} spatial "
                       f"coordinate(s) {tuple(spatial)}")
        if self.temporal is None:
            return out
        if self.temporal and "t" not in coords:
            out.append(f"dimension.temporal=true but the problem has no 't' coordinate {tuple(coords)}")
        if not self.temporal and "t" in coords:
            out.append("dimension.temporal=false but the problem has a 't' coordinate; set temporal: true or "
                       "leave it unset to take it from the problem")
        return out


# --------------------------------------------------------------------------------------------------- problem
@dataclass
class ProblemConfig(_Block):
    """Where the governing equations come from.

    ``source="preset"``     a named problem of ``pinneapple_physics.pde_environment`` (``pp.list_presets()``),
                            e.g. ``preset: burgers_1d`` with ``preset_kwargs: {nu: 0.01}``.
    ``source="spec"``       a ``ProblemSpec`` built with ``ProblemBuilder`` or ``pp.get_preset`` and handed to the
                            constructor (``problem: <object>``); ``equations``/``constants`` do not apply.
    ``source="symbolic"``   equations written as text, parsed by ``pinneapple_physics.symbolic_pde``::

                                source: symbolic
                                coords: [x, y, t]
                                fields: [u]
                                equations: ["Derivative(u, t) - alpha * (Derivative(u, x, 2) + Derivative(u, y, 2))"]
                                constants: {alpha: 0.01}

    ``formulation`` is recorded with the run (strong form is what the pipeline trains) and ``regime``
    (``steady``/``transient``) is taken from the coordinates unless stated.
    """

    source: str = "preset"
    preset: str = ""
    preset_kwargs: dict[str, Any] = field(default_factory=dict)
    problem: Any = None                      # a ProblemSpec / PhysicalProblem, only for source="spec"
    coords: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    domain_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    equations: tuple[str, ...] = ()
    constants: dict[str, float] = field(default_factory=dict)
    conditions: tuple[dict[str, Any], ...] = ()      # symbolic conditions: {"kind", "on", "equation"}
    formulation: str = "strong"
    regime: str = ""                          # "" = from the coordinates; or steady | transient
    problem_type: str = "forward"
    units: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        self.source = str(self.source).lower()
        self.equations, self.conditions = _as_tuple(self.equations), _as_tuple(self.conditions)
        if self.source not in ("preset", "spec", "symbolic"):
            raise ConfigError(f"problem.source must be preset, spec or symbolic, got {self.source!r}")
        if self.source == "preset" and not self.preset:
            raise ConfigError("problem.source is 'preset' but problem.preset is empty; pass a name from "
                              "pp.list_presets()")
        if self.source == "spec" and self.problem is None:
            raise ConfigError("problem.source is 'spec' but problem.problem is None; pass the ProblemSpec or "
                              "PhysicalProblem object")
        if self.source == "symbolic":
            if not self.equations:
                raise ConfigError("problem.source is 'symbolic' but problem.equations is empty")
            if not (self.coords and self.fields and self.domain_bounds):
                raise ConfigError("a symbolic problem needs problem.coords, problem.fields and problem.domain_bounds")
            missing = [c for c in self.coords if c not in self.domain_bounds]
            if missing:
                raise ConfigError(f"problem.domain_bounds has no range for coordinate(s) {missing}")
        if self.regime and self.regime not in ("steady", "transient"):
            raise ConfigError(f"problem.regime must be steady or transient, got {self.regime!r}")
        if self.formulation != "strong":
            raise ConfigError(f"the pipeline trains the strong form of the equations; formulation={self.formulation!r} "
                              "is accepted in the schema but not implemented")


# --------------------------------------------------------------------------------------------------- physics
@dataclass
class PhysicsConfig(_Block):
    """The physical hypotheses the model rests on, so that the ``physics`` gate can say when they stop holding.

    ``assumptions`` are names of ``pinneapple_physics.physics_audit.assumptions`` (``incompressible``, ``laminar``,
    ``turbulent``, ``creeping_flow``, ``continuum``, ``small_strain``, ``linear_elastic``, ``diffusion_dominated``,
    ``advection_dominated``, ``attached_flow``, ``shallow_water``, ``boussinesq``, ``constant_properties``,
    ``lumped_capacitance``, ``forced_convection``). ``features`` are the numbers they are checked against
    (``{Re: 1200}``). An assumption whose feature is missing is reported as not evaluated, never as satisfied.
    """

    domains: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    features: dict[str, float] = field(default_factory=dict)
    conservation_laws: tuple[str, ...] = ()
    nondimensionalization: str = "auto"

    def validate(self) -> None:
        self.domains, self.assumptions = _as_tuple(self.domains), _as_tuple(self.assumptions)
        self.conservation_laws = _as_tuple(self.conservation_laws)
        self.features = {str(k): float(v) for k, v in _mapping(self.features).items()}
        if self.assumptions:
            from pinneapple_physics.physics_audit import assumptions as _a
            known = {n for n in dir(_a) if not n.startswith("_") and callable(getattr(_a, n))}
            unknown = [a for a in self.assumptions if a not in known]
            if unknown:
                raise ConfigError(f"physics.assumptions names {unknown}, which physics_audit does not define; "
                                  f"known: {sorted(n for n in known if not n[0].isupper())[:20]}...")


# --------------------------------------------------------------------------------------------------- BC/IC
@dataclass
class BoundaryConditionConfig(_Block):
    """A condition added to (or replacing) the problem's own, without editing the preset.

    ``kind``: ``dirichlet`` | ``neumann`` | ``robin`` | ``initial``. ``source``: ``constant`` | ``symbolic`` |
    ``callable``. ``on``: a face selector, ``"x_min"``/``"x_max"``/... for one coordinate face, or the list
    ``["y_min", "y_max"]`` for several. ``expression`` is a Python expression in the coordinates and ``t``
    (``"300 + 10*sin(t)"``); ``fn`` is ``fn(X) -> (n, len(fields))`` for anything an expression cannot say.
    """

    name: str = ""
    kind: str = "dirichlet"
    field: str = ""
    source: str = "constant"
    value: float = 0.0
    expression: str = ""
    fn: Any = None
    on: Any = "boundary"
    weight: float = 1.0

    def validate(self) -> None:
        if self.kind not in ("dirichlet", "neumann", "robin", "initial"):
            raise ConfigError(f"boundary condition {self.name!r}: kind must be dirichlet, neumann, robin or initial, "
                              f"got {self.kind!r}")
        if self.source not in ("constant", "symbolic", "callable"):
            raise ConfigError(f"boundary condition {self.name!r}: source must be constant, symbolic or callable, "
                              f"got {self.source!r}")
        if not self.field:
            raise ConfigError(f"boundary condition {self.name!r}: field is empty")
        if self.source == "symbolic" and not self.expression:
            raise ConfigError(f"boundary condition {self.name!r}: source is symbolic but expression is empty")
        if self.source == "callable" and not callable(self.fn):
            raise ConfigError(f"boundary condition {self.name!r}: source is callable but fn is not")

    def to_condition(self, coords: Sequence[str], bounds: Mapping[str, tuple[float, float]]):
        """A ``ConditionSpec}`` for this declaration, with its selector built from ``on``."""
        import numpy as np

        from pinneapple_physics.pde_environment import ConditionSpec

        mask = _face_selector(self.on, coords, bounds)
        if self.source == "constant":
            value_fn = lambda X, ctx, _v=float(self.value): np.full((len(X), 1), _v)   # noqa: E731
        elif self.source == "symbolic":
            value_fn = _expression_fn(self.expression, coords)
        else:
            value_fn = lambda X, ctx, _f=self.fn: np.asarray(_f(X), dtype=np.float32).reshape(len(X), -1)  # noqa: E731
        return ConditionSpec(name=self.name or f"{self.kind}_{self.field}", kind=self.kind, fields=(self.field,),
                             selector_type="callable", selector=mask, value_fn=value_fn, weight=float(self.weight))


def _face_selector(on: Any, coords: Sequence[str], bounds: Mapping[str, tuple[float, float]]):
    """``(X) -> bool mask`` for ``"x_min"``/``"x_max"``/``[...]``/``"boundary"`` (every spatial face)."""
    import numpy as np

    faces = [on] if isinstance(on, str) else list(on)
    if faces == ["boundary"]:
        faces = [f"{c}_{s}" for c in coords if c != "t" for s in ("min", "max")]
    parsed = []
    for f in faces:
        try:
            coord, side = str(f).rsplit("_", 1)
        except ValueError:
            raise ConfigError(f"condition 'on' must be 'boundary', 'x_min', 'x_max' or a list of those, "
                              f"got {f!r}") from None
        if coord not in coords or side not in ("min", "max"):
            raise ConfigError(f"condition 'on' names face {f!r}, which the problem (coords {tuple(coords)}) does "
                              f"not have")
        parsed.append((list(coords).index(coord), side, bounds[coord]))

    def mask(X):
        X = np.asarray(X, dtype=float)
        m = np.zeros(len(X), dtype=bool)
        for i, side, (lo, hi) in parsed:
            tol = 1e-6 * max(abs(hi - lo), 1e-12)
            m |= X[:, i] <= lo + tol if side == "min" else X[:, i] >= hi - tol
        return m

    return mask


def _expression_fn(expression: str, coords: Sequence[str]):
    """``value_fn`` from a Python expression in the coordinate names (``t`` included)."""
    import numpy as np

    code = compile(expression, "<condition expression>", "eval")

    def fn(X, ctx=None):
        X = np.asarray(X, dtype=float)
        env = {str(c): X[:, i] for i, c in enumerate(coords)}
        env.update(np=np, sin=np.sin, cos=np.cos, tan=np.tan, exp=np.exp, log=np.log, sqrt=np.sqrt,
                   pi=np.pi, abs=np.abs, tanh=np.tanh)
        out = eval(code, {"__builtins__": {}}, env)               # noqa: S307 - the expression is the user's own config
        arr = np.asarray(out, dtype=float)
        if arr.ndim == 0:
            arr = np.full(len(X), float(arr))
        return arr.reshape(len(X), -1)

    return fn


# --------------------------------------------------------------------------------------------------- parameters
@dataclass
class ParameterSpaceConfig(_Block):
    """The family of problems the surrogate must cover: which parameters vary, and over what.

    ``parameters`` maps a name of the problem's own parameters to its range::

        parameters:
          nu: {min: 0.005, max: 0.06, scale: linear}        # or simply [0.005, 0.06]
          material: {kind: categorical, values: [steel, aluminium]}

    ``constraints`` are expressions that must hold between parameters (``"temperature - 273.15 > 0"``), evaluated
    by the design of experiments of ``pinneapple_data.doe``. An empty space is a one-case pipeline (the preset's
    own parameter values).
    """

    parameters: dict[str, Any] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    budget: int | None = None

    def validate(self) -> None:
        self.constraints = _as_tuple(self.constraints)
        for name, spec in _mapping(self.parameters).items():
            if isinstance(spec, (list, tuple)) and len(spec) == 2:
                lo, hi = float(spec[0]), float(spec[1])
                if not lo < hi:
                    raise ConfigError(f"parameter {name!r}: range {spec!r} is not increasing")
            elif isinstance(spec, Mapping):
                kind = str(spec.get("kind", "continuous"))
                if kind not in ("continuous", "integer", "categorical", "constant"):
                    raise ConfigError(f"parameter {name!r}: kind must be continuous, integer, categorical or "
                                      f"constant, got {kind!r}")
                if kind == "categorical" and not spec.get("values"):
                    raise ConfigError(f"parameter {name!r}: a categorical parameter needs `values`")
            else:
                raise ConfigError(f"parameter {name!r}: give [min, max] or a mapping with min/max/kind")

    def to_space(self, name: str = "pipeline_parameters"):
        """A ``pinneapple_data.contracts.ParameterSpace`` (``None`` when the space is empty)."""
        if not self.parameters:
            return None
        from pinneapple_data.contracts import Parameter, ParameterSpace

        params = []
        for pname, spec in self.parameters.items():
            if isinstance(spec, (list, tuple)):
                spec = {"min": spec[0], "max": spec[1]}
            spec = dict(spec)
            for src, dst in (("min", "low"), ("max", "high")):
                if src in spec and dst not in spec:
                    spec[dst] = spec.pop(src)
            params.append(Parameter(name=pname, **{k: v for k, v in spec.items()
                                                   if k in ("low", "high", "kind", "values", "log", "value", "unit",
                                                            "group", "default", "description")}))
        return ParameterSpace(name=name, parameters=params, constraints=list(self.constraints),
                              budget=self.budget)

    @property
    def names(self) -> list[str]:
        return list(self.parameters)


# --------------------------------------------------------------------------------------------------- sampling
@dataclass
class SamplingConfig(_Block):
    """How many cases, and where they are.

    ``design_sampling``: the cases of the parametric study (``method``: ``lhs``, ``maximin``, ``sobol``, ``halton``,
    ``random``, ``full_factorial``, ``full_factorial_1level``; ``n_samples``). ``collocation_sampling``: points
    inside one problem for a physics-only model (``n_interior``). ``validation_sampling``: the held-out fraction of
    cases, and whether a distribution-shift set is carved out of the corner of the space (``split.shift``).
    """

    design_sampling: dict[str, Any] = field(default_factory=lambda: {"method": "lhs", "n_samples": 16})
    collocation_sampling: dict[str, Any] = field(default_factory=lambda: {"n_interior": 2048})
    validation_sampling: dict[str, Any] = field(default_factory=lambda: {"test_fraction": 0.2,
                                                                        "val_fraction": 0.15, "shift": "auto"})
    seed: int = 0

    def validate(self) -> None:
        method = str(self.design_sampling.get("method", "lhs"))
        from pinneapple_data.doe.designs import METHODS
        if method not in METHODS:
            raise ConfigError(f"sampling.design_sampling.method must be one of {METHODS}, got {method!r}")
        n = int(self.design_sampling.get("n_samples", 16))
        if n < 12:
            raise ConfigError(f"sampling.design_sampling.n_samples must be at least 12: with the default 15 % "
                              f"distribution-shift set, a supervised study needs 10 cases left to train, validate and "
                              f"test on (got {n})")
        self.design_sampling["method"], self.design_sampling["n_samples"] = method, n
        self.collocation_sampling.setdefault("n_interior", 2048)
        v = self.validation_sampling
        test, val = float(v.get("test_fraction", 0.2)), float(v.get("val_fraction", 0.15))
        if not (0 < test < 1 and 0 <= val < 1 and test + val < 1):
            raise ConfigError(f"sampling.validation_sampling fractions must satisfy 0 < test < 1, 0 <= val and "
                              f"test + val < 1, got test={test}, val={val}")


# --------------------------------------------------------------------------------------------------- solver
@dataclass
class SolverConfig(_Block):
    """The reference solution the data comes from (or is validated against).

    ``method`` is a name of ``pinneapple_physics.solving``: ``reference`` (the problem's own classical solver),
    ``analytic`` (a closed form the library knows), ``fem`` or ``exact``. ``options`` are that method's own
    options; ``fidelity`` is recorded with the data (``reference`` and ``analytic`` are different grounds for
    trusting a number, and the report says which one was used).
    """

    method: str = "reference"
    options: dict[str, Any] = field(default_factory=dict)
    fidelity: str = "high"
    on_failure: str = "fail"          # fail | skip

    def validate(self) -> None:
        from pinneapple_physics.solving import list_methods
        known = list_methods()
        if isinstance(self.method, str) and self.method not in known:
            raise ConfigError(f"solver.method must be one of {known}, got {self.method!r}")
        if self.on_failure not in ("fail", "skip"):
            raise ConfigError(f"solver.on_failure must be 'fail' or 'skip', got {self.on_failure!r}")


# --------------------------------------------------------------------------------------------------- data
@dataclass
class DataConfig(_Block):
    """What the model learns from.

    ``source="physics_only"``: nothing but the equations (a PINN); no reference solve, no dataset split.
    ``source="simulation"``: the parameter space is sampled, each case is solved with ``solver``, and the outputs are
    the quantities of interest of ``outputs`` (``"field"`` = the solution sampled on the probe points, or explicit
    names of ``PhysicalProblem.quantities``).
    ``source="file"``: an ``.npz``/``.csv`` dataset with ``inputs`` and ``outputs`` columns (``X``/``Y`` arrays).
    """

    source: str = "simulation"
    path: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    probes: dict[str, int] = field(default_factory=dict)          # coordinate -> number of probe values
    probe_at: dict[str, float] = field(default_factory=dict)      # coordinate -> the one value it is held at
    input_variables: tuple[str, ...] = ()
    output_variables: tuple[str, ...] = ()

    def validate(self) -> None:
        self.source = str(self.source).lower()
        if self.source not in ("physics_only", "simulation", "file"):
            raise ConfigError(f"data.source must be physics_only, simulation or file, got {self.source!r}")
        self.inputs, self.outputs = _as_tuple(self.inputs), _as_tuple(self.outputs)
        self.probes = {str(k): int(v) for k, v in _mapping(self.probes).items()}
        self.probe_at = {str(k): float(v) for k, v in _mapping(self.probe_at).items()}
        both = sorted(set(self.probes) & set(self.probe_at))
        if both:
            raise ConfigError(f"data.probes and data.probe_at both set coordinate(s) {both}: give a coordinate a "
                              "count (it varies) or a value (it is held fixed), not both")
        if self.source == "file" and not self.path:
            raise ConfigError("data.source is 'file' but data.path is empty")

    @property
    def is_physics_only(self) -> bool:
        return self.source == "physics_only"


# --------------------------------------------------------------------------------------------------- architecture
@dataclass
class ArchitectureConfig(_Block):
    """The model family: one name of ``pp.list_models()``, or ``family="auto"`` to try ``candidates``.

    ``parameters`` are handed to the registry as given (``hidden_dim``, ``n_layers``, ``modes``, ``width``...). For
    the parametric surrogate the family is usually one of ``gp``, ``ridge``, ``mlp``, ``fno``, ``deeponet`` — the
    methods ``pinneapple_analysis.surrogate`` knows how to fit, split and gate; for the physics-only path it is a
    registry architecture (``modified_mlp``, ``siren``, ``vanilla_pinn``...).
    """

    family: str = "auto"
    candidates: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.candidates = _as_tuple(self.candidates)
        if self.family == "auto" and not self.candidates:
            self.candidates = ("gp", "mlp")

    @property
    def is_auto(self) -> bool:
        return self.family == "auto"


# --------------------------------------------------------------------------------------------------- training
@dataclass
class TrainingConfig(_Block):
    """Training budget and optimiser settings, in the vocabulary of the backends that run them."""

    epochs: int = 2000
    lr: float = 1e-3
    batch_size: int = 128
    n_collocation: int = 2048
    hidden_dim: int | None = None
    n_layers: int | None = None
    optimizer: str = "adam"
    scheduler: str = "none"
    grad_clip: float = 0.0
    seed: int = 0
    device: str = "auto"
    budget_seconds: float | None = None          # per training run, for the study backend
    lbfgs_steps: int = 0                         # symbolic backend: quasi-Newton polish after Adam
    early_stopping: bool = False
    patience: int = 100

    def validate(self) -> None:
        if self.epochs < 1:
            raise ConfigError(f"training.epochs must be at least 1, got {self.epochs}")
        if self.lr <= 0:
            raise ConfigError(f"training.lr must be positive, got {self.lr}")
        if self.optimizer not in ("adam", "adamw", "lbfgs"):
            raise ConfigError(f"training.optimizer must be adam, adamw or lbfgs, got {self.optimizer!r}")
        if self.scheduler not in ("none", "cosine", "plateau"):
            raise ConfigError(f"training.scheduler must be none, cosine or plateau, got {self.scheduler!r}")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ConfigError(f"training.device must be auto, cpu, cuda or mps, got {self.device!r}")
        if self.early_stopping:
            raise ConfigError("training.early_stopping is not implemented: the backends train for a fixed budget "
                              "(use training.epochs, or training.budget_seconds on the study backend)")

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            from pinneapple_neural.trainer import best_device
            return str(best_device())
        except Exception:                                            # pragma: no cover - torch always present here
            return "cpu"


# --------------------------------------------------------------------------------------------------- loss
@dataclass
class LossConfig(_Block):
    """The weights of the physics loss, when the backend takes them (``w_pde``, ``w_bc``, ``w_ic``, ``w_data``)."""

    weights: dict[str, float] = field(default_factory=lambda: {"w_pde": 1.0, "w_bc": 10.0, "w_ic": 10.0,
                                                              "w_data": 1.0})
    balancing: str = "fixed"
    normalization: str = "physical_scale"

    def validate(self) -> None:
        from pinneapple_physics.pinn_solver.compiler.loss import LossWeights
        known = {f.name for f in dataclasses.fields(LossWeights)}
        unknown = sorted(set(self.weights) - known)
        if unknown:
            raise ConfigError(f"loss.weights has {unknown}; the compiled physics loss takes {sorted(known)}")
        if self.balancing not in ("fixed",):
            raise ConfigError(f"loss.balancing={self.balancing!r} is not implemented; only 'fixed' weights are "
                              "applied by the compiled loss")

    def to_weights(self):
        from pinneapple_physics.pinn_solver.compiler.loss import LossWeights
        return LossWeights(**{k: float(v) for k, v in self.weights.items()})


# --------------------------------------------------------------------------------------------------- execution
@dataclass
class ExecutionConfig(_Block):
    """Where the run writes and how much wall clock it may use.

    ``out_dir``: every artefact of the run (checkpoints, dataset, figures, the evidence report) goes under
    ``<out_dir>/<name>/``. ``profile`` adjusts thread counts and defaults but never a tolerance: a profile cannot
    loosen verification. ``resume`` re-uses a finished stage whose artefacts are still on disk.
    """

    out_dir: str = "runs/surrogate"
    profile: str = "balanced"
    cache: bool = True
    resume: bool = True
    threads: int = 1
    max_wall_time: float | None = None

    def validate(self) -> None:
        if self.profile not in ("fast", "balanced", "accurate", "low_memory"):
            raise ConfigError(f"execution.profile must be fast, balanced, accurate or low_memory, got "
                              f"{self.profile!r}")
        if self.threads < 1:
            raise ConfigError(f"execution.threads must be at least 1, got {self.threads}")


# --------------------------------------------------------------------------------------------------- verification
_GATE_NAMES = ("problem", "geometry", "discretization", "simulation", "dataset", "model", "physics", "applicability",
               "export", "inference")


@dataclass
class VerificationConfig(_Block):
    """VeriPhysics: what must hold, and what happens when it does not.

    ``mode``: ``strict`` fails on any ``WARN``; ``balanced`` fails only on ``FAIL`` (the default); ``report`` never
    fails the run and records every result. ``acceptance`` sets the thresholds each gate compares against, per
    quantity: ``relative_l2``, ``nrmse``, ``residual``, ``bc``, ``export_equivalence``. The numbers below are the
    defaults measured by the analysis modules, not universal tolerances — set them for the use case::

        verification:
          mode: strict
          acceptance: {relative_l2: 0.02, residual: 0.05}
          fail_policy: block
    """

    enabled: bool = True
    mode: str = "balanced"
    verify_each_stage: bool = True
    fail_policy: str = "block"                 # block | warn | report
    acceptance: dict[str, float] = field(default_factory=dict)
    gates: dict[str, bool] = field(default_factory=dict)     # name -> run it (default: every applicable gate)
    validation_queries: int = 256
    run_checks: bool = True                    # also run `pinneapple check` over the run folder

    def validate(self) -> None:
        if self.mode not in ("strict", "balanced", "report"):
            raise ConfigError(f"verification.mode must be strict, balanced or report, got {self.mode!r}")
        if self.fail_policy not in ("block", "warn", "report"):
            raise ConfigError(f"verification.fail_policy must be block, warn or report, got {self.fail_policy!r}")
        unknown = sorted(set(self.gates) - set(_GATE_NAMES))
        if unknown:
            raise ConfigError(f"verification.gates names {unknown}; the gates are {list(_GATE_NAMES)}")
        self.acceptance = {str(k): float(v) for k, v in _mapping(self.acceptance).items()}
        for k, v in self.acceptance.items():
            if v < 0:
                raise ConfigError(f"verification.acceptance[{k!r}] must be non-negative, got {v}")

    def threshold(self, name: str, default: float) -> float:
        return float(self.acceptance.get(name, default))

    def wants(self, gate: str) -> bool:
        return self.gates.get(gate, True)


# --------------------------------------------------------------------------------------------------- uncertainty
@dataclass
class UncertaintyConfig(_Block):
    """Spread of the prediction: seed-to-seed for a study (each seed refits the model), or a deep ensemble for a
    physics-only model (each member retrained from a different seed)."""

    enabled: bool = False
    method: str = "seed_ensemble"
    n_models: int = 3
    confidence_level: float = 0.95

    def validate(self) -> None:
        if self.method not in ("seed_ensemble",):
            raise ConfigError(f"uncertainty.method must be seed_ensemble (the only method wired), got {self.method!r}")
        if self.enabled and self.n_models < 2:
            raise ConfigError(f"uncertainty.n_models must be at least 2 for an ensemble, got {self.n_models}")
        if not 0 < self.confidence_level < 1:
            raise ConfigError(f"uncertainty.confidence_level must be in (0, 1), got {self.confidence_level}")


# --------------------------------------------------------------------------------------------------- export
@dataclass
class ExportConfig(_Block):
    """What leaves the pipeline: the checkpoint always, the portable formats when asked, and the card that says what
    the model is (and is not) validated for. ``verify_export`` compares the exported model against the in-memory one
    and fails the ``export`` gate when they disagree."""

    enabled: bool = True
    formats: tuple[str, ...] = ("pytorch",)
    include_metadata: bool = True
    include_verification: bool = True
    verify_export: bool = True
    tolerance: float = 1e-5

    def validate(self) -> None:
        self.formats = _as_tuple(self.formats)
        if not self.formats and not self.enabled:
            return
        known = ("pytorch", "torchscript", "onnx", "npz")
        unknown = [f for f in self.formats if f not in known]
        if unknown:
            raise ConfigError(f"export.formats has {unknown}; known: {list(known)}")


# --------------------------------------------------------------------------------------------------- visualization
@dataclass
class VisualizationConfig(_Block):
    """Figures written next to the artefacts (parity, predicted vs reference field, training loss, error map).
    A missing matplotlib is not an error: the gate records that the figures were not produced."""

    enabled: bool = True
    plots: tuple[str, ...] = ("training_loss", "parity", "error_map")
    formats: tuple[str, ...] = ("png",)
    dpi: int = 130

    def validate(self) -> None:
        self.plots, self.formats = _as_tuple(self.plots), _as_tuple(self.formats) or ("png",)


# --------------------------------------------------------------------------------------------------- aggregate
@dataclass
class SurrogateConfig(_Block):
    """Everything a :class:`~pinneapple_pipeline.SurrogatePipeline` needs, loadable from YAML or JSON.

    Required: ``name`` and ``problem``. Every other block has a default that a small problem can run with::

        name: burgers_in_viscosity
        problem: {source: preset, preset: burgers_1d}
        parameters: {nu: [0.005, 0.06]}
        data: {source: simulation}
        training: {epochs: 800}
    """

    name: str = ""
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    dimension: DimensionConfig = field(default_factory=DimensionConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    boundary_conditions: tuple[BoundaryConditionConfig, ...] = ()
    parameters: ParameterSpaceConfig = field(default_factory=ParameterSpaceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    description: str = ""

    # ---- nested blocks are built from their dataclass when given as a mapping
    _BLOCKS = ("problem", "dimension", "physics", "parameters", "sampling", "solver", "data", "architecture",
               "training", "loss", "execution", "verification", "uncertainty", "export", "visualization")

    def __post_init__(self) -> None:
        for name in self._BLOCKS:
            value = getattr(self, name)
            if isinstance(value, Mapping):
                setattr(self, name, _BLOCK_TYPES[name].from_dict(value))
        if self.boundary_conditions:
            self.boundary_conditions = tuple(
                c if isinstance(c, BoundaryConditionConfig) else BoundaryConditionConfig.from_dict(c)
                for c in self.boundary_conditions)

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("name is required (it is the run's folder under execution.out_dir)")
        if not self.name.replace("_", "").replace("-", "").isalnum():
            raise ConfigError(f"name {self.name!r} must be letters, digits, '-' and '_' (it becomes a folder name)")
        for name in self._BLOCKS:
            getattr(self, name).validate()
        for c in self.boundary_conditions:
            c.validate()
        if self.data.source == "simulation" and not self.parameters.parameters:
            raise ConfigError("data.source is 'simulation' but parameters is empty: a supervised surrogate needs a "
                              "parameter to vary (or set data.source: physics_only)")
        if self.data.source == "physics_only" and self.architecture.is_auto:
            self.architecture.candidates = ("modified_mlp", "siren")
        if self.parameters.parameters and not self.data.is_physics_only:
            unknown = sorted(set(self.parameters.parameters) - set(self._problem_parameter_names()))
            if unknown and self.problem.source == "preset":
                known = sorted(self._problem_parameter_names())
                raise ConfigError(f"parameters names {unknown}, which preset {self.problem.preset!r} does not have; "
                                  f"it has {known}")

    def _problem_parameter_names(self) -> set[str]:
        if self.problem.source != "preset":
            return set()
        try:
            import pinneapple as pp
            spec = pp.get_preset(self.problem.preset, **dict(self.problem.preset_kwargs))
            return set(spec.pde.params)
        except Exception:
            return set()

    # ---- files
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> SurrogateConfig:
        """Read a YAML (or JSON) pipeline file; ``overrides`` replace top-level blocks by name."""
        import yaml
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"{p}: a pipeline file is a mapping, got {type(data).__name__}")
        return cls.from_dict({**data, **overrides})

    from_json = from_yaml

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None):
        data = dict(data or {})
        known = {f.name for f in dataclasses.fields(cls)} - {"_BLOCKS"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown top-level key(s) {unknown}; a pipeline file takes {sorted(known)}")
        obj = cls(**data)
        obj.validate()
        return obj

    def to_yaml(self, path: str | Path | None = None) -> str:
        import yaml
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=100)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for f in dataclasses.fields(self):
            if f.name == "_BLOCKS":
                continue
            v = getattr(self, f.name)
            if dataclasses.is_dataclass(v):
                v = v.to_dict()
            elif isinstance(v, tuple) and v and all(dataclasses.is_dataclass(x) for x in v):
                v = [x.to_dict() for x in v]
            elif isinstance(v, tuple):
                v = list(v)
            elif dataclasses.is_dataclass(v):
                v = v.to_dict()
            elif isinstance(v, (Path,)):
                v = str(v)
            out[f.name] = v
        return json.loads(json.dumps(out, default=str))


#: ``pp.PipelineConfig``: the aggregate under a name that does not collide with
#: ``pinneapple_design.design_optimizer.surrogate.SurrogateConfig`` (already exported as ``pp.SurrogateConfig``).
PipelineConfig = SurrogateConfig

#: Which dataclass each block name builds (used by :meth:`SurrogateConfig.__post_init__`).
_BLOCK_TYPES: dict[str, type] = {
    "problem": ProblemConfig, "dimension": DimensionConfig, "physics": PhysicsConfig,
    "parameters": ParameterSpaceConfig, "sampling": SamplingConfig, "solver": SolverConfig, "data": DataConfig,
    "architecture": ArchitectureConfig, "training": TrainingConfig, "loss": LossConfig,
    "execution": ExecutionConfig, "verification": VerificationConfig, "uncertainty": UncertaintyConfig,
    "export": ExportConfig, "visualization": VisualizationConfig,
}
