"""The configuration objects of :class:`~pinneapple_pipeline.SurrogatePipeline`.

One dataclass per block of the pipeline, each with ``to_dict()`` and ``from_dict()`` so a pipeline can be written
as YAML, checked in, and re-run elsewhere. Nothing here builds or trains anything: a configuration says *what* is
to be done, and :mod:`pinneapple_pipeline.adapters` maps it onto the modules that already know how to do it
(``pinneapple_physics.pde_environment`` presets, ``pinneapple_data.doe`` designs, ``pinneapple_physics.solving``
reference solvers, ``pinneapple_neural`` architectures and trainer, ``pinneapple_check``, ``pinneapple_tools``
model export).

Every block here is run by the pipeline: a configuration field that silently does nothing is worse than an absent
one, so a block that the library cannot back yet raises :class:`ConfigError` with the reason instead of being
accepted (``geometry.repair``, ``materials.temperature_dependent``).
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DimensionConfig", "ProblemConfig", "PhysicsConfig", "BoundaryConditionConfig", "GeometryConfig",
    "MaterialConfig", "ParameterSpaceConfig", "SamplingConfig", "SolverConfig", "DataConfig",
    "ArchitectureConfig", "TrainingConfig", "LossConfig", "OptimizationConfig", "AdaptiveLearningConfig",
    "ExecutionConfig", "VerificationConfig", "UncertaintyConfig", "DeploymentConfig", "ExportConfig",
    "VisualizationConfig", "SurrogateConfig", "PipelineConfig", "DIMENSION_KEYS",
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

    A symbolic problem may also declare the exact solution it is supposed to have, which is what turns the ``model``
    gate from *not evaluated* into a measured error: ``exact_expression`` is a Python expression in the coordinates
    (``"exp(-2*alpha*pi**2*t) * sin(pi*x) * sin(pi*y)"``), or pass a callable ``X -> (n, len(fields))`` as
    ``exact`` from Python. It is used by ``validate_pinn`` (the ``pinn.exact`` check) and therefore by the gate.
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
    exact_expression: str = ""
    exact: Any = None                         # X -> (n, len(fields)), for a manufactured/known solution
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
        if self.exact_expression and self.source != "symbolic":
            raise ConfigError("problem.exact_expression only applies to a symbolic problem (source: symbolic)")
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


def _expression_fn(expression: str, coords: Sequence[str], constants: Mapping[str, float] | None = None):
    """``value_fn`` from a Python expression in the coordinate names (``t`` included) and the problem's constants."""
    import numpy as np

    code = compile(expression, "<condition expression>", "eval")

    def fn(X, ctx=None):
        X = np.asarray(X, dtype=float)
        env = {str(c): X[:, i] for i, c in enumerate(coords)}
        env.update(np=np, sin=np.sin, cos=np.cos, tan=np.tan, exp=np.exp, log=np.log, sqrt=np.sqrt,
                   pi=np.pi, abs=np.abs, tanh=np.tanh)
        env.update({str(k): float(v) for k, v in dict(constants or {}).items()})
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
    """Training budget and optimiser settings, in the vocabulary of the backends that run them.

    ``early_stopping`` stops the run when the monitored quantity stops improving for ``patience`` checks:

    * ``preset``/``spec`` problems: a held-out physics batch (the same compiled loss on points the model is not
      trained on), every ``check_every`` epochs — ``Trainer`` + ``EarlyStopping``;
    * ``symbolic`` problems: the PINN validation suite every ``check_every`` steps — ``PINNMonitor``, which stops on
      a repeated *physics* failure (a trivial solution, boundary or initial conditions lost), not on the loss;
    * parametric surrogates: the study backend already stops at ``budget_seconds`` per candidate, so the flag is
      recorded and has no second meaning there.
    """

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
    patience: int = 5                            # checks without improvement before stopping
    check_every: int = 100                       # epochs between two checks
    min_delta: float = 0.0

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
            if self.patience < 1:
                raise ConfigError(f"training.patience must be at least 1 check, got {self.patience}")
            if self.check_every < 1:
                raise ConfigError(f"training.check_every must be at least 1 epoch, got {self.check_every}")
            if self.check_every >= self.epochs:
                raise ConfigError(f"training.check_every ({self.check_every}) is not smaller than training.epochs "
                                  f"({self.epochs}): no check would ever run")

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

#: ``pinneapple_check``'s own physics names (``verification.check_physics``).
_CHECK_PHYSICS = ("none", "incompressible-flow", "heat", "elasticity", "conservation", "darcy", "maxwell")


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
    check_physics: str = "none"                # the model's own physics, for pinneapple check's PC2xx/PC3xx
    check_queries: int = 32                    # queries the check runs through the model

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
        if self.check_physics not in _CHECK_PHYSICS:
            raise ConfigError(f"verification.check_physics must be one of {list(_CHECK_PHYSICS)}, got "
                              f"{self.check_physics!r}")
        if self.check_queries < 1:
            raise ConfigError(f"verification.check_queries must be at least 1, got {self.check_queries}")

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
    plots: tuple[str, ...] = ("training_loss", "parity", "error_map", "geometry")
    formats: tuple[str, ...] = ("png",)
    dpi: int = 130

    def validate(self) -> None:
        self.plots, self.formats = _as_tuple(self.plots), _as_tuple(self.formats) or ("png",)


# --------------------------------------------------------------------------------------------------- geometry
@dataclass
class GeometryConfig(_Block):
    """A geometry read from a file, checked before anything is trained on it.

    ``source="file"`` loads a surface with ``pinneapple_data.cae.io.read_stl`` and runs
    ``pinneapple_data.cae.quality.surface_checks`` on it: watertightness, open and non-manifold edges, the bounding
    box and the area. The result is the ``geometry`` gate's evidence — a surface that is not closed cannot bound a
    volume, and the run says so instead of meshing it. ``source="none"`` (the default) means the problem is a box
    and there is nothing to check.

    ``units`` scales the file to metres (``mm``, ``cm``, ``m``, ``in``); ``scale`` multiplies on top of it, for a
    file whose units are unknown. This pipeline verifies the surface, it does not mesh it: a volume mesh is
    ``pinneapple_design.geometry.mesh``'s job, called from a stage of your own.
    """

    source: str = "none"
    path: str = ""
    format: str = "stl"
    units: str = "m"
    scale: float = 1.0
    repair: bool = False
    regions: tuple[str, ...] = ()
    builder: str = ""                          # source="builder": cylinder | box | sphere | annulus
    parameters: dict[str, Any] = field(default_factory=dict)

    _UNITS = {"m": 1.0, "cm": 0.01, "mm": 1e-3, "um": 1e-6, "in": 0.0254, "ft": 0.3048}
    _BUILDERS = ("cylinder", "box", "sphere", "annulus")

    def validate(self) -> None:
        self.source = str(self.source).lower()
        self.regions = _as_tuple(self.regions)
        if self.source not in ("none", "file", "builder"):
            raise ConfigError(f"geometry.source must be 'none' (a box), 'file' or 'builder', got {self.source!r}")
        if self.source == "builder":
            if self.builder not in self._BUILDERS:
                raise ConfigError(f"geometry.builder must be one of {list(self._BUILDERS)}, got {self.builder!r}")
            self.parameters = {str(k): v for k, v in _mapping(self.parameters).items()}
        if self.source == "file":
            if not self.path:
                raise ConfigError("geometry.source is 'file' but geometry.path is empty")
            if self.format.lower() != "stl":
                raise ConfigError(f"geometry.format {self.format!r} is not read yet: the surface loader here is STL "
                                  "(pinneapple_data.cae.io.read_stl); convert the file or load it in a stage of your "
                                  "own")
            self.format = "stl"
        if self.units not in self._UNITS:
            raise ConfigError(f"geometry.units must be one of {sorted(self._UNITS)}, got {self.units!r}")
        if self.scale <= 0:
            raise ConfigError(f"geometry.scale must be positive, got {self.scale}")
        if self.repair:
            raise ConfigError("geometry.repair is not implemented: a surface that is not watertight is reported by "
                              "the geometry gate, not silently stitched (pinneapple_design.geometry has the repair "
                              "tools if you want to run them yourself)")

    @property
    def length_scale(self) -> float:
        """Metres per unit of the file."""
        return self._UNITS[self.units] * float(self.scale)


# --------------------------------------------------------------------------------------------------- materials
@dataclass
class MaterialConfig(_Block):
    """A material from the catalogue (``pinneapple_design.intent.materials.MATERIALS``, 7 of them) applied to the
    problem's own parameters.

    ``materials`` maps a body name to a catalogue key (``steel_1018``, ``al6061_t6``, ``ss304``, ``zamak3``,
    ``pla_4043d``, ``balsa``, ``al6063_t5``) or to explicit properties::

        materials:
          materials:
            body: al6061_t6                      # everything the problem declares, from the catalogue
            insert: {youngs_modulus: 1.2e11, poisson: 0.34}
          assignments: {body: {E: youngs_modulus, nu: poisson}}

    A property is applied to a problem parameter of the same name when the problem has it; ``assignments`` adds
    explicit pairs for a problem that names its parameters differently (``E``, ``k_eff``...). Every value comes with
    the catalogue's provenance (the datasheet it was read from), which the model card records.
    """

    materials: dict[str, Any] = field(default_factory=dict)
    assignments: dict[str, dict[str, str]] = field(default_factory=dict)
    properties: tuple[str, ...] = ()
    temperature_dependent: bool = False

    def validate(self) -> None:
        self.materials = {str(k): v for k, v in _mapping(self.materials).items()}
        self.assignments = {str(body): {str(p): str(prop) for p, prop in _mapping(pairs).items()}
                            for body, pairs in _mapping(self.assignments).items()}
        self.properties = _as_tuple(self.properties)
        if self.temperature_dependent:
            raise ConfigError("materials.temperature_dependent is not implemented: the catalogue carries "
                              "single-temperature properties, and a temperature-dependent law needs its own model")
        unknown = sorted(set(self.assignments) - set(self.materials))
        if unknown:
            raise ConfigError(f"materials.assignments names {unknown}, which materials.materials does not define")
        for body, spec in self.materials.items():
            if isinstance(spec, str):
                continue
            if not isinstance(spec, Mapping) or not spec:
                raise ConfigError(f"material {body!r} must be a catalogue key (a string) or a mapping of property "
                                  "values")

    def resolve(self) -> dict[str, dict[str, Any]]:
        """``{body: {property: value}}`` from the catalogue (with provenance) or from the config's own values."""
        from pinneapple_design.intent.materials import MATERIALS

        out: dict[str, dict[str, Any]] = {}
        for body, spec in self.materials.items():
            if isinstance(spec, str):
                if spec not in MATERIALS:
                    raise ConfigError(f"material {body!r}: {spec!r} is not in the catalogue {sorted(MATERIALS)}")
                material = MATERIALS[spec]
                out[body] = {name: {"value": prop.value, "unit": prop.unit, "source": prop.source,
                                    "label": material.label, "key": spec}
                             for name, prop in material.props.items()}
            else:
                out[body] = {str(k): {"value": float(v), "unit": "", "source": "the pipeline configuration"}
                             for k, v in spec.items()}
        return out

    _ALIASES = {"E": "youngs_modulus", "young": "youngs_modulus", "youngs_modulus": "youngs_modulus",
                "nu": "poisson", "poisson": "poisson", "rho": "density", "density": "density",
                "k": "conductivity", "conductivity": "conductivity", "thermal_conductivity": "conductivity",
                "sy": "yield_strength", "yield_strength": "yield_strength",
                "su": "tensile_strength", "tensile_strength": "tensile_strength"}

    def targets(self, parameter_names: set[str] | Sequence[str]) -> dict[str, dict[str, Any]]:
        """``{problem parameter: {value, unit, source, body}}`` — what the materials set on the problem itself.

        A catalogue property lands on the parameter of **the same name** — only the exact names, because ``nu`` is
        the Poisson ratio of a solid and the kinematic viscosity of a flow, and guessing from the name is how a
        material silently changes the physics. Anything named differently goes through ``assignments``. The Lamé
        constants derived from ``E`` and ``nu`` (``lame_lambda``/``mu``) are recomputed when the problem declares
        them, so changing the material cannot leave them stale.
        """
        names = set(parameter_names)
        resolved = self.resolve()
        out: dict[str, dict[str, Any]] = {}
        for body, props in resolved.items():
            pairs: dict[str, str] = dict(self.assignments.get(body, {}))
            for param in names - set(pairs):
                if param in props:
                    pairs[param] = param
            for param, prop in pairs.items():
                if prop not in props:
                    raise ConfigError(f"material {body!r} has no property {prop!r}; it has {sorted(props)}")
                if param not in names:
                    raise ConfigError(f"materials.assignments sends {body!r}.{prop} to parameter {param!r}, which the "
                                      f"problem does not declare; it declares {sorted(names) or 'nothing'}")
                if param in out:
                    raise ConfigError(f"parameter {param!r} is set by both {out[param]['body']!r} and {body!r}")
                out[param] = {**props[prop], "body": body, "property": prop}
        youngs = next((p for p in out if out[p]["property"] == "youngs_modulus"), None)
        poisson = next((p for p in out if out[p]["property"] == "poisson"), None)
        if youngs is not None and poisson is not None:
            e, nu = float(out[youngs]["value"]), float(out[poisson]["value"])
            if not -1.0 < nu < 0.5:
                raise ConfigError(f"the Poisson ratio of {out[poisson]['body']!r} is {nu}: it must be in (-1, 0.5) for "
                                  "the Lamé constants to exist")
            lame_lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            derived = {"lame_lambda": lame_lambda, "lambda_lame": lame_lambda, "mu": e / (2.0 * (1.0 + nu))}
            for param, value in derived.items():
                if param not in names or param in out:
                    continue
                out[param] = {"value": value, "unit": "", "body": out[youngs]["body"],
                              "property": f"derived from {youngs} and {poisson}",
                              "source": "Lamé constants computed from the catalogue's E and nu"}
        return out


# --------------------------------------------------------------------------------------------------- optimization
@dataclass
class OptimizationConfig(_Block):
    """Hyperparameter search over the training and architecture settings, scored on validation data only.

    ``strategy="grid"`` (every point of ``search_space``) or ``"random"`` (``n_trials`` samples; a pair
    ``[min, max]`` is sampled uniformly, a list is sampled from). The objective is the held-out error of each trial —
    ``relative_l2`` against the reference on a validation set for a physics model, the validation split error for a
    parametric surrogate — never the test data. ``n_jobs`` runs the trials with
    ``pinneapple_neural.trainer.run_parallel_sweep``; keep it at 1 when the trials use torch on CPU.

    The parameter names are the ones the backend takes: ``epochs``, ``lr``, ``n_collocation``, ``hidden_dim``,
    ``n_layers``, ``architecture`` (physics) or a method's own options (supervised).
    """

    enabled: bool = False
    strategy: str = "grid"
    n_trials: int = 8
    n_jobs: int = 1
    objectives: tuple[str, ...] = ("validation_error",)
    search_space: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.strategy not in ("grid", "random"):
            raise ConfigError(f"optimization.strategy must be grid or random, got {self.strategy!r}")
        if not self.search_space:
            raise ConfigError("optimization.enabled is true but optimization.search_space is empty")
        for name, values in self.search_space.items():
            if not isinstance(values, (list, tuple)) or not values:
                raise ConfigError(f"optimization.search_space[{name!r}] must be a non-empty list of values or a "
                                  "[min, max] pair")
        if self.strategy == "random" and self.n_trials < 1:
            raise ConfigError(f"optimization.n_trials must be at least 1, got {self.n_trials}")
        if self.n_jobs < 1:
            raise ConfigError(f"optimization.n_jobs must be at least 1, got {self.n_jobs}")

    def grid(self) -> list[dict[str, Any]]:
        """The parameter sets to try (the grid, or ``n_trials`` random draws)."""
        import numpy as np
        names = list(self.search_space)
        if self.strategy == "grid":
            import itertools
            return [dict(zip(names, combo, strict=True))
                    for combo in itertools.product(*(self.search_space[n] for n in names))]
        rng = np.random.default_rng(self.seed)
        out = []
        for _ in range(self.n_trials):
            trial = {}
            for name in names:
                values = self.search_space[name]
                if len(values) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                    lo, hi = float(values[0]), float(values[1])
                    trial[name] = float(rng.uniform(lo, hi)) if not (lo == int(lo) and hi == int(hi)) \
                        else int(rng.integers(int(lo), int(hi) + 1))
                else:
                    trial[name] = values[int(rng.integers(0, len(values)))]
            out.append(trial)
        return out


# ----------------------------------------------------------------------------------------------- adaptive learning
@dataclass
class AdaptiveLearningConfig(_Block):
    """Add cases where the surrogate is worst, until it reaches the target or the budget runs out.

    Each round asks ``pinneapple_data.doe.design(method="infill")`` for the points farthest from the cases already
    solved (weighted by where the outputs vary most among neighbours), solves them with the reference solver,
    appends them to the dataset and retrains. The loop ends on the held-out target
    (``verification.acceptance.nrmse`` unless ``target_error`` says otherwise), on ``max_rounds`` or on
    ``max_total_simulations``. Every round is recorded in the report, with the error that justified the next one.

    Only the parametric path can adapt: a physics-only model has no case to add.
    """

    enabled: bool = False
    strategy: str = "infill"
    max_rounds: int = 3
    samples_per_round: int = 4
    max_total_simulations: int = 60
    target_error: float | None = None

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.strategy != "infill":
            raise ConfigError(f"adaptive_learning.strategy must be 'infill' (the only one wired), got "
                              f"{self.strategy!r}")
        if self.max_rounds < 1:
            raise ConfigError(f"adaptive_learning.max_rounds must be at least 1, got {self.max_rounds}")
        if self.samples_per_round < 1:
            raise ConfigError(f"adaptive_learning.samples_per_round must be at least 1, got {self.samples_per_round}")
        if self.target_error is not None and self.target_error <= 0:
            raise ConfigError(f"adaptive_learning.target_error must be positive, got {self.target_error}")


# --------------------------------------------------------------------------------------------------- deployment
@dataclass
class DeploymentConfig(_Block):
    """Serve the trained model through ``pinneapple_tools.model_services``.

    The service answers ``predict`` with the model and a ``trust`` verdict computed from the validated domain and
    the gates of the run: a query outside the parameter space is refused (``on_ood="reject"``) or flagged
    (``"flag"``), and a run whose gates did not pass is served with ``trust.decision`` set to that verdict rather
    than hidden. ``target="local"`` builds the ``ModelService`` and returns it; ``target="http"`` also starts
    ``make_server`` (``serve_in_thread``) or writes a ``serve.py`` next to the model.
    """

    enabled: bool = False
    target: str = "local"
    host: str = "127.0.0.1"
    port: int = 0                                # 0: the server picks a free port
    api_key: str = ""
    on_ood: str = "reject"
    enforce_validated_domain: bool = True
    serve_in_thread: bool = True

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.target not in ("local", "http"):
            raise ConfigError(f"deployment.target must be local or http, got {self.target!r}")
        if self.on_ood not in ("reject", "flag"):
            raise ConfigError(f"deployment.on_ood must be reject or flag, got {self.on_ood!r}")
        if self.target == "http" and not (0 <= self.port <= 65535):
            raise ConfigError(f"deployment.port must be 0..65535, got {self.port}")


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
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    materials: MaterialConfig = field(default_factory=MaterialConfig)
    parameters: ParameterSpaceConfig = field(default_factory=ParameterSpaceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    adaptive_learning: AdaptiveLearningConfig = field(default_factory=AdaptiveLearningConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    description: str = ""

    # ---- nested blocks are built from their dataclass when given as a mapping
    _BLOCKS = ("problem", "dimension", "physics", "geometry", "materials", "parameters", "sampling", "solver",
               "data", "architecture", "training", "loss", "optimization", "adaptive_learning", "execution",
               "verification", "uncertainty", "deployment", "export", "visualization")

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
        if self.optimization.enabled and not self.data.is_physics_only:
            raise ConfigError("optimization searches the PINN's training knobs (epochs, lr, n_collocation, "
                              "hidden_dim, n_layers). A parametric surrogate is already searched by the study — "
                              "architecture.candidates over methods and training.budget_seconds — and there is no "
                              "second set of hyperparameters to sweep here")
        if self.adaptive_learning.enabled:
            if self.data.is_physics_only:
                raise ConfigError("adaptive_learning needs cases to add, and data.source is 'physics_only': it "
                                  "applies to a parametric surrogate (data.source: simulation)")
            planned = int(self.sampling.design_sampling.get("n_samples", 0) or 0)
            if planned > self.adaptive_learning.max_total_simulations:
                raise ConfigError(f"sampling.design_sampling.n_samples ({planned}) already exceeds "
                                  f"adaptive_learning.max_total_simulations "
                                  f"({self.adaptive_learning.max_total_simulations})")
        if self.materials.materials and self.problem.source == "preset":
            names = self._problem_parameter_names()
            if names and not self.materials.targets(names):
                raise ConfigError(f"materials sets no parameter of preset {self.problem.preset!r}: its parameters are "
                                  f"{sorted(names)}; name the target explicitly under materials.assignments")

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
    "geometry": GeometryConfig, "materials": MaterialConfig,
    "parameters": ParameterSpaceConfig, "sampling": SamplingConfig, "solver": SolverConfig, "data": DataConfig,
    "architecture": ArchitectureConfig, "training": TrainingConfig, "loss": LossConfig,
    "optimization": OptimizationConfig, "adaptive_learning": AdaptiveLearningConfig,
    "execution": ExecutionConfig, "verification": VerificationConfig, "uncertainty": UncertaintyConfig,
    "deployment": DeploymentConfig, "export": ExportConfig, "visualization": VisualizationConfig,
}
