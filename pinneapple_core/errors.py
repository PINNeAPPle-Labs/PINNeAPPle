"""Errors that teach: every message says what broke *and* what to do about it (#46, roadmap A5).

A bare ``ValueError("repeats must be >= 1")`` tells a user that they were wrong and leaves them to guess what
"right" looks like. The errors here carry a second half -- the action -- and every one of them renders as::

    repeats must be >= 1
      Fix: pass repeats=5 (the default) or any integer >= 1
      See: docs/dev/errors_that_teach.md

Three things follow from that shape:

- the *type* is unchanged where it matters: each class also inherits the standard exception a caller already
  catches (``BadValue`` is a ``ValueError``, ``MissingExtra`` an ``ImportError``, ``UnknownName`` a ``KeyError``,
  ``NotConfigured`` a ``RuntimeError``, ``NotSupported`` a ``NotImplementedError``), so ``except ValueError`` keeps
  working and nobody's error handling breaks because a message got better;
- the action is data (``err.fix``), not prose buried in a string, so a tool can group failures by remedy and the
  audit below can be checked mechanically;
- raising one is one line: ``raise BadValue("what broke", fix="what to do")``.

The audit -- the 30 most common errors a user of this library actually hits -- lives in :data:`AUDITED` and is
rendered into ``docs/dev/errors_that_teach.md`` by :func:`audit_markdown`. Each entry names the site, the failure,
and the sentence that now tells the user what to do; ``tests/test_errors_that_teach.py`` triggers every one of them
and refuses an entry whose error does not carry a fix, which is what keeps this file honest instead of decorative.

What counts as teachable, and what does not: a message names an *action* when the user can take it (install the
extra, pass a valid argument, call ``fit`` first, widen the budget). It is not an action to say "invalid input" or
to list the valid values without saying which call takes them. Internal invariant failures (``assert``-style
"this should never happen") are out of scope: there is nothing for the user to do.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AUDITED", "AuditEntry", "BadType", "BadValue", "MissingExtra", "NotConfigured", "NotSupported",
    "OutOfValidityDomain", "PINNeAPPleError", "ShapeMismatch", "UnknownName", "audit", "audit_markdown",
    "install_hint",
]

FIX_LABEL = "Fix:"
DOCS_LABEL = "See:"
THIS_DOC = "docs/dev/errors_that_teach.md"


def install_hint(extra: str, packages: str = "") -> str:
    """The command that installs one extra, the same sentence everywhere: ``pip install "pinneapple[extra]"``."""
    extra = str(extra).strip("[]")
    cmd = f'pip install "pinneapple[{extra}]"'
    return f"{cmd}  (installs {packages})" if packages else cmd


class PINNeAPPleError(Exception):
    """An error with a second half: ``what`` broke, and the ``fix`` the user can apply.

    ``fix`` is required to be non-empty by :func:`audit`, not by the constructor: a site that genuinely has no
    action for the user omits it, and then it must not be listed as an audited error.
    """

    def __init__(self, what: str, *, fix: str = "", docs: str = ""):
        self.what = str(what)
        self.fix = str(fix)
        self.docs = str(docs)
        lines = [self.what]
        if self.fix:
            lines.append(f"  {FIX_LABEL} {self.fix}")
        if self.docs:
            lines.append(f"  {DOCS_LABEL} {self.docs}")
        self.message = "\n".join(lines)
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.what!r}, fix={self.fix!r})"


class BadValue(PINNeAPPleError, ValueError):
    """A value the call cannot work with. Caught by ``except ValueError``."""


class BadType(PINNeAPPleError, TypeError):
    """An argument of the wrong type. Caught by ``except TypeError``."""


class UnknownName(PINNeAPPleError, KeyError):
    """A name (method, problem, preset, solver) that is not registered. Caught by ``except KeyError``."""

    def __str__(self) -> str:  # KeyError would render the message as a repr, quotes and all
        return self.message


class MissingExtra(PINNeAPPleError, ImportError):
    """An optional dependency the requested feature needs. Caught by ``except ImportError``."""


class NotConfigured(PINNeAPPleError, RuntimeError):
    """The object is not ready for this call (not fitted, no domain, no backend). Caught by ``except RuntimeError``."""


class NotSupported(PINNeAPPleError, NotImplementedError):
    """The combination is not implemented, and the message names what is. ``except NotImplementedError``."""


class ShapeMismatch(BadValue):
    """Two arrays whose shapes have to agree do not."""


class OutOfValidityDomain(BadValue):
    """The request sits outside the range where the model is claimed to hold (issue #58)."""


# --------------------------------------------------------------------------- the audit
@dataclass(frozen=True)
class AuditEntry:
    """One audited error site.

    ``where`` is ``module:callable``, importable as written so the test can reach the site and trigger it; ``fix``
    is the sentence the error now carries. ``was`` records what the message used to say, so the change is visible
    in review; an empty ``was`` means the site already pointed at the action and was only brought under the common
    ``Fix:`` shape.
    """
    id: str
    where: str
    failure: str
    was: str
    fix: str
    kind: str = "value"     # value | type | unknown-name | missing-extra | not-configured | not-supported | domain


def _e(id, where, failure, fix, was="", kind="value") -> AuditEntry:
    return AuditEntry(id, where, failure, was, fix, kind)


AUDITED: tuple[AuditEntry, ...] = (
    _e("cost-repeats", "pinneapple_analysis.cost.profiler:measure",
       "measure() called with repeats < 1", "pass repeats=5 (the default) or any integer >= 1",
       was="repeats must be >= 1"),
    _e("cost-sizes", "pinneapple_analysis.cost.profiler:scaling_study",
       "scaling_study() with fewer than three distinct sizes", "pass at least 3 distinct sizes spanning a factor of 10, e.g. [10**3, 10**4, 10**5]",
       was="need at least 3 distinct sizes"),
    _e("cost-complexity", "pinneapple_analysis.cost.profiler:expected_exponent",
       "an unknown complexity string", "use a number or one of '1', 'log n', 'n', 'n log n', 'n^2', 'n^3'",
       was="unknown complexity {declared!r}; use a number or one of [...]"),
    _e("cost-budget", "pinneapple_analysis.cost.profiler:Budget.validate",
       "an operation over its budget", "raise the limit in Budget(...) or reduce the problem size; the violations above name the quantity",
       was="exceeds ... (no next step)"),
    _e("budget-empty", "pinneapple_analysis.cost.equal_budget:ComputeBudget.__post_init__",
       "a ComputeBudget with no limit set", "set at least one of wall_s=..., flops=..., steps=...",
       was="a ComputeBudget needs at least one of wall_s, flops, steps"),
    _e("budget-nonpositive", "pinneapple_analysis.cost.equal_budget:ComputeBudget.__post_init__",
       "a negative or zero budget limit", "use a positive limit; omit the field (None) when that quantity should not be limited",
       was="{name} must be positive, got {v}"),
    _e("l2-shape", "pinneapple_analysis.cost.equal_budget:relative_l2",
       "pred and true of different shapes", "reshape or resample one to the other; relative_l2 compares point by point",
       was="shape mismatch {p.shape} vs {t.shape}"),
    _e("per-query", "pinneapple_analysis.cost.equal_budget:AmortizedCost.per_query",
       "per_query(0) (or any n < 1)", "pass n_queries >= 1: the cost per query is undefined for zero queries",
       was="n_queries must be >= 1"),
    _e("method-unknown", "pinneapple_catalog.methods:get_method",
       "an id that is not in the catalog", "call pinneapple_catalog.methods.list_methods() and use one of those ids",
       was="unknown method 'X'", kind="unknown-name"),
    _e("problem-unknown", "pinneapple_analysis.benchmark.registry:get_problem",
       "a benchmark problem that is not registered", "call pinneapple_analysis.benchmark.problems() and use one of those ids",
       was="unknown problem 'X'; registered: [...]", kind="unknown-name"),
    _e("bench-method-unknown", "pinneapple_analysis.benchmark.registry:get_method",
       "a benchmark method that is not registered", "call pinneapple_analysis.benchmark.methods() and use one of those ids",
       was="unknown method 'X'; registered: [...]", kind="unknown-name"),
    _e("bench-space", "pinneapple_analysis.benchmark.registry:Method.space",
       "asking a method for the hyper-parameter grid it does not declare", "train this method without a search, or add space_<method id>() to the problem",
       was="X has no hyper-parameter space for Y", kind="not-supported"),
    _e("bench-train", "pinneapple_analysis.benchmark.registry:Method.train",
       "training a method on a problem that has no implementation of it", "pick a method the problem implements (see problem.pairs()) or add train_<method id>()",
       was="X cannot train on Y", kind="not-supported"),
    _e("bench-config-budget", "pinneapple_analysis.benchmark.config:BenchmarkConfig",
       "a benchmark config without a budget", "pass budget={'wall_s': 60} (or flops=... / steps=...)",
       was="budget is required (wall_s, flops or steps)"),
    _e("bench-config-fields", "pinneapple_analysis.benchmark.config:BenchmarkConfig.from_dict",
       "an unknown key in the benchmark config", "drop or rename the key: the error lists the known ones",
       was="unknown benchmark config fields: [...]", kind="unknown-name"),
    _e("solver-unknown", "pinneapple_simulation.numerical_solvers.registry:SolverRegistry.spec",
       "an unregistered solver name", "call SolverRegistry.list() and use one of those names",
       was="Unknown solver 'X'. Available: [...]", kind="unknown-name"),
    _e("solver-ctor", "pinneapple_simulation.numerical_solvers.registry:SolverRegistry.build",
       "keyword arguments the solver's constructor does not take", "pass the constructor's own arguments: the error prints the signature",
       was="SolverRegistry.build('X'): ... The constructor is ...", kind="type"),
    _e("trust-not-fitted", "pinneapple_analysis.trust.trust_gate:TrustGate.ood_p_values",
       "scoring OOD before fitting the reference distribution", "call fit(training_coords) or fit_from_bounds(domain_bounds) first",
       was="TrustGate.fit()/.fit_from_bounds() must be called before OOD scoring.", kind="not-configured"),
    _e("trust-no-bounds", "pinneapple_analysis.trust.trust_gate:TrustGate.coord_bounds_from_checkpoint",
       "a checkpoint that records no domain bounds", "re-save the checkpoint with metadata['domain_bounds'], or build the gate with fit(training_coords)",
       was="checkpoint metadata has no 'domain_bounds' entry.", kind="unknown-name"),
    _e("trust-ood-level", "pinneapple_analysis.trust.trust_gate:TrustGate.__init__",
       "an out-of-range ood_level", "pass ood_level between 0 and 1 exclusive, e.g. 0.01 for the 99 % region",
       was="ood_level must be in (0, 1); got {v}"),
    _e("unit-convert-from", "pinneapple_core.physics_tensor:PhysicsTensor.convert",
       "converting a tensor that carries no unit", "build it with pp.tensor(..., units='m/s'), or drop convert()",
       was="'name' has no unit to convert from"),
    _e("unit-convert-across", "pinneapple_core.physics_tensor:PhysicsTensor.convert",
       "converting between two different dimensions", "convert only within a dimension (m -> km); a different dimension is a different quantity",
       was="cannot convert 'm/s' to 'Pa': the dimensions differ"),
    _e("unit-product", "pinneapple_core.physics_tensor:_no_unit_product",
       "multiplying two dimensioned tensors", "drop the labels with .without_labels() first, then label the product with pp.tensor(..., units=...)",
       was="cannot apply 'mul' to ...: a unit algebra would be needed"),
    _e("backend-unknown", "pinneapple_analysis.dynamics._backends:require",
       "asking for a backend name the toolkit does not have", "use one of the four it knows ('torch', 'jax', 'sympy', 'scipy'); capabilities() lists them",
       was="unknown backend {backend!r}; one of [...]"),
    _e("missing-pyarrow", "pinneapple_data.contracts.tables:_pyarrow",
       "writing or reading a Parquet table without pyarrow", "pip install pyarrow",
       was="Parquet needs pyarrow: pip install pyarrow", kind="missing-extra"),
    _e("missing-h5py", "pinneapple_data.contracts.tables:_h5py",
       "writing or reading an HDF5 table without h5py", "pip install h5py",
       was="HDF5 needs h5py: pip install h5py", kind="missing-extra"),
    _e("missing-hub", "pinneapple_catalog.resources:fetch",
       "fetching a hosted resource without the hub extra", install_hint("hub", "huggingface_hub"),
       was='fetch requires the "hub" extra: pip install "pinneapple[hub]"', kind="missing-extra"),
    _e("missing-onnx", "pinneapple_analysis.surrogate.model_source:onnx_function",
       "opening an ONNX model without onnxruntime", "pip install onnxruntime (or the [onnx] extra); TorchScript and a plain callable need nothing",
       was="an ONNX model needs onnxruntime ...", kind="missing-extra"),
    _e("missing-retrieval", "pinneapple_analysis.retrieval.index:_require_sentence_transformers",
       "building a retrieval index without the retrieval extra", install_hint("retrieval", "sentence-transformers"),
       was="requires the 'retrieval' extra ...", kind="missing-extra"),
    _e("backend-missing", "pinneapple_analysis.dynamics._backends:require",
       "asking a dynamics method for a backend that is not installed", "install it with the command the error prints, or pick an installed one from capabilities()",
       was="BackendUnavailable: {backend} is not installed. {cmd}", kind="missing-extra"),
)


def audit() -> tuple[AuditEntry, ...]:
    """The audited errors, checking what a table of promises can be checked for: unique ids, a fix per entry, and
    an importable site."""
    seen: set[str] = set()
    for e in AUDITED:
        if e.id in seen:
            raise BadValue(f"duplicate audit id {e.id!r}", fix="give each AUDITED entry its own id")
        seen.add(e.id)
        if not e.fix.strip():
            raise BadValue(f"audited error {e.id!r} carries no fix",
                           fix=f"write the action the user takes, or drop the entry from AUDITED ({THIS_DOC})")
        if ":" not in e.where:
            raise BadValue(f"audited error {e.id!r} does not name module:callable",
                           fix="write where='my.module:function' so the test can import and trigger it")
    return AUDITED


def audit_markdown() -> str:
    """The audit table as the documentation file carries it (regenerated by ``scripts/build_error_audit.py``)."""
    lines = [
        "# Errors that teach (#46, roadmap A5)",
        "",
        "Every failure names what broke and what to do next. The messages live in the code (",
        "`pinneapple_core/errors.py`), and each one is raised through a class that also inherits the standard",
        "exception a caller already catches, so ``except ValueError`` / ``except ImportError`` keep working.",
        "",
        "This file is generated from `pinneapple_core.errors.AUDITED`; run",
        "`python scripts/build_error_audit.py` after changing the table. `tests/test_errors_that_teach.py`",
        "triggers every entry below and fails if the error comes back without its fix.",
        "",
        f"{len(AUDITED)} audited errors.",
        "",
        "| # | id | site | the failure | was | now says |",
        "|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(AUDITED, start=1):
        was = e.was.replace("|", "\\|") or "(already carried the action: brought under the common shape)"
        lines.append(f"| {i} | `{e.id}` | `{e.where}` | {e.failure} | {was} | {e.fix.replace('|', chr(92) + '|')} |")
    lines += [
        "",
        "## Not audited, on purpose",
        "",
        "- internal invariant failures (\"this should never happen\"): the user has no action, so there is nothing to",
        "  teach; they stay plain asserts inside private functions.",
        "- numerical failures already handled by a check that reports instead of raising (the benchmark ``Check``",
        "  objects, ``pp.doctor()``): those carry their own ``fix`` field and are not errors.",
        "",
    ]
    return "\n".join(lines)
