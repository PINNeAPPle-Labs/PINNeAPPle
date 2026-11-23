"""Every audited error carries the action that clears it (#46, roadmap A5).

The table lives in ``pinneapple_core.errors.AUDITED``; this file is what keeps it honest. For each of the thirty
entries it makes the failure happen for real, then asks the raised error two questions: does it name an action
(``.fix``, or a ``Fix:`` line), and is it still caught by the exception every caller already catches
(``ValueError``, ``ImportError``, ``KeyError``, ``RuntimeError``, ``NotImplementedError``)?

The last test refuses a table that has drifted: each entry's site has to be importable and its source has to carry
``fix=``, so an entry cannot point at a raise that was never rewritten. The documentation page is compared with
``audit_markdown()`` too, so the table, the code and ``docs/dev/errors_that_teach.md`` cannot disagree.
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from pinneapple_core import errors as E
from pinneapple_core.errors import AUDITED, audit, audit_markdown

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "dev" / "errors_that_teach.md"
AUDIT_IDS = [e.id for e in AUDITED]


def _missing(monkeypatch, *modules):
    """Make ``import <module>`` fail as it does on a machine without it (the sys.modules None trick)."""
    for m in modules:
        monkeypatch.setitem(sys.modules, m, None)


# --------------------------------------------------------------------------- triggers
def _cost_budget():
    from pinneapple_analysis.cost import Budget, measure
    Budget(max_time_s=0.0).validate(measure(lambda: sum(range(10_000)), repeats=1, memory=False, flops=False))


def _bench_space():
    from pinneapple_analysis.benchmark.registry import Method, Problem
    Method().space(Problem())


def _bench_train():
    from pinneapple_analysis.benchmark.registry import Method, Problem
    Method().train(Problem(), REPO, {}, 0, None)


def _unit_binop():
    import torch

    from pinneapple_core.physics_tensor import tensor
    a = tensor(torch.zeros(2), units="m/s")
    b = tensor(torch.zeros(2), units="Pa")
    return a * b


def _backend_missing(monkeypatch):
    from pinneapple_analysis.dynamics import _backends
    monkeypatch.setattr(_backends.importlib.util, "find_spec", lambda name: None if name == "jax" else
                        importlib.util.find_spec(name))
    _backends.require("jax", purpose="the Fisher information")


def _missing_hub(monkeypatch):
    from pinneapple_catalog import resources as R
    _missing(monkeypatch, "huggingface_hub")
    hosted = [r for r in R._RESOURCES if getattr(r, "hf_repo", None) and not getattr(r, "research_only", False)]
    if not hosted:
        pytest.skip("no openly licensed hub-hosted resource in the catalog")
    R.fetch(hosted[0].id)




def _solver_registry():
    return importlib.import_module("pinneapple_simulation.numerical_solvers.registry")


def _unit_tensor(**kw):
    import torch

    from pinneapple_core.physics_tensor import tensor
    return tensor(torch.zeros(2), **kw)


def _missing_pyarrow(monkeypatch):
    _missing(monkeypatch, "pyarrow", "pyarrow.parquet")
    importlib.import_module("pinneapple_data.contracts.tables")._pyarrow()


def _missing_h5py(monkeypatch):
    _missing(monkeypatch, "h5py")
    importlib.import_module("pinneapple_data.contracts.tables")._h5py()


def _missing_onnx(monkeypatch):
    _missing(monkeypatch, "onnxruntime")
    importlib.import_module("pinneapple_analysis.surrogate.model_source").onnx_function("m.onnx", ["x"], ["y"])


def _missing_retrieval(monkeypatch):
    _missing(monkeypatch, "sentence_transformers")
    importlib.import_module("pinneapple_analysis.retrieval.index")._require_sentence_transformers()


TRIGGERS = {
    "cost-repeats": lambda: importlib.import_module("pinneapple_analysis.cost.profiler")
    .measure(lambda: 1, repeats=0),
    "cost-sizes": lambda: importlib.import_module("pinneapple_analysis.cost.profiler")
    .scaling_study(lambda n: (lambda: sorted(range(n))), [10, 10, 20]),
    "cost-complexity": lambda: importlib.import_module("pinneapple_analysis.cost.profiler")
    .expected_exponent("n^pi", [10, 100, 1000]),
    "cost-budget": _cost_budget,
    "budget-empty": lambda: importlib.import_module("pinneapple_analysis.cost.equal_budget").ComputeBudget(),
    "budget-nonpositive": lambda: importlib.import_module("pinneapple_analysis.cost.equal_budget")
    .ComputeBudget(wall_s=-1.0),
    "l2-shape": lambda: importlib.import_module("pinneapple_analysis.cost.equal_budget")
    .relative_l2(np.zeros((2, 3)), np.zeros((2, 4))),
    "per-query": lambda: importlib.import_module("pinneapple_analysis.cost.equal_budget").AmortizedCost()
    .per_query(0),
    "method-unknown": lambda: importlib.import_module("pinneapple_catalog.methods").get_method("S999"),
    "problem-unknown": lambda: importlib.import_module("pinneapple_analysis.benchmark.registry")
    .get_problem("no_such_problem"),
    "bench-method-unknown": lambda: importlib.import_module("pinneapple_analysis.benchmark.registry")
    .get_method("no_such_method"),
    "bench-space": _bench_space,
    "bench-train": _bench_train,
    "bench-config-budget": lambda: importlib.import_module("pinneapple_analysis.benchmark.config")
    .BenchmarkConfig(budget={}),
    "bench-config-fields": lambda: importlib.import_module("pinneapple_analysis.benchmark.config")
    .BenchmarkConfig.from_dict({"nope": 1}),
    "solver-unknown": lambda: _solver_registry().SolverRegistry.spec("no_such_solver"),
    "solver-ctor": lambda: (_solver_registry().register_all(),
                            _solver_registry().SolverRegistry.build("fft", no_such_argument=1)),
    "trust-not-fitted": lambda: importlib.import_module("pinneapple_analysis.trust.trust_gate")
    .TrustGate().ood_p_values(np.zeros((1, 2))),
    "trust-no-bounds": lambda: importlib.import_module("pinneapple_analysis.trust.trust_gate")
    .TrustGate.coord_bounds_from_checkpoint({"metadata": {}}),
    "trust-ood-level": lambda: importlib.import_module("pinneapple_analysis.trust.trust_gate")
    .TrustGate(ood_level=2.0),
    "unit-convert-from": lambda: _unit_tensor().convert("m"),
    "unit-convert-across": lambda: _unit_tensor(units="m/s").convert("Pa"),
    "unit-product": _unit_binop,
    "backend-unknown": lambda: importlib.import_module("pinneapple_analysis.dynamics").require("no_such_backend"),
    "backend-missing": _backend_missing,
    "missing-pyarrow": _missing_pyarrow,
    "missing-h5py": _missing_h5py,
    "missing-hub": _missing_hub,
    "missing-onnx": _missing_onnx,
    "missing-retrieval": _missing_retrieval,
}


# A trigger may take the monkeypatch fixture; these are the ones that do.
NEEDS_FIXTURE = {"backend-missing", "missing-pyarrow", "missing-h5py", "missing-hub", "missing-onnx",
                 "missing-retrieval"}

EXPECTED_TYPE = {
    "cost-repeats": ValueError, "cost-sizes": ValueError, "cost-complexity": ValueError,
    "budget-empty": ValueError, "budget-nonpositive": ValueError, "l2-shape": ValueError,
    "per-query": ValueError, "trust-ood-level": ValueError, "unit-convert-from": ValueError,
    "unit-convert-across": ValueError, "unit-product": ValueError,
    "method-unknown": KeyError, "problem-unknown": KeyError, "bench-method-unknown": KeyError,
    "solver-unknown": KeyError, "trust-no-bounds": KeyError, "bench-config-fields": KeyError,
    "bench-space": NotImplementedError, "bench-train": NotImplementedError,
    "trust-not-fitted": RuntimeError, "solver-ctor": TypeError,
    "missing-pyarrow": ImportError, "missing-h5py": ImportError, "missing-hub": ImportError,
    "missing-onnx": ImportError, "missing-retrieval": ImportError, "backend-missing": ImportError,
    "backend-unknown": ValueError, "cost-budget": AssertionError, "bench-config-budget": ValueError,
}


def _raise(entry_id: str, monkeypatch=None):
    trigger = TRIGGERS[entry_id]
    with pytest.raises(BaseException) as box:  # noqa: PT011 - the type is checked in its own test
        trigger(monkeypatch) if entry_id in NEEDS_FIXTURE else trigger()
    return box.value


# --------------------------------------------------------------------------- the tests
def test_the_audit_lists_thirty_errors():
    assert len(audit()) == 30, f"the audit promises thirty errors, found {len(audit())}"


def test_every_audited_error_has_a_trigger():
    assert sorted(TRIGGERS) == sorted(AUDIT_IDS), (
        "an audited error without a trigger is an unproven claim (and a trigger without an entry is undocumented): "
        f"{sorted(set(AUDIT_IDS) - set(TRIGGERS))} / {sorted(set(TRIGGERS) - set(AUDIT_IDS))}")


@pytest.mark.parametrize("entry_id", AUDIT_IDS)
def test_the_error_names_the_action(entry_id, monkeypatch):
    err = _raise(entry_id, monkeypatch)
    fix = getattr(err, "fix", "")
    assert fix.strip() or "Fix:" in str(err), f"{entry_id}: the message names no action: {err}"
    assert len(str(err)) > 20, f"{entry_id}: the message says too little to act on: {err}"


@pytest.mark.parametrize("entry_id", AUDIT_IDS)
def test_the_standard_exception_type_is_preserved(entry_id, monkeypatch):
    """Improving a message must not break ``except ValueError`` (or the type the caller already catches)."""
    err = _raise(entry_id, monkeypatch)
    assert isinstance(err, EXPECTED_TYPE[entry_id]), (
        f"{entry_id}: {type(err).__name__} is no longer a {EXPECTED_TYPE[entry_id].__name__}")


@pytest.mark.parametrize("entry", AUDIT_IDS)
def test_each_site_carries_the_rewrite(entry):
    """The table points at real code, and that code passes ``fix=``: no entry can outlive its rewrite."""
    site = next(e for e in AUDITED if e.id == entry)
    module_name, _, attr = site.where.partition(":")
    obj = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    assert "fix=" in inspect.getsource(obj), f"{site.where} raises without fix= any more"


def test_the_documentation_matches_the_table():
    assert DOC.exists(), f"{DOC} is generated from pinneapple_core.errors.AUDITED; run scripts/build_error_audit.py"
    assert DOC.read_text(encoding="utf-8") == audit_markdown(), (
        "docs/dev/errors_that_teach.md is stale: run python scripts/build_error_audit.py")


def test_the_audit_rejects_an_entry_without_a_fix():
    bad = E.AuditEntry("no-fix", "pinneapple_core.errors:audit", "a failure", "", "  ")
    original = E.AUDITED
    try:
        E.AUDITED = (*original, bad)
        with pytest.raises(ValueError, match="carries no fix"):
            E.audit()
    finally:
        E.AUDITED = original
