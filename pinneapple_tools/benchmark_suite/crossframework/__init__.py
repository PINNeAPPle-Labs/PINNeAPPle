"""Neutral parts of the cross-framework benchmark (PINNeAPPle vs PhysicsNeMo / PhysicsNeMo-Sym, DeepXDE,
neuraloperator): problem definitions and references, the shared point sets and initial weights, the one evaluator,
statistics over seeds, run records and the operator-learning data and training loop.

Nothing here imports a competitor framework; the runners in ``benchmarks/vs_frameworks/`` do, one per framework and
track, and export plain arrays that :mod:`.evaluate` scores. The protocol is ``benchmarks/vs_frameworks/PROTOCOL.md``.

>>> from pinneapple_tools.benchmark_suite.crossframework import problems, evaluate
>>> p = problems.get_problem("poisson")
>>> X = problems.test_points(p)
>>> evaluate.evaluate_pinn(p, p.exact(X))["rel_l2"]
0.0
"""
from . import evaluate, problems, protocol, records, weights  # noqa: F401

__all__ = ["evaluate", "problems", "protocol", "records", "weights"]
