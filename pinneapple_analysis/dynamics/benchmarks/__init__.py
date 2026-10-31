"""Reproducible benchmarks of the dynamics toolkit against known answers.

Each benchmark returns :class:`BenchmarkCase` records: the reference value and where it comes from, the computed
value, the error, the Monte Carlo standard error when the method is random, a convergence record, the wall time and
the limitations of the comparison. A benchmark that passes shows agreement with that reference under those settings;
it is not a proof that the method is valid elsewhere.

``python -m pinneapple_analysis.dynamics.benchmarks first_passage [--quick]`` prints the table in Markdown.
"""
from __future__ import annotations

from ._case import BenchmarkCase, to_markdown

__all__ = ["BenchmarkCase", "to_markdown"]
