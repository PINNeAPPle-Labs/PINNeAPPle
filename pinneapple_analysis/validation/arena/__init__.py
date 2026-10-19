"""Red team / blue team for validation (#707): adversarial generators of wrong-but-plausible cases against the
checks, and a loop that hardens them.

Built on the fault library of :mod:`pinneapple_analysis.validation.fault_injection` (#697)::

    from pinneapple_analysis.validation import arena
    rep = arena.run_arena("darcy_2d", budget="small", alpha=0.05)     # or arena("darcy_2d", ...) / pp.validate.arena
    print(rep.to_markdown())          # escapes, false alarms, mutation score, ROC/AUC, minimal detectable magnitude
    rep.save_cases("darcy_cases.json")    # replayed in CI by `pinneapple check --arena-cases darcy_cases.json`

Modules: :mod:`.search` (differential evolution, CMA-ES, L-BFGS-B on the unit box), :mod:`.generators` (mutation
operators and physics-consistent attacks), :mod:`.red` (evasion and false-alarm search, mutation score),
:mod:`.blue` (split-conformal combined test, hardening), :mod:`.cases` (regression cases, replay), :mod:`.loop`
(rounds and report). The module itself is callable: ``arena(problem, checks, budget=...)`` is ``run_arena``.
"""
from __future__ import annotations

import sys
import types

from .blue import Defence, calibrate, calibrate_scores, harden, score_matrix
from .cases import ReplayResult, builtin_cases, load_cases, load_defence, replay, save_cases
from .generators import GENERATORS, Generator, generators_for, get_generator
from .loop import ArenaBudget, ArenaReport, ModelAudit, RoundReport, audit_model, run_arena
from .red import Finding, evasion_search, false_alarm_search, mutation_escapes, mutation_score
from .search import METHODS, SearchResult, cmaes, minimize

__all__ = ["run_arena", "audit_model", "ModelAudit", "load_defence", "ArenaBudget", "ArenaReport", "RoundReport", "Defence", "calibrate", "calibrate_scores",
           "harden", "score_matrix", "Finding", "evasion_search", "false_alarm_search", "mutation_score",
           "mutation_escapes", "Generator", "GENERATORS", "generators_for", "get_generator", "replay",
           "ReplayResult", "save_cases", "load_cases", "builtin_cases", "minimize", "cmaes", "METHODS",
           "SearchResult"]


class _CallableModule(types.ModuleType):
    """``pinneapple_analysis.validation.arena(problem, checks, budget=...)`` runs the arena."""

    def __call__(self, problem="darcy_2d", checks=None, **kwargs):
        return run_arena(problem, checks, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
