"""Contact, as a capability of the library: seven cases with a closed-form answer, and the decks PrePoMax opens
unchanged. Six are friction cases; the seventh lets the plate yield.

Each case is one geometry, one contact interface and one question that has an exact answer independent of the solver:

===========================  ==========================================================================
``bloco_plano``              a slider pushed sideways on a fixed plate: the joint transmits ``mu*N`` and no more,
                             so the tangential force saturates exactly there
``junta_dupla``              a plate pulled between two clamped plates: two interfaces share the load, so the
                             pull at slip is ``2*mu*N`` while each support carries ``mu*N``
``cilindro_hertz``           a rigid cylinder pressed into a plate: a non-conforming pair, so the peak pressure
                             and the half-width follow Hertz instead of the nominal contact area
``rampa``                    the same slider on an inclined interface: it stays put while ``tan(theta) <= mu``
                             and slides when it does not — the self-locking test
``parafuso_pretensao``       two plates clamped by a bolt whose preload is a ``*PRE-TENSION SECTION``, so the clamp
                             force is the preload and the slip load is ``mu`` times the preload *measured*
``ajuste_forcado``           a ring shrunk onto a hollow shaft: the interference is geometric and the interface
                             pressure is Lamé's, not load/area
``plastificacao_hertz``      the Hertz cylinder on an elastic-perfectly-plastic plate: yield begins at
                             ``p0 = k*sigma_y`` with Johnson's ``k``, and the equivalent stress then stops there
===========================  ==========================================================================

The pieces are separable on purpose: :mod:`.mesh` (structured blocks and CalculiX's brick face numbering),
:mod:`.deck` (the keyword blocks PrePoMax reads), :mod:`.verify` (the closed forms and the comparisons) and
:mod:`.run` (the solver, with the results read by :mod:`pinneapple_data.cae`, never by a parser of its own).

    from pinneapple_simulation.external_solvers.calculix.contact import CASES, build, run_case
    case, params, built = build("bloco_plano", {"friccao": 0.3})
    result = run_case("bloco_plano", {"friccao": 0.3}, "runs/bloco")

``run_case`` returns the numbers as read and the checks; a report, a route and an evidence record are written from it
and cannot change what the checks saw. Without a ``ccx`` binary the deck is still written and the result says so.
"""
from . import deck, mesh, run, verify
from .base import Built, Case, Param, contact_of, mesh_size, reaction, stack
from .cases import CASES, DEFAULT_CASE, build, catalog, get
from .run import (
           SolverMissing,
           ccx_binary,
           footprint,
           measured,
           read_results,
           run_case,
           solve,
           version,
)

__all__ = ["CASES", "DEFAULT_CASE", "build", "catalog", "get", "Case", "Built", "Param", "contact_of", "mesh_size",
           "reaction", "stack", "deck", "mesh", "verify", "run", "run_case", "read_results", "measured", "footprint", "solve",
           "ccx_binary", "version", "SolverMissing"]
