"""Physics Query Engine: ask a physical quantity at a stable physical reference, under conditions, to an accuracy;
get the value with its unit, uncertainty, validity domain, provenance and the method that produced it, and whether it
was *read* from a solved state or *predicted* for an unsolved one.

    from pinneapple_condense.query import package_engine
    engine, model = package_engine(pkg)
    a = engine.query("T", "die@0.3,0.7", conditions={"power": P}, accuracy=0.1)
    a.value, a.unit, a.uncertainty, a.kind, a.method

Generic parts here (:mod:`.core`: query, offers, planner, answers; :mod:`.refs`: physical references); the methods of
a given physics live with it (chip package: :mod:`pinneapple_condense.query`).
"""
from .core import Answer, Engine, Method, Offer, Planner, Query, condition_key
from .refs import PhysicalRef, parse_ref

__all__ = ["Answer", "Engine", "Method", "Offer", "Planner", "Query", "PhysicalRef", "condition_key", "parse_ref"]
