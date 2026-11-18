"""The built-in study families, and the registry they plug into.

    from pinneapple_intent.families import builtin
    [f.name for f in builtin()]     # ['beam_cantilever', 'heat_wall', 'laminar_pipe']

A new domain adds a module here that subclasses
:class:`~pinneapple_intent.family.StudyFamily`, implements ``evaluate``, and is returned by :func:`builtin`. The
guide (``docs/guides/intent_to_study.md``) walks through it.
"""
from __future__ import annotations

from .beam_cantilever import BEAM
from .heat_wall import WALL
from .laminar_pipe import PIPE


def builtin() -> tuple:
    """The families this release ships, in the order a request is matched against them."""
    return (WALL, PIPE, BEAM)


__all__ = ["builtin", "WALL", "PIPE", "BEAM"]
