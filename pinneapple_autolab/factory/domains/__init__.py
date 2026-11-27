"""The cases of the factory. Each one is a spec plus the ends it can supply; none of them re-implements a stage.

``ugs`` is the case whose solver is inside this library (:mod:`pinneapple_simulation.ugs`): the whole chain runs.
``pipe_erosion`` is the product's case whose solver is not here: it stops at that arrow and says why.
"""
from __future__ import annotations

from . import pipe_erosion, ugs

__all__ = ["pipe_erosion", "ugs"]
