"""Built-in problem templates. A new domain plugs in by subclassing
:class:`~pinneapple_design.intent.template.ProblemTemplate` and calling :func:`pinneapple_design.intent.register`."""
from __future__ import annotations


def builtin() -> list:
    from .bracket import BracketTemplate
    from .glider_wing import GliderWingTemplate
    from .heatsink import HeatSinkTemplate
    from .loop_car import LoopCarTemplate
    return [LoopCarTemplate(), BracketTemplate(), HeatSinkTemplate(), GliderWingTemplate()]
