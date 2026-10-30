"""pinneapple_core.onboarding — the first hour with PINNeAPPle.

Everything a new user needs before writing physics code:

* :func:`~pinneapple_core.onboarding.doctor.doctor` checks the installation against a *profile*
  (``start``, ``cfd``, ``research``, ``enterprise``) and prints one fix command per problem.
* :func:`~pinneapple_core.onboarding.recommend.recommend` turns a problem described in one
  sentence, in English or Portuguese, into a preset, a method and a runnable snippet, and says why.
* :func:`~pinneapple_core.onboarding.catalog.card` prints what a preset takes, what it gives back
  and the smallest program that uses it.
* :func:`~pinneapple_core.onboarding.newproject.new_project` writes a standalone project (not an
  example inside this repository) with a case file, a runner and a README, at three levels of
  programming: ``plain`` (edit a YAML, never touch Python), ``research`` (experiment manifests and
  a notebook) and ``cfd`` (geometry and solver campaigns).

The four are also reachable from the command line as ``pinneapple doctor``, ``pinneapple
recommend``, ``pinneapple card`` and ``pinneapple new``.

>>> import pinneapple as pp
>>> pp.doctor()                       # is this installation ready for what I want to do?
>>> pp.recommend("heat conduction in a fin")   # which preset and method should I start from?
>>> pp.card("burgers_1d")             # what does it take, what does it give back?
>>> pp.new("my_study", template="plain")       # a folder I can grow into a paper or a product
"""
from __future__ import annotations

from .catalog import Card, card, describe, format_card, search, snippet
from .doctor import DoctorReport, doctor, profiles
from .newproject import TEMPLATES, new_project
from .recommend import Recommendation, recommend

__all__ = [
    "Card", "card", "describe", "format_card", "search", "snippet",
    "DoctorReport", "doctor", "profiles",
    "Recommendation", "recommend",
    "TEMPLATES", "new_project",
]
