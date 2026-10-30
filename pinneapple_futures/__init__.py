"""PINNeAPPle Futures Lab (``pp.futures``): Speculative Physics & Engineering.

A speculative system -- a world, a material, a habitat, a technology that does not exist -- written down as
physics and investigated with the same checks as anything else in PINNeAPPle. The Lab is an application layer of
the library, not a fork of it: its experiments are :class:`pinneapple_science.core.Experiment` subclasses, its
results carry the same physics checks and the same trust report, and the solvers it uses are the library's.

    import pinneapple as pp
    brief = pp.futures.ask("What if Earth's gravity were 2.5 times stronger?")
    print(brief)                              # what changed, which equations contain it, what can run it
    r = brief.run(quick=True)                 # the experiment, with the scenario attached
    print(r.assessment)                       # six evidence axes, two verdicts, and what was not assessed

Four things make it a laboratory rather than a generator of settings:

* **a schema** (:mod:`.schema`) in which every number carries its provenance -- a constant, a measurement, a
  design choice, a derivation or a hypothesis -- and every law its epistemic class;
* **a What If Engine** (:mod:`.whatif`) that says which equations contain the parameter a question changes,
  including when the honest answer is "none of them";
* **a plausibility framework** (:mod:`.plausibility`) that scores six axes of evidence separately and refuses to
  let a simulation fill the two only a measurement or a build can fill;
* **demonstrators** (:mod:`.labs`) whose every result is compared with a closed form, with the observed order of
  the discretisation reported next to it.

Command line: ``python -m pinneapple_futures atlas | list | run NAME | ask QUESTION | publish NAME --out DIR``.
Guide: docs/guides/futures_lab.md.
"""
from .atlas import Area, Atlas, load as atlas, validate
from .core import SpeculativeExperiment, SpeculativeResult, experiments, get, names, register, run
from .plausibility import AXES, ENGINEERING_VERDICTS, PHYSICS_VERDICTS, Assessment, AxisScore, assess
from .schema import (PLAUSIBILITY_CLASSES, PROVENANCE_KINDS, STAGES, Claim, Derived, Hypothesis, Input, Proposal,
                     Scenario, classification, weakest)
from .whatif import DEPENDENCIES, RULE_FAMILIES, Brief, ask

__all__ = ["atlas", "validate", "Atlas", "Area", "ask", "Brief", "DEPENDENCIES", "RULE_FAMILIES",
           "run", "get", "names", "experiments", "register", "SpeculativeExperiment", "SpeculativeResult",
           "assess", "Assessment", "AxisScore", "AXES", "PHYSICS_VERDICTS", "ENGINEERING_VERDICTS",
           "Scenario", "Proposal", "Hypothesis", "Claim", "Input", "Derived", "weakest", "classification",
           "PLAUSIBILITY_CLASSES", "PROVENANCE_KINDS", "STAGES"]
