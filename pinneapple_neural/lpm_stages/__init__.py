"""The design-space pipeline, stage by stage — the shape the campaign is meant to have.

The first release of the large-physics-model work grew inside two modules: ``lpm.py`` held the model and the fit, and
``lpm_pipeline.py`` held the five stages (draw, build, write, fit, card) glued to a CLI. That was enough for eleven
design spaces and not enough for the pipeline the campaign is now asked to be. This package keeps the same code and
gives each stage a boundary, so a stage can be replaced — a different sampler, a different reference, a model from
another framework — without touching the ones around it:

0. **pergunta** — a structured specification of the problem (``pinneapple_intent`` writes one; this package reads it)
1. :mod:`~.space` — the design space and how it is sampled: lattice (LHS/Sobol) or **active**, driven by where the
   model is least sure
2. **geometria** — ``pinneapple_design.geometry.morph`` turns an entry of the space into a body, and
   ``morph/validity`` refuses the ones a reference cannot represent
3. **referência** — the family's own solver or closed form, and the dataset it writes (``lpm_pipeline.generate_dataset``,
   with the neighbourhood features cached once per design)
4. **âncoras** — measurements, as evidence and never as training data (``pinneapple_veriphysics.validation_case``)
5. **treino** — ``lpm.fit`` and the PhysicsNeMo models behind the same interface (:mod:`.training`)
6. :mod:`~.trust` — what the model is worth: held-out error against the constant-field baseline, a **calibrated**
   confidence, and the gates that ask whether the answer obeys the physics
7. :mod:`~.card` — the versioned record of all of it, and :mod:`~.registry` the index over the records
8. **entrega** — the Studio and the panel read the card; the API returns the field *and* the verdict — and
   :mod:`~.shift`, the other half of what a model is worth: an outside corpus (Luminary's SHIFT datasets), read off
   disk for prediction only, under the licence the corpus carries

What is here today: ``space``, ``trust``, ``card``, ``registry``, ``shift``, ``catalog``, ``anchors`` and the
PhysicsNeMo backend of ``training``. The rest stays in ``lpm_pipeline``
until its stage is extracted, which is deliberate — the campaign runs on that module and it keeps running. The
comparison itself lives beside the benchmarks (``benchmarks/lpm_compare.py``), because a table that scores models is a
benchmark, not a stage.
"""
from __future__ import annotations

__all__ = ["anchors", "card", "catalog", "registry", "shift", "space", "training", "trust"]
