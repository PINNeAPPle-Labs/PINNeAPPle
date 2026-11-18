- `pp.study("...")` and `pp.compile_study("...")`: text to simulation and text to Physics AI. A sentence is compiled
  into a `StudyContract` (`pinneapple.study/1`) that carries the provenance of every value — stated, answered,
  defaulted with its source, or missing — and refuses to invent the ones nobody gave. The contract runs a fidelity
  ladder (closed form, a real solver, the same solver refined, and in `physics_ai` mode a surrogate scored on designs
  held out of its fit), is decided by `pinneapple_analysis.trust.TrustReport`, and writes `contract.json`, `run.json`,
  `report.md` and `trust.md`. `pinneapple_intent.replay` rebuilds a run from those files alone, without an agent, and
  reports every number that did not come back the same. Three families ship: `heat_wall` (series resistance against
  immersed finite volume), `laminar_pipe` (Hagen-Poiseuille against finite differences on the radial momentum
  balance) and `cantilever_beam` (Euler-Bernoulli in plane strain against immersed elasticity). Text-to-science is
  recognised and refused with what the literature stage would need. Flow nodes `intent.compile` and `intent.study`,
  the shell `python -m pinneapple_intent compile|run|replay|families`, and
  [the guide](docs/guides/intent_to_study.md).
