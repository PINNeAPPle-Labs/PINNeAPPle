- `pinneapple_neural.architectures.neural_operators.transolver`: `PhysicsAttention.forward` and
  `TransolverBlock.forward` take a `mask`, so the padded part of a batch (a geometry with fewer nodes than the widest
  one) is left out of the attention instead of pulling every prediction towards it. The parameter is keyword-only with a
  default of `None`, so every existing call is unchanged; `api/core.json` records the two new signatures.
