# Large Physics Models (LPM): native integration

This is the **first integration slice**, not a completed solver-backed LPM release.
The original experimental patch includes OpenFOAM, CalculiX and FEniCS
adapters, campaign orchestration, model training, and a configurable
VeriPhysics hook. Those components require adaptation to existing PINNeAPPLe
APIs before they can be considered natively integrated.

## Native boundary

- Import campaign configuration from `pinneapple.lpm`.
- Reuse `pinneapple_simulation` for physical solver execution and
  `pinneapple_solvers` compatibility paths, rather than introducing
  an independent solver runtime.
- Reuse `pinneapple_neural` for model architectures and training.
- Integrate with `pinneapple_veriphysics` using an explicitly tested
  adapter. Do not equate a configurable hook with completed validation.
- Use existing registry and orchestration facilities before adding
  duplicate model stores or schedulers.

## Example

```python
from pinneapple.lpm import Campaign

campaign = Campaign.load("examples/lpm/cfd01_cavity_lowRe.json")
campaign.validate()
```

## Outstanding work

1. Migrate the three solver adapters into the native simulation integration layer.
2. Adapt the restartable campaign runner, deterministic splits, and provenance.
3. Wire the training families (MLP, 2-D FNO, graph) to native model APIs.
4. Wire data-quality and holdout gates to native VeriPhysics.
5. Port the ten reference campaign configurations and tests.
6. Run actual solver-based integration tests and publish measured results.

**Evidence boundary:** The experimental package generated 240 case
inputs, but this is not evidence of 240 completed physical simulations.
Do not publish performance or physical-validity claims until real
solver runs and independent checks are recorded.
