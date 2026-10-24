# 16 Verified heat sink in Omniverse: PINNeAPPle + NVIDIA USD BRep + PhysicsNeMo + Omniverse

**What.** The design loop of example 10, shorter (60 training designs), proposes the lightest heat sink; PINNeAPPle
verifies it; the accepted design is tessellated by USD BRep and every vertex gets the temperature of the model that
accepted it (base at the fin-root temperature, fins along the 1-D fin solution). The OpenUSD stage references the
native B-rep and carries the mesh with `temperature`, the design and the trust decision.

**Why.** The colours in a scene should be the field the decision rests on. PINNeAPPle checks that the field written to
USD reproduces the rating (its fin average equals the fin efficiency, its root and tip match the model).

| Step | Who | Ran here |
|---|---|---|
| exact solids, tessellation of the accepted design, native B-rep USD | USD BRep 1.0.1 | yes |
| closed-form rating, verification, temperature field, field-consistency checks | PINNeAPPle | yes |
| surrogate and search | PhysicsNeMo 2.2.2 (CPU) | yes |
| OpenUSD stage, validation, read-back preview | OpenUSD (`usd-core`) | yes |
| RTX render in Omniverse Kit (`results/kit_render.py`) | Omniverse | **no** (needs an RTX GPU) |

```bash
PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/16_omniverse_physicsnemo_brep_heatsink/run.py   # ~1 min
```

## Results (this run: 4-core Xeon 2.1 GHz)

- Surrogate held-out error at most 0.41 %. First proposal (15.0 mm fins) verified at 75.30 C and rejected; second
  (14 fins, 15.5 x 1.00 mm) verified at **74.20 C**, dp 8.7 Pa, mass 195.1 g: accepted.
- Field on 57 262 triangles (chord error 7.7 um): fin average equals the fin efficiency 0.9786 to 1.3e-9; root and tip
  equal the model to round-off; temperatures 66.1 to 67.5 C on the sink.
- **Decision: APPROVED**; the stage validates cleanly.

Figures in `results/`: `architecture.png`, `usd_temperature.png` (read back from `heatsink.usda`), `checks.png`.
