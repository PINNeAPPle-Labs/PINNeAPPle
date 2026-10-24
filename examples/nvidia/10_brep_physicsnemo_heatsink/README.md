# 10 Heat-sink design from exact geometry: PINNeAPPle + NVIDIA USD BRep + PhysicsNeMo

**What.** An extruded 6063 plate-fin heat sink (100 x 100 mm base, 1 mm fin-root fillets) cools a 60 W, 40 x 40 mm
source in ducted 3 m/s air. Each candidate (fin count, height, thickness) is built as an exact B-rep and rated by
PINNeAPPle's closed-form model; a PhysicsNeMo network learns the design space; the lightest design with
T_source <= 75 C and dp <= 30 Pa is searched on the network and accepted only after verification.

**Why.** Surrogate-driven design looks for the edge of the feasible region, which is where the surrogate is least
reliable. PINNeAPPle re-evaluates every proposal with the exact geometry and the physics model before accepting it.

| Step | Who | Ran here |
|---|---|---|
| 150 exact solids (with root fillets): volume, area, mass | USD BRep 1.0.1 | yes |
| thermal / hydraulic rating (Teertstra forced convection, fin efficiency, spreading, pressure drop) | PINNeAPPle | yes |
| FullyConnected 3 x 64, 15 000 steps, (n, H, t) -> T rise, dp, mass | PhysicsNeMo 2.2.2 (CPU) | yes |
| held-out errors, monotonic trends, verification of proposals | PINNeAPPle | yes |

```bash
PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/10_brep_physicsnemo_heatsink/run.py    # ~1.5 min
```

## Results (this run: 4-core Xeon 2.1 GHz)

- B-rep volumes equal the closed form with the fillets to 7e-9 for all designs; the fillets add up to 0.8 % mass that
  the closed-form thermal model leaves out.
- Surrogate on 30 held-out designs: relative RMS error 0.19 % (T rise), 0.37 % (dp), 0.08 % (mass); every physical
  trend holds on the grid (T falls with fin height, mass and dp grow with fin count).
- First proposal (14 fins, 15.0 x 1.00 mm): predicted 74.99 C, **verified 75.30 C: rejected**. Second proposal
  (14 fins, 15.5 x 1.00 mm): predicted 73.98 C, verified **74.20 C**, dp 8.7 Pa, exact mass **195.1 g**: accepted.
- **Decision: APPROVED** for the second proposal. The closed form assumes ducted flow (its warning is in the summary).

Figures in `results/`: `architecture.png`, `parity.png`, `design_space.png` (rejected and accepted proposals),
`checks.png`; `designs.json` holds every design.
