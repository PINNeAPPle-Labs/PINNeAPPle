# 10 Heat-sink design from exact geometry: PINNeAPPle + NVIDIA USD BRep + PhysicsNeMo

**What.** An extruded plate-fin heat sink (100 x 100 mm base, 1 mm fin-root fillets) cools **two** chips: an 80 W
power module on 40 x 40 mm and a 15 W gate driver on 15 x 15 mm, in 3 m/s air with a partial shroud, so part of the
array sees the room and radiates. Each candidate (fin count, height, thickness, base thickness, material) is built as
an exact B-rep, rated by PINNeAPPle's closed-form network *and* by a 3-D finite-volume solve of the base with both
footprints on it, and a PhysicsNeMo network learns the design space over seven variables -- the five shape ones plus
the film coefficient and the interface resistance, so the study spans the uncertainty an engineer lives with. The
lightest design with T_source <= 75 C and dp <= 30 Pa is searched on the network and accepted only after verification.

**Why.** Surrogate-driven design looks for the edge of the feasible region, which is where the surrogate is least
reliable; and a resistance network with one equivalent source cannot say how much hotter the small chip runs than the
large one. PINNeAPPle therefore re-evaluates every proposal with the exact geometry and *two* independent physics
models before accepting it, and reports the gap between those two models as a check of its own.

| Step | Who | Ran here |
|---|---|---|
| exact solids with root fillets: volume, area, mass | USD BRep 1.0.1 | needs `usd-brep` |
| network: Teertstra forced convection, fin efficiency, spreading, pressure drop, radiation | PINNeAPPle | yes |
| 3-D finite volumes of the base: both chips, per-chip hot spot, its own energy balance | PINNeAPPle | yes |
| FullyConnected 7 x 64, 15 000 steps, (n, H, t, base, k, h, R''_TIM) -> rise, dp, mass | PhysicsNeMo 2.2.2 (CPU) | yes |
| held-out errors, trend direction across each range, verification of proposals | PINNeAPPle | yes |

```bash
PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/10_brep_physicsnemo_heatsink/run.py   # ~1 min
python examples/nvidia/10_brep_physicsnemo_heatsink/run.py --smoke    # seconds, no B-rep needed
```

Without `usd-brep` the run still completes: the mass comes from the closed form, and `run_info.json` records the
B-rep step as not run with that reason. In a run with `usd-brep` the exact volumes equal the closed form with the
fillets to 7e-9 and the fillets add up to ~1 % mass that the closed-form thermal model leaves out.

## Results (this run: macOS arm64, 10 CPU cores, no GPU, PhysicsNeMo 2.2.2, torch 2.15.0; `usd-brep` absent, so the
mass in this run is the closed form's)

- **The two physics models agree on every design**: median 0.6 %, p90 1.1 %, worst 5.3 % of the temperature rise.
  The network is optimistic exactly where a small chip sits next to a large one, which is the reason the 3-D solve
  is the number this example reports.
- **The 3-D solves close their own energy balance** to 1e-12 (heat out of the fin-root face against chip power).
- Surrogate on 40 held-out designs: relative RMS error 0.51 % (rise), 0.70 % (dp), 0.69 % (mass). Every physical
  direction holds across the whole box, including the two that are easy to get backwards: the hot spot falls with the
  base thickness (more spreading beats the extra 1-D resistance here) and grows with the interface resistance.
- Accepted design (the first proposal already survived verification): 21 fins, 18.1 mm tall, 0.83 mm thick, 3.5 mm
  base, Al 1050 -- **181 g**, 3-D hot spot **74.2 C** (small chip 62.6 C), dp 13.6 Pa, fin efficiency 96 %,
  radiation 5 % of the surface coefficient.
- **Decision: APPROVED.** Every check passed: the two models agree, the surrogate's held-out error is under 1 %, all
  ten physical directions hold across the whole box, and the accepted design meets both limits under the 3-D solve.
- The two warnings the model carries are the model's own limits: the slots of an array see each other, so only the
  `sky_fraction = 0.25` of the area with an unobstructed view radiates, and the flow is assumed ducted -- without a
  shroud, bypass can cost 20-50 %.

Figures in `results/`: `architecture.png`, `parity.png` (surrogate against the models), `models.png` (network against
the 3-D solve -- the cross-check this example is about), `design_space.png` (rejected and accepted proposals),
`checks.png`; `designs.json` holds every design.
