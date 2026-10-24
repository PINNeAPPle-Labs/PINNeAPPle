Generative design is easy to demo and hard to trust: a search will happily exploit any loophole in the model that scores it.

We tried this on a small, honest test bed: interlocking toy bricks. `pinneapple_design.bricks` edits builds (add, move, rotate, brace the joint that failed, add a counterweight) and every candidate is judged by physics. A linear programme does the statics with a 0.98 N clutch per stud (the value cited by StableLego), NVIDIA Newton replays the rejected builds, and evolution or a Gaussian-process-guided search keeps what stands. Three goals, 5 seeds x 400 physics evaluations each:

- Cantilever, at most 8 parts: 135.9 mm past a table edge at 34% of the stud clutch. Unconnected blocks of the same length reach 86.7 mm.
- Bridge over a 96 mm gap carrying 1 N: 3 parts, capacity 1.46 N.
- Tower that survives a 10 mN push: 153.6 mm with 16 parts, 1.64x the push.

Why we would trust these numbers:
- 9 of 9 closed-form checks pass (harmonic stacking, stud tension, tipping push, a beam deflection within 0.4% of Timoshenko).
- Statics and Newton agree on 90%, 100% and 75% of searched candidates; the disagreements are listed and explained where we can.
- Limits are written down: the bridge fails at 0.5x the assumed clutch, the clutch has no published uncertainty, and nothing was compared with physical builds. With 5 seeds, the two search strategies are not separated.

Builds export to LDraw, glTF and STL, with step-by-step instructions.

Guide and code: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/interlocking_bricks.md

#GenerativeDesign #ComputationalMechanics #Simulation #StructuralEngineering #Optimization #OpenSource #PINNeAPPle

---
## Media

1. `media/interlocking-bricks/interlocking_bricks.mp4` (54.7 s, 1080x1350) — Title card, then the search evolving a cantilever, a bridge and a tower (colour = stud tension / clutch, rejected builds replayed in Newton), a Cycles turntable of the best cantilever, and an end card with checks, limits and the repo URL.

## Sources

All paths relative to the repo root (main, PR #805).

- 0.98 N clutch per stud -> `examples/bricks/_out/report.md` ("Joint model", line "Clutch per stud: 0.98 N"); `docs/guides/interlocking_bricks.md` "Joint model".
- 5 seeds x 400 evaluations -> `examples/bricks/_out/summary.json` keys `seeds` = [0,1,2,3,4], `budget` = 400.
- 8 parts max (cantilever), 96 mm gap / 1 N (bridge, gap=12 studs x 8 mm), 16 parts / 10 mN (tower) -> `examples/bricks/self_modifying_bricks.py` lines 40-42; guide line 117-119.
- 135.9 mm, 34% of clutch -> `summary.json` `best.cantilever.score` = 135.9, `best.cantilever.metrics.utilisation` = 0.339 (explanation "34% of the 0.98 N clutch").
- 86.7 mm unconnected blocks -> `docs/guides/interlocking_bricks.md` line 124; `report.md` "harmonic_unconnected 86.7 mm".
- Bridge 3 parts, 1.46 N -> `summary.json` `best.bridge.n_bricks` = 3, `best.bridge.metrics.capacity_n` = 1.4615.
- Tower 153.6 mm, 16 parts, 1.64x, 16.4 mN -> `summary.json` `best.tower.score` = 153.6, `best.tower.n_bricks` = 16, `best.tower.metrics.load_factor` = 1.637, `push_capacity_n` = 0.01637.
- 9 of 9 closed-form checks -> `summary.json` `validation` (9 entries, all `passed: true`); guide line 106.
- Beam deflection within 0.4% of Timoshenko -> `summary.json` `validation[6].rel_error` = 0.00403.
- Statics vs Newton agreement 90/100/75% -> `summary.json` `cross_checks.{cantilever,bridge,tower}.agreement` = 0.9 / 1.0 / 0.75.
- Bridge fails at 0.5x clutch (load factor 0.80), stands from 0.75x -> `report.md` bridge sensitivity table (clutch 0.49 N: joint_failure, 0.797); guide line 128.
- Video only: Newton displacement 0.00 / 0.00 / 0.03 mm in 0.6 s -> guide line 132.
- Video only: tower stands over 0.5x-1.5x clutch -> `report.md` tower sensitivity table.
