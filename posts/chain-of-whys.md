Why does a roof slope? Why is there a ceiling under it? Every element of a working design answers a failure of a simpler one. That reasoning mostly stays in engineers' heads.

We built pinneapple_reason to compute it. Give it requirements and a naive concept: it finds the first failure (mechanism plus the numbers that prove it), tries library countermeasures on a copy, keeps the cheapest that fixes it, and re-checks everything until all pass. One call: reason(*cases.roof()).

Four cases:
- Roof: flat clay tiles let rain through (capillary 14 mm + wind 6.15 mm + ponded storm 25 mm against a 0 mm lap rise). Pitch to 22°, a ventilated ceiling (tile underside 14.9 °C under a 16.4 °C dew point), a radiant barrier (ceiling 12 K above the air, limit 5 K).
- 30 m footbridge: deflection 133 mm > 83.3 mm, deeper girders, lateral-torsional buckling (123 vs 1660 kN·m), 10 cross-frames, then the deeper girders put f1 at 2.05 Hz: 1.70 m/s² > 0.7. A 164 kg tuned mass damper beat 1606 kg of extra steel.
- Power board: a 20 A MOSFET on the bare board reaches T_j 626 °C; heat sink, 218 °C; forced air, 109 °C < 125 °C.
- Pumped water main: NPSH 2.41 m vs 4.5 m, so lower the pump; a 2 s valve closure gives a 1.81 MPa surge on a 1 MPa pipe; a 15 s closure gives 0.80 MPa.

Why trust it: an LLM may propose countermeasures but never supplies a number. Modes are tested against known answers and solvers against closed forms (FD beam, MOC vs Joukowsky); 18 of 18 checks pass. Limits are stated: engineering models with ranges, not code sign-off; costs are relative indices.

Guide: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/chain_of_whys.md

#Engineering #StructuralEngineering #Hydraulics #ThermalManagement #BuildingPhysics #ScientificComputing #Python #OpenSource

---
## Media

1. `media/chain-of-whys/01_intro.png` — Why does a roof slope? The loop: naive concept, check, failure with its number, countermeasure, check again.
2. `media/chain-of-whys/02_roof.png` — Roof: flat tiles -> 22° pitch -> ventilated ceiling -> radiant barrier, with the number behind each step.
3. `media/chain-of-whys/03_footbridge.png` — 30 m footbridge: deflection, lateral-torsional buckling, then resonance with footfall, and the alternatives rejected on cost.
4. `media/chain-of-whys/04_power_board.png` — 20 A power board: junction temperature from 626 °C to 218 °C to 109 °C.
5. `media/chain-of-whys/05_water_main.png` — Pump and water main: cavitation, then water hammer, then a 15 s closure.
6. `media/chain-of-whys/06_water_hammer.png` — Head at the valve (method of characteristics): 2 s closure vs 15 s closure against the pipe class.
7. `media/chain-of-whys/07_trust.png` — Why trust the chain: known answers, solver vs closed form, trust decision, stated limits.

## Sources

All case numbers are in the committed reports `examples/chain_of_whys/_out/<case>/report.md` and `chain.json` (`steps[i].finding.verification.demand` / `.capacity`, `steps[i].options[*].cost.quantity`), and were reproduced exactly by a re-run of `reason(*cases.get(name))` for the four cases (same demand/capacity to 4 decimals).

- 14 mm, 6.15 mm, 25 mm, 45.1 mm head, 0 mm lap rise -> `examples/chain_of_whys/_out/roof/report.md` (Version 0; chain.json `steps[0].finding.verification.quantities` h_capillary / h_wind / h_film, demand 45.127, capacity 0)
- 22° pitch, +7.9 % roof area -> roof/report.md, Version 0 options table (increase_slope)
- 14.9 °C underside, 16.4 °C dew point -> roof/report.md Version 1 (chain.json steps[1] demand 16.3994, capacity 14.9462)
- 12 K, limit 5 K, 78.7 W/m² -> roof/report.md Version 2 (steps[2] demand 12.0078, capacity 5.0)
- radiant barrier emissivity 0.05 -> roof/report.md Version 2 options table
- v3: 28.1 vs 20.7 mm, 19.2 vs 16.4 °C, 2.591 K < 5 K -> roof/report.md "Checks of the final version" table
- 133 mm > L/360 = 83.3 mm -> `_out/footbridge/report.md` Version 0 (steps[0] demand 133.0665, capacity 83.3333)
- girders 1.05 m deep, +1653 kg -> footbridge/report.md Version 0 options table
- 123 vs 1660 kN·m -> footbridge/report.md Version 1 (steps[1] capacity 122.9092, demand 1660.0924)
- 10 cross-frames, 204 kg; wider flanges +15617 kg -> footbridge/report.md Version 1 options table
- 2.05 Hz, 1.70 m/s² > 0.7 -> footbridge/report.md Version 2 (steps[2] demand 1.7028, capacity 0.7)
- TMD 164 kg vs deeper girders +1606 kg -> footbridge/report.md Version 2 options table (cost.quantity 164.43)
- v3: 0.241 m/s², 69.8 mm, 1786 kN·m -> footbridge/report.md final checks table
- T_j 626 °C (625.8), 15.9 K/W, loop gain 0.763, 36.8 W -> `_out/power_board/report.md` Version 0
- 40 thermal vias do not fix it; heat sink 40x40x20 mm envelope -> power_board/report.md Version 0 options table; `pinneapple_reason/cases.py` power_board (hs_max_w/d/h)
- T_j 218 °C (217.8), 10.3 K/W -> power_board/report.md Version 1
- 0.5 m/s air -> power_board/report.md Version 1 options table
- T_j 109 °C (109.3), 12.0 W, 5.751 K/W, limit 125 °C -> power_board/report.md final checks + measured numbers
- NPSH 2.41 m (2.405) vs 4.5 m -> `_out/pumped_main/report.md` Version 0
- 2.2 m suction lift, 33.6 m³ excavation -> pumped_main/report.md Version 0 options table
- 2 s closure, 3.94 s round trip, 1.81 MPa (1.805) > 1 MPa -> pumped_main/report.md Version 1
- 15 s closure; thicker wall +110977 kg -> pumped_main/report.md Version 1 options table
- 0.80 MPa (0.795), NPSH 5.2 m (5.205), no column separation -> pumped_main/report.md final checks table
- 2.4 km main, 2 s valve, pump 5 m above sump -> `pinneapple_reason/cases.py` pumped_main (main_length 2400, valve_close_time 2.0, suction_lift 5.0)
- 18 of 18 checks pass -> `examples/chain_of_whys/experiment.json` `checks` (18 entries, all `passed: true`)
- 4 / 4 trust decision APPROVED -> experiment.json checks "<case>: trust report decision" (detail "APPROVED"); also re-run `trust_report(chain).decisao`
- Water hammer figure -> regenerated in English with `write_report(chain, ..., lang="en")`; same data as committed `examples/chain_of_whys/_out/pumped_main/figures/water_hammer.png`
