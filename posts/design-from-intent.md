"A heat sink for a 40 W chip in still air" is a request with a hole in it: what temperature may the chip reach? A tool that quietly fills it gives a confident wrong answer.

`pp.design(text)` in PINNeAPPle (also `pinneapple design "<goal>"` on the command line and a `design` tool over MCP) turns a plain-language goal into one of three replies:

- a question, when a value the physics needs has no safe default. The heat sink asks for the case temperature limit; the bracket asks how far from the mounting face the load acts.
- a design package: geometry (STEP/STL/glTF), bill of materials with cited properties, operating window, report and check files. Optimised on cheap models, then checked once with the full solver.
- a refusal that names what is missing. "An electric motor for my drone" returns: needs an electromagnetics template with current-density, thermal and demagnetisation limits.

Measured:
- Heat sink at 85 °C (my run): copper, 14 fins. 3-D finite volume 77.7 °C vs resistance network 77.6 °C, energy balance 1.4e-12, still under the limit with the convection coefficient 15 % low (83.9 °C).
- Bracket for 2 kN: CalculiX peak stress 94.3 MPa vs 96.8 MPa from the reduced FE, P(success) 0.9985. Model calls: 12000 closed form, 19 reduced, 1 CalculiX.
- Toy loop car: NVIDIA Newton gives 1.507 m/s at the top vs a 1.023 m/s worst-case envelope.

Every value says whether it was stated (with the quote), answered or defaulted (with its source). Limits are written down: the Newton run is frictionless because its contact loses 5.7 % of the launch energy on the curved mesh, and the heat sink has no CFD of the air.

Guide: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/design_from_intent.md

#EngineeringDesign #Simulation #FEA #Verification #Optimization #ScientificML #OpenSource #PINNeAPPle

---
## Media

1. `media/design-from-intent/01_title.png` — Three replies to three goals: a question, a delivered design, a refusal.
2. `media/design-from-intent/02_questions.png` — Real CLI output: the heat sink asks for the case temperature limit, the bracket for the arm length.
3. `media/design-from-intent/03_provenance.png` — The report's requirements table: each value stated (with quote), answered or defaulted (with source).
4. `media/design-from-intent/04_heatsink_checks.png` — The delivered heat sink and its verification table against the 3-D finite-volume solver.
5. `media/design-from-intent/05_bracket_calculix.png` — CalculiX von Mises on the 2 kN bracket, with the checks and model-call counts.
6. `media/design-from-intent/06_loop_newton.png` — The toy loop car in NVIDIA Newton, judged against the worst-case envelope, with the solver's stated limit.
7. `media/design-from-intent/07_refusal.png` — The out-of-scope reply for a drone motor, and where to find the guide and examples.

## Sources

Repo paths relative to the root (main; PRs #820, #821). "My run" = `python -m pinneapple_design.intent.cli` from the repo root on 2026-10-24, outputs in `scratchpad/fp/work_dfi/` (heat sink: `--answer t_limit=85°C --out hs`; matches the committed `examples/design_from_intent/_out/summary.json` `heatsink` entry).

- Heat-sink question text, bracket question text, motor refusal text -> my run (`work_dfi/out_a_heat_sink_.txt`, `out_a_bracket_th.txt`, `out_an_electric_.txt`); same text in `examples/design_from_intent/_out/summary.json` `heatsink.first_reply`, `bracket.first_reply`, `out_of_scope.reply.reason`.
- Exit codes 3 (questions) / 4 (refusal) -> my run; `pinneapple_design/intent/cli.py` last lines.
- 85 °C limit -> answer given (`summary.json` `heatsink.answers.t_limit`).
- Copper, 14 fins, 100 x 100 x 6.5 mm base, fins 1.1 x 47 mm -> my run `hs/design.json` `design`; `summary.json` `heatsink.design`.
- 77.7 °C (full) vs 77.6 °C (reduced) -> `hs/design.json` `verification.comparison[0].full` = 77.676, `.reduced` = 77.592; `summary.json` `heatsink.verification.comparison[0]`.
- Energy balance 1.4e-12 -> `hs/design.json` `verification.checks[energy_balance].value` = 1.4456e-12.
- 83.9 °C with h 15 % low, limit 85 -> `hs/design.json` `verification.checks[5].value` = 83.853, `limit` = 85.0.
- P(success) 1.000 (0.998–1.000), APPROVED (image 4) -> `hs/design.json` `robustness.p_success`, `ci95`; `summary.json` `heatsink.probability/ci95/trust`.
- Bracket 94.3 vs 96.8 MPa -> `summary.json` `bracket.verification.comparison[0].full` = 94.26, `.reduced` = 96.76.
- Bracket P(success) 0.9985 -> `summary.json` `bracket.probability`.
- Bracket model calls 12000 / 19 / 1 -> `summary.json` `bracket.calls.calls`.
- Image 5 only: deflection 0.380 vs 0.380 mm -> `summary.json` `bracket.verification.comparison[1]`; reaction = load 1.15e-7, mesh convergence 1.2 % -> `examples/design_from_intent/_out/bracket/design.json` `verification.checks`; 5th-percentile bolt margin 0.017 -> guide "(b) Bracket" and `bracket/design.json` `robustness.margin_p05.margin_bolts` = 0.0165; t 17 / w 36 mm, Al 6061-T6 -> `summary.json` `bracket.design`; worst corner (+10 % load, -0.2 mm, 90 % yield) -> guide "(b) Bracket".
- Loop v_top 1.507 m/s vs 1.023 m/s -> `summary.json` `loop.verification.comparison[0].full` = 1.5068, `.reduced` = 1.0234 (kind `at_least`); guide table row (a).
- 5.7 % energy loss in Newton contact -> `docs/guides/design_from_intent.md` "(a) Loop-the-loop car", *Full* paragraph.
- Image 6 only: R 8.4 cm, launch 2.47 m/s, 31 g -> `summary.json` `loop.design.radius` = 0.0839, `v0` = 2.465, `mass` = 0.0310.
- Image 7 only: weapons refused, template `check_scope` refusals (bracket under fatigue, liquid-cooled heat sink) -> guide "(e) Out of scope".
