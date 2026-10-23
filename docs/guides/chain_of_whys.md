# Chain of whys: why each element of a design exists, computed

Why does a roof slope? A flat tiled roof lets the rain through: water climbs the gap of each lap by capillarity, the
wind pushes it further and, with nothing to drain it, the storm ponds on top. Why a ceiling? On a clear night the tile
radiates to the sky, its underside drops below the dew point of the room air and drips. `pinneapple_reason` turns
that way of explaining a design into a computation that works for any project and any scale:

```
requirement -> naive concept -> physical check (numbers) -> failure mode (mechanism + the number that proves it)
            -> countermeasure (new element, cost, expected effect) -> check every requirement again -> ... -> all pass
```

Every claim in a chain comes from a computed check: a closed-form model with its source, or a solver of the library
(finite-difference beam, 3-D board finite volumes, plate-fin heat-sink correlations, method of characteristics). An
LLM can propose candidates; it never supplies a number.

```python
from pinneapple_reason import cases, reason, write_report

chain = reason(*cases.roof())          # requirements, the naive concept, a title
print(chain.explain("en"))             # version 0 -> version N, and why
write_report(chain, "roof_report")     # chain.json, report.md / .html, figures, trust.json
```

`python -m pinneapple_reason run roof --out roof_report` does the same from a terminal; `python -m
pinneapple_reason modes` lists the library.

## The data model (`pinneapple_reason.chain`)

| Object | What it holds |
|---|---|
| `Requirement` | what the design must do (`text`, both languages), the failure modes that would stop it (`modes`), limit overrides |
| `Concept` | one version of the design: parameters in SI units, elements, and the source of every value (`sources`: a premise, a cited standard value, or the countermeasure that set it) |
| `Verification` | one failure mode on one concept: `demand` vs `capacity` (the two numbers that decide), `margin`, `mechanism` in words built from the numbers, every measured quantity, `method`, `assumptions`, `sources`, applicability range (`validity`) and internal checks (`self_checks`); `.check()` is its `pinneapple_analysis.trust.Check` |
| `Finding` | the first failing verification and its proof |
| `Option` | a countermeasure tried: changes, cost (quantity x unit cost), target margin before / after, whether it fixes the failure, what it breaks, who proposed it |
| `Step` / `Chain` | every version with all its checks, the options and the one chosen, and why; `to_dict` / `from_dict` / `save` / `load` round-trip through JSON |

## The library of failure modes (`pinneapple_reason.library`, `pinneapple_reason.modes`)

| Domain | Failure mode | Check | Countermeasures |
|---|---|---|---|
| hydraulics / capillarity | `lap_water_ingress` | capillary rise between plates + wind head + run-off film (Nusselt) or ponded storm vs lap rise `L sin(alpha)` (Garden, CBD-40) | `increase_slope` (sized to the material's minimum slope), `longer_lap` |
| hygrothermal | `underside_condensation` | steady roof network on a clear night (Swinbank sky, Watmuff wind, ISO 6946 surfaces, grey plates) vs Magnus dew point | `ceiling_ventilated_attic`, `ceiling_insulation` |
| thermal comfort | `radiant_heat_gain` | same network on a sunny day: ceiling surface - air vs 5 K warm-ceiling asymmetry (ASHRAE 55) | `ceiling_ventilated_attic`, `radiant_barrier`, `ceiling_insulation` |
| electrothermal | `joule_heating` | `I^2 R_on(T)` coupled to the 3-D board model (`pinneapple_physics.pcb`) or the plate-fin heat sink (`pinneapple_physics.closed_form.plate_fin_heatsink`) until `T_j` is consistent; loop gain >= 1 is runaway | `thermal_vias`, `heatsink_natural` (best fins in the envelope), `forced_air` (sized air speed) |
| structures | `bending_strength`, `deflection`, `lateral_torsional_buckling`, `flexural_buckling`, `fatigue` | EN 1990 / EN 1991-2 loads, `5qL^4/384EI` checked by the FD beam solver, EN 1993-1-1 6.3 with Timoshenko's `M_cr`, Basquin | `deepen_girders`, `lateral_bracing`, `wider_flanges`, `column_midheight_brace`, `thicker_section` |
| structural dynamics | `pedestrian_resonance` | first vertical mode (checked by Rayleigh's method on the FD deflection), steady response to one walker, harmonics 1-3 (Bachmann & Ammann), 0.7 m/s^2 (EN 1990 A2) | `tuned_mass_damper` (Den Hartog), `deepen_girders` |
| wind | `wind_uplift` | `|Cp| 1/2 rho V^2` vs weight + fixings | `tile_clips` |
| hydraulics / cavitation | `pump_cavitation` | NPSH available (ISO 2533 atmosphere, Magnus vapour pressure, Colebrook suction loss) vs required + margin | `lower_pump`, `larger_suction_pipe` |
| hydraulic transients | `water_hammer`, `column_separation` | method of characteristics (`pinneapple_simulation.numerical_solvers.water_hammer_moc`), Korteweg wave speed; Joukowsky / Michaud as references | `slow_valve_closure`, `higher_pressure_class` |
| corrosion | `galvanic_corrosion` | O2-diffusion-limited cathode x area ratio, Faraday's law vs corrosion allowance | `insulating_kit`, `coat_cathode` |

A new mode is a function `evaluate(params, limits) -> Verification` registered with `register_mode(FailureMode(...))`;
it declares the parameters it needs (`Need`), its sources, the premises whose spread the uncertainty check moves
(`uncertain`) and known-answer cases (`known_answers`, run by the tests and by the trust card). A countermeasure is
`register_countermeasure(Countermeasure(name, element, title, effect, addresses, apply, cost, ...))`; `apply` may size
itself from the failing verification (`modes._util.size_to_pass` tries values until the mode passes with the case's
`design_margin`), `requires` / `excludes` name elements, and `cost` returns the physical quantity the change costs
(extra area, steel mass, excavation, a fan). The cost index is that quantity times a relative unit cost declared as a
premise (override per case with the parameter `cost.<countermeasure>`).

## The engine (`pinneapple_reason.engine`)

At each step: run every failure mode of every requirement (in the order given) and take the first that fails; apply
every countermeasure for it (plus the advisor's proposals) to a copy of the concept and re-run every check on the copy;
keep the cheapest option that fixes the failure without breaking anything that passed (ties within 1 %: most checks
passing, then largest margin). If none fixes it alone, keep the one that improves the margin most and go on; an option
that returns to an earlier version is refused. The chain ends `resolved`, `unresolved` (nothing in the library improves
the failure) or at `max_steps`.

With an LLM (`pinneapple_reason.llm.LLMAdvisor(pinneapple_design.intent.CallLLM(...))`, or any object with
`extract(text, schema)`), the advisor proposes countermeasures from the library and a first concept; unknown names
and parameters no mode reads are dropped, and proposals are tried and judged like any other option
(`Option.proposed_by == "llm"`).

## The four worked cases (`pinneapple_reason.cases`, `examples/chain_of_whys/`)

Numbers below are what the library computes for the premises listed in each report.

| Case | Version 0 | Chain |
|---|---|---|
| `roof` | clay tiles laid flat, no ceiling | rain: capillarity 14 mm + wind 6.2 mm + ponded storm 25 mm against a 0 mm lap rise -> pitch to 22° (lap rises 28.1 mm vs 20.7 mm head) -> clear night: tile underside 14.9 °C below the 16.4 °C dew point -> ventilated ceiling (ceiling underside 18.1 °C) -> sunny day: ceiling 12 K above the air, 78.7 W/m^2 into the room -> radiant barrier (2.6 K) |
| `footbridge` | 30 m, two welded girders sized for strength | deflection 133 mm > L/360 = 83.3 mm -> girders 1.05 m deep -> lateral-torsional buckling: 123 kN m resistance vs 1660 kN m -> 10 cross-frames (1786 kN m) -> first mode 2.05 Hz in the walking band, 1.70 m/s^2 > 0.7 -> tuned mass damper, mu = 1 %, 164 kg (0.24 m/s^2); deepening the girders again also fixed it, with 1606 kg more steel |
| `power_board` | 20 A D2PAK MOSFET on a 4-layer board | board alone: 15.9 K/W, loop gain 0.76, T_j 626 °C -> best heat sink in a 40 x 40 x 20 mm envelope, natural convection: 218 °C -> forced air at 0.5 m/s: 109 °C < 125 °C |
| `pumped_main` | pump 5 m above the sump; valve closing in 2 s | NPSH available 2.41 m < 4.5 m -> pump lowered to 2.2 m lift (5.2 m) -> MOC peak 1.81 MPa > PN10 -> 15 s closure: 0.80 MPa, no column separation |

Run all four with `python examples/chain_of_whys/chain_of_whys.py` (reports in `examples/chain_of_whys/_out/`; the
`.html` reports are self-contained, so they are regenerated rather than committed).

## Reports and trust

`write_report(chain, out_dir, lang="pt" | "en")` writes:

* `figures/timeline.png`: version cards; each arrow names the countermeasure and its cost, each card the failure and
  its numbers and the state of every requirement;
* `figures/margins.png` (PASS / FAIL with demand / capacity per check and version), `figures/demand_capacity.png`
  (one panel per failure mode) and domain figures: minimum pitch by covering, natural frequency against the walking
  bands, head at the valve after the closure starts;
* `report.md` and a self-contained `report.html` (light and dark), `chain.json`, `trust.json`.

The trust report (`pinneapple_reason.trust.trust_report`) is a `TrustReport` over the final version: every
requirement's check (critical), the known-answer cases of the modes used, the internal checks (energy-balance
closure, equilibrium, fixed-point convergence, MOC surge decay), the solver-vs-closed-form agreements, the
applicability ranges and the worst margin when each uncertain premise moves by its spread. `trust_card` answers the
six questions of `pinneapple_lab.curation.trust_stages` from those checks. A chain whose margins do not cover the
spread of its premises gets REVIEW, not APPROVED.

## Integrations

* Flow Studio: node types `reason.requirement`, `reason.concept`, `reason.verify`, `reason.failure_mode`,
  `reason.countermeasure`, `reason.chain`, `reason.report` (palette section Optimization & control), port types
  `requirement`, `concept`, `chain`; example flow `pinneapple_flow/flows/reason_roof.json`.
* MCP server: `reason` (job), `reason_library`, `reason_check` ([MCP server](mcp_server.md)).
* Agent workbench: `reason_chain`, `reason_library`, `reason_check` for the physics agent
  ([agent workbench](agent_workbench.md)).

## Limits

The checks are engineering models with stated ranges, not code sign-off: one-dimensional steady roof network with a
clear-sky model, single-pedestrian steady resonance (an upper bound), no cavity model after column separation, a
linear `R_on(T)`. Case inputs are premises written in each report; replace them with your data. Costs are relative
indices meant to rank options for the same finding, not estimates.
