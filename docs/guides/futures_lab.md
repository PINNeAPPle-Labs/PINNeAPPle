# Futures Lab: the physics of possible worlds

*Exploring the physics of possible worlds.*

The Futures Lab turns a speculative system -- a world with another gravity, a material that does not exist, a
settlement on Mars -- into physics, and then investigates it with the same checks as anything else in PINNeAPPle.
It is deliberately **not** a worldbuilder and **not** a generator of plausible-sounding answers: every number it
prints is compared with a closed form, and every claim about what the result supports is separated from what it
cannot support.

```python
import pinneapple as pp

brief = pp.futures.ask("What if Earth's gravity were 2.5 times stronger?")
print(brief)                     # what changed, which equations contain it, and what can run it
res = brief.run(quick=True)
print(res.assessment)            # six evidence axes, two verdicts, and the axes that were not assessed
```

## Why it is a laboratory and not a generator

Three decisions do the work.

**Every number carries its provenance.** An `Input` is a constant, a measurement, a design choice, a derivation or
a hypothesis -- and the five are not interchangeable. The reference conductivity of a material that does not exist
is a `hypothesis`, which is what makes every result built on it hypothetical at best; the surface gravity of Mars
is `measured`, with the source. A number without a source cannot be constructed at all:

```python
import pinneapple_futures as F

F.Input("k", 0.015, "W/m/K", "design_input", "an aerogel-class insulation: a choice of this experiment")
# F.Input("k", 0.015, "W/m/K", "design_input", "")     -> ValueError: a number needs its source
```

**Every law has an epistemic class.** A proposal is classified by its weakest claim, never by its best:

| Class | Meaning |
|---|---|
| `established` | compatible with laws and phenomena that are well established |
| `extrapolated` | a known model extended to conditions that are still poorly studied |
| `hypothetical` | depends on physics that has not been demonstrated |
| `fictional` | introduces laws or mechanisms with no known scientific grounding |

**Six axes of evidence, scored only from what the run recorded**, and two verdicts that are never averaged into
one number:

| Axis | What fills it |
|---|---|
| `mathematical_consistency` | a residual, an analytic solution on the same grid, a bound |
| `numerical_stability` | a refinement ladder with its observed order, or the same answer on a different mesh |
| `physical_plausibility` | conservation, an asymptotic limit, a symmetry |
| `experimental_validation` | a measurement -- a bench or a published table |
| `technological_feasibility` | that the thing can be built, made or powered |
| `uncertainty_sensitivity` | how much the answer moves when the inputs move |

An axis with no evidence is **not assessed**: never zero, never one. A simulation therefore cannot fill
`technological_feasibility` or `experimental_validation`, and the report says so in as many words:

```
physics SUPPORTED | engineering NOT ASSESSED | class extrapolated
  ok   mathematical_consistency   1.0  <- heat flux through the wall vs closed_form, ...
  ok   numerical_stability        1.0  <- the wall solution does not depend on the mesh
  ok   physical_plausibility      1.0  <- film-to-film energy balance, ...
    -  experimental_validation    not assessed
    -  technological_feasibility  not assessed
  note: technological feasibility was not assessed: a simulation cannot establish it
```

That is the caution the Lab is built on: *a model that is mathematically consistent and numerically stable shows
the behaviour of the equations that were adopted; it does not show that the technology can be built or that the
world behaves so.*

## The What If Engine

`pp.futures.ask` reads a question and returns a `Brief`: the scenario with the change applied and converted, the
eight steps of the plan, and -- the part that makes it scientific -- **which equations contain the parameter that
changed**. A parameter that appears in no equation cannot be investigated by changing it, and the engine says that
instead of simulating something else and reporting a number.

```
What if Earth's gravity were 2.5 times stronger?

  surface gravity               24.5166 m/s^2  2.5 g

  which equations contain it:
    hydrostatic column               H = R T / (M g),  p(z) = p0 exp(-z/H)
                                     -> H is inversely proportional to g
    weight and structural load       W = m g
                                     -> every mass weighs g/g_earth times as much
    buoyancy and natural convection  Ra = g beta dT L^3 / (nu alpha)
                                     -> the Rayleigh number is linear in g
    radiative equilibrium            T_eq = [S (1-a)/(4 sigma)]^(1/4)
                                     -> g does not appear: no effect at all
```

It reads gravity, temperature, pressure, insolation and albedo, in English or Portuguese, with the number in any
of the usual forms (`2.5 times`, `24.5 m/s^2`, `2,5 vezes maior`, `5 kPa`). A question it cannot read comes back
with `understood == False` and the list of what it does read; `Brief.run()` then raises rather than inventing an
answer. It runs the experiment with the world that experiment already declares, so that only the asked parameter
changes -- a question about pressure does not silently reset the temperature to Earth's.

## The three demonstrators

| Experiment | Direction | What it settles |
|---|---|---|
| `futures.exoplanet_laboratory` | planetary systems | radiative equilibrium, the latitudinal energy balance against its Legendre solution, the scale height and what gravity does and does not touch |
| `futures.mars_habitat` | future engineering | a three-layer envelope against its resistance network, the heater budget, the radiative boundary against its linearisation, the pressure hull and the shield load |
| `futures.programmable_matter` | advanced materials | conductivities that depend on the temperature, on the position or on a transition, against their closed forms, and the point where the material stops conducting |

```python
import pinneapple_futures as F

print(F.names())
print(F.atlas())                       # the eight directions, with what is available and what is planned
assert F.validate() == []              # every pointer names code that exists

res = F.run("futures.exoplanet_laboratory", dim="1D", quick=True)
print(res)                             # every comparison, every check, the trust decision and the assessment
```

A few results worth reading in the source, because they are the arguments rather than the decorations:

* **the equilibrium temperature does not contain `g`.** A world can be heavier without being hotter; what changes
  is the atmosphere, whose scale height falls as `1/g`. The run checks both: `T_eq` identical, `H` exactly halved
  at double gravity.
* **a spacecraft's hull does not get thicker in a heavier world.** The hoop stress balances the *internal*
  pressure; the regolith shield above it does get heavier, in direct proportion. The experiment measures the two
  separately, and verifies the membrane formula as the thin-wall limit of the thick-wall Lamé solution.
* **a programmable conductivity cannot regulate all the way.** `dq/dT1 = k0 (1 + beta T1)/L` falls with a falling
  `beta`, but reaches zero exactly where `k` itself vanishes -- and the solver refuses to run there. The demo
  reports the factor it *can* buy (two thirds at `beta = -1/(3 T1)`) and where the wall is.

## Publishing an experiment

```bash
python -m pinneapple_futures publish all --out out/futures
```

writes, per experiment, `result.json`, the figures and a `README.md` publication: the world with the provenance of
every number, the equations, the measured quantities, the comparisons with their sources, the physics checks, the
evidence axes with their two verdicts, and a section -- **What this does not show** -- that no experiment of the Lab
may omit. The same page is what a research note would publish, so the Lab's public output is always the checked
one.

## Command line

```bash
python -m pinneapple_futures atlas [--markdown] [--check]    # the eight directions; --check exits 1 on a bad pointer
python -m pinneapple_futures list [--area AREA]              # the registered experiments
python -m pinneapple_futures run NAME [--quick] [--set k=v] [--out DIR]
python -m pinneapple_futures ask "What if ... ?" [--run] [--json]
python -m pinneapple_futures publish NAME|all --out DIR
```

## In a flow

`futures.ask`, `futures.run` and `futures.atlas` are Flow Studio nodes, so a what-if question can be a step of a
larger flow, with the tables and metrics every other node produces.

## What the Lab is not

* It does not claim that a mathematically consistent model is a realisable technology.
* It does not compare with a measurement unless the run actually performed one, and it says which of its
  demonstrators do.
* It does not turn a parameter that appears in no equation into a result: the engine's answer to "what does
  gravity do to the equilibrium temperature?" is *nothing*, and it says so.
* It is an application layer: the experiments are `pinneapple_science` experiments, the results carry the same
  physics checks and the same trust report, and the solvers are the library's. Nothing here is a fork.

## References

* North, G. R. (1975). *Theory of energy-balance climate models.* Journal of the Atmospheric Sciences 32(11),
  2033-2043 -- the diffusive energy-balance model and its Legendre solution.
* Stefan-Boltzmann law; σ = 5.670374419 × 10⁻⁸ W m⁻² K⁻⁴ (CODATA 2022).
* ISO 2533:1975 standard atmosphere -- the sea-level and 1000 m pressures the isothermal column is compared with.
* NASA NSSDC Mars Fact Sheet -- the Martian surface gravity, mean pressure and mean temperature.
* Thin-wall pressure vessel: hoop stress `σ = p r / t` (membrane theory), and its thick-wall limit, the Lamé
  solution.
