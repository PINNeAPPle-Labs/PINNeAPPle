# Digital Twin Library: five systems, each with its own conservation laws

`pinneapple_systems.digital_twin.library` holds reference models of five engineering systems —
an FPSO with its subsea well, a blast furnace, a chemical synthesis loop, a reusable booster and a
lunar base. They are the bottom rung of the fidelity ladder the Digital Twin Library issues
describe: cheap enough to run in a test suite, explicit about their assumptions, and the source of
the solver data a trained surrogate would later be fitted to.

Every one of them takes SI parameters with declared ranges, returns a machine-readable result with
its conservation balances, its invariants, its assumptions, its limits and what it did **not**
verify, and states where every external number came from.

```python
from pinneapple_systems.digital_twin.library import catalogue, run, describe, compare

print(compare())            # every system on its declared default, with its verdict
result = run("fpso_subsea") # one system, default scenario
print(result.report())      # outputs, balances, invariants, limits, verdict
print(describe("fpso_subsea"))   # parameters with ranges, assumptions, sources
result.save("fpso.json")    # the whole run as JSON, provenance included
```

## Why balances and not just outputs

A model that only prints numbers cannot be wrong on its own terms. Each system here computes its
conservation laws **twice, by different routes**, and reports the residual:

| System | Balances | The two routes |
|---|---|---|
| `fpso_subsea` | oil, gas, lift, energy | tank pressure decline against the integrated oil rate; endpoint enthalpy against the integrated heat flux along the line |
| `blast_furnace` | iron, carbon, oxygen, mass, raceway energy | iron charged against iron tapped and lost; total mass from molar masses, independent of the stoichiometry algebra |
| `chemical_plant` | A atoms, B atoms, energy | make-up against product plus purge, at the steady state the loop solves for |
| `reusable_booster` | mass, momentum, energy | acceleration integrated along the path against the endpoint velocity; specific power against kinetic plus potential energy |
| `lunar_base` | energy, oxygen, water | generation against day load plus storage; regolith yield against crew consumption |

A residual above the declared tolerance fails the run, and `result.ok` is `False`. `run.require()`
raises `BalanceFailed` with every failing line named.

The invariants are separate: they are things that must hold for *any* parameter set — the well
produces nothing at zero drawdown, the booster cannot beat its own Tsiolkovsky limit, the lunar
night is half a synodic month, a wet blast runs the raceway colder.

## What the default scenarios answer

| System | Default scenario | The answer |
|---|---|---|
| `fpso_subsea` | 32 MPa reservoir, 2 500 m tubing, 6 km flowline, 380 m water depth | 1 329 m³/d at start-up, 27 % recovery in a year, 54 °C arrival, 22 K of wax margin |
| `blast_furnace` | 330 kg/thm coke, 170 kg/thm coal, 1 050 Nm³/thm blast at 1 200 °C | RAFT 2 453 K, 40 % direct reduction, gas utilisation 0.52, 271 kg/thm of slag |
| `chemical_plant` | 80 bar, 480 K inlet, `A + 2B ⇌ C` | 20 % per pass, 93 % per mole of make-up, 611 K at the bed outlet |
| `reusable_booster` | 400 t stage, 1 900 m/s at MECO, return to launch site | 139 km apex, 1 697 m/s of boostback, 1.1 m/s touchdown, 17 t of propellant left |
| `lunar_base` | 4 crew, 1 200 m² of array, 28 kW at night | 354 h of night, 11.7 MWh of storage, 37 m² of radiator, 3 t of regolith per lunar day |

The numbers are engineering-plausible and *not* plant records: every system carries a
`not_verified` list that says exactly that. The reactor's enthalpy and entropy of reaction are
declared illustrative inputs; set them from your own catalyst data.

## Coupling the systems the way the physics couples them

The interesting part of each model is what it solves rather than computes:

* the FPSO finds its operating point where the reservoir's inflow curve crosses the line's
  outflow curve, by bisection on the bottom-hole pressure;
* the blast furnace's top-gas composition and degree of direct reduction fall out of the carbon
  and oxygen balances together;
* the chemical loop's recycle composition is solved in closed form, so the steady state is exact
  rather than the limit of a slow iteration;
* the booster's boostback burn is solved so the stage lands on its own pad — the burn is commanded
  in velocity, and the last step of the burn is trimmed to it, because a burn that always ends on
  a whole time step quantises downrange to hundreds of metres;
* the lunar base is sized by its night, and the night comes from the synodic month.

## Ranges, extrapolation and honesty

A parameter outside its declared range raises `ParameterError` unless the caller asks for it:

```python
model = build("lunar_base")
model.simulate(crew=40)                 # ParameterError: outside [1, 12]
model.simulate(crew=40, strict=False)   # runs, and records the extrapolation in the result
result.extrapolated                     # ['crew = 40.0 - outside [1.0, 12.0]']
```

The range is the range in which the model's equations are the right physics, not where the
arithmetic survives: the hydrostatic line is valid at 50 bar and arithmetic at 5 000 bar.

## The ladder

Each entry in the catalogue carries its fidelity and the next rung:

```python
for entry in catalogue():
    print(entry.slug, entry.fidelity, entry.status)
```

The 0D/1D models here are the first rung of the Digital Twin Library issues (`#1770`, `#1777`,
`#1779`, `#1784`, `#1801`). What comes above them — 1D transients, 2D/3D solvers, trained
surrogates with an OOD domain, VeriPhysics gates, the Engineering API workers and the Studio
scenes — is tracked in those issues, and none of it is claimed here.

## Tests

`tests/test_digital_twin_library.py` runs every system and checks, for each one, that its balances
close and its invariants hold, that a run is reproducible, and — separately — that the model
reproduces answers computed by hand in the test: Vogel's published curve, a straight-line
productivity index as its zero-exponent special case, LMTD for a line with no Joule-Thomson
cooling, the raceway balance from published enthalpies, Gibbs energy for the equilibrium constant,
van't Hoff on the function actually used, the exponential atmosphere's scale height, Stefan-Boltzmann
for the radiator and the FeO oxygen yield.
