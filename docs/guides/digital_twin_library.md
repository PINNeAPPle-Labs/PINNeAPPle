# Digital Twin Library: eleven systems, each with its own conservation laws

`pinneapple_systems.digital_twin.library` holds reference models of eleven engineering systems —
an FPSO with its subsea well, a blast furnace, a chemical synthesis loop, a reusable booster, a
lunar base, a gas pipeline network, a compression station, a green hydrogen plant, a seawater
desalination train, an industrial refrigeration plant and a data centre. They are the bottom rung of the fidelity ladder the Digital Twin Library issues
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
| `pipeline_network` | mass, energy, linepack | the stations' polytropic shaft work against `cp` times the temperature rise it produces; the closed-form linepack integral against a 200-node quadrature of the pressure profile |
| `compression_station` | mass, energy, cooling | the shaft work from `cp ΔT` against the integral of `v dp` along the polytropic path, stage by stage; the intercoolers against the enthalpy the previous stage added |
| `green_hydrogen_plant` | electricity, energy, water, oxygen | the power asked for against cells × voltage × current; hydrogen at its heating value plus the heat the overpotential rejects against the electricity in; the water the stoichiometry demands against the mass of the two gases it became |
| `desalination_plant` | water, salt, energy | salt in the feed against salt in permeate plus concentrate, with the concentrate's concentration *derived* from that balance and then used to set the osmotic pressure; the pump less the energy-recovery device against the specific energy times the permeate flow |
| `industrial_refrigeration` | energy, duty | the compressor's work as the isentropic enthalpy rise against the integral of `v dp` along the same path; the first law round the loop against the condenser's own enthalpy route |
| `data_center_cooling` | energy | the facility built up component by component against the heat the cooling plant must move. In a data centre every watt leaves as heat, so this is a *completeness* check and the one that fails when a PUE is wrong |

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
| `pipeline_network` | 120 km of 0.9 m line, two stations at 1.35, 95 bar in and 60 bar out | 2 210 t/h, 63 MW of compression, 69 °C discharge, 5 175 t of linepack |
| `compression_station` | 120 kg/s of 0.65-gravity gas, 30 to 90 bar in two stages | 22.4 MW of shaft work, 82 °C hottest discharge, 20.2 MW of cooling, 1 784 kg/h of fuel |
| `green_hydrogen_plant` | 100 MW stack at 1.78 V and 1.5 A/cm², 2 % purge, product at 30 bar | 48 t/day of hydrogen, 54.1 kWh/kg at the fence, 61.6 % LHV efficiency, 16.8 MW of heat, 16.3 t/h of oxygen |
| `desalination_plant` | 1 000 m³/h of 35 g/L seawater at 45 % recovery, 95 % energy recovery | 450 m³/h of permeate at 175 mg/L, 63.5 g/L brine, 65.9 bar applied, 2.83 kWh/m³ against a 1.10 kWh/m³ floor |
| `industrial_refrigeration` | 5 MW at −8 °C, ammonia, 30 °C cooling medium | COP 3.31 against a Carnot ceiling of 5.85 (56.6 % of it), 13.2 bar condensing, 137 °C discharge, 6.4 MW rejected |
| `data_center_cooling` | 20 MW of IT, 250 m³/h of air per kW, 24 °C wet bulb | 13.3 K air rise, chiller COP 8.1, PUE 1.25 at design and 1.23 annual, 215 GWh/yr |

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
* the lunar base is sized by its night, and the night comes from the synodic month;
* the pipeline's throughput is the flow at which the pressure the line needs at its inlet is exactly
  the pressure the supply offers — the stations are set to a ratio and the line takes what it can;
* the compression station *proves* the equal-ratio split instead of asserting it: it searches the
  split and checks that no other division of the same total ratio costs less work;
* the electrolyser's hydrogen is what Faraday's law says it is, and its heat is what the cell
  voltage says is left over — the two share no algebra, so the energy balance can fail;
* the desalination train derives its brine concentration from the salt balance and then uses it to
  set the osmotic pressure the pump has to beat, and every run is checked against the reversible
  separation work computed as an integral rather than quoted;
* the refrigeration cycle is built from correlations that can each be checked against published
  data — Antoine reproduces ammonia's normal boiling point, Watson is anchored on its latent heat
  there — and the condenser balance reports the property model's own inconsistency (a few per cent)
  instead of hiding it;
* the data centre states its own limitation: in a hall where every watt leaves as heat the first law
  is a tautology, so the single balance is a completeness check on the PUE roll-up rather than two
  dressed-up copies of the same sum.

## Named scenarios and the registry's status

A parameter set that says something is worth keeping, so it has a name:

```python
from pinneapple_systems.digital_twin.library import SCENARIOS, run_scenario, manifest, save_scenarios

run_scenario("pipeline_network/low_delivery_pressure")   # a case, not a keyword soup
manifest()["summary"]                                    # every twin and every case, with verdicts
save_scenarios("my_cases.json")                          # hand a case to somebody else
```

Each scenario is a slug, a name, a few overrides and a line saying where the numbers came from. It
runs through the same `simulate` as anything else, so a scenario can never bypass a range check.

Four of the eleven shipped scenarios **trip an invariant on purpose**, and they are the most useful
ones: a 3:1 ratio in a single compression stage discharges at 143 °C, above the machine limit; a 69 K
lift in a single-stage ammonia machine discharges above 150 °C, which is where a two-stage machine
starts; a heavier booster runs out of landing propellant; and a cold-climate data hall has an
economiser that costs more than the chiller it replaces, because the fixed free-cooling share is not
sized against the plant. They carry `expect="finding"` instead of being tuned until they pass, and
the check is strict in both directions — a finding that stops being reported is a regression too.

`python -m pinneapple_systems.digital_twin.library --check` is the CI hook: it exits 1 if
a case that had to pass failed, and prints what the library's own status page says.

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

The 0D/1D models here are the first rung of the Digital Twin Library issues (`#1770`, `#1772`,
`#1774`, `#1777`, `#1779`, `#1784`, `#1801`, `#1813`, `#1814`, `#1807`, `#1819`). The registry and
the scenario manifest are the discoverability half of the shared-specification issue (`#1802`); the
typed unit algebra, the component graph with ports, the solver adapter protocol, the cross-fidelity
consistency checks and the VeriPhysics applicability gate are not here, and that issue is not closed
by them. What comes above them — 1D transients, 2D/3D solvers, trained
surrogates with an OOD domain, VeriPhysics gates, the Engineering API workers and the Studio
scenes — is tracked in those issues, and none of it is claimed here.

## Tests

`tests/test_digital_twin_library.py` runs every system and checks, for each one, that its balances
close and its invariants hold, that a run is reproducible, and — separately — that the model
reproduces answers computed by hand in the test: Vogel's published curve, a straight-line
productivity index as its zero-exponent special case, LMTD for a line with no Joule-Thomson
cooling, the raceway balance from published enthalpies, Gibbs energy for the equilibrium constant,
van't Hoff on the function actually used, the exponential atmosphere's scale height, Stefan-Boltzmann
for the radiator and the FeO oxygen yield, the isothermal line equation written out factor by factor,
the linepack integral collapsed both to `p V / Z R T` at a flat pressure and to the mean of a
parabolic pressure profile, the polytropic stage work for one stage and the hand sum for three, and
Colebrook-White solved for the friction factor the run reports, Faraday's law and the two water
voltages, the nine kilograms of water per kilogram of hydrogen, van't Hoff's thirty bar for
seawater, the separation-work integral against a quadrature of the same integrand, Antoine's normal
boiling point and its inverse, Watson anchored at the boiling point, and the data centre's air
temperature rise and PUE recomputed from their own definitions.
