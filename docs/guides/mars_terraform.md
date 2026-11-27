# Terraforming Mars: a globe warming over centuries, and where the model runs out

`pinneapple_physics.terraform` is a zonal energy-balance model of Mars with the CO2 cycle bolted to it, integrated
for centuries, and an ensemble whose spread is the honest answer to the question in the title. It is not a Mars
climate model. It is the smallest model that has the three things this question turns on: Mars' own insolation, a
CO2 cap that cannot be warmed above its frost point, and a published reservoir of CO2 that runs out.

The verdict is the paper's, not the model's: **terraforming Mars is not possible with the CO2 that is there**
(Jakosky & Edwards 2018, *Inventory of CO2 available for terraforming Mars*, Nature Astronomy **2**, 634-639,
doi:10.1038/s41550-018-0529-6). The model is built to check that verdict rather than to be talked out of it, and
every number below is something it measured.

```
the sun and the greenhouse heat each latitude
    -> the surface warms
        -> where it warms above the CO2 frost point at the current pressure, the cap sublimates,
           which takes latent heat, which holds the cap at the frost point while any ice is left
        -> the gas goes into the atmosphere and the pressure rises
            -> which raises the frost point and the greenhouse
```

## The four things the model has to get right before it says anything

**The insolation is integrated, not fitted.** `ebm.annual_insolation` averages the daily box formula over the orbit,
with Mars' eccentricity (0.0934) and obliquity (25.19°) from the NASA fact sheet, and with the areocentric solar
longitude running from the true anomaly. The solar longitude of perihelion is *derived* from the pole direction and
the J2000 elements — `(RA 317.681, Dec 52.887)` and `longitude of perihelion 336.04°` — and comes out at **250.99°**,
which is the value Mars climate work quotes. It matters: it makes the southern summer the short hot one, and the
south polar annual mean comes out **14 % above** the north. The check that the integration is right is that the area
average of the answer is `S0 / (4 sqrt(1 - e^2))`: **147.22 W m⁻² measured, 147.17 expected**, a 4e-4 difference
which is the orbit quadrature and not the physics.

**The frost point is Clausius-Clapeyron, and it is the whole coupling.** From the CO2 triple point (216.58 K,
5.185 bar) and the enthalpy of sublimation (25.23 kJ/mol), both NIST:

```
ln(p / p_t) = -(dH_sub / R) (1/T - 1/T_t)
```

Two checks that it is the real curve and not a convenient one. At 194.7 K it gives 1.07 bar against the measured
1 atm — **6 %**. And at the present 6.36 mbar it gives **146.50 K**, which is the temperature the Martian seasonal
cap is observed to form at. Its slope, `R T²/dH`, is **7.4 K per e-fold of pressure**, and that slope is the whole
argument: from 6.36 to 20 mbar the frost point rises 8.5 K while the CO2 greenhouse adds 4 K, so **a greenhouse warms
Mars's surface without warming its cap**. The cap does not melt because the planet is warmer; it melts because
somebody heats *it*.

**The greenhouse saturates.** The published ends (Jakosky & Edwards: 20 mbar warms by "less than 10 K"; about 1 bar
brings the surface "close to the melting point", i.e. the ~60 K the paper calls necessary) fix a power law
`G = 60 K (p/1 bar)^0.485`, and the third point is a prediction: **5.16 K at today's 6.36 mbar**, which is the
greenhouse Mars is observed to have. It is applied the way a grey atmosphere applies it, as an optical depth
`tau = (1 + G/T_eff)^4 - 1` that divides the outgoing longwave radiation — a *fraction*, so it cannot compound with
the temperature. Written as an additive flux it does compound, and this model was wrong that way before it was
right: 1.2 bar came out at 324 K instead of 275 K.

**The transport conserves the global mean exactly.** `D (1 - x²) dT/dx` in finite-volume form on equal-area bands:
the area-weighted sum of the divergence is zero to roundoff, because `1 - x²` vanishes at both cell faces. That is a
test, not a hope.

With those four in place, the model's own global mean at the present 6.36 mbar is **214.29 K** against the fact
sheet's **214 K**. Nothing was fitted to that point.

## The scenario

`terraforming.TERRAFORMING_SCENARIO`: an engineered greenhouse of 3 W m⁻² and 0.06 of dust on the caps, ramped in
over 150 years and held for 350 more. The dust is the method the paper describes ("using explosives to raise dust
into the atmosphere so that it would deposit on the polar caps, effectively decreasing their surface albedo"); the
gas is a declared what-if, because the paper explicitly leaves it out ("these are short-lived and without a feasible
source using current technologies").

| year | global mean | CO2 pressure | ice left | cap area frosted | cap edge |
| --- | --- | --- | --- | --- | --- |
| 0 | 210.96 K | 6.05 mbar | 9.00 mbar | 100 % | 70.0° |
| 50 | 213.3 K | 9.07 mbar | 5.98 mbar | 66 % | 73.7° |
| 100 | 215.65 K | 11.19 mbar | 3.86 mbar | 43 % | 76.9° |
| 150 | 217.26 K | 12.80 mbar | 2.25 mbar | 25 % | 80.0° |
| 500 | 217.26 K | 12.80 mbar | 2.25 mbar | 25 % | 80.0° |

The caps retreat, the pressure climbs — and then it **stops**, with 2.25 mbar of CO2 still sitting as ice on a pole
that has no reason to give it up. The atmosphere plus that residue is 15.05 mbar, which is Jakosky & Edwards'
"less than 15 mbar total", reached from the other direction.

## The ensemble, which is the actual answer

Sixty-four planets. The frost albedo (0.70-0.95), the poleward diffusivity (0.001-0.020 W m⁻² K⁻¹), the regolith
reservoir (0-40 mbar), the CO2 greenhouse at one bar (40-80 K) and the strength of the engineering itself (0-1.5× the
scenario) are drawn from their ranges; every member starts from the *same* observed planet, so the spread that
follows is the model's uncertainty and not the initial condition's.

**The declared threshold is 2 K**, about a third of the warming the scenario itself delivers: past that point the
ensemble cannot tell the warming the scenario predicts from the uncertainty of the model that predicted it. The
spread reaches it after **44 years**.

That is the number this whole exercise exists to produce, and it is the honest punchline: *a model of a planet we
are deliberately changing stops being able to say what the planet is doing within a human career.* The spread peaks
at 2.01 K around the horizon and then falls back to 1.75 K, because after that the reservoir, not the parameters,
decides the answer.

The bigger question has no horizon at all. The same spread never reaches 6 K, a tenth of the ~60 K liquid water
needs, over the full five centuries: **the model is unsure how much Mars would warm, never whether the attempt
works.**

## The verdict, measured

Every published reservoir put into the air at once — no engineering at all, just the CO2 — and settled for 1500
years:

| inventory | CO2 | global mean | warming | what it is |
| --- | --- | --- | --- | --- |
| today | 6.36 mbar | 214.3 K | — | what Mars has now |
| mobilizable | 20.0 mbar | 218.1 K | +3.8 K | Jakosky & Edwards' "representative maximum" |
| accessible top | 67.4 mbar | 225.3 K | +11.0 K | every accessible reservoir at its upper bound |
| everything | 217.4 mbar | 237.7 K | +23.4 K | plus the 150 mbar clathrate extreme upper limit |
| deep carbonates | 1217 mbar | 274.9 K | +60.7 K | the '~1 bar' their figure 3 calls inaccessible |

The last row is the only one that reaches the melting point, it needs the reservoir the paper lists as "not a
legitimate source for replenishing the Martian atmosphere", and it would take "regional- or planet-scale strip
mining". Everything that can actually be reached warms Mars by 11 K of the 60 it needs. The model reproduces the
published conclusion, from the published CO2 and the published greenhouse, without being pointed at it.

## The picture

`examples/mars_terraform` writes `figures/mars_globe.gif` — a rotating Mars with the caps retreating, the numbers on
the right, and a **confidence bar under the globe that is `1 - spread / 2 K`**: it reaches zero exactly at year 44
and the frame after that says "no trust left". The projection is `pinneapple_physics.weather.viz.orthographic`, the
same one the weather examples use.

The **topography is synthetic and the figures say so**. The repo has no MOLA data and this package does not invent
a map of Mars: `globe.synthetic_topography` places a Tharsis-sized bulge, a volcano, a rift and a basin roughly where
they are on the real planet so the globe reads as a globe, and every frame carries the words "synthetic
topography". `figures/mars_centuries.png` is the four curves the animation is a view of.

## What is physics and what is assumption

Physics, and checkable against something published:

* the annual insolation, from Mars' orbit — checked against `S0/(4 sqrt(1-e²))` and against the equinox and solstice
  limits;
* the CO2 frost point — checked against the dry-ice sublimation point and against the temperature the seasonal cap
  forms at;
* the CO2 reservoirs — every one of them is a page of Jakosky & Edwards, and `co2.RESERVOIRS` carries the quote;
* the energy balance and its transport — the global mean is conserved to roundoff, and the mean temperature at
  6.36 mbar comes out at 214.3 K against the fact sheet's 214 K;
* the mass of CO2 — conserved exactly, `atmosphere + caps + regolith`, in every run of every test.

Assumptions, listed in `ebm.ASSUMPTIONS` and worth repeating because they are the model's limits:

* **annual means only.** No seasons. This is the largest assumption and it has a visible cost: the model's polar
  band settles at 128.5 K, where the observed annual mean is nearer 150 K, because on Mars the polar night, the
  seasonal CO2 cycle and the latent heat it carries do most of the polar energy budget. The model therefore needs a
  **weak** poleward transport (0.004 W m⁻² K⁻¹, against Earth's Budyko 0.6) to hold a cap at all, and that number is
  a calibration to the observed cap, not a measurement;
* **the greenhouse is a power law** through two published ends. Its shape between them, and beyond 1 bar, is an
  assumption. Its agreement at 6.36 mbar is a prediction, but one point is not a curve;
* **the frost albedo is nobody's measurement** for a dusty cap. Clean CO2 snow is 0.93-0.98 broadband, and 0.01 %
  of dust takes ~60 % off the visible albedo; the central 0.90 and the ensemble's 0.70-0.95 are a declaration. The
  model is *bistable* in it: a cap that has melted does not come back, because the dark ground it exposes keeps the
  pole warm. That is a result as much as a limitation;
* **the cap is a zonal film of uniform thickness**, sized to the published 9 mbar, and it is not allowed to grow past
  that — which keeps a very cold pole from freezing the whole atmosphere out, something Mars' seasonal cycle prevents
  and this model cannot represent;
* **the regolith releases on one e-folding time**, 10⁴ years, from Jakosky & Edwards' estimate that heat takes about
  that long to cross 100 m of regolith. Over the five centuries of the scenario that is 5 % of 40 mbar: the biggest
  reservoir is also the one the timescale puts out of reach, and that is part of the answer;
* **the engineering is a flux.** No chemistry, no lifetime, no source, no cost. The paper's own position is that
  there is no feasible source; the model's position is "grant it anyway, and watch what the CO2 does";
* the cell is one mixed layer of atmosphere and regolith skin at every latitude, and the planet is zonally symmetric
  — the two caps in the model are identical, and the globe is drawn from one hemisphere of a one-dimensional model.

## Running it

```
python examples/mars_terraform/mars_terraform.py            # the whole thing, about 25 s
python examples/mars_terraform/mars_terraform.py --quick    # no GIF, 32 members
```

Results go to `examples/mars_terraform/results/mars_terraform.json`, figures to
`examples/mars_terraform/figures/`.

```python
from pinneapple_physics.terraform import terraforming as tf

model = tf.TerraformModel()
run = model.run(tf.TERRAFORMING_SCENARIO)
run.pressure_pa[-1] / 100.0           # 12.80 mbar, and it stops there
ensemble = tf.run_ensemble(members=64)
ensemble.horizon()                    # 44.0 years, at the declared 2 K
```

## References

* B. M. Jakosky & C. S. Edwards, *Inventory of CO2 available for terraforming Mars*, Nature Astronomy **2**, 634-639
  (2018), doi:10.1038/s41550-018-0529-6 — every reservoir, the 20 mbar representative maximum, the two ends of the
  greenhouse, the 10⁴-year regolith timescale and the conclusion.
* NASA Mars Fact Sheet, `nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html` — radius, gravity, Bond albedo, solar
  irradiance, black-body temperature, surface pressure, mean temperature, atmospheric mass, and the J2000 elements
  from which the solar longitude of perihelion is derived.
* NIST Chemistry WebBook — the CO2 triple point (216.58 K, 5.185 bar) and the enthalpy of sublimation.
