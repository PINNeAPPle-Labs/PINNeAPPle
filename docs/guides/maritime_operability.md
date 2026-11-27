# An operability window with a measured confidence: a buoy's sea, a vessel's answer, and the hours left

`pinneapple_physics.maritime` takes a buoy record and ends on one sentence a deck crew can use: **work allowed until
hour N, at a confidence the forecast was measured to have**. Everything between the two is measured rather than
assumed -- the spectrum against NDBC's own numbers, the section against the limits any correct solution has to pass, the
mooring against its own closed forms, and the confidence against what the sea did next.

```
buoy 41010   45 days, 2080 measured spectra, 46 bands          the sea
sea state    swell decays faster at high frequency              what it will do in 24 h
section      added mass, damping, excitation from a 2-D solve   what the vessel answers
motion       heave, roll, pitch at a crane and at a bow         what the load feels
mooring      quasi-static catenary, 404 kN at rest              what holds it
residual     slam, mooring step, or a lying instrument          what happened
window       Hs, acceleration, relative velocity                whether to work, and how sure
```

```python
from pinneapple_physics.maritime import ndbc, seastate, rao, motion, mooring, operability

buoy = ndbc.load("examples/maritime_operability/data/ndbc_41010.npz")
time, freq, measured = buoy.spectra()                     # 2080 spectra on NDBC's band grid

model = seastate.SeaStateModel(swell_decay_hours=192.0, windsea_growth_hours=16.0)
ahead = model.forecast(freq, measured[-1], buoy.wind_at(time[-1]), leads=range(1, 25))

section  = rao.Section.from_gm(beam=20.0, draught=6.0, gm=1.5, depth=30.0)
transfer = rao.sweep(section).with_pitch(rao.pitch_transfer_function(40.0, 6.0, 30.0, kg=7.06))

crane = motion.significant_acceleration(freq, measured[-1], transfer, motion.Point(x=0.0, y=8.0))
print(operability.opening(operability.Criteria(hs_max=1.5, acceleration_max=0.2), ahead.leads_hours, ahead.hs,
                          [crane] * ahead.leads_hours.size)["sentence"])
```

## The sea state, measured against the buoy

The measured spectra are the ones NOAA publishes on NDBC's 46-band grid, and they are enough to check the two shapes
against a real sea without any fitting to a target:

| what was checked | measured |
| --- | --- |
| `4 sqrt(m0)` over the 46 bands, against the buoy's own `WVHT` | **0.974** -- the published bands are 2.6 % short of the number NDBC reports, which integrates bands below 0.033 Hz that the file does not carry |
| JONSWAP fitted to each of 120 spectra, RMS of `log S` | **28 %**, against **43 %** for Pierson-Moskowitz on the same spectra |
| the peak enhancement the fit finds | **gamma = 1.8** -- between the 1.0 of a fully developed sea and the 3.3 the JONSWAP average is quoted with |
| a synthesized record against the spectrum it was cut from | `4 sigma` of the record is the `Hs` it was built with, to 0.2 % |

The 0.974 is why the analysis scales every spectrum so that its own `4 sqrt(m0)` matches the height the buoy reports: a
forecast that starts 2.6 % low has a bias that has nothing to do with the model.

## The forecast with a horizon, and what it is worth

Two mechanisms move a spectrum at a fixed point. The **swell** decays, and it decays faster at high frequency, so `Hs`
falls while `Tp` rises -- the signature of an ageing sea that carrying the last observation forward cannot produce. The
**wind sea** is carried with the wind it was measured in, because holding a measured wind is not a forecast that it will
keep blowing, and only a sea above what that wind can hold is allowed to decay.

Two coefficients, fitted on the first 22 days and scored on the 22 after them:

| lead | Hs RMSE | Hs against persistence | Tp RMSE | Tp against persistence |
| --- | --- | --- | --- | --- |
| 1 h | 0.222 m | +0.2 % | 0.83 s | **-8.0 %** |
| 6 h | 0.384 m | +1.7 % | 1.08 s | +4.0 % |
| 12 h | 0.527 m | +2.0 % | 1.41 s | +3.9 % |
| 24 h | 0.753 m | +2.2 % | 1.84 s | +4.7 % |

Those numbers are the honest headline, and they are not flattering: **on this record a single buoy's own spectrum
carries almost nothing about tomorrow's significant height.** The two per cent at 24 h is real and small, and the
-8 % at one hour is the price of a forecast that has to start from an analysis whose peak period is read off 46 coarse
bands: the buoy's own `DPD` comes from the full spectrum at a resolution the file does not carry, and at a one-hour lead
the forecast is the analysis.

The `Tp` skill is where the spectral evolution pays, and it depends on what is done with the wind sea:

| wind sea | Hs skill at 12 h | Tp skill at 12 h | Hs bias at 12 h |
| --- | --- | --- | --- |
| **carried** with the wind (the default) | +2.0 % | +3.9 % | -5 cm |
| relaxed toward a fully developed Pierson-Moskowitz sea | +1.5 % | **+16.4 %** | +3 cm |

Relaxing the wind sea toward what the wind could make buys four times the period skill and costs no height skill at
all, because the peak period of that closure follows the wind (`Tp = 7.16 U/g`). It is not the default because at this
site the measured wind sea is a fifth to a half of the fully developed one -- the sea 20 miles offshore is
fetch-limited -- so the closure is fighting the data even when it scores; `SeaStateModel(windsea_growth=True)` turns it
on, and that row is the measurement.

The group velocity is what turns a distance into a lead time, and it is what says how much of the sea a forecast can
possibly know about: a 14 s swell covers 39 km in an hour, a 5 s sea 14 km, so the same 39 km is one hour of travel for
the swell and 2.8 for the wind sea.

## The section: solved, not tabulated

Added mass and damping of a rectangular section are not closed form, and a table recalled from memory is a number with
nothing to check it against. So the two-dimensional free-surface problem is solved: `div grad phi = 0` on a graded grid,
the linearised free surface `dphi/dz = -(omega^2/g) phi`, the hull as a Neumann surface, a sponge three wavelengths long
in which the free-surface coefficient acquires a large imaginary part, and the force from the pressure at the wetted
faces. One matrix factorisation per frequency, four right-hand sides -- sway, heave, roll and the diffraction problem --
and then `(-omega^2 (M + A) - i omega B + C) xi = F_exc` for the response.

The numbers, for a 20 m beam and 6 m draught section with the 1.5 m metacentric height of a loaded support vessel:

| check | measured |
| --- | --- |
| heave per unit wave amplitude at an 80 s period | **1.0009** -- a body small against the wavelength moves with the water |
| roll per radian of wave slope at 80 s | **1.062** |
| heave damping across the band | positive everywhere, and falling to zero at both ends |
| 40 nodes per wavelength, and a four-wavelength sponge | change the heave response by **under 2 %** |
| roll resonance | **14.53 s**, against the **11.47 s** the hydrostatic period gives |

That last row is the measurement worth keeping. The hydrostatic period is `2 pi k_xx / sqrt(g GM)` with `k_xx = 0.35 B`
-- the estimate the naval module already uses -- and the solve says the effective gyration is 0.43 B, because it
measures **A44 = 0.48 I44**: the water adds nearly half again the roll inertia of the hull, and no rule of thumb that
fixes `k_xx` at a fraction of the beam can contain that. The difference is one second of roll period per metre of
`k_xx`, which is the difference between a crane limit with margin and one without.

The **roll damping** the potential solve gives is `zeta = 4e-6` of critical -- nothing -- and that is a known limit of
linear potential theory rather than a bug: a real roll is damped by bilge viscosity, eddy shedding at the bilge keels
and lift on the skeg, none of which are in the equations here. The section therefore carries an imposed 8 % of critical,
which is the number a vessel of this kind is quoted with, and the sensitivity to it is one line of the example.

**Pitch is not a two-dimensional mode** and the module does not pretend otherwise: what it offers is the roll problem of
the *longitudinal* section, with the longitudinal metacentric height, which is what a strip calculation uses. It has the
right limits (1.015 at long waves, 0.29 at 6 s) and it is not a pitch RAO of a ship.

## The mooring: the load is not linear in the offset

A chain of constant submerged weight between an anchor on the seabed and the fairlead, quasi-static -- no line dynamics,
no drag along the line, no snap load, and the guide says so because a quasi-static solution under-predicts the peak. The
closed forms are checked against each other: a line hanging vertically carries its own weight, the two branches agree at
the span where the line lifts off the seabed, and a line too short to reach is refused.

| quantity | measured |
| --- | --- |
| eight 76 mm chains, 437 m long, in 120 m of water, 400 m of span | **404 kN** at rest -- 6.9 % of the 5.83 MN breaking load |
| stiffness at rest, with 166 m of chain on the seabed | **34.8 kN/m** |
| the same 400 kN environmental load, eight lines | **512 kN** on the worst line, 5.15 m off |
| the same load, two symmetric lines lost | **598 kN**, 8.86 m off -- 10.3 % of breaking |

The comparison is at constant load rather than constant offset because the load is what the weather decides: with two
lines gone the vessel sits further off until the six that are left carry it, and at that offset each of them is stiffer.

## What happened: a slam, a mooring step, or a sensor that is lying

The residual is the arithmetic first -- what the model says the instruments should have read, minus what they read, in
units of each instrument's own noise -- and then the *kind* of event is read off which channels disagree and what the
physics was doing:

| scenario | what the residual said |
| --- | --- |
| a quiet 4 m sea | nothing |
| the bow hitting the water for five seconds | **slamming** -- both accelerometers, and `v_rel` at the bow over Ochi's `0.093 sqrt(g L)` at the time |
| a tether that changes at 500 s | **mooring** -- a persistent step in the tension with the accelerometers inside their noise |
| an accelerometer spiking 2 % of the time | **sensor: crane** -- it disagrees with the model while the centre agrees with it |
| an accelerometer gone flat | **sensor: centre** -- the channel has no variance at all |

Five records, five correct diagnoses. The rule that keeps a quiet sea quiet is a *share* rather than a maximum: one
sample beyond four sigma happens once in thirty thousand by definition, so a window of a few hundred samples would fire
on noise alone unless a fraction of its samples has to be outside the band.

The relative velocity at the bow is the one number in this chain that is an estimate rather than a solve. The hull and
the surface are taken **in phase** -- the response ratios the section gives are magnitudes, and the phase a
three-dimensional solve would give them is not carried -- which reproduces the right trend (zero where the hull rides
the water, the surface's own where it cannot) and cannot give a slam count.

## The window, and the confidence that falls across it

The criterion is the operator's: `Hs <= 1.5 m`, significant vertical acceleration at the crane pedestal
`<= 0.2 g`, and the relative velocity at the bow under Ochi's threshold. The defaults are illustrative and say so; they
are the kind of limit a 40 m support vessel works to, not a classification rule.

What the package adds is a confidence that is *measured*: on the 254 analyses of the record's second half, the verdict
the forecast gives is compared with the verdict the buoy's own numbers would have given, and the fraction that matched
is the confidence of the window at that lead.

| lead | 1 h | 3 h | 6 h | 12 h | 18 h | 24 h |
| --- | --- | --- | --- | --- | --- | --- |
| verdicts that matched | **96 %** | 92 % | 89 % | 83 % | 79 % | **76 %** |

On the day the example reports -- 10 November, an analysis of 1.4 m -- the answer is **"work is allowed for the whole
24 h forecast, confidence 76 %"**, and the buoy then measured 1.7 m for the first hour and 1.6 m for two more: the
window was open on paper and shut in fact for four of the twenty-four hours, which is the 83 % agreement of that day
against the 96 % the forecast manages at a one-hour lead over the whole record. That is what a confidence is, and the
example prints both numbers rather than the flattering one.

The criterion that binds first is worth knowing. For this vessel, in a 1.5 m / 8 s sea, the significant acceleration at
the pedestal is **0.11 g** and the relative velocity at the bow **1.41 m/s**, both inside their limits: what closes this
window is the operator's 1.5 m and not the vessel, which could work to about 1.9 m before the crane or the bow said no.
That is a sentence an operator can argue with, which is the point of measuring it.

## What this is not

* **It is not a three-dimensional seakeeping computation.** The coefficients are a section's, which is where strip
  theory's come from; the response in a real beam sea at a real heading needs the hull's three-dimensional radiation and
  diffraction, and the coupling between surge, sway, yaw and the rest that a section does not have. Sway is reported and
  diverges at long periods, because a free section has nothing to restore it -- that is a property of the problem, not a
  defect of the solve, and it is why the mooring here takes its offset as an input rather than computing it.
* **It is not a mooring analysis.** Quasi-static, one plane, no line dynamics, no seabed friction, and the slowly
  varying drift that dominates a real mooring load is not computed at all: the environmental load is an input.
* **The slamming number is an estimate.** In-phase hull and surface, magnitudes only; it cannot count slams.
* **The forecast is a 24-hour one from a single point.** No wind forecast, no neighbouring spectra, no assimilation
  beyond anchoring the analysis to the buoy's own height. That is why the height skill is two per cent and not twenty.
* **The validation is one station, 45 days, one season.** October and November off Cape Canaveral, a swell-dominated
  site. Nothing here says how a fetch-limited or a monsoon sea would score.

## Running it

```
python examples/maritime_operability/build_dataset.py                    # fetch the buoy (only step that needs network)
python examples/maritime_operability/maritime_operability.py             # everything, about 55 s
python examples/maritime_operability/maritime_operability.py --quick     # ten leads, coarser calibration grid
```

The results go to `examples/maritime_operability/results/maritime_operability.json` and the five figures to
`figures/`: the measured spectra against the fitted shape, the skill against persistence, the transfer functions with
the resonance marked, the mooring load against the offset with the two equilibria, and the window with the sea that
followed it.
