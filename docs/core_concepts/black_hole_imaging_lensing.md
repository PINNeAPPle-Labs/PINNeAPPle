# Black-hole imaging and strong lensing: what the data decide, what the prior decides

Two inverse problems in which the forward model is exact and the answer is not. Interferometric data fix a few
arcs of the sky's Fourier transform and leave the image to a regulariser and a prior; a lens model turns an arc into
a mass profile that is not unique either. This page is the measured half of issue
[#390](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/390): the numbers below were produced by
`pinneapple_physics/astro/eht_imaging.py` and `pinneapple_physics/astro/lens_model.py` on this machine, and the page
says for each one whether it was **measured**, **predicted** from published values, or **published**.

## What is synthetic here, and what is not

Neither the EHT visibility releases nor the HST/Euclid lens catalogues are bundled with PINNeAPPle, and this work
does not download them. So:

| Thing | Where it comes from |
|---|---|
| M87* visibilities, Sgr A* visibilities | **simulated**: the transform of a stated image model, on a stated array, with a stated noise model |
| Lens images | **simulated**: renderings of the declared SIE + shear model |
| The 42.0 ± 3.0 µas M87* ring, the 51.8 ± 2.3 µas Sgr A* ring | published (EHT 2019, ApJL 875, L1; EHT 2022, ApJL 930, L12) — **inputs**, not measurements made here |
| Mass 6.5 × 10⁹ M<sub>☉</sub>, distance 16.8 Mpc, and the 39.69 µas shadow they predict | published mass and distance (EHT 2019 L5, L6), shadow from the closed form 2√27 GM/c²D |
| Station coordinates of the 2017 array | published (EHT 2019, ApJL 875, L4), rounded to ~100 m |
| Per-station SEFDs, 2 GHz bandwidth, 10 s coherence | a **model** of the array's sensitivity, not the campaign's measured Tsys |

The consequence, stated once and repeated where it matters: the ring diameter recovered below is a **closure test**.
It says the forward model, the array model and the fitter are consistent with each other and land on the published
value. It says nothing new about M87*.

```python
from pinneapple_physics.astro.eht_imaging import (uv_coverage, observe, fit_crescent,
                                                  reproduce_ring_diameter, prior_dependence_study)

uv  = uv_coverage(12.3911)                       # M87*, 230 GHz, eight stations, +-3.5 h
obs = observe(uv, seed=0)                        # synthetic visibilities from the published ring
fit = fit_crescent(obs.visibilities, uv, obs.sigma, prior={})
print(fit)      # diameter 41.95 +- 0.16 uas, width 7.99, asymmetry 0.40, pa 89.8, flux 0.593 Jy, chi2/dof 0.39
```

## Part 1 — the array, and what a night of it measures

Earth rotation turns each station pair into an arc of the (u, v) plane. With both stations required above 10°
elevation — the horizon is not negotiable — the 2017 array over ±3.5 h at 230 GHz gives:

| quantity | value (measured) |
|---|---|
| visibilities | 327 |
| scans | 24 (hour angle −3.5 h … +3.5 h, 18 min apart) |
| distinct baseline pairs | 21 of the 28 possible |
| longest projected baseline | 7.98 Gλ |
| thermal σ per visibility (radiometer equation) | 1.2 mJy … 15.0 mJy, median 5.5 mJy |
| median amplitude S/N | 5.8 |

A ring of diameter *d* has visibility amplitudes J₀(2πbR): the first null sits at
`visibility_null_baseline_gl(d) = 2.4048 / (2πR)`, which for 42 µas is **3.76 Gλ** — inside the sampled range, which
is why the diameter is measurable at all. The null scales as 1/*d*.

### The forward model, and the check that it is right

The crescent is a ring of diameter *d* and radial Gaussian width *w*, with brightness modulated by
`1 + a cos(φ − pa)` — the family the EHT's own geometric modelling uses. Its visibility separates in polar
coordinates:

```
V(b, φ_b) = F [ S0(b) − i a cos(φ_b − pa) S1(b) ],   Sk(b) = ∫ r p(r) Jk(2πbr) dr / ∫ r p(r) dr
```

The asymmetry term is purely imaginary and carries `cos` of the angle between the baseline and the bright side, so a
pair perpendicular to the bright side sees a real visibility. The integral is a 96-node Gauss–Legendre quadrature in
`u = (r − R)/w`.

Checked against the discrete transform of the rendered image (a 192² render, a 2-D sum, nothing shared with the
quadrature): **max |V_direct − V_closed| / max |V| = 2.1 × 10⁻⁷** over the 327 sampled baselines.

## Part 1 — reproducing the M87* ring diameter and its uncertainty

Twenty noise realisations of the synthetic data set, each fitted independently (no prior):

| quantity | value (µas) | provenance |
|---|---|---|
| ring diameter | **42.029 ± 0.017** | recovered here (standard error of the mean of 20 realisations) |
| per-realisation scatter | 0.074 | recovered here |
| ring diameter, 95 % distribution-free | 42.029 ± 0.191 | split-conformal quantile of the 20 deviations |
| published ring diameter | 42.0 ± 3.0 | EHT 2019 L1 |
| shadow diameter 2√27 GM/c²D | 39.69 | closed form at the published mass and distance |
| (recovered − published)/σ_published | 0.010 | |

The published ring is 0.77σ *larger* than the shadow: the two are different quantities, and the closed-form shadow
lands inside the published error bar, which is the one comparison on this page that involves no code of ours.

**The uncertainty does not reproduce.** The synthetic data constrain the diameter to 0.07 µas; the published error is
3 µas. That is not a tuning problem: EHT's ±3 µas is dominated by the spread between analysis methods, not by thermal
noise, and a data set whose image is exactly in the fitted model family cannot contain that spread. The module reports
both numbers instead of inflating the noise until they agree — with calibration errors included (5 % amplitude,
0.1 rad phase per station per scan) the median per-visibility uncertainty rises from 5.5 mJy thermal to 13.9 mJy,
and the diameter error is still only 0.07 µas.

## Part 1 — which features survive with the physics prior switched off

The prior is a *prediction*, not a fit: the ring sits at the shadow diameter of the published mass and distance,
39.69 ± 4 µas, and its radial width at 8 ± 4 µas (the range the published crescent fits report). Both enter the
least-squares fit as Gaussian rows next to the data rows, so switching the prior on and off changes one argument and
nothing else, and the two fits see identical visibilities.

The shift is paired: the difference of the two answers on the same noise realisation, whose standard error is free of
the noise common to both. Two yardsticks are reported for each feature:

* **moved** = |shift| / σ(prior off) — how much of the feature's own uncertainty the prior accounts for. An analysis
  quoting the prior-on value would be wrong by this fraction if the prior were wrong.
* **shrink** = σ(prior off) / σ(prior on) — how much precision the prior bought.

Verdict: *prior-driven* when `moved ≥ 0.5`, *prior-influenced* when `moved < 0.5` but `shrink ≥ 1.5`, *data-determined*
otherwise. The thresholds are a stated convention; both numbers are printed so another one can be applied.

### At the sensitivity of the campaign

| feature | prior off | prior on | paired shift | moved | shrink | verdict |
|---|---|---|---|---|---|---|
| diameter (µas) | 42.029 ± 0.074 | 42.025 ± 0.074 | −0.004 ± 0.000 | 0.05 | 1.00 | data-determined |
| width (µas) | 7.984 ± 0.056 | 7.987 ± 0.055 | +0.002 ± 0.000 | 0.04 | 1.00 | data-determined |
| asymmetry *a* | 0.402 ± 0.007 | 0.402 ± 0.007 | −0.000 ± 0.000 | 0.01 | 1.00 | data-determined |
| position angle (°) | 89.96 ± 0.60 | 89.96 ± 0.60 | +0.001 ± 0.000 | 0.00 | 1.00 | data-determined |
| flux (Jy) | 0.591 ± 0.006 | 0.591 ± 0.006 | +0.000 ± 0.000 | 0.02 | 1.00 | data-determined |

All five features survive: the prior moves none of them by more than a twentieth of its uncertainty and buys no
precision. (The paired standard errors are tiny — 20 identical noise realisations make even a 0.004 µas move
statistically visible — which is exactly why the verdict uses *moved* and not the significance of the move.)

### Degrading the data is what makes the prior matter

`sigma_scale` multiplies the thermal noise; since a visibility's thermal noise falls as 1/√t, 100× the noise is a
campaign with a hundredth of the integration time.

| thermal noise | median S/N | diameter prior off | diameter prior on | shift | moved | shrink | verdict |
|---|---|---|---|---|---|---|---|
| ×1 | 5.80 | 42.029 ± 0.074 | 42.025 ± 0.074 | −0.004 | 0.05 | 1.00 | data-determined |
| ×10 | 1.42 | 42.028 ± 0.417 | 41.985 ± 0.411 | −0.043 | 0.10 | 1.02 | data-determined |
| ×100 | 0.15 | 41.772 ± 3.683 | 40.630 ± 1.480 | −1.142 | 0.31 | 2.49 | prior-influenced |

| thermal noise | width prior off | width prior on | shift | moved | shrink | verdict |
|---|---|---|---|---|---|---|
| ×1 | 7.984 ± 0.056 | 7.987 ± 0.055 | +0.002 | 0.04 | 1.00 | data-determined |
| ×10 | 7.926 ± 0.265 | 7.943 ± 0.262 | +0.017 | 0.06 | 1.01 | data-determined |
| ×100 | 7.027 ± 2.758 | 7.678 ± 1.436 | +0.651 | 0.24 | 1.92 | prior-influenced |

The asymmetry (0.416 ± 0.333 → 0.402 ± 0.326 at ×100) is never moved by this prior at all, the flux only barely
(0.578 ± 0.064 → 0.582 ± 0.059), and the position angle is unconstrained at that noise level (±66° off, ±85° on) —
neither number means anything. **The prior is what the data are not.** Where the data run out — a hundredth
of the integration time — the diameter error is 2.5× smaller with the prior and the central value moves by 1.14 µas;
above that the data own the answer outright.

### The one feature the prior cannot rescue

Fitting visibility **amplitudes only** — the situation the EHT's geometric modelling was in, since the atmospheric
phase at 1.3 mm is not measured absolutely:

| feature | prior off | prior on | truth | verdict |
|---|---|---|---|---|
| diameter (µas) | 42.010 ± 0.105 | 42.002 ± 0.105 | 42.0 | data-determined |
| width (µas) | 7.969 ± 0.057 | 7.973 ± 0.057 | 8.0 | data-determined |
| asymmetry *a* | 0.404 ± 0.014 | 0.403 ± 0.014 | 0.40 | data-determined |
| position angle (°) | **−90.10 ± 2.35** | **−90.01 ± 2.40** | **+90** | data-determined |
| flux (Jy) | 0.596 ± 0.007 | 0.596 ± 0.007 | 0.6 | data-determined |

An asymmetric ring's visibility amplitude is invariant to the *sign* of the asymmetry, so the amplitudes do not say
which side is bright. The fit lands on the reflected crescent — 180° from the truth — and reports a formal error of
2.4°. The prior used here says nothing about the direction, so it cannot pull it back either; that decision needs an
external statement (which side the jet points to). This is the case where the honest answer to "does the feature
survive without the prior" is *no, and the error bar does not show it*.

## Part 1 — the ray-traced link: the ring is not the shadow

`raytraced_ring` reuses the null-geodesic tracer of `pinneapple_physics.astro.lensing` (`trace_thin_disc` +
`disc_image`: RK4 on the Binet equation, Keplerian gas, relativistic beaming), and rescales the render to M87*'s
published mass and distance so that the shadow has its published angular size.

| quantity | value | provenance |
|---|---|---|
| shadow radius in the render, from the captured-ray mask | 19.3 µas | measured here (analytic value 19.84; one 1.09 µas pixel) |
| shadow diameter, closed form | 39.69 µas | predicted |
| ring diameter of the ray-traced disc image | 69.37 ± 0.06 µas | measured here, crescent fit to the image's visibilities |
| flux-weighted mean diameter of the image | 72.5 µas | measured here, no model |
| asymmetry of the ray-traced image | 0.20 at pa 175° | measured here (beaming plus lensing) |

The disc annulus is 6–8 GM/c², a stated choice; the ring that comes out sits at **1.75× the shadow diameter**, because
the emission ring is not the photon ring. So using the shadow as a prior on an image feature is a modelling
assumption, not a measurement — which is precisely why the two are reported separately here.

## Part 2 — strong lensing: the lens model and the lenses

`lens_model` fits a singular isothermal ellipsoid plus external shear (Keeton 2001, astro-ph/0102341, eqs. 2–3; the
pseudo-elliptical isothermal of Kormann et al. 1994), with b as the Einstein radius of the equivalent isothermal
sphere (Narayan & Bartelmann 1996, astro-ph/9606001; Schneider, Ehlers & Falco 1992). Convergence is
κ = b / (2√(q²x′² + y′²)) > 0, and the external shear adds no convergence at all.

The three lenses in `LENSES` are **generated**: their Einstein radii are the published isothermal formula evaluated at
a stated velocity dispersion and pair of redshifts, and the images are renderings of the same model. For each, the
declared parameters and what the fit recovers from its own image:

| lens | σ (km/s) | z_l / z_s | θ_E formula | θ_E recovered | q | pa (°) | γ₁, γ₂ | ⟨θ⟩ critical curve | κ_min | images in/out | rms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| galaxy-scale | 250 | 0.30 / 1.20 | 0.8344″ | 0.8344″ | 0.78 | 35.0 | 0.05, −0.02 | 1.0112″ | 0.178 | 4 / 2 | 9 × 10⁻¹⁶ |
| quad-with-shear | 290 | 0.45 / 2.00 | 0.7567″ | 0.7567″ | 0.62 | −20.0 | 0.09, 0.06 | 1.1053″ | 0.187 | 4 / 2 | 3 × 10⁻¹⁵ |
| group-scale | 550 | 0.20 / 1.50 | 5.3225″ | 5.3225″ | 0.85 | 70.0 | 0.03, 0.04 | 6.0345″ | 0.181 | 4 / 2 | 5 × 10⁻¹⁵ |

Every parameter comes back to 10⁻⁵ and the residual is round-off, because the image is a rendering of the model that
is being fitted. That is what "generated rather than observed" means, and it is the reason the table proves the
fitter and the renderer agree — not that any real lens has these parameters.

`θ_E` is the Einstein radius of the equivalent *sphere*; the mean radius of the tangential critical curve is larger
(1.01″, 1.11″, 6.03″) because the ellipse and the shear spread it. Both are reported; the published SIE fits quote the
first.

### The physics checks, and one place where the check's convention fails

* **positive convergence** — κ sampled on a 41² grid (the singular centre excluded): minimum 0.178, 0.187, 0.181 over
  the three lenses. External shear leaves κ identical, which
  `test_shear_adds_no_convergence_and_the_map_stays_physical` asserts directly.
* **the images solve the lens equation** — every image pushed back through the lens map lands on the source to better
  than 10⁻⁶ arcsec.
* **multiplicity against the caustics** — along six directions a source at 0.02–0.05 θ_E (inside the tangential
  caustic) is seen in four images and one at 0.85–0.95 θ_E (outside) in two, with the distance to the nearest caustic
  reported for every probe. For `quad-with-shear` the count drops 4 → 2 as the source crosses the caustic at 0.78 θ_E
  and reaches 1 image at 1.6 θ_E, where the shear leaves nothing else.

The check compares the measured count with the *ellipsoidal* expectation (4 in, 2 out). That convention fails for a
nearly spherical lens, and the report says so rather than passing it: at q = 0.95 the tangential caustic reaches only
0.051 θ_E, so the probe at 0.02 θ_E is inside it (four images) and the one at 0.05 θ_E is outside (two) —
`inner_counts = [2, 4]`, `multiplicity_ok = False`. Two images is the right answer there; the mismatch is the
convention's, and it is reported.

## Reproduced, declared, synthetic — the summary

| claim | status |
|---|---|
| the 2017 array's (u, v) coverage at 230 GHz, out to 7.98 Gλ | computed from published station coordinates (rounded) |
| the crescent visibility closed form | verified against a direct transform (2 × 10⁻⁷) |
| the ring diameter of the published M87* ring, recovered from synthetic data | 42.029 ± 0.017 µas — **closure test**, not a measurement of M87* |
| the published ±3 µas | **not** reproduced, and cannot be: it is systematic-dominated |
| the shadow diameter from the published mass and distance | 39.69 µas, 0.77σ below the published ring |
| the prior's effect on each feature | measured: ≤ 0.05 of the feature's uncertainty at campaign sensitivity |
| the ray-traced image's ring vs the shadow | measured: 69.4 vs 39.7 µas, i.e. a factor 1.7 |
| lens parameters of three lenses | recovered from **generated** images to 10⁻⁵ |
| real EHT visibilities, real HST/Euclid lens images | **not used**; `data_requirements()` and the two loaders name the release, file and size and refuse |

## Limitations

* **Synthetic data only.** Every fitted visibility is the transform of an image we rendered. Where real data would
  bring flagging, RFI, polarisation leakage, bandpass and a time-variable source, this has Gaussian noise and one
  gain and phase per station per scan.
* **The noise level is matched to no campaign.** The SEFDs are order-of-magnitude values; the 15.5 mJy median
  per-visibility uncertainty is the model's, not the release's.
* **A 5-parameter crescent is not an image.** The features reported are the ones this family has. A non-parametric
  reconstruction would add the questions the EHT's imaging papers are about (regularisation weight, field of view,
  multi-scale) and none of them are measured here.
* **No lensing substructure.** The issue asks for a pixelated or network correction on top of the SIE; what is here
  is the parametric part and its checks.
* **The multiplicity convention** is the ellipsoidal one (see above).

## Reproducing these numbers

```bash
PYTHONPATH=. python -m pytest tests/test_astro_lensing_imaging.py -q     # 24 tests, ~30 s
PYTHONPATH=. python -c "from pinneapple_physics.astro.eht_imaging import report; from pprint import pprint; pprint(report('M87*', n_realizations=20))"
```

`report()` returns the reproduction table, the three-rung prior ladder, the amplitude-only table and the ray-traced
comparison in one dictionary; every table on this page is a transcription of that output with the seed fixed
(`seed=0`).
