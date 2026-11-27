# Exoplanet transits: the model, the star around it, and what vetting actually buys (#386)

A transit is a clean measurement inside a dirty signal. The dip is set by geometry — the Mandel-Agol model with limb
darkening — and around it sits everything else: spots, granulation, momentum dumps, scattered light. Telling a planet
from a look-alike is the library's "regime change or sensor noise" question in its astronomical form.

**There is no network here.** The TESS and Kepler light curves and the NASA Exoplanet Archive labels are not available,
so `load_light_curve` names the product it would need and fails cleanly (measured: it raises naming `astropy`, which is
not installed either — `pip install "pinneapple[astro]"`). Every rate on this page comes from a **labelled set
generated here with a stated model**, which is what makes the answer measurable at all: the truth is returned with the
data.

## The labelled set

`transits.synthetic_light_curve(kind, seed)` returns the flux, the error, the centroid and its error, and a `truth`
block with the injected period, depth, `k`, impact parameter and centroid shift. Six kinds, three cases each, 18 in
total:

| kind | label | what it is |
|---|---|---|
| `planet` | 1 | a transit at a known period, depth and impact parameter |
| `eb_alternating` | 0 | an eclipsing binary whose odd and even eclipses differ in depth |
| `eb_secondary` | 0 | an eclipsing binary with a second dip at phase 0.5 |
| `eb_grazing` | 0 | a grazing eclipse with a V-shaped bottom |
| `blend` | 0 | a background eclipsing binary seen as a centroid shift |
| `systematic_sinusoid` | 0 | an instrumental sinusoid that looks periodic |

Stellar variability (621,7 ppm in the planet case measured) and the instrument systematics — momentum dumps of
372,6 ppm on a 0,41 d recovery, scattered light of 289,8 ppm on a 13,7 d period — are injected by the same generator.

## The three pipelines, measured

BLS is the real baseline: it searches the period grid and cuts on the signal-to-noise. `bls+fit` adds the transit model.
`bls+fit+vetting` adds the checks the issue names.

| pipeline | recovery | false positive | of which planets | of which look-alikes |
|---|---|---|---|---|
| `bls` | **1,000** (3/3) | **0,800** (12/15) | 3 | 15 |
| `bls+fit` | **1,000** (3/3) | **0,800** (12/15) | 3 | 15 |
| `bls+fit+vetting` | **1,000** (3/3) | **0,000** (0/15) | 3 | 15 |

AUC of the vetting score: **0,889**.

The honest reading:

- **BLS finds every planet and calls four out of five look-alikes planets too.** That is the baseline the issue asks to
  be compared against, and it is not a straw man: BLS is genuinely good at finding periodic dips.
- **Fitting the transit model does not help.** The false-positive rate is exactly the same 12 of 15 — a fit tells you
  the dip has a shape, not that the shape is a planet's.
- **The vetting is what removes them: 12 of 15 to 0 of 15, with the recovery intact.** The checks that fire, by name:
  `odd_even` (the two eclipses differ in depth), `secondary` (a second dip at phase 0.5), `v_shape` (no flat bottom, the
  ingress is a large fraction of the eclipse), `shape_inconsistent` (the profile is not a transit's) and `centroid_shift`
  (the flux centroid moves during the dip — a background eclipsing binary).
- **The three sinusoids were not detected at all**, so they are false negatives of the *search*, not false positives of
  the vetting: their dips sit below the signal-to-noise the fit needs. That is a different failure and it is reported as
  such (`not detected: the dip is below the signal-to-noise the fit needs`), not folded into the false-positive count.

Every verdict carries the reason in words next to the flags, so a candidate is never a bare score.

## The physics check

`analyze` returns a `MeasurementCheck`, the library-wide verdict object: the fitted radius in Earth radii, the depth
against its own uncertainty, the period agreement with BLS, and the flags. A signal is called physical when no check
fired, and each check that fired is named in the verdict.

## The animation

`phase_fold_gif` folds the light curve on the recovered period and draws the fitted transit over it:
[`docs/assets/transit_phase_fold.gif`](../assets/transit_phase_fold.gif), 36 frames, generated from the synthetic
planet at seed 7 (466 kB).

## What is not here

- **Observed light curves.** MAST and the Exoplanet Archive are unreachable; `LIGHT_CURVE_RELEASES` and
  `MAST_PRODUCT_URL` record what a loader would fetch.
- **A real recovery rate.** The 1,000 / 0,000 above are on 18 synthetic cases with Wilson intervals that say what that
  size supports (recovery CI95 [0,438; 1,000] — three planets cannot pin a rate). The benchmark reports the intervals
  precisely so the numbers are not read as more than they are.
