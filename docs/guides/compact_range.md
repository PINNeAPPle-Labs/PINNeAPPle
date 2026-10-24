# Compact range quiet zone: feed gain × reflector size

A compact range turns the spherical wave of a feed into a plane wave with an offset paraboloid. The antenna under
test sits in the **quiet zone** (QZ), where the field must stay within a few tenths of a dB and a few degrees of a
plane wave. This guide sweeps the two inputs a range designer controls first, the feed gain (the illumination taper)
and the reflector size, plus the rolled-edge size, computes the QZ metrics by physical optics, validates the PO code
against closed forms, fits a surrogate with held-out error and draws the trade-off map.

Everything lives in `pinneapple_simulation.electromagnetics.reflector`; the example
`examples/use_cases/antennas/compact_range_qz.py` only calls it.

```bash
python examples/use_cases/antennas/compact_range_qz.py --quick     # 3 x 3 designs, coarser mesh (about 3 min)
python examples/use_cases/antennas/compact_range_qz.py             # 7 gains x 6 sizes x 2 rolled edges
```

## Model

* **Reflector** (`geometry.OffsetReflector`): the paraboloid `z = (x² + y²)/(4F)`, a section whose projection is a
  superellipse (half sizes, exponent `n`; large `n` = a square with rounded corners) centred `centre_y` above the
  axis. Past the junction an optional **rolled edge** leaves the paraboloid tangentially and with its curvature,
  blends to a constant curvature `-1/roll` turning away from the QZ and stops at `end_angle`; `roll_aspect`
  stretches it in depth (an elliptic-like roll). This is a blended rolled edge in the spirit of Gupta and Burnside;
  the exact profile of a particular range is a parameter, not built in.
* **Feed** (`po.cosq_feed`): an ideal `cos^q` pattern at the focus, aimed at the section centre, Ludwig-3
  polarised, normalised to 1 W, with `D = 2(2q + 1)`; the feed gain sets `q` (`po.q_from_gain`).
* **Physical optics** (`po.po_currents`, `po.radiate`): `J = 2 n × H_inc` on the lit triangles, and its field at
  the QZ points with the complete dyadic Green's function (no far-field approximation): the QZ is in the reflector's
  near field. Centroid quadrature on λ/8 triangles. PO leaves out the fringe edge currents (PTD) and multiple
  reflections; the edge diffraction it contains is that of the truncated PO current. The feed's direct radiation into
  the QZ is zero for this feed model (the QZ lies behind it).
* **Quiet-zone metrics** (`qz.qz_metrics`) over a disk of the QZ diameter: peak-to-peak amplitude and phase, the
  taper (a quadratic fitted to the amplitude in dB), the ripple (peak-to-peak residual about it), the phase ripple,
  the cross-polar level, and the **usable diameter**: the largest centred disk within ±0.5 dB and ±5° (the field is
  computed over 1.6 × the QZ diameter to find it).

## Validation

1. **PO integrator against a closed form.** The PO current of a normally incident plane wave on a flat disk has an
   exact on-axis field, `E_x = −E0 [e^{−jkz} − ½(1 + z²/R_a² + j a²/(k R_a³)) e^{−jkR_a}]` (`analytic.disk_on_axis`,
   derived in its docstring). Measured maximum relative error over z = 2–40 λ for a 3 λ disk: VALID_DISK.
2. **PO against geometrical optics.** In a compact range the smooth part of the QZ amplitude is the GO aperture taper
   of the feed (`sqrt(D(ψ))/ρ`, `analytic.go_aperture_taper_db`); the PO field adds the edge-diffraction ripple. The
   difference between the quadratic fit of the PO amplitude and the GO taper shrinks as the rolled edge grows:
   VALID_GO.
3. **Mesh convergence** of the QZ metrics: VALID_MESH.

No published compact-range measurement was compared here.

## Results (generic example)

RESULTS_GENERIC

## API

```python
from pinneapple_simulation.electromagnetics.reflector import CompactRange, OffsetReflector, evaluate, scaled
from pinneapple_simulation.electromagnetics.reflector.study import qz_figure, qz_study, validate

cr = CompactRange(freq_hz=3e9, reflector=OffsetReflector(focal=1.5, half_x=0.6, half_y=0.6, centre_y=0.9,
                                                         n=6, roll=0.15), feed_gain_dbi=12, qz_z=3.0, qz_diameter=0.6)
evaluate(cr)                                     # metrics, triangles, seconds
validate(cr)                                     # disk closed form, PO vs GO, mesh convergence
qz_study(cr, gains=[8, 12, 16, 20], sides=[0.9, 1.2, 1.5], rolls=[0.0, 0.15], out="cr/")   # sweep, GP, maps
```

`study.fit_surrogates` fits a Gaussian process per metric on (feed gain, side) for each rolled edge
(`antenna.surrogates.GPModel` with `features=columns("feed_gain_dBi", "side")`) and reports two held-out RMSEs:
a random quarter of the designs (interpolation inside the grid) and every design of one size at a time
(interpolation across sizes, the harder case). `study.tradeoff_map` draws the predicted ripple and the predicted
usable QZ diameter over (feed gain, size), the simulated designs, the required diameter and the feasible region (or
says that it is empty).
