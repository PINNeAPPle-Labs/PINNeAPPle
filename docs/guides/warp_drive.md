# The Alcubierre warp bubble: the curvature, the energy conditions, and the bill

`pinneapple_physics.relativity` takes the Alcubierre metric and differentiates it. It does not quote the paper's
energy density; it computes the Einstein tensor of the metric as written and checks it against the closed form. The
difference, on twelve thousand points in the wall, is `3e-11` relative -- and sympy, handed the same metric with an
abstract `f(r_s)`, simplifies the same difference to exactly zero.

```
ds^2 = -dt^2 + (dx - v_s f(r_s) dt)^2 + dy^2 + dz^2
```

`f` is 1 inside the bubble and 0 outside, `r_s = |x - x_s(t)|`, and the bubble moves along `x` at `v_s`, which may be
larger than `c`, because nothing inside the bubble moves.

```python
from pinneapple_physics.relativity import warp

bubble = warp.WarpMetric(velocity=1.0, radius=10.0, wall=2.0)
curvature = warp.einstein_tensor(bubble, points)      # points are (t, x, y, z)
curvature.energy_density                              # T^00, negative in the wall
warp.worst_null_energy(bubble, points)                # the null energy condition, searched over directions
warp.integrated_energy(bubble)                        # the bill, in kilograms and solar masses
```

## Three pictures, and what they mean

`examples/warp_drive` writes `figures/warp_maps.png` with the three maps on the `z = 0` plane:

* **the wall** -- `f(r)`, 1 inside and 0 outside;
* **the expansion** -- `theta = v_s (x - x_s)/r_s f'(r_s)`, the York time. It is **negative ahead of the bubble and
  positive behind it**: the space in front contracts and the space behind expands, which is the picture, and here it
  is a number, `-0.250` at `x = +R` and `+0.250` at `x = -R`;
* **the energy density** -- negative in the two lobes above and below the direction of travel, and **exactly zero on
  the axis** `y = z = 0`, because the closed form carries a factor `(y^2 + z^2)`. The axis is not exotic; the wall
  beside it is.

`figures/warp_turntable.gif` is the same energy density as a rotating 3-D surface, so the shell can be seen from
every side.

## The energy conditions, measured

The null energy condition says `T_{mu nu} k^mu k^nu >= 0` for **every** null vector `k`. Two things had to be right
before the number meant anything:

* `k` has to be null. The obvious choice `k = (1, n)` is not null in this metric; the module builds `k = n + s` from
  the Eulerian normal `n^mu = (1, v_s f, 0, 0)` and a unit spatial direction, and checks `g(k, k) = 0` to roundoff
  before returning a value. With the naive `k`, `g(k, k)` is of order 1 and the "energy" is meaningless;
* the minimum has to be searched, not guessed: `worst_null_energy` sweeps a sphere of directions per point.

At every one of the 12 866 wall points the minimum is negative; the global minimum is `-5.95e-3`. The condition is
violated, and the library points at where.

The **weak** condition fails twice over, which is worth separating. It asks for `T_{mu nu} u^mu u^nu >= 0` for every
*unit timelike* `u`, and `u = (n + speed s)/sqrt(1 - speed^2)` covers them. At `speed = 0` this is exactly `T^{00}`
of the closed form and it is already `-6.2e-4`. A boosted observer, moving at `0.9 c` relative to the bubble in the
`-z` direction, sees `-2.77e-2` -- forty-five times worse. The negative energy is not an artefact of a badly chosen
frame; it gets worse in the frames a falling observer would occupy.

## The bill

A 10 m bubble at the speed of light needs

| quantity | value |
| --- | --- |
| total `T^00` | `-1.407 m` (geometric units, `G = c = 1`) |
| mass | `-1.8947e27 kg` |
| solar masses | `-9.53e-4` |
| Jupiter masses | `-1.00` |

The peak density in the wall is `-6.2e-4 m^-2`, or `-8.4e23 kg/m^3`. The dark side of the number is that it does not
go away with a cleverer profile:

| knob | scaling | measured |
| --- | --- | --- |
| velocity | `E ~ v_s^2` | 0.1 c: `-1.89e25 kg`; 1 c: `-1.89e27 kg`; 5 c: `-4.74e28 kg` |
| radius | `E ~ R^2` | 5 m: `-5.05e26 kg`; 20 m: `-7.45e27 kg`; 50 m: `-4.59e28 kg` |
| wall thickness | `E ~ 1/wall` | 1 m: `-3.73e27 kg`; 4 m: `-1.01e27 kg` |

A thin wall is *worse*: `f'` grows as `1/wall` and the integral scales with `f'^2` times the volume of the shell. The
only way to zero is `v_s = 0` or `R = 0`. A Gaussian profile does better than the tanh -- `-5.27e26 kg`, 0.28 times as
much -- and still not zero.

## Three things the measurement corrected

1. **The interior of the bubble is not flat.** The usual picture says the metric is flat inside and curved only in the
   wall. With a smooth tanh wall `f` reaches 1 only at the centre, so the interior carries residual curvature: 0.3 %
   of the wall's peak at `r = 0.1 R`, 1.6 % at `0.3 R`, 10 % at `0.5 R`, and it is gone only where `f` is exactly 1.
   A compactly supported shape function would be genuinely flat inside; the smooth one is not.
2. **The top-hat is out of scope, and the module says so.** Alcubierre's original profile has `df/dr = 0` everywhere
   except on the shell, where it is a delta function. Its analytic derivative is 0, so the closed form says the total
   is 0, while the numerical curvature depends on whether a grid point lands on `r = R`: across four grids the total
   comes out `-7.1e31`, `-3.3e32`, `0` and `-4.1e31` kg. A discontinuous profile is a surface density this
   representation cannot resolve. Only the smooth shapes are meaningful here.
3. **The naive null vector is a trap.** `k = (1, n)` looks like a null vector and is not one in a metric with a
   shift. Every number computed with it is a number about a spacelike or timelike direction wearing the name. The
   module refuses to return a value unless `g(k, k)` is zero.

## What this is not

It is not a solution of the quantum inequalities (Pfenning-Ford), which bound how negative the energy can be for how
long, and it is not a proposal for how to build one. It is the metric, its curvature, the sign of the answer, and the
number -- and the number is the point.

## Running it

```
python examples/warp_drive/warp_drive.py            # maps, the turntable and the JSON, about 80 s
python examples/warp_drive/warp_drive.py --quick    # no GIF
```

The results go to `examples/warp_drive/results/warp_drive.json`, the figures to `examples/warp_drive/figures/`.
