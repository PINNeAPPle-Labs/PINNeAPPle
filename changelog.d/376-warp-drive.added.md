- `pinneapple_physics.relativity` differentiates the Alcubierre metric instead of quoting it: `einstein_tensor` is
  the Einstein tensor of the metric as written, computed by fourth-order central differences in the frame that moves
  with the bubble, and checked against the closed form
  `-(v_s^2/32pi)(y^2+z^2)/r_s^2 (df/dr_s)^2` to `3e-11` relative on twelve thousand points in the wall -- and against
  sympy, which is handed the same metric with an abstract `f(r_s)` and simplifies the difference to exactly zero.
- the expansion (`theta = v_s (x-x_s)/r_s f'(r_s)`, the York time) contracts ahead of the bubble and expands behind
  it, `-0.250` at `x = +R` and `+0.250` at `x = -R` for `R = 10 m` and `v_s = c`.
- the energy density is negative in the wall and **exactly zero on the axis**, which is the factor `(y^2+z^2)` of the
  closed form rather than a coincidence -- and the check that catches a wrong index permutation in the Christoffel
  assembly, which is how this module was wrong before it was right.
- the null energy condition, measured rather than asserted: `worst_null_energy` sweeps a sphere of null directions at
  every point, and the null vectors are built as `n + s` from the Eulerian normal and checked to satisfy `g(k,k) = 0`
  to roundoff. `k = (1, n)`, the obvious guess, is not null in a metric with a shift and silently measures nothing.
  Every one of the 12 866 sampled wall points has a direction with `T_{mu nu} k^mu k^nu < 0`.
- both energy conditions, not just the null one: `weak_energy` contracts the Einstein tensor with a unit timelike
  vector `u = (n + speed s)/sqrt(1-speed^2)` and `worst_weak_energy` searches over directions and speeds. The weak
  condition fails at every wall point twice over -- the Eulerian observer's own density is `-6.2e-4`, and a boosted
  observer at `0.9 c` relative to the bubble sees `-2.77e-2`, forty-five times worse.
- the bill: `-1.8947e27 kg` of negative energy for a 10 m bubble at the speed of light -- `-9.53e-4` solar masses,
  one Jupiter mass. It scales as `v_s^2 R^2 / wall`, so a thinner wall costs more, and no choice of profile or
  thickness brings it to zero.
- two measured negatives: the smooth tanh wall **leaks curvature into the bubble** (1.6 % of the wall's peak at
  `r = 0.3 R`), so the interior is not flat as the picture suggests; and Alcubierre's original top-hat is out of scope
  for this representation -- its derivative is a delta function on the shell and the numerical total depends on
  whether a grid point lands on it.
- `examples/warp_drive` with three maps (the wall, the expansion, the energy density with the negative region in red)
  and a turntable render of the negative-energy shell, plus 26 tests in `tests/test_warp_bubble.py`.
