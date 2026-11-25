# A surrogate conditioned on the geometry, and the error on a body it never saw

Roadmap item **G11** ([#83](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/83)) asks for one thing: a case
with a geometry outside the training set, and the error reported on it. This page is that case, with the number.

**The published number: relative L2 of the pressure-coefficient field on the new bodies, 12.0 % (mean over three
training seeds, 11.5 % to 12.8 %), worst body 17.2 %.** On bodies inside the training box but still unseen it is
8.7 % (8.2 % to 9.4 %), and on the bodies it trained on 6.9 % (6.4 % to 7.6 %). The same model keeps the ranking of
the designs by lift: flip rate 0.7 % of the design pairs, Spearman 0.996, no change of the chosen design in any
seed.

```bash
python -m benchmarks.geometry_conditioned_surrogate          # the published protocol, about 5 minutes on one CPU
python -m benchmarks.geometry_conditioned_surrogate --quick  # the test budget, under a minute, not a benchmark
```

The result file is `benchmarks/_out/geometry_conditioned_surrogate.json` and the labelled design table is next to
it (`..._designs.csv`). The gate that keeps this page honest is `tests/test_geometry_conditioned_surrogate.py`: it
re-runs the panel frame check (a predicted field integrated with the solver's own pressure integration must give the
solver's lift), the split (a "new geometry" has a weight outside the training box by construction), the report, and
the protocol of the committed result.

## What "conditioned on the geometry" means here

The surrogate never receives the six numbers that define the body. It receives the **geometry**: the panel
mid-points of the body, one outward normal per panel, and a geometric code pooled from them by the DeepSets encoder
of [`pinneapple_neural.lpm`](../api/pinneapple_neural/index.md), with multi-scale neighbourhoods at each query
point. That is the difference between a parameter study and a geometry-conditioned surrogate: the shape reaches the
model as a point set, so the same model can answer for a body that arrives as a mesh, an STL or a drawing, and the
six CST weights are only how this page happens to generate the family.

## The family and the labels

* **geometry** — closed 2-D bodies of the CST (Kulfan) basis the library's airfoil work uses: thickness weights
  `T0..T2` and camber weights `M0..M2`, from `pinneapple_design.aero.geometry` with its documented bounds;
* **label** — the pressure coefficient `Cp` on every panel, from the Hess-Smith panel method
  (`pinneapple_design.aero.panel.panel_solve`): the inviscid potential-flow reference, which is what any
  potential-flow or neural model can at best reach. 100 panels per body, 0.06 s for the whole design set;
* **the split** — training bodies are a Latin hypercube inside the *training box*: the documented parameter ranges
  pulled back by 20 % at each end. Held out are bodies inside that box (interpolation), **new geometries** drawn in
  the outer band so that at least one weight is outside everything the model was trained on, and two named real
  airfoils fitted to the same basis: NACA 2412 (inside the box) and NACA 4412, whose `M1`/`M2` fall outside both the
  training box and the documented bounds of that weight.

| setting | published protocol |
|---|---|
| bodies trained on | 24 |
| bodies held out inside the box | 4 |
| new geometries | 4 drawn in the outer band + NACA 2412 + NACA 4412 |
| panels per body | 100 (50 per side, blunt trailing edge closed) |
| training | 1500 epochs, Adam with cosine decay, lr 2e-3, ensemble of 3 bagged heads, width 96, code 32 |
| seeds | 7, 8, 9 (same design set, three trainings) |
| angle of attack | 0 deg |

## The measured error

| body | group | outside the training box | relative L2 | Cp min (solver -> surrogate) | Cl (solver -> surrogate) | OOD |
|---|---|---|---|---|---|---|
| cst00 | unseen | - | 0.0939 | -0.529 -> -0.548 | 0.125 -> 0.152 | 0.39 |
| cst01 | unseen | - | 0.0735 | -0.606 -> -0.610 | 0.313 -> 0.299 | 0.30 |
| cst02 | unseen | - | 0.1267 | -0.559 -> -0.506 | 0.198 -> 0.175 | 0.92 |
| cst03 | unseen | - | 0.0529 | -0.505 -> -0.530 | 0.116 -> 0.126 | 0.34 |
| cst00 | new | T0, T2, M0, M1 | 0.1339 | -0.838 -> -0.731 | 0.208 -> 0.226 | 0.53 |
| cst01 | new | T1 | 0.0909 | -0.441 -> -0.432 | 0.116 -> 0.103 | 0.53 |
| cst02 | new | T1, T2, M1, M2 | 0.1466 | -0.724 -> -0.661 | 0.369 -> 0.314 | 1.17 |
| cst03 | new | T0, M2 | 0.1388 | -0.368 -> -0.348 | -0.026 -> -0.026 | 0.37 |
| NACA 2412 | new | - | 0.0388 | -0.555 -> -0.580 | 0.243 -> 0.252 | 0.37 |
| NACA 4412 | new | M1, M2 | 0.1716 | -0.731 -> -0.720 | 0.485 -> 0.539 | 0.91 |

`relative L2` is the mean over the three seeds; `Cp min` is the suction peak, the number that sets the lift; `Cl` is
integrated from the *predicted* field with the solver's own pressure integration, so coefficient and field never
disagree; `OOD` is the library's own geometry-code distance to the training set, relative to the usual distance
between training designs (the model's own warning, not the error).

Three things the table says plainly:

* the error grows as the body gets further from the training set -- 6.9 % on the trained bodies, 8.7 % on unseen
  bodies inside the box, 12.0 % on bodies in the outer band -- and the worst body is the one that is outside the
  documented bounds of the family (NACA 4412, 17.2 %);
* a real airfoil that happens to sit inside the trained box (NACA 2412) is the *easiest* body of the held-out set
  (3.9 %), which is the useful direction: the model does not care that the weights were written by NACA or by a
  sampler;
* the model's own OOD score does not track the error (the worst body scores 0.91, the hardest in-box body 0.92):
  a geometry distance is a warning about the design space, not a prediction of accuracy.

## The decision level

A field error does not say whether the surrogate changes the design. Over the 10 held-out bodies, asking "which
body carries the most lift" (`pinneapple_analysis.decision`):

| metric | value |
|---|---|
| flip rate (pairs of designs whose order is reversed) | 0.7 % |
| Spearman rho / Kendall tau-b | 0.996 / 0.985 |
| top-k kept (k = 1) | 100 % |
| regret (the design the surrogate picks, against the best) | 0 |
| the chosen design changed | in none of the three seeds |

## What this case is not

* **The reference is inviscid.** The model learns the panel method's `Cp`, not a viscous flow: the absolute error
  levels are not comparable with a RANS benchmark, and no claim is made that they are.
* **One family.** CST sections at 0 deg, one chord, no compressibility, no viscous decambering. Generalisation is
  measured *inside the family and its outer band*, not onto an arbitrary shape.
* **The 2-D bodies come from `pinneapple_design.aero.geometry`, not from `pinneapple_design.geometry_library`.**
  The library's entries are 3-D assemblies whose solid kernel is `manifold3d` (not installed in this environment),
  and labelling a 3-D body needs a field solver that costs minutes per design, which would make the case a campaign
  rather than a case. The step that would connect the library to this model is a section of a library body used as
  the body's point set; the surrogate itself needs nothing else.
* **Two digits are stable, three are not.** The spread over seeds is reported next to every mean for that reason.
