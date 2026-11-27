# Morphing a surface mesh: a design space on a body that already exists

The library could generate a shape only through a parametric family — a CST airfoil, the airliner, the Ahmed body —
so a design study was limited to the shapes somebody had hand-coded, and `pp.cfd.ExternalFlow` could only explore
them. Roadmap item **P1** ([#292](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/292)) asks for the other half:
a generic way to deform an arbitrary surface mesh, with the deformation as the design variable, and a check that
decides whether the result is still a body.

The result, measured on the road car of `data/geometries` (10 898 vertices, 21 760 triangles):

| Quantity | Measured |
|---|---|
| FFD identity at zero parameters | max vertex error **6.2e-15 m** |
| Cost of one FFD deformation (392 variables, 245 control points) | **2.1 ms** |
| Cost of one RBF deformation (4 handles, 92 fixed points, TPS) | **0.38 ms** |
| RBF handle displacement error / fixed-point motion | **2.2e-16 m** / **6.3e-16 m** |
| Reference measurement for the validity gate (once per study) | **0.6–0.9 s** |
| Validity gate per sample (watertight, volume, area, self-intersection, flipped faces) | **0.6–1.0 s** |
| Valid samples from the declared band (±35 % of a lattice cell) | **24 / 24**, volume 0.949–1.038 of the original |
| Valid samples from a wide band (±2 cells) | **3 / 8**, with the reason for each refusal |
| FFD of the Ahmed slant against the library's own 35° body | **3.7 % relative L2**, worst vertex **4.1 mm** |

```bash
python examples/geometry/09_surface_morphing_design_space.py        # ~45 s, writes _out/morphing/summary.json
```

The gate that keeps this page honest is `tests/pinneapple_design/test_geometry_morph.py`: identity at zero
parameters, a deformation never larger than its control-point displacement, symmetry holding the mirror plane,
keep-out leaving its region alone, RBF interpolating its handles exactly, a rigid motion reproduced to round-off, and
the four refusals — volume out of band, a pair of faces that crosses, a face turned inside out, a point outside the
declared box.

## The three morphings, and when to use which

| | design variables | use it when | cost per design (this mesh) |
|---|---|---|---|
| `FFD` | control-point displacements of a lattice around the body | you want a smooth, global change and a bounded box to sample (the usual aero/design study) | 2.1 ms |
| `RBFMorph` | displacements of handles placed on chosen surface points | you know *what* has to move (a lip, a trailing edge) and what must not (a mounting face) | 0.38 ms |
| `ShapeModes` | coefficients of a PCA of a registered mesh family | the family already exists (variants from CAD, AhmedML, DrivAerML) and you want to stay inside it | a matrix product of the kept modes |

All three map a parameter vector to `(vertices, faces)` with the **topology unchanged**, which is the property that
makes fields comparable across designs: vertex `i` is the same material point before and after.

```python
import pinneapple as pp

car = pp.geom.morph.Surface.load("data/geometries/road_car/road_car.light.stl")
ffd = pp.geom.morph.FFD.fit(car.reference, control=(6, 4, 6), symmetry="y")
space = pp.geom.morph.DesignSpace(ffd, car.faces, car.reference,
                                  gate={"max_volume_change": 0.25, "contact_tolerance": 1.0})
report = space.sample_valid(24, seed=0)
V, F = space.build(report.parameters[0]).as_tuple()      # ready for pp.cfd.ExternalFlow
```

## How the deformation is parameterised

**FFD** is the tensor-product Bernstein map of Sederberg & Parry (1986). A vertex at normalised coordinates
`(s, t, u)` moves by the Bernstein-weighted sum of the control-point displacements. Two properties are why this is a
design space and not a toy: the weights are non-negative and sum to one, so a zero vector returns the mesh *exactly*
(6.2e-15 m here) and no vertex moves further than the largest control-point displacement in the band; and the map is
smooth, so a lattice perturbation moves a region instead of denting vertices.

The weights depend on the mesh and the lattice, never on the parameters, so they are computed once and a design is
then one matrix product. That is the 2.1 ms.

**Symmetry and keep-out are constraints on the control points, resolved when the lattice is built**, not checks after
the fact: a mirror plane identifies each control point with its image (one of the pair carries the parameters, the
other follows with the normal component flipped, and a point on the plane loses that component), and a keep-out box or
a frozen point removes it from the parameter vector. So the sampler never proposes a direction the geometry cannot
take, and the parameter count drops with the symmetry (735 → 392 for the car, 53 %).

Two measured caveats, both worth knowing before designing a study:

- **Keep-out is exact at the control points, and only there.** With the lattice flush against the body
  (`padding=0`) a frozen layer leaves its vertices at 0.0 m; with the default 5 % padding the extreme vertex still
  feels the free neighbours and moves 13 % of the maximum. When the region must not move at all, `DesignSpace`
  enforces it geometrically with a tolerance (`keep_out=`), and refuses the design that violates it.
- **The least-squares inverse of a shape change is degenerate.** Fitting the parameters so the 25° Ahmed body becomes
  the library's 35° body works (3.7 % relative L2, 4.1 mm at the worst vertex over a 51.9 mm change) — but the design
  vector that does it has control-point displacements of 1.2 m, and refining the lattice to reach 0.02 mm makes them
  hundreds of metres that cancel. A global Bernstein basis correlates its parameters; the design space is therefore a
  **bounded box**, and the bounds are part of the geometry, not advice to the sampler.

**RBF morphing** moves the points an engineer names. The field is
`u(x) = Σ wᵢ φ(|x − hᵢ|) + a₀ + a₁x + a₂y + a₃z`, solved from the handle values, with the points that must stay put
entering as handles at zero. The polynomial part is not decoration: without it the interpolant cannot reproduce a
translation, and with it a rigid motion of the handles reproduces the rigid motion exactly — measured at 4.4e-16 m
with the handles free. With the ground face pinned the same translation is resisted (5.0e-2 m of mismatch), which is
what pinning means. The default kernel is the thin-plate spline, the minimum-bending interpolant; `multiquadric`,
`gaussian` and `linear` are available. At least four handles (or handles plus fixed points) are required, and they
must not be coplanar — below that the linear part is unconstrained and a rigid motion is only reproduced on the plane
through the handles, so the class refuses the system instead of returning a plausible wrong field.

**Shape modes** are the PCA of a registered family: `mean + Σ cₖ modeₖ`, with `project`/`residual` to fit a new shape
into the space and measure what the kept modes cannot express. The reference is the family's *mean*, not the first
sample. On a family of 24 shapes randomly sampled from the car's FFD it takes 21 modes to reach 99 % and the first six
carry 71 % — a random design space is high-dimensional by construction, and that is the honest reading: what the modes
compress is a family of real variants with few directions, not a random box.

## The validity gate

A deformation of a closed surface keeps its winding and its closed boundary, so the failures that matter are the ones
a solver would only find later. `check_surface` reports, with the number behind each verdict:

| check | source | default limit |
|---|---|---|
| watertight, non-manifold edges, orientation, degenerate faces, enclosed volume | `pinneapple_data.cae.quality.surface_checks` — the same checks the mesh quality app reports | watertight, positive finite volume |
| faces turned inside out (normal against the reference) | this module | 0 |
| volume and frontal-area change against the reference | this module | ±50 % (configurable) |
| **new** self-intersecting face pairs | this module | 0 |
| wall thickness from inward rays | this module | off by default |
| keep-out region displacement | `DesignSpace` | 0 within its tolerance |

The self-intersection test is the one the CAE layer did not have. Candidate pairs come from a uniform grid and an
AABB pass (1.3 M → 48 k on the car), neighbours sharing a vertex are dropped, and the rest go through a vectorised
Möller-Trumbore over both directions plus a 2-D crossing test for the coplanar case the ray test cannot see.

Two decisions make it usable on real STLs:

- **Only *new* intersections count.** The car STL is watertight and already has **1708** intersecting face pairs —
  overlapping shells, a wheel against its arch, glass against the body. A gate that counted them again would be
  measuring the model, not the deformation. `Reference` records the pairs once and the gate subtracts them.
- **`contact_tolerance`** ignores a pair whose faces were already touching in the reference (centroids closer than
  the tolerance times their two diagonals). Under any body-wide deformation a wheel-arch gap of a few millimetres
  closes; at tolerance 0 the car refuses every sample, at 1.0 it accepts the declared band. The ignored pairs are
  reported in the metrics, so the setting is visible in the study, not buried in a default.

## Cost, and what it means for a study

On the road car: 2.1 ms for an FFD design and 0.38 ms for an RBF one, against 0.6–1.0 s for the validity gate — the
deformation is roughly a thousand times cheaper than proving it is safe. That ratio is the point: a study can afford
thousands of morphings and the gate is what costs, so the gate is measured once per reference and run once per
sample, and the expensive checks (thickness rays) are opt-in.

A 200-sample sweep gated per sample costs about three minutes on this mesh, against the minutes *per sample* of the
`snappyHexMesh` + `simpleFoam` run that follows it. What this page does not contain is that sweep: the example writes
the OpenFOAM case of one deformed design (`pp.cfd.ExternalFlow`, reference area 2.7 m², the triSurface written), and
the drag over the design space needs the solver installed — named as the next step on the issue, together with the
Ahmed slant/rear study through the CFD path.

## What is not here

- Re-meshing: a deformation large enough to make the mesh unusable is **rejected**, not repaired. The gate's
  `thickness` rays flag a collapsed wall; nothing closes it.
- Registration: `ShapeModes` refuses meshes that are not registered instead of guessing a correspondence. Producing
  the correspondence — a family of CAD variants, or morphings of one reference — is the caller's job, and the FFD and
  RBF morphings are how you do it.
- A volume mesh: `FFD.deform(x, vertices=...)` and `RBFMorph.deform(x, vertices=...)` apply the same field to another
  point set inside the lattice (a boundary-layer block, a probe cloud), but nothing here generates that mesh.
