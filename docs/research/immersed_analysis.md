# Immersed analysis on raw geometry (issue #966)

The question: can PINNeAPPle analyse a part straight from its geometry (a signed distance, a voxel mask, an STL)
without generating a conforming mesh, accurately enough and fast enough to matter? The earlier meshless attempt
(RBF-FD and GFDM in [condensation_falsification.md](condensation_falsification.md), M1–M4) lost to finite volumes at
equal unknowns. This part tests the other well-established family: immersed / finite cell methods.

Every number below was measured in this container (4 cores, one thread per run, Python 3.13, NumPy / SciPy,
pyamg 5.3, gmsh 4.15, CalculiX 2.21) by the commands given with each table.

## Method (`pinneapple_simulation.immersed`)

- **Geometry**: any level set `phi(X)` (negative inside). Sources: analytic SDFs and their CSG (`ball`, `box`,
  `cylinder`, `torus`, `annulus`, `|`, `&`, `-`), any vectorised SDF (e.g. `pinneapple_design.geometry.gen.sdf_shapes`),
  a voxel mask (`VoxelGeometry`: Euclidean distance transforms of material and void, interpolated), or an STL
  (`STLGeometry`). For the STL, the inside test is the generalised winding number (Jacobson et al. 2013) with a
  dipole tree (Barill et al. 2018), and the distance is the exact point-triangle distance to candidates from a
  KD-tree of surface samples. In the near band of a watertight, consistently oriented surface, the sign comes from
  angle-weighted pseudonormals (Bærentzen & Aanæs 2005); there it is identical to the winding number (0 mismatches on
  200 000 points) and about 4× cheaper. A surface with holes uses the winding number everywhere.
- **Discretisation**: uniform background grid, Q1 or Q2 Lagrange elements; unknowns only on the active cells.
- **Cut cells**: adaptive octree, `depth` levels (default 2 for Q1, 3 for Q2). Leaves still cut are split into Kuhn
  simplices and clipped by the level set. The surface points are put on the true surface by regula falsi. The fine
  rule of each cut cell is compressed by moment fitting onto (2p+1)^d Gauss points, which is exact for the stiffness
  and mass integrands (Q_2p). The cut-cell matrices are then one matrix product.
- **Small cuts**: ghost penalty on the faces of cut cells (Burman 2010; coefficient 0.1 for Q1, 0.01 for Q2), or the
  finite cell method's fictitious material α = 1e-6 (Parvizian, Düster & Rank 2007), or none. The condition number is
  computed on request (Lanczos plus shift-invert, raw and Jacobi-scaled).
- **Boundary conditions**: Nitsche for temperatures. The penalty is β k / h with ghost penalty, or comes from a
  per-cell eigenvalue problem without it (Embar, Dolbow & Harari 2010). Heat flux and convection use surface
  quadrature on the clipped facets. Conditions are selected by `where(X, n)`.
- **Checks returned with every solve**: an energy balance (heat generated against heat leaving through each
  condition, using Nitsche's consistent flux) and, on request, the condition number.

## Verification

### Manufactured solutions on curved domains

`u = sin 2x cos 3y (sin z) + x²` with the matching source. Relative L2 and H1 errors are measured on the fine octree
rule. The rate is the slope between successive grids. Command: `verification.convergence(verification.manufactured(...), order, ns)`.

| Case | Order | n (cells across) | Rates L2 | Rates H1 |
|---|---|---|---|---|
| 2-D ring, Robin | Q1 | 10, 20, 40, 80 | 2.33, 2.12, 2.11 | 0.99, 1.00, 1.00 |
| 2-D ring, Robin | Q2 | 10, 20, 40, 80 | 3.18, 3.13, 3.09 | 2.09, 2.06, 2.03 |
| 2-D ring, Dirichlet | Q1 | 10, 20, 40, 80 | 2.26, 2.06, 2.08 | 1.00, 1.00, 1.00 |
| 2-D ring, Dirichlet | Q2 | 10, 20, 40, 80 | 3.21, 3.15, 3.09 | 2.10, 2.07, 2.03 |
| 3-D sphere, Dirichlet | Q1 | 6, 12, 24, 48 | 2.09, 2.25, 2.20 | 1.02, 1.03, 1.01 |
| 3-D sphere, Dirichlet | Q2 | 4, 8, 16, 24 | 2.97, 3.18, 3.16 | 2.04, 2.10, 2.08 |
| 3-D sphere, Robin | Q1 | 6, 12, 24, 48 | 2.18, 2.28, 2.20 | 0.99, 1.00, 1.00 |
| 3-D sphere, Robin | Q2 | 4, 8, 16, 24 | 2.84, 3.11, 3.09 | 1.96, 2.09, 2.08 |
| 3-D torus, Dirichlet | Q1 | 6, 12, 24, 48 | 1.80, 2.00, 2.21 | 0.93, 1.00, 1.01 |
| 3-D torus, Dirichlet | Q2 | 4, 8, 16, 24 | 2.14, 2.94, 3.06 | 1.82, 2.10, 2.08 |

The rates are optimal (p + 1 in L2, p in H1). The energy balance closes to between 1e-16 and 1e-10 in every run.

### Sphere and cylinder in closed form

Uniform generation q, convection h on the surface. Sphere: `T = T∞ + qR/3h + q(R² − r²)/6k`. Finite cylinder with
convection on the mantle and adiabatic caps: `T = T∞ + qR/2h + q(R² − r²)/4k` (Incropera et al., §3.5).

| Case | Order | n | Unknowns | L2 rel | H1 rel |
|---|---|---|---|---|---|
| sphere | Q1 | 6, 12, 24, 48 | 350 … 69 092 | 3.45e-3, 7.40e-4, 1.58e-4, 3.56e-5 | 2.12e-1 … 2.68e-2 (rate 1.0) |
| sphere | Q2 | 4, 8, 16, 24 | 879 … 74 231 | 7.32e-5, 1.92e-5, 4.74e-6, 2.10e-6 | 4.21e-4 … 2.15e-5 |
| cylinder | Q1 | 6, 12, 24, 48 | | L2 rates 2.31, 2.38, 1.59 | H1 rates 0.97, 0.98, 0.99 |
| cylinder | Q2 | 4, 8, 16, 24 | 847 … 70 839 | 4.48e-4, 2.37e-4, 1.17e-4, 8.35e-5 | 1.63e-2 … 6.12e-3 |

Two measured limitations:

- **Geometry-limited Q2.** The exact sphere field is quadratic, so Q2 can represent it exactly. What remains is the
  O((h/2^depth)²) error of the clipped surface, and the L2 rate is 2.0, not 3.
- **Sharp edges.** On the cylinder, Q2 drops to an L2 rate of about 1. A manufactured Dirichlet field on the same
  cylinder also loses its rate at the finest grid (L2 rates 2.93, 0.76). The level set has a kink at the rim, the
  linear clipping inside a leaf chamfers it, and the facets of the chamfer take the mantle or the cap condition by
  their normal. Edge-preserving clipping (one level set per smooth face) is the known remedy; it is not implemented
  here.

### Small cuts and conditioning

2-D manufactured case on a disk (n = 10). The grid is placed so that the disk pokes `offset × h` past a grid line,
which leaves slivers in the cells it enters. Command: `verification.conditioning(order)`.

| offset | order | smallest cut fraction | κ, none | κ, FCM (Jacobi-scaled) | κ, ghost (Jacobi-scaled) | L2 rel, none / FCM / ghost |
|---|---|---|---|---|---|---|
| 0.1 | Q1 | 3.3e-04 | 1.1e+09 | 1.2e+08 (3.9e+01) | 3.3e+02 (2.5e+01) | 9.8e-03 / 9.8e-03 / 1.2e-02 |
| 0.01 | Q1 | 4.1e-06 | 3.8e+13 | 8.5e+08 (6.1e+01) | 3.1e+02 (2.4e+01) | 9.5e-03 / 9.5e-03 / 1.2e-02 |
| 0.001 | Q1 | 4.2e-08 | 2.3e+18 | 2.2e+09 (2.8e+01) | 2.8e+02 (2.4e+01) | 9.6e-03 / 9.6e-03 / 1.2e-02 |
| 0.0001 | Q1 | 4.2e-10 | singular | 4.4e+09 (2.8e+01) | 2.6e+02 (2.4e+01) | 9.6e-03 / 9.6e-03 / 1.2e-02 |
| 0.1 | Q2 | 3.3e-04 | singular | 2.7e+08 (1.7e+05) | 3.7e+04 (2.6e+02) | 3.1e-04 / 3.1e-04 / 4.3e-04 |
| 0.01 | Q2 | 4.1e-06 | singular | 1.2e+09 (3.5e+05) | 6.3e+04 (2.8e+02) | 3.1e-04 / 3.1e-04 / 4.4e-04 |
| 0.001 | Q2 | 4.2e-08 | 1.8e+23 | 5.7e+09 (8.6e+03) | 6.4e+04 (2.9e+02) | 3.1e-04 / 3.1e-04 / 4.5e-04 |
| 0.0001 | Q2 | 4.2e-10 | singular | 1.5e+10 (5.2e+02) | 6.2e+04 (2.9e+02) | 3.1e-04 / 3.1e-04 / 4.5e-04 |

Without stabilisation, Q2 is numerically singular. The finite cell method keeps the solution accurate, but the raw
condition number grows to 1e8–1e10. The ghost penalty keeps it bounded and independent of the cut: about 3e2 for Q1
and 6e4 for Q2 (2.4e1 and 2.9e2 after Jacobi scaling). Its cost is the error: 25 % above FCM for Q1, 40–45 % for Q2.

The two coefficients were chosen together. For Q2 a ghost coefficient of 0.1 gives about 4× the FCM error. With 0.01
and the Nitsche penalty β = 40, the matrix was **indefinite** at offset 0.01 (two negative eigenvalues), because the
weak ghost penalty no longer controls the gradient on the sliver. β = 100 is positive definite at every offset from
0.3 to 1e-4 (`tests/test_immersed.py::test_ghost_nitsche_matrix_stays_positive_definite_on_slivers`). The Q2 defaults
are ghost 0.01 and β = 100. Every Q2 table in this report was measured with them; the convection cases have no
Nitsche term and do not depend on β.

### Other geometry inputs

Sphere with convection, Q1 and Q2 at h = R/8 (`tests/test_immersed.py`, measured once here):

| Input | Volume error | L2 rel (Q1 / Q2) | Quadrature time (Q1 / Q2) |
|---|---|---|---|
| SDF | −2.3e-4 | 3.8e-4 / 4.7e-6 | 1.2 / 5.3 s |
| STL (icosphere, 20 480 triangles) | −7.7e-4 (faceting) | 4.2e-4 / 4.4e-5 | 8.3 / 32.4 s |
| Voxels (R/40) | −5.7e-4 | 1.1e-3 / 9.7e-4 | 0.7 / 4.2 s |

The STL and voxel errors are dominated by the input itself: the facets of the icosphere, and the staircase of the
voxels smoothed over one voxel. STL quadrature is 6× slower than the SDF because every level-set value is a
nearest-triangle query. In the near band the pseudonormal sign brought it down from 116 s to 32 s for Q2; the winding
number alone was 27 of 36 s in a Q1 profile.

## Equal accuracy against conforming FEM

Command: `python -m pinneapple_simulation.immersed.compare --parts sphere_mms sphere --out DIR`. Raw rows are in
[`immersed/compare_results.json`](immersed/compare_results.json). Same part, same exact field, one thread.

- **Conforming:** P1 / P2 tetrahedra from gmsh (P2 nodes on the CAD sphere), solved by the same Python assembly and
  the same linear solver: SuperLU up to 3e5 non-zeros, otherwise CG with smoothed-aggregation AMG. CalculiX 2.21
  (C3D10) runs on the identical P2 mesh. It agrees with the internal P2: L2 3.0e-5 against 2.8e-5 at h = 0.2. Its
  error stops falling near 1e-6 at the two finest meshes, which looks like the output precision of the .frd file.
- **Times:** each method's geometry phase (cut-cell quadrature for immersed, gmsh meshing and optimisation for
  conforming) plus assembly and solve. The energy-balance check is left out for both methods.
- **Equal accuracy:** log-log interpolation between measured runs only; targets outside the measured range are not
  reported.

![sphere manufactured](immersed/sphere_mms_equal_accuracy.png)
![sphere convection](immersed/sphere_equal_accuracy.png)

| Part | Target L2 | Method | Unknowns | Compute [s] | of which geometry [s] |
|---|---|---|---|---|---|
| sphere, manufactured (non-polynomial) | 1e-3 | immersed Q2 | 2.7e3 | 1.38 | 0.78 (quadrature) |
| | | conforming P2 | 3.0e3 | 0.77 | 0.32 (meshing) |
| | | immersed Q1 | 2.4e4 | 6.58 | 4.44 |
| | 1e-4 | immersed Q2 | 1.73e4 | 6.45 | 3.29 |
| | | conforming P2 | 3.27e4 | 5.50 | 2.56 |
| sphere, convection (quadratic field) | 1e-3 | immersed Q1 | 1.26e3 | 0.50 | 0.32 |
| | | conforming P1 | 1.12e3 | 0.66 | 0.54 |
| | 1e-4 | immersed Q1 | 1.81e4 | 3.90 | 2.92 |
| | 1e-5 | immersed Q2 | 9.6e3 | 3.37 | 2.08 |
| | | conforming P2 | 2.75e3 | 0.46 | 0.15 |
| | | CalculiX C3D10 | 2.9e3 | 0.53 | 0.17 |

Run-to-run variation in this container is about ±30 % on these timings: the same sweep measured 10.7 s and 8.2 s for
the 1e-4 manufactured row in an earlier run. The ratios are stable.

## Verdict (part 1, steady heat)

Measured on two parts, a sphere with a smooth field and a sphere with generation and convection.

**Immersed wins where:**
- **Unknowns, smooth non-polynomial fields:** Q2 reaches 1e-4 with 1.73e4 unknowns against 3.27e4 for conforming P2.
  The Cartesian grid spends fewer nodes on the same volume.
- **Raw geometry:** an STL (winding number), a voxel mask or an SDF is solved directly, with no meshing step and no
  meshing failure mode. A surface with 10 missing triangles still gets the right inside test.
- **Conditioning:** with the ghost penalty, κ stays at 3e2 (Q1) / 6e4 (Q2) for cut fractions down to 4e-10.
- **Matching low order:** Q1 vs P1 at 1e-3 on the convection sphere is a draw (0.50 s vs 0.66 s, similar unknowns).

**Immersed loses where:**
- **Wall time at equal accuracy on clean CAD:** 1.2–1.8× slower than conforming P2 on the smooth field (6.45 vs
  5.50 s at 1e-4, 1.38 vs 0.77 s at 1e-3). No meshing time is saved there: the Python cut-cell quadrature took as long
  as gmsh's compiled meshing (3.3 s vs 2.6 s).
- **Fields conforming P2 represents exactly:** on the quadratic sphere field, immersed Q2 is geometry-limited (L2 rate
  2.0). It needs 3.5× the unknowns and 7× the time of P2 or CalculiX to reach 1e-5.
- **Sharp edges with a change of boundary condition:** on the cylinder rim, Q2 falls to L2 rate ≈ 1 (8.4e-5 at
  7e4 unknowns). Meshing is not the bottleneck there; edge-preserving clipping would be needed.
- **STL input cost:** 6× the SDF quadrature time (32 s for Q2 at h = R/8 on a 20 480-triangle sphere).

Not yet measured (later parts of #966): parts whose meshing is the hard step (an STL heat sink and bracket, topology
sweeps), elasticity, transient heat.


## Part 2: linear elasticity and two STL parts

### Elasticity verification

`immersed.solve_elasticity` reuses the discretisation, the stabilisation and the Nitsche machinery. Displacements can
be imposed on all components or a subset; loads are tractions, pressures, or a total force spread over the area the
selector finds (`Force`). Every solve returns an equilibrium check: applied loads against the reactions computed
from Nitsche's consistent flux. Commands: `verification.elastic_convergence(verification.elastic_manufactured(dim), order, ns)`
and `verification.solve_elastic_case(verification.lame_sphere(), n, order)`.

| Case | Order | n | L2 rel | Rates L2 | Rates H1 |
|---|---|---|---|---|---|
| 2-D disk, plane waves, ν = 0.3 | Q1 | 10, 20, 40, 80 | 1.36e-2 … 1.59e-4 | 2.21, 2.11, 2.10 | 1.03, 1.02, 1.01 |
| 2-D disk, plane waves | Q2 | 10, 20, 40, 80 | 5.08e-4 … 5.56e-7 | 3.44, 3.23, 3.16 | 2.18, 2.08, 2.05 |
| 3-D sphere, plane waves | Q1 | 6, 12, 24 | 3.04e-2 … 1.47e-3 | 2.11, 2.26 | 1.06, 1.06 |
| 3-D sphere, plane waves | Q2 | 4, 8, 12, 16 | 5.84e-3 … 5.94e-5 | 3.23, 3.26, 3.57 | 2.16, 2.16, 2.23 |
| Lamé thick sphere (a = 0.3, b = 0.6), inner pressure | Q1 | 6, 12, 24 | 1.80e-1 … 8.14e-3 | 1.97, 2.50 | 1.21, 1.17 |
| Lamé thick sphere | Q2 | 4, 8, 12, 16 | 1.18e-1 … 1.91e-3 | 2.83, 3.15, 3.11 | 1.87, 2.31, 2.43 |

Equilibrium closes to between 1e-16 and 1e-9. FCM sits at 2e-8 because of its ~1e10 condition number. On the Lamé
sphere the pressure has no net force, so the imbalance is relative to the sum of the load magnitudes.

PARTS

## Part 3: transient heat

TRANSIENT
