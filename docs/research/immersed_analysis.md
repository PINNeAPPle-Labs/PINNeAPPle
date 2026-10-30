# Immersed analysis on raw geometry: steady heat (issue #966, part 1)

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
| 2-D ring, Dirichlet | Q2 | 10, 20, 40, 80 | 3.20, 3.15, 3.09 | 2.12, 2.07, 2.04 |
| 3-D sphere, Dirichlet | Q1 | 6, 12, 24, 48 | 2.09, 2.25, 2.20 | 1.02, 1.03, 1.01 |
| 3-D sphere, Dirichlet | Q2 | 4, 8, 16, 24 | 2.86, 3.15, 3.15 | 2.00, 2.11, 2.09 |
| 3-D sphere, Robin | Q1 | 6, 12, 24, 48 | 2.18, 2.28, 2.20 | 0.99, 1.00, 1.00 |
| 3-D sphere, Robin | Q2 | 4, 8, 16, 24 | 2.84, 3.11, 3.09 | 1.96, 2.09, 2.08 |
| 3-D torus, Dirichlet | Q1 | 6, 12, 24, 48 | 1.80, 2.00, 2.21 | 0.93, 1.00, 1.01 |
| 3-D torus, Dirichlet | Q2 | 4, 8, 16, 24 | 2.23, 2.90, 3.00 | 1.90, 2.16, 2.07 |

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

CONDITIONING_TABLE

Without stabilisation, Q2 is numerically singular. The finite cell method keeps the solution accurate, but the raw
condition number grows to 1e9–1e10. The ghost penalty keeps it bounded and independent of the cut. Its coefficient
trades accuracy against conditioning: for Q2 on the 2-D disk, 0.1 gives about 3× the FCM error, while 0.01 comes
within 25 % at κ ≈ 3e4 (sweep in the PR description).

### Other geometry inputs

Sphere with convection, Q1 and Q2 at h = R/8 (`tests/test_immersed.py`, measured once here):

| Input | Volume error | L2 rel (Q1 / Q2) | Quadrature time (Q1 / Q2) |
|---|---|---|---|
| SDF | −2.3e-4 | 3.8e-4 / 4.7e-6 | 1.2 / 7.8 s |
| STL (icosphere, 20 480 triangles) | −7.7e-4 (faceting) | 4.2e-4 / 4.4e-5 | 7.7 / 30 s after the pseudonormal change |
| Voxels (R/40) | −5.7e-4 | 1.1e-3 / 9.7e-4 | 0.8 / 6.9 s |

The STL and voxel errors are dominated by the input itself: the facets of the icosphere, and the staircase of the
voxels smoothed over one voxel.

## Equal accuracy against conforming FEM

COMPARISON

## Verdict (part 1, steady heat)

VERDICT
