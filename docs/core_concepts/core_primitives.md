# Core primitives: Domain, Geometry, Mesh, Field

`pinneapple_core` (also `pp.core`) is the shared vocabulary of tensors, geometry and topology. One object
answers the four questions every physics workflow asks of a quantity: its derivative, its divergence, its value
at an arbitrary point, and its integral.

| Class | What it is |
|---|---|
| `Domain` | A box in any dimension, or a 2D CSG shape. `contains`, `sdf`, `sample_interior`, `sample_boundary(return_normals=True)`, `volume`. |
| `Geometry` | A `Domain` with named boundaries (`x_min`, `x_max`, ... for boxes) and `mesh(h)`. |
| `Mesh` | Simplex mesh (interval, triangle, tetrahedron). Exact P1 `gradient`, `integrate`, `interpolate`, `boundary_nodes`. |
| `Field` | Values on a `Grid`, a `Mesh` or a `PointCloud`. |
| `FunctionField` | A torch callable with the same calls, derivatives by autograd. |

```python
import numpy as np
from pinneapple_core import Field, Geometry

geom = Geometry.box([0, 0], [1, 1])
mesh = geom.mesh(h=0.05)

u = Field.on_mesh(mesh, lambda p: np.sin(np.pi * p[:, 0]) * p[:, 1])
u.gradient()                      # Field with values (N, 2)
u.integrate()                     # exact for the piecewise-linear interpolant
u.interpolate(np.array([[0.3, 0.4]]))
```

Values have shape `(N, *components)`: `(N,)` scalar, `(N, d)` vector, `(N, c, d)` gradient of a vector.
`gradient()` of a scalar gives `(N, d)`; `divergence()` needs a vector with `d` components.

## How each support computes the four operations

| Support | gradient | interpolate | integrate |
|---|---|---|---|
| `Grid` | central differences (second-order edges), non-uniform axes allowed | multilinear | trapezoid |
| `Mesh` | volume-weighted average of the exact cell gradients (exact for linear data) | barycentric (NaN outside, or `fill="nearest"`) | exact for P1 |
| `PointCloud` | weighted local least-squares plane over the k nearest neighbours | Delaunay-linear in d >= 2, `np.interp` in 1D | `weights`, else Monte Carlo over a `Domain` |
| `FunctionField` | autograd, graph kept (`create_graph=True`) | evaluation | Monte Carlo over the `Domain`, or points + weights |

## Notes and limits

- `Mesh` stores simplices only; surface meshes embedded in 3D (`MeshData`) are a different object.
- `Geometry.mesh` is a structured mesh for boxes and a Delaunay mesh of a lattice plus boundary samples for 2D CSG
  shapes. Good for prototyping; use gmsh for production quality.
- The point-cloud Monte Carlo integral assumes the points are uniform samples of the domain.

## Operators

`pinneapple_core.operators` (also `pp.grad`, `pp.div`, ...) gives one function per operator, the
`torch.nn.functional` of physics. The same call works on a grid, mesh or point-cloud `Field`
(no point argument, returns a `Field`) and on a continuous field (a torch callable or `FunctionField`,
takes the points `x`, returns a tensor that keeps the autograd graph).

| Operator | Result |
|---|---|
| `grad(f)` | `(N, d)` scalar, `(N, c, d)` vector |
| `div(f)` | scalar, from a vector field |
| `curl(f)` | `(N, 3)` in 3D, scalar `dv/dx - du/dy` in 2D |
| `laplacian(f)` | scalar, or one per component of a vector |
| `jacobian(f)` | `(N, c, d)`, `J[n, i, j] = d f_i / d x_j` (`c = 1` for a scalar) |
| `hessian(f)` | `(N, d, d)` of a scalar |
| `integrate(f)` | number; continuous fields take `n=` (Monte Carlo) or `points=`, `weights=` |
| `flux(f, where=None)` | outward flux `integral f.n dS`; `where` is a box face (`"x_max"`) or a predicate |

```python
import pinneapple as pp

u = Field.on_mesh(mesh, lambda p: p[:, 0] ** 2 + p[:, 1])
pp.laplacian(u)                    # Field of 2.0
v = Field.on_mesh(mesh, lambda p: p)
pp.flux(v) == pp.integrate(pp.div(v))   # divergence theorem, exact for P1 data
```

Accuracy against manufactured solutions (`tests/pinneapple_core/test_operators.py`, interior of the unit
square, max error relative to the max of the exact value): grid 81x81 ~ 4e-4, mesh 40x40 ~ 3e-3,
4000 scattered points ~ 4e-2 for second derivatives and 3e-3 for first, continuous exact to round-off.
On a mesh the second derivatives are two rounds of gradient recovery, so they are accurate in the
interior and degrade at the boundary; a point cloud uses a local quadratic fit. `flux` is exact on a
mesh for P1 fields, trapezoid on a grid, and Monte Carlo (error ~ 1/sqrt(n)) on a point cloud or a
continuous field.

## Derivatives through solvers: `pp.func`

`pinneapple_core.func` (`pp.func`) wraps `torch.func` (`grad`, `jacobian`, `jacrev`, `jacfwd`, `hessian`, `vmap`)
and adds `wrt`, the argument to differentiate by name or position:

```python
from pinneapple_core import func
func.jacobian(model, wrt="geometry")(geometry, mu)     # d model / d geometry
```

- `func.implicit_solve(residual, x0, *params, solver=None)` solves `residual(x, *params) = 0` and differentiates
  the solution with the implicit function theorem, `dx/dp = -(dF/dx)^-1 dF/dp`. The backward pass is one adjoint
  solve, independent of the forward iterations, and `solver=` accepts any external code (never differentiated).
  The Jacobian `dF/dx` is dense; second derivatives through the solve ignore its dependence on `params`.
- `pinneapple_core.fem` is a differentiable P1 finite-element Poisson solver (dense matrices, up to a few
  thousand nodes). Because the vertex coordinates are torch tensors, `loss.backward()` returns the gradient with
  respect to the geometry: write `points = f(design_parameters)` and differentiate. Tested against central
  finite differences of the same pipeline (relative agreement 1e-6 or better for a 1D length and for 2D
  width, height and bump parameters).

Examples: `examples/core_primitives/01_poisson_pinn.py` (PINN) and `02_operator_poisson_1d.py` (neural operator).
