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
- Higher-order operators (curl, jacobian, hessian, flux) are the scope of roadmap item X4.

Examples: `examples/core_primitives/01_poisson_pinn.py` (PINN) and `02_operator_poisson_1d.py` (neural operator).
