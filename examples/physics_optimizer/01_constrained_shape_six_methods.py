"""One constrained design problem through one interface, six optimizers.

Torsional rigidity of an a x b rectangle is the integral of u for -laplace(u) = 1, u = 0 on the boundary (solved by the
differentiable finite-element solver). Maximise it at fixed area a*b = 1, with 0.5 <= a, b <= 2: among rectangles of equal
area the square is the best, so the answer is a = b = 1.

    PYTHONPATH=. python examples/physics_optimizer/01_constrained_shape_six_methods.py
"""
import torch

from pinneapple_core import Mesh
from pinneapple_core.fem import integrate_p1, solve_poisson
from pinneapple_core.optim import PhysicsOptimizer, eq

torch.set_default_dtype(torch.float64)
ref = Mesh.structured([0, 0], [1, 1], (10, 10))
base, boundary = torch.tensor(ref.points), ref.boundary_nodes()


def rigidity(x):
    pts = base * x                                           # the geometry is the design variable
    return integrate_p1(pts, ref.cells, solve_poisson(pts, ref.cells, 1.0, boundary))


opt = PhysicsOptimizer({"a": (0.5, 2.0, 1.6), "b": (0.5, 2.0, 0.8)},          # name: (lo, hi, start)
                       objective=lambda x: -rigidity(x),
                       constraints=[eq(lambda x: x[0] * x[1] - 1.0)])
adjoint = lambda z: torch.func.grad(lambda x: -rigidity(x))(torch.tensor(z)).numpy()   # any adjoint code fits here

for result in opt.compare(grad=adjoint).values():
    print(result)
