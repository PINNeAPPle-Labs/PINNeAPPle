import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import pinneapple as pp
from pinneapple_core import Mesh
from pinneapple_core.fem import solve_poisson
from pinneapple_core.module import Hybrid, Lambda, PhysicsModule, Sequential, SolverModule

D = torch.float64


@pytest.fixture(autouse=True)
def _dtype():
    old = torch.get_default_dtype()
    torch.set_default_dtype(D)
    yield
    torch.set_default_dtype(old)


def test_sequential_chains_modules_solvers_and_functions_and_describes_itself():
    net = Sequential(nn.Linear(2, 4), nn.Tanh(), SolverModule(lambda z: 2 * z), lambda z: z.sum(dim=1))
    assert net(torch.ones(3, 2)).shape == (3,)
    assert len(net) == 4 and isinstance(net[3], Lambda)
    text = net.describe()
    assert "Sequential" in text and "SolverModule [solver]" in text and "Linear [neural] params=12" in text
    named = Sequential(("lift", nn.Linear(1, 2)), ("out", nn.Linear(2, 1)))
    assert list(named.components()) == ["lift", "out"]
    with pytest.raises(ValueError):
        Sequential()


def test_custom_physics_module_subclass_and_fit():
    class Quadratic(PhysicsModule):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(()))

        def forward(self, x):
            return self.w * x ** 2

    x = torch.linspace(-1, 1, 40)
    m = Quadratic()
    hist = m.fit(x, 3.0 * x ** 2, epochs=300, lr=0.1)
    assert hist[-1] < 1e-6 < hist[0] and m.w.item() == pytest.approx(3.0, abs=1e-3)
    assert m.num_parameters() == 1


def test_solver_module_trainable_parameter_is_identified_through_a_fem_solve():
    mesh = Mesh.interval(0.0, 1.0, 20)
    pts, bn = torch.tensor(mesh.points), mesh.boundary_nodes()
    solve = lambda f, kappa: solve_poisson(pts, mesh.cells, f, bn, kappa=kappa)
    truth = solve(torch.ones(21), torch.tensor(2.5))
    solver = SolverModule(solve, params={"kappa": 1.0}, trainable=["kappa"])
    assert [n for n, _ in solver.named_parameters()] == ["params.kappa"]
    for _ in range(300):
        solver.zero_grad()
        loss = ((solver(torch.ones(21)) - truth) ** 2).sum()
        loss.backward()
        with torch.no_grad():
            solver.params["kappa"] -= 2.0 * solver.params["kappa"].grad
    assert solver.params["kappa"].item() == pytest.approx(2.5, rel=1e-3)
    const = SolverModule(solve, params={"kappa": 1.0})
    assert const.num_parameters() == 0
    with pytest.raises(ValueError):
        SolverModule(solve, params={"kappa": 1.0}, trainable=["nope"])


def _coarse_fem(n):
    m = Mesh.interval(0.0, 1.0, n)
    pts = torch.tensor(m.points)
    xc = (pts[1:] + pts[:-1]).T / 2
    k = lambda th, x: 1 + 0.8 * torch.sin(2 * np.pi * th[:, :1] * x)

    def solve(th):
        u = torch.stack([solve_poisson(pts, m.cells, 1.0, m.boundary_nodes(), kappa=kk) for kk in k(th, xc)])
        return torch.nn.functional.interpolate(u[:, None], size=65, mode="linear", align_corners=True)[:, 0]
    return solve, k


def test_hybrid_coarse_solver_plus_neural_correction_trains_end_to_end():
    torch.manual_seed(0)
    coarse, k = _coarse_fem(4)
    fine, _ = _coarse_fem(64)
    xf = torch.linspace(0, 1, 65)
    theta = torch.rand(48, 1) * 3 + 2
    truth = fine(theta)
    net = Sequential(nn.Conv1d(3, 16, 5, padding=2), nn.Tanh(), nn.Conv1d(16, 16, 5, padding=4, dilation=2), nn.Tanh(),
                     nn.Conv1d(16, 1, 5, padding=2), lambda c: c[:, 0])
    model = Hybrid(SolverModule(coarse), net, corrector_input=lambda a, u0: torch.stack([u0, k(a[0], xf), 0 * u0 + xf], 1))
    rel = lambda p, t: ((p - t).norm(dim=1) / t.norm(dim=1)).mean().item()
    with torch.no_grad():
        before = rel(model(theta), truth)
    hist = model.fit(theta[:36], truth[:36], epochs=120, lr=1e-2, batch_size=12)
    assert hist[-1] < 0.3 * hist[0]
    with torch.no_grad():
        held = rel(model(theta[36:]), truth[36:])
        coarse_only = rel(model.solver(theta[36:]), truth[36:])
    assert held < 0.8 * coarse_only                       # generalises: better than the solver alone on unseen inputs
    parts = model(theta[:2], return_parts=True)
    assert torch.allclose(parts["output"], parts["solver"] + parts["correction"])
    assert before > 0  # sanity


def test_hybrid_modes_and_detach_and_validation():
    solver, corr = (lambda x: 2 * x), nn.Linear(1, 1)
    nn.init.zeros_(corr.weight); nn.init.constant_(corr.bias, 0.5)
    x = torch.ones(4, 1)
    assert torch.allclose(Hybrid(solver, corr, "residual")(x), 2 * x + 0.5)
    assert torch.allclose(Hybrid(solver, corr, "multiplicative")(x), 2 * x * 1.5)
    assert torch.allclose(Hybrid(solver, corr, "replace")(x), torch.full((4, 1), 0.5))
    # detach_solver: no gradient reaches a trainable solver parameter
    s = SolverModule(lambda x, a: a * x, params={"a": 1.0}, trainable=["a"])
    Hybrid(s, corr, detach_solver=True)(x).sum().backward()
    assert s.params["a"].grad is None and corr.bias.grad is not None
    s2 = SolverModule(lambda x, a: a * x, params={"a": 1.0}, trainable=["a"])
    Hybrid(s2, corr)(x).sum().backward()
    assert s2.params["a"].grad is not None
    with pytest.raises(ValueError):
        Hybrid(solver, corr, "bogus")


def test_any_pp_solve_method_is_a_component():
    from pinneapple_physics.pde_environment.builder import ProblemBuilder

    spec = (ProblemBuilder("p").domain(x=(0, 1), y=(0, 1)).fields("u").pde("poisson")
            .bc("dirichlet", field="u", value=0.0, on="x_boundary").bc("dirichlet", field="u", value=0.0, on="y_boundary").build())
    prob = pp.PhysicalProblem.from_pde_spec(spec)
    src = lambda X, ctx=None: -2 * np.pi ** 2 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    fem = SolverModule.from_method(prob, "fem", ctx={"source_fn": src}, n=8)       # coarse classical solver
    truth_fn = lambda X: torch.sin(np.pi * X[:, :1]) * torch.sin(np.pi * X[:, 1:])
    X = torch.rand(300, 2)
    net = nn.Sequential(nn.Linear(3, 24), nn.Tanh(), nn.Linear(24, 1))
    model = Hybrid(fem, net, corrector_input=lambda a, u0: torch.cat([a[0], u0], dim=1))
    err0 = ((model(X) - truth_fn(X)) ** 2).mean().item()
    model.fit(X, truth_fn(X), epochs=150, lr=1e-2)
    assert ((model(X) - truth_fn(X)) ** 2).mean().item() < err0
    assert fem.solution.kind == "classical"


def test_public_namespace():
    assert pp.PhysicsModule is PhysicsModule and pp.Hybrid is Hybrid and pp.Sequential is Sequential
