"""The PINN-solver examples (``examples/pinn_solver``): the symbolic loss of ``PINNFactory`` evaluated on a field
(``01_symbolic_loss_basic``). The compiler examples (Poisson 2-D, Burgers 1-D, wave 1-D) and the inverse heat problem
are flows of the generic equations / train.pinn / inverse.discover nodes."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("pinn_solver.factory_loss", "Symbolic PINN loss (PINNFactory) on the zero field", category="PINN",
      outputs=[Port("metrics", "metrics")],
      params=[Param("residuals", "equations", ["Derivative(u(t,x), t) + u(t,x)"],
                    help="SymPy residual strings that should vanish at the collocation points"),
              Param("independent_vars", "list", ["t", "x"]), Param("dependent_vars", "list", ["u"]),
              Param("n_points", "int", 32, minimum=1, help="collocation points, standard normal per variable"),
              Param("seed", "int", 0, minimum=0)],
      calls="pinneapple_physics.pinn_solver.factory.pinn_factory.PINNProblemSpec + PINNFactory."
            "generate_loss_function")
def factory_loss(ctx, inputs, params):
    """A PINNFactory loss evaluated for the zero field. The residual strings are compiled by SymPy into a torch
    loss and evaluated at random collocation points: the total and each component."""
    import torch

    from pinneapple_physics.pinn_solver.factory.pinn_factory import PINNFactory, PINNProblemSpec
    ind = [str(v) for v in params["independent_vars"]]
    dep = [str(v) for v in params["dependent_vars"]]
    spec = PINNProblemSpec(pde_residuals=[str(r) for r in params["residuals"]], conditions=[],
                           independent_vars=ind, dependent_vars=dep, inverse_params=[], verbose=False)
    loss_fn = PINNFactory(spec).generate_loss_function()

    class _Zero(torch.nn.Module):
        def forward(self, *xs):
            return torch.zeros((xs[0].shape[0], len(dep)), device=xs[0].device, dtype=xs[0].dtype)

    torch.manual_seed(int(params["seed"]))
    pts = tuple(torch.randn(int(params["n_points"]), 1, requires_grad=True) for _ in ind)
    loss, comps = loss_fn(_Zero(), {"collocation": pts})
    ctx.metric("loss", float(loss.item()))
    for k, v in comps.items():
        ctx.metric(f"component.{k}", float(v))
    ctx.table("loss components", ["component", "value"], [[k, float(v)] for k, v in comps.items()])
    return {"metrics": ctx.report["metrics"]}


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    register_capability(Capability(
        id="pinn_solver.factory_loss", title="A symbolic PINN loss from residual strings (PINNFactory)",
        domain="PINN", example="pinn_solver",
        keywords={"pinnfactory": 4, "symbolic loss": 4, "residual strings": 3, "sympy residual": 3,
                  "symbolic pinn loss": 4},
        min_score=3, nodes=("pinn_solver.factory_loss",), references=("raissi2019pinn",),
        template="pinn_solver/01_symbolic_loss_basic", quick=False,
        description="examples/pinn_solver/01_symbolic_loss_basic.py"))


_register_capabilities()
