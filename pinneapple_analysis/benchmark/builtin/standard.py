"""The standard operator-learning problems (Li et al. 2021, "Fourier Neural Operator for Parametric PDEs"), data from
``pinneapple_simulation.numerical_solvers.operator_benchmarks``:

* ``burgers``: u_t + (u^2/2)_x = nu u_xx, periodic [0, 1), u(., 0) -> u(., T).
* ``darcy``: -div(a grad u) = 1 on the unit square, u = 0 on the boundary, a -> u.
* ``navier_stokes``: 2-D vorticity, periodic, Li et al. forcing, w(., 0) -> w(., T).

Each has its PINN formulation (hard initial/boundary conditions, PINNFactory residual) and an exact solution of its
PDE that the residual gate runs through the same compiled residual.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..registry import Check, register_problem
from .fields import FieldOperatorProblem

SPLIT_SEEDS = {"train": 11, "val": 12, "test": 13, "ood": 14}
CHUNK = 25


def _single_thread():
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def _run_chunks(fn, chunks, n_workers):
    if n_workers <= 1:
        return [fn(c) for c in chunks]
    with ProcessPoolExecutor(n_workers) as ex:
        return list(ex.map(fn, chunks))


# ------------------------------------------------------------------ chunk solvers (module level: picklable)
def _burgers_chunk(args):
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import burgers_1d
    _single_thread()
    u0, o = args
    t0 = time.process_time()
    _, u = burgers_1d(u0, o["nu"], o["t_end"], dt=o["dt"])
    dt = time.process_time() - t0
    s = o["n_solve"] // o["n_out"]
    return u[:, 0, ::s], u[:, -1, ::s], dt


def _darcy_chunk(args):
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import (
        darcy_2d,
        threshold_coefficient,
    )
    _single_thread()
    fields, o = args
    a = threshold_coefficient(fields, o["high"], o["low"])
    t0 = time.process_time()
    u = np.stack([darcy_2d(ai) for ai in a])
    dt = time.process_time() - t0
    m = o["n_solve"] // o["n_out"]
    pool = lambda z: z.reshape(len(z), o["n_out"], m, o["n_out"], m).mean(axis=(2, 4))  # noqa: E731
    return pool(a), pool(u), dt


def _ns_chunk(args):
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import (
        li_forcing,
        navier_stokes_2d_vorticity,
    )
    _single_thread()
    w0, o = args
    t0 = time.process_time()
    _, w = navier_stokes_2d_vorticity(w0, o["nu"], o["t_end"], forcing=li_forcing(o["n"]), dt=o["dt"])
    dt = time.process_time() - t0
    return w[:, 0], w[:, -1], dt


class _GRFProblem(FieldOperatorProblem):
    chunk_fn = None

    def inputs(self, split: str, n: int) -> np.ndarray:
        raise NotImplementedError

    def generate(self, split, n, n_workers):
        X = self.inputs(split, n)
        chunks = [(X[i:i + CHUNK], self.options) for i in range(0, n, CHUNK)]
        res = _run_chunks(type(self).chunk_fn, chunks, n_workers)
        return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]), sum(r[2] for r in res)


# ====================================================================== Burgers
@register_problem("burgers")
class Burgers(_GRFProblem):
    title = "Burgers 1-D"
    qoi_label = "Energy ∫u²"
    comparison_absent = ("the field is a 1-D profile u(x) at the final time: there is no second spatial axis, so the "
                         "standard reference / surrogate / error panels have nothing to put on the vertical one")
    chunk_fn = staticmethod(_burgers_chunk)
    space_fno = {"width": [32, 64], "modes": [16, 24], "lr": [1e-3, 3e-3]}

    def default_options(self):
        return {**super().default_options(), "nu": 0.01, "t_end": 1.0, "n_solve": 1024, "n_out": 256, "dt": 1e-4,
                "alpha": 2.0, "tau": 5.0, "sigma": 25.0, "sigma_ood": 50.0}

    def smoke_options(self):
        return {**super().smoke_options(), "n_solve": 256, "n_out": 64, "dt": 5e-4}

    def inputs(self, split, n):
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import periodic_grf
        o = self.options
        sig = o["sigma_ood"] if split == "ood" else o["sigma"]
        return periodic_grf(n, (o["n_solve"],), alpha=o["alpha"], tau=o["tau"], sigma=sig, seed=SPLIT_SEEDS[split])

    def qoi(self, field):
        f = field.reshape(field.shape[0], -1).astype(np.float64)
        return (f ** 2).mean(1)

    def verify_solver(self):
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import burgers_1d
        o = self.options
        nu, n = o["nu"], 128
        x = np.arange(n) / n

        def exact(t):                                          # Cole-Hopf: phi = 2 + exp(-nu 4 pi^2 t) cos(2 pi x)
            e = np.exp(-nu * 4 * np.pi ** 2 * t)
            return 4 * np.pi * nu * e * np.sin(2 * np.pi * x) / (2 + e * np.cos(2 * np.pi * x))
        errs = []
        for dt in (0.02, 0.01, 0.005):
            _, u = burgers_1d(exact(0.0), nu, 0.5, dt=dt)
            errs.append(float(np.abs(u[0, -1] - exact(0.5)).max()))
        order = math.log2(errs[0] / errs[1]) if errs[1] > 0 else float("inf")
        ok = errs[-1] < 1e-8 or order > 3.5
        return [Check("Burgers vs Cole-Hopf exact solution", "problem", self.id, errs[-1] < 1e-6, value=f"{errs[-1]:.2e}",
                      threshold=1e-6, detail=f"max error at t = 0.5 for dt = 0.02, 0.01, 0.005: {[f'{e:.1e}' for e in errs]}"),
                Check("ETDRK4 observed order in time (>= 3.5 or at round-off)", "problem", self.id, ok,
                      value=round(order, 2) if math.isfinite(order) else "round-off", threshold=3.5)]

    def validation(self, out):
        from pinneapple_physics.physics_audit import advection_dominated
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import (
            burgers_1d,
            periodic_grf,
        )

        from ..validation import field_validation, gci_check, physics_audit_check
        o = self.options

        def study():                                           # energy at T on 3 grids, one input of the train law
            u0 = periodic_grf(1, (o["n_solve"] * 2,), alpha=o["alpha"], tau=o["tau"], sigma=o["sigma"], seed=99)[0]
            vals = []
            for k in (1, 2, 4):
                _, u = burgers_1d(u0[::k], o["nu"], o["t_end"], dt=o["dt"])
                vals.append(float((u[0, -1] ** 2).mean()))
            n = o["n_solve"] * 2
            return gci_check(self.id, "the energy at T", vals, [1 / n, 2 / n, 4 / n])
        amp = float(np.abs(periodic_grf(8, (o["n_solve"],), alpha=o["alpha"], tau=o["tau"], sigma=o["sigma"], seed=1)).max())
        return field_validation(self, out, study) + [physics_audit_check(
            self.id, "viscous Burgers, steep fronts", "solver", ["u_t + u u_x = nu u_xx"], [advection_dominated(10.0)],
            {"Pe": amp / o["nu"]})]

    def pinn_setup(self, x_in, cfg):
        import torch

        from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, PINNProblemSpec
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import FourierInterpolant
        nu = self.options["nu"]
        init = FourierInterpolant(x_in)
        spec = PINNProblemSpec(
            pde_residuals=["Derivative(u(x, t), t) + u(x, t)*Derivative(u(x, t), x) - nu*Derivative(u(x, t), (x, 2))"],
            independent_vars=["x", "t"], dependent_vars=["u"], constants={"nu": nu})
        n = x_in.shape[0]
        xg = (torch.arange(n, dtype=torch.float32) / n).reshape(-1, 1)

        def sample(m, g):
            return (torch.rand(m, 1, generator=g).requires_grad_(True), torch.rand(m, 1, generator=g).requires_grad_(True))

        def evaluate(model):
            with torch.no_grad():
                return model(xg, torch.ones_like(xg)).numpy().reshape(n)

        class Exact(torch.nn.Module):                          # Cole-Hopf solution of the same PDE
            def forward(self, X):
                x, t = X[:, :1], X[:, 1:2]
                e = torch.exp(-nu * 4 * math.pi ** 2 * t)
                return 4 * math.pi * nu * e * torch.sin(2 * math.pi * x) / (2 + e * torch.cos(2 * math.pi * x))
        return PINN(HardNet("burgers", cfg, init)), spec, sample, evaluate, PINN(Exact())


# ====================================================================== Darcy
@register_problem("darcy")
class Darcy(_GRFProblem):
    title = "Darcy 2-D"
    qoi_label = "Compliance ∫u"
    chunk_fn = staticmethod(_darcy_chunk)
    space_fno = {"width": [24, 32], "modes": [12, 16], "lr": [1e-3, 3e-3]}

    def default_options(self):
        return {**super().default_options(), "n_solve": 128, "n_out": 64, "alpha": 2.0, "tau": 3.0, "tau_ood": 6.0,
                "high": 12.0, "low": 3.0}

    def inputs(self, split, n):
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import periodic_grf
        o = self.options
        tau = o["tau_ood"] if split == "ood" else o["tau"]
        return periodic_grf(n, (o["n_solve"],) * 2, alpha=o["alpha"], tau=tau, seed=SPLIT_SEEDS[split])

    def comparison_labels(self):
        return "u", ""                                     # the pressure head, dimensionless: -div(a grad u) = 1

    def verify_solver(self):
        from pinneapple_analysis.verification.convergence import richardson_extrapolate
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import darcy_2d
        errs, mids = [], []
        for n in (16, 32, 64):
            h = 1.0 / n
            xc = (np.arange(n) + 0.5) * h
            X, Y = np.meshgrid(xc, xc, indexing="ij")
            a = 1 + X
            ex = np.sin(np.pi * X) * np.sin(np.pi * Y)
            f = -(np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y)) + a * 2 * np.pi ** 2 * ex
            u = darcy_2d(a, f)
            errs.append(float(np.abs(u - ex).max()))
            mids.append(float(u.mean()))
        cr = richardson_extrapolate(mids[0], mids[1], mids[2], 2.0)
        order = math.log2(errs[1] / errs[2])
        return [Check("Darcy manufactured solution, observed order (max error)", "problem", self.id, order > 1.8,
                      value=round(order, 2), threshold=1.8, detail=f"max errors {[f'{e:.1e}' for e in errs]}"),
                Check("Darcy grid convergence of the mean (Richardson / GCI)", "problem", self.id,
                      cr.observed_order > 1.5, value=f"p = {cr.observed_order:.2f}, GCI_fine = {cr.gci_fine:.2e}",
                      threshold="p > 1.5", detail="pinneapple_analysis.verification.richardson_extrapolate")]

    def validation(self, out):
        from pinneapple_physics.physics_audit import diffusion_dominated
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import (
            darcy_2d,
            periodic_grf,
            threshold_coefficient,
        )

        from ..validation import field_validation, gci_check, physics_audit_check
        o = self.options

        def study():                                           # mean u on 3 grids, the same coefficient field pooled
            n = o["n_solve"] * 2
            a = threshold_coefficient(periodic_grf(1, (n, n), alpha=o["alpha"], tau=o["tau"], seed=98)[0], o["high"], o["low"])
            vals = []
            for k in (1, 2, 4):
                ak = a.reshape(n // k, k, n // k, k).mean(axis=(1, 3))
                vals.append(float(darcy_2d(ak).mean()))
            return gci_check(self.id, "the mean of u", vals, [1 / n, 2 / n, 4 / n])
        return field_validation(self, out, study) + [physics_audit_check(
            self.id, "Darcy flow", "solver", ["-div(a grad u) = 1"], [diffusion_dominated(1.0)], {"Pe": 0.0})]

    def pinn_setup(self, x_in, cfg):
        import torch

        from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, PINNProblemSpec
        a = Bilinear(x_in)
        spec = PINNProblemSpec(
            pde_residuals=["a(x, y)*(Derivative(u(x, y), (x, 2)) + Derivative(u(x, y), (y, 2)))"
                           " + ax(x, y)*Derivative(u(x, y), x) + ay(x, y)*Derivative(u(x, y), y) + 1"],
            independent_vars=["x", "y"], dependent_vars=["u"], exogenous={"a": a.value, "ax": a.dx, "ay": a.dy})
        n = x_in.shape[0]
        c = (torch.arange(n, dtype=torch.float32) + 0.5) / n
        X, Y = torch.meshgrid(c, c, indexing="ij")

        def sample(m, g):
            return (torch.rand(m, 1, generator=g).requires_grad_(True), torch.rand(m, 1, generator=g).requires_grad_(True))

        def evaluate(model):
            with torch.no_grad():
                return model(X.reshape(-1, 1), Y.reshape(-1, 1)).numpy().reshape(n, n)
        exact = None
        if np.allclose(x_in, x_in.flat[0]):                       # constant coefficient: u = x(1 - x) / (2a)
            av = float(x_in.flat[0])

            class Exact(torch.nn.Module):
                def forward(self, Xy):
                    return Xy[:, :1] * (1 - Xy[:, :1]) / (2 * av)
            exact = PINN(Exact())
        return PINN(HardNet("darcy", cfg)), spec, sample, evaluate, exact

    def checks_pinn(self, prepared, cfg):
        """The residual gate needs an exact solution: run it on a constant coefficient (a = 3) of the same PDE."""
        from .fields import pinn_checks
        C = pinn_checks(self, prepared, cfg)
        import torch

        from pinneapple_physics.pinn_solver.factory.pinn_factory import PINNFactory
        _, spec, sample, _, exact = self.pinn_setup(np.full((self.options["n_out"],) * 2, 3.0), cfg)
        g = torch.Generator().manual_seed(0)
        batch = {"collocation": tuple(t.detach().clone().requires_grad_(True) for t in sample(512, g))}
        res = float(PINNFactory(spec).generate_loss_function()(exact, batch)[1]["pde"])
        C = [c for c in C if c.name != "PDE residual of an exact solution ~ 0"]
        C.append(Check("PDE residual of an exact solution ~ 0", "architecture", f"{self.id}/pinn", res < 1e-6,
                       value=f"{res:.2e}", threshold=1e-6, detail="a = 3 everywhere, u = x(1 - x)/(2a)"))
        return C


# ====================================================================== Navier-Stokes
@register_problem("navier_stokes")
class NavierStokes(_GRFProblem):
    title = "Navier-Stokes 2-D"
    qoi_label = "Enstrophy ∫ω²"
    chunk_fn = staticmethod(_ns_chunk)
    space_fno = {"width": [24, 32], "modes": [12, 16], "lr": [1e-3, 3e-3]}

    def default_options(self):
        return {**super().default_options(), "nu": 1e-3, "t_end": 10.0, "n": 64, "dt": 5e-3, "alpha": 2.5,
                "tau": 7.0, "tau_ood": 14.0}

    def inputs(self, split, n):
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import periodic_grf
        o = self.options
        tau = o["tau_ood"] if split == "ood" else o["tau"]
        return periodic_grf(n, (o["n"],) * 2, alpha=o["alpha"], tau=tau, seed=SPLIT_SEEDS[split])

    def qoi(self, field):
        f = field.reshape(field.shape[0], -1).astype(np.float64)
        return (f ** 2).mean(1)

    def comparison_labels(self):
        return "ω", "1/s"                                  # the vorticity, the field the operator maps to

    def verify_solver(self):
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import (
            li_forcing,
            navier_stokes_2d_vorticity,
        )
        n, nu, t = 32, 0.01, 1.0
        x = np.arange(n) / n
        X, Y = np.meshgrid(x, x, indexing="ij")
        w0 = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        _, w = navier_stokes_2d_vorticity(w0, nu, t, dt=0.01)
        err = float(np.abs(w[0, -1] - w0 * np.exp(-nu * 8 * np.pi ** 2 * t)).max())
        o = self.options
        f = li_forcing(o["n"])                                    # steady state of the forcing mode: w = f / (8 pi^2 nu)
        ws = f / (8 * np.pi ** 2 * o["nu"])
        _, w2 = navier_stokes_2d_vorticity(ws, o["nu"], 1.0, forcing=f, dt=o["dt"])
        err2 = float(np.abs(w2[0, -1] - ws).max() / np.abs(ws).max())
        return [Check("Taylor-Green vortex decays exactly", "problem", self.id, err < 1e-8, value=f"{err:.1e}",
                      threshold=1e-8),
                Check("forced steady state is steady (this config)", "problem", self.id, err2 < 1e-8, value=f"{err2:.1e}",
                      threshold=1e-8, detail=f"nu = {o['nu']}, dt = {o['dt']}, grid {o['n']}^2")]

    def validation(self, out):
        from pinneapple_physics.physics_audit import continuum, laminar
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import (
            li_forcing,
            navier_stokes_2d_vorticity,
            periodic_grf,
        )

        from ..validation import field_validation, gci_check, physics_audit_check
        o = self.options

        def study():                                           # enstrophy at T on 3 grids (n/2, n, 2n)
            n2 = o["n"] * 2
            w0 = periodic_grf(1, (n2, n2), alpha=o["alpha"], tau=o["tau"], seed=97)[0]
            vals = []
            for k in (1, 2, 4):
                _, w = navier_stokes_2d_vorticity(w0[::k, ::k], o["nu"], o["t_end"], forcing=li_forcing(n2 // k),
                                                  dt=o["dt"] * k / 2)
                vals.append(float((w[0, -1] ** 2).mean()))
            return gci_check(self.id, "the enstrophy at T", vals, [1 / n2, 2 / n2, 4 / n2])
        return field_validation(self, out, study) + [physics_audit_check(
            self.id, "2-D incompressible Navier-Stokes", "solver", ["w_t + u.grad w = nu lap w + f"],
            [laminar(re_crit=2300.0, geometry="periodic"), continuum()], {"Re": 1.0 / o["nu"], "Kn": 0.0})]

    def pinn_setup(self, x_in, cfg):
        import torch

        from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, PINNProblemSpec
        from pinneapple_simulation.numerical_solvers.operator_benchmarks import (
            FourierInterpolant,
            li_forcing,
        )
        o = self.options
        nu, T = o["nu"], o["t_end"]
        init = FourierInterpolant(x_in)
        n = x_in.shape[0]
        f = FourierInterpolant(li_forcing(n))
        spec = PINNProblemSpec(
            pde_residuals=[
                "Derivative(w(x, y, t), t) + Derivative(psi(x, y, t), y)*Derivative(w(x, y, t), x)"
                " - Derivative(psi(x, y, t), x)*Derivative(w(x, y, t), y)"
                " - nu*(Derivative(w(x, y, t), (x, 2)) + Derivative(w(x, y, t), (y, 2))) - f(x, y)",
                "Derivative(psi(x, y, t), (x, 2)) + Derivative(psi(x, y, t), (y, 2)) + w(x, y, t)"],
            independent_vars=["x", "y", "t"], dependent_vars=["w", "psi"], constants={"nu": nu},
            exogenous={"f": lambda x, y, t: f(x, y).detach()})
        g1 = torch.arange(n, dtype=torch.float32) / n
        X, Y = torch.meshgrid(g1, g1, indexing="ij")

        def sample(m, g):
            return (torch.rand(m, 1, generator=g).requires_grad_(True), torch.rand(m, 1, generator=g).requires_grad_(True),
                    (T * torch.rand(m, 1, generator=g)).requires_grad_(True))

        def evaluate(model):
            with torch.no_grad():
                xx, yy = X.reshape(-1, 1), Y.reshape(-1, 1)
                return model(xx, yy, torch.full_like(xx, T))[:, 0].numpy().reshape(n, n)

        class Exact(torch.nn.Module):                          # steady forced mode: w = f/(8 pi^2 nu), psi = w/(8 pi^2)
            def forward(self, X3):
                s = 2 * math.pi * (X3[:, :1] + X3[:, 1:2])
                fv = 0.1 * (torch.sin(s) + torch.cos(s))
                w = fv / (8 * math.pi ** 2 * nu)
                return torch.cat([w, w / (8 * math.pi ** 2)], 1)
        return PINN(HardNet("navier_stokes", cfg, init, t_end=T)), spec, sample, evaluate, PINN(Exact())


# ====================================================================== PINN building blocks
class Bilinear:
    """Bilinear interpolant of a cell-centred n x n field on the unit square and its gradient; values outside the
    outermost centres are held constant. No autograd: the coefficient does not depend on the network."""

    def __init__(self, a: np.ndarray):
        import torch
        self.a = torch.from_numpy(np.asarray(a, np.float32))
        self.n = a.shape[0]

    def _locate(self, x, y):
        n = self.n
        sx, sy = x.detach().reshape(-1) * n - 0.5, y.detach().reshape(-1) * n - 0.5
        gx, gy = sx.clamp(0, n - 1 - 1e-6), sy.clamp(0, n - 1 - 1e-6)
        i, j = gx.floor().long().clamp(max=n - 2), gy.floor().long().clamp(max=n - 2)
        fx, fy = gx - i, gy - j
        A = self.a
        return A[i, j], A[i + 1, j], A[i, j + 1], A[i + 1, j + 1], fx, fy, (sx > 0) & (sx < n - 1), (sy > 0) & (sy < n - 1)

    def value(self, x, y):
        a00, a10, a01, a11, fx, fy, _, _ = self._locate(x, y)
        return ((1 - fx) * (1 - fy) * a00 + fx * (1 - fy) * a10 + (1 - fx) * fy * a01 + fx * fy * a11).reshape(-1, 1)

    def dx(self, x, y):
        a00, a10, a01, a11, fx, fy, ix, _ = self._locate(x, y)
        return ((((1 - fy) * (a10 - a00) + fy * (a11 - a01)) * self.n) * ix).reshape(-1, 1)

    def dy(self, x, y):
        a00, a10, a01, a11, fx, fy, _, iy = self._locate(x, y)
        return ((((1 - fx) * (a01 - a00) + fx * (a11 - a10)) * self.n) * iy).reshape(-1, 1)


def HardNet(problem: str, cfg: dict, init=None, t_end: float = 10.0):
    """A library NeuralNetwork with input features and hard initial/boundary conditions:

    burgers: (x, t) -> u = u0(x) + t N(sin 2pi kx, cos 2pi kx, 2t - 1), k = 1, 2
    darcy: (x, y) -> u = 16 x(1-x) y(1-y) 0.1 N(2x - 1, 2y - 1)
    navier_stokes: (x, y, t) -> w = w0(x, y) + (t/T) N_w, psi = N_psi / (4 pi^2)
    """
    import torch
    import torch.nn as nn

    from pinneapple_physics.pinn_solver.factory.pinn_factory import NeuralNetwork

    class _HardNet(nn.Module):
        def __init__(self):
            super().__init__()
            n_in = {"burgers": 5, "darcy": 2, "navier_stokes": 9}[problem]
            n_out = 2 if problem == "navier_stokes" else 1
            self.mlp = NeuralNetwork(num_inputs=n_in, num_outputs=n_out, num_layers=cfg["depth"],
                                     num_neurons=cfg["width"], activation=nn.Tanh())

        @staticmethod
        def _per(z):
            w = 2 * math.pi
            return [torch.sin(w * z), torch.cos(w * z), torch.sin(2 * w * z), torch.cos(2 * w * z)]

        def forward(self, X):
            if problem == "burgers":
                x, t = X[:, :1], X[:, 1:2]
                return init(x) + t * self.mlp(torch.cat(self._per(x) + [2 * t - 1], 1))
            if problem == "darcy":
                x, y = X[:, :1], X[:, 1:2]
                return 16 * x * (1 - x) * y * (1 - y) * 0.1 * self.mlp(torch.cat([2 * x - 1, 2 * y - 1], 1))
            x, y, t = X[:, :1], X[:, 1:2], X[:, 2:3]
            n = self.mlp(torch.cat(self._per(x) + self._per(y) + [2 * t / t_end - 1], 1))
            return torch.cat([init(x, y) + (t / t_end) * n[:, :1], n[:, 1:2] / (4 * math.pi ** 2)], 1)
    return _HardNet()
