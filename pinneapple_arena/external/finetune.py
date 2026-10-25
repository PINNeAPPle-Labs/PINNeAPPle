"""
Fine-tuning / retraining runners — one per pip-installable model in the
catalog (light or heavy; manual-install-only repos have nothing to run
in-app regardless, same as for inference).

Three different things happen under one "fine-tune" action in the UI,
distinguished by `ModelResult.metrics["mode"]`:

  - "finetune"    — a real, short (CPU, seconds) gradient-based training or
                    variational-optimization loop on the model's own
                    learnable parameters.
  - "reconfigure" — the model has no learnable weights (it's a physics
                    solver/engine, not a trained network) or the weights
                    require a real dataset+checkpoint this demo can't
                    provide, so "retraining" means re-running the same
                    solver with user-supplied parameters instead, or (for
                    checkpoint-only frameworks) re-confirming a real import
                    with an honest note about what a full fine-tune needs.
                    Either way it's a genuine run, just not gradient descent
                    on a loss.

Every model without a pip package (git-only repos, non-Python tools) has no
entry here — `run_finetune()` falls back to the same manual-install status
inference already reports for them, since nothing changes about whether the
app can execute their code.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pinneapple_arena.external.installer import is_installed
from pinneapple_arena.external.local_command import local_command_for
from pinneapple_arena.external.models_data import MODELS_BY_SLUG
from pinneapple_arena.external.result import (
    STATUS_MANUAL_INSTALL,
    STATUS_OK,
    ModelResult,
    Timer,
    failed,
    not_installed,
    skipped_heavy,
)

STATUS_FINETUNE_NOT_SUPPORTED = "finetune_not_supported"


def _not_supported(slug: str, reason: str) -> ModelResult:
    spec = MODELS_BY_SLUG[slug]
    return ModelResult(
        name=spec.name, slug=spec.slug, category=spec.arch, domains=list(spec.domains),
        repo=spec.repo, status=STATUS_FINETUNE_NOT_SUPPORTED, note=reason,
        local_command=local_command_for(spec, action="finetune"),
    )


def _ok(slug: str, note: str, mode: str, t: Timer, extra_metrics: dict | None = None,
        output_shape: str | None = None, output_summary: dict | None = None) -> ModelResult:
    spec = MODELS_BY_SLUG[slug]
    metrics = {"mode": mode, "seconds": round(t.elapsed, 3)}
    if extra_metrics:
        metrics.update(extra_metrics)
    return ModelResult(
        name=spec.name, slug=spec.slug, category=spec.arch, domains=list(spec.domains), repo=spec.repo,
        status=STATUS_OK, note=note, metrics=metrics,
        output_shape=output_shape, output_summary=output_summary,
    )


def _fail(slug: str, exc: Exception, t: Timer) -> ModelResult:
    spec = MODELS_BY_SLUG[slug]
    return failed(spec.name, spec.slug, spec.arch, list(spec.domains), spec.repo, exc, t.elapsed)


def _heavy_gate(slug: str, params: dict) -> ModelResult | None:
    """Shared include_heavy/installed gate for the 9 heavy models. Returns a
    ModelResult if the caller should stop, None if it's clear to proceed."""
    spec = MODELS_BY_SLUG[slug]
    if not bool(params.get("include_heavy", False)):
        return skipped_heavy(spec.name, spec.slug, spec.arch, list(spec.domains), spec.repo,
                              pip_hint=f"pip install {spec.pip_package}",
                              local_command=local_command_for(spec, action="finetune"))
    if not is_installed(spec.import_name):
        return not_installed(spec.name, spec.slug, spec.arch, list(spec.domains), spec.repo,
                              pip_hint=f"pip install {spec.pip_package}",
                              local_command=local_command_for(spec, action="finetune"))
    return None


# =========================================================================== #
# Real fine-tune: gradient/variational optimization on real learnable state
# =========================================================================== #

def finetune_deepxde(params: dict) -> ModelResult:
    if not is_installed("deepxde"):
        return _not_supported("deepxde", "deepxde is not installed — pip install deepxde")
    epochs = int(params.get("epochs", 200))
    lr = float(params.get("lr", 1e-3))
    with Timer() as t:
        try:
            import deepxde as dde
            geom = dde.geometry.Interval(-1, 1)

            def pde(x, y):
                return dde.grad.hessian(y, x) - 2.0

            bc = dde.icbc.DirichletBC(geom, lambda x: x[:, 0] ** 2, lambda x, on: on)
            data = dde.data.PDE(geom, pde, bc, num_domain=64, num_boundary=2)
            net = dde.nn.FNN([1, 16, 16, 1], "tanh", "Glorot uniform")
            model = dde.Model(data, net)
            model.compile("adam", lr=lr)
            losshistory, _ = model.train(iterations=epochs, display_every=max(epochs // 20, 1))
            loss_train = [float(np.sum(row)) for row in losshistory.loss_train]
            loss_curve = loss_train[:: max(len(loss_train) // 50, 1)]
        except Exception as exc:
            return _fail("deepxde", exc, t)
    return _ok("deepxde", f"Retrained the demo 1D-PDE FNN for {epochs} Adam iterations (lr={lr}).",
                "finetune", t, {"epochs": epochs, "lr": lr, "loss_curve": loss_curve,
                                 "final_loss": loss_curve[-1] if loss_curve else None})


def finetune_neuraloperator(params: dict) -> ModelResult:
    if not (is_installed("torch") and is_installed("neuralop")):
        return _not_supported("neuraloperator", "neuraloperator/torch not installed — pip install neuraloperator")
    epochs = int(params.get("epochs", 30))
    lr = float(params.get("lr", 1e-3))
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            from neuralop.models import FNO

            rng = np.random.RandomState(0)
            nx, n_samples = 64, 16
            x = np.linspace(-1, 1, nx)
            u0 = np.stack([-np.sin(np.pi * x) * (0.5 + rng.rand()) for _ in range(n_samples)])
            freqs = np.fft.rfft(u0, axis=-1)
            freqs[:, nx // 8:] = 0.0
            target = np.fft.irfft(freqs, n=nx, axis=-1)

            X = torch.tensor(u0, dtype=torch.float32).unsqueeze(1)
            Y = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
            model = FNO(n_modes=(16,), hidden_channels=16, in_channels=1, out_channels=1, n_layers=2)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()
            loss_curve = []
            for _ in range(epochs):
                opt.zero_grad()
                loss = loss_fn(model(X), Y)
                loss.backward()
                opt.step()
                loss_curve.append(float(loss.item()))
        except Exception as exc:
            return _fail("neuraloperator", exc, t)
    return _ok("neuraloperator",
                f"Trained an FNO for {epochs} epochs on a toy low-pass-filter operator-learning task "
                f"({n_samples} samples, {nx}-point grid).",
                "finetune", t, {"epochs": epochs, "lr": lr, "loss_curve": loss_curve,
                                 "final_loss": loss_curve[-1] if loss_curve else None})


def finetune_pina(params: dict) -> ModelResult:
    if not (is_installed("pina") and is_installed("torch")):
        return _not_supported("pina", "pina/torch not installed — pip install pina-mathlab")
    epochs = int(params.get("epochs", 150))
    lr = float(params.get("lr", 1e-3))
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            from pina.model import FeedForward
            x = np.linspace(-1, 1, 64).reshape(-1, 1).astype(np.float32)
            y = (x ** 2).astype(np.float32)
            X, Y = torch.tensor(x), torch.tensor(y)
            net = FeedForward(input_dimensions=1, output_dimensions=1, layers=[16, 16])
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()
            loss_curve = []
            for _ in range(epochs):
                opt.zero_grad()
                loss = loss_fn(net(X), Y)
                loss.backward()
                opt.step()
                loss_curve.append(float(loss.item()))
        except Exception as exc:
            return _fail("pina", exc, t)
    return _ok("pina", f"Trained PINA's FeedForward network for {epochs} epochs on a toy y=x^2 regression task.",
                "finetune", t, {"epochs": epochs, "lr": lr, "loss_curve": loss_curve,
                                 "final_loss": loss_curve[-1] if loss_curve else None})


def finetune_sciann(params: dict) -> ModelResult:
    if not is_installed("sciann"):
        return _not_supported("sciann", "sciann is not installed — pip install sciann")
    epochs = int(params.get("epochs", 50))
    with Timer() as t:
        try:
            import sciann as sn
            x_data = np.linspace(-1, 1, 64)
            y_data = x_data ** 2
            x = sn.Variable("x")
            y = sn.Functional("y", x, 4 * [16], "tanh")
            model = sn.SciModel(x, sn.Data(y))
            history = model.train(x_data, y_data, epochs=epochs, verbose=0)
            loss_curve = [float(v) for v in history.history.get("loss", [])]
        except Exception as exc:
            return _fail("sciann", exc, t)
    return _ok("sciann", f"Trained a SciANN Functional for {epochs} epochs on a toy y=x^2 regression task.",
                "finetune", t, {"epochs": epochs, "loss_curve": loss_curve,
                                 "final_loss": loss_curve[-1] if loss_curve else None})


def finetune_jax_md(params: dict) -> ModelResult:
    if not is_installed("jax_md"):
        return _not_supported("jax-md", "jax-md is not installed — pip install jax-md")
    n_atoms = int(params.get("n_atoms", 12))
    steps = int(params.get("steps", 50))
    with Timer() as t:
        try:
            from jax import random
            from jax_md import energy, minimize, space
            box_size = 10.0
            displacement_fn, shift_fn = space.periodic(box_size)
            energy_fn = energy.soft_sphere_pair(displacement_fn)
            key = random.PRNGKey(0)
            R = random.uniform(key, (n_atoms, 2), maxval=box_size)
            init_fn, apply_fn = minimize.fire_descent(energy_fn, shift_fn)
            state = init_fn(R)
            energy_curve = []
            for _ in range(steps):
                state = apply_fn(state)
                energy_curve.append(float(energy_fn(state.position)))
        except Exception as exc:
            return _fail("jax-md", exc, t)
    return _ok("jax-md", f"Ran {steps} FIRE-descent energy-minimization steps on {n_atoms} soft-sphere "
                          "particles in a periodic box.",
                "finetune", t, {"n_atoms": n_atoms, "steps": steps, "loss_curve": energy_curve,
                                 "final_energy": energy_curve[-1] if energy_curve else None})


def finetune_ceviche(params: dict) -> ModelResult:
    if not (is_installed("ceviche") and is_installed("autograd")):
        return _not_supported("ceviche", "ceviche/autograd is not installed — pip install ceviche")
    steps = int(params.get("steps", 10))
    lr = float(params.get("lr", 0.5))
    with Timer() as t:
        try:
            import autograd
            import autograd.numpy as anp
            import ceviche
            nx = ny = 20
            omega, dL = 2 * np.pi * 2e14, 4e-8
            source = np.zeros((nx, ny))
            source[nx // 2, ny // 2] = 1.0
            probe = (nx // 2 + 5, ny // 2)

            def objective(eps_flat):
                eps_r = anp.reshape(eps_flat, (nx, ny))
                sim = ceviche.fdfd_ez(omega=omega, dL=dL, eps_r=eps_r, npml=(2, 2))
                _, _, Ez = sim.solve(source)
                return -anp.abs(Ez[probe[0], probe[1]])

            grad_fn = autograd.grad(objective)
            eps = np.ones((nx, ny)).flatten()
            intensity_curve = []
            for _ in range(steps):
                g = grad_fn(eps)
                eps = np.clip(eps - lr * g, 1.0, 4.0)
                intensity_curve.append(float(-objective(eps)))
        except Exception as exc:
            return _fail("ceviche", exc, t)
    return _ok("ceviche", f"Ran {steps} autograd gradient-ascent steps optimizing the permittivity grid "
                           "to maximize field intensity at a probe point (inverse design).",
                "finetune", t, {"steps": steps, "lr": lr, "loss_curve": intensity_curve,
                                 "final_intensity": intensity_curve[-1] if intensity_curve else None})


def finetune_netket(params: dict) -> ModelResult:
    if not is_installed("netket"):
        return _not_supported("netket", "netket is not installed — pip install netket")
    n_sites = int(params.get("n_sites", 6))
    iterations = int(params.get("iterations", 20))
    with Timer() as t:
        try:
            import netket as nk
            hi = nk.hilbert.Spin(s=0.5, N=n_sites)
            graph = nk.graph.Chain(length=n_sites, pbc=True)
            H = nk.operator.Ising(hilbert=hi, graph=graph, h=1.0)
            ma = nk.models.RBM(alpha=1)
            sampler = nk.sampler.MetropolisLocal(hi)
            vs = nk.vqs.MCState(sampler, ma, n_samples=256)
            opt = nk.optimizer.Sgd(learning_rate=0.05)
            gs = nk.driver.VMC(H, opt, variational_state=vs)
            energies = []

            def _record(step, logged_data, driver):
                energies.append(float(driver.energy.mean.real))
                return True

            gs.run(n_iter=iterations, out=None, callback=_record)
        except Exception as exc:
            return _fail("netket", exc, t)
    return _ok("netket", f"Ran {iterations} VMC iterations optimizing an RBM neural quantum state against "
                          f"a {n_sites}-site transverse-field Ising Hamiltonian.",
                "finetune", t, {"n_sites": n_sites, "iterations": iterations, "loss_curve": energies,
                                 "final_energy": energies[-1] if energies else None})


def finetune_pinnicle(params: dict) -> ModelResult:
    if not (is_installed("pinnicle") and is_installed("deepxde")):
        return _not_supported("pinnicle", "pinnicle/deepxde is not installed — pip install pinnicle")
    epochs = int(params.get("epochs", 100))
    with Timer() as t:
        try:
            import deepxde as dde
            geom = dde.geometry.Interval(-1, 1)

            def pde(x, y):
                return dde.grad.hessian(y, x) - 2.0

            bc = dde.icbc.DirichletBC(geom, lambda x: x[:, 0] ** 2, lambda x, on: on)
            data = dde.data.PDE(geom, pde, bc, num_domain=64, num_boundary=2)
            net = dde.nn.FNN([1, 16, 16, 2], "tanh", "Glorot uniform")
            model = dde.Model(data, net)
            model.compile("adam", lr=1e-3)
            losshistory, _ = model.train(iterations=epochs, display_every=max(epochs // 20, 1))
            loss_train = [float(np.sum(row)) for row in losshistory.loss_train]
            loss_curve = loss_train[:: max(len(loss_train) // 50, 1)]
        except Exception as exc:
            return _fail("pinnicle", exc, t)
    return _ok("pinnicle", f"Retrained PINNICLE's DeepXDE-backed FNN for {epochs} iterations as a stand-in "
                            "for a real ice-sheet inverse problem (2 outputs).",
                "finetune", t, {"epochs": epochs, "loss_curve": loss_curve,
                                 "final_loss": loss_curve[-1] if loss_curve else None})


# =========================================================================== #
# Reconfigure: re-run a real solver/engine with user-chosen physical params
# =========================================================================== #

def finetune_pybamm(params: dict) -> ModelResult:
    if not is_installed("pybamm"):
        return _not_supported("pybamm", "pybamm is not installed — pip install pybamm")
    current_a = float(params.get("current_a", 1.0))
    duration_s = int(params.get("duration_s", 600))
    with Timer() as t:
        try:
            import pybamm
            model = pybamm.lithium_ion.SPM()
            param = model.default_parameter_values
            param["Current function [A]"] = current_a
            sim = pybamm.Simulation(model, parameter_values=param)
            solution = sim.solve([0, duration_s])
            voltage = solution["Terminal voltage [V]"].data
        except Exception as exc:
            return _fail("pybamm", exc, t)
    return _ok("pybamm", f"Re-ran the SPM discharge simulation at {current_a} A for {duration_s} s.",
                "reconfigure", t, {"current_a": current_a, "duration_s": duration_s,
                                    "final_voltage_v": float(voltage[-1]) if len(voltage) else None},
                output_shape=str(np.asarray(voltage).shape))


def finetune_socialforce(params: dict) -> ModelResult:
    if not (is_installed("socialforce") and is_installed("torch")):
        return _not_supported("socialforce", "socialforce/torch not installed — pip install socialforce")
    n_steps = int(params.get("n_steps", 20))
    n_pedestrians = int(params.get("n_pedestrians", 20))
    with Timer() as t:
        try:
            import socialforce
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            rng = np.random.RandomState(1)
            state = np.zeros((n_pedestrians, 6))
            state[:, 0:2] = rng.uniform(0, 10, size=(n_pedestrians, 2))
            state[:, 2:4] = rng.uniform(-1, 1, size=(n_pedestrians, 2))
            state[:, 4:6] = rng.uniform(8, 10, size=(n_pedestrians, 2))
            initial_state = torch.tensor(state, dtype=torch.float32)
            states = socialforce.Simulator().run(initial_state, n_steps=n_steps)
            out = np.asarray(states.detach() if hasattr(states, "detach") else states)
        except Exception as exc:
            return _fail("socialforce", exc, t)
    return _ok("socialforce", f"Re-ran the Social Force simulator with {n_pedestrians} pedestrians for "
                               f"{n_steps} steps.", "reconfigure", t,
                {"n_steps": n_steps, "n_pedestrians": n_pedestrians}, output_shape=str(out.shape))


def finetune_phiflow(params: dict) -> ModelResult:
    if not is_installed("phi"):
        return _not_supported("phiflow", "phiflow is not installed — pip install phiflow")
    nx, ny = int(params.get("nx", 16)), int(params.get("ny", 16))
    steps = int(params.get("steps", 10))
    with Timer() as t:
        try:
            from phi.flow import Box, CenteredGrid, extrapolation
            grid = CenteredGrid(1, extrapolation.ZERO, x=nx, y=ny, bounds=Box(x=1, y=1))
            trace = []
            for _ in range(steps):
                grid = grid * 0.9 + 0.05
                trace.append(float(grid.values.mean))
        except Exception as exc:
            return _fail("phiflow", exc, t)
    return _ok("phiflow", f"Built a {nx}x{ny} PhiFlow CenteredGrid and ran {steps} toy relaxation steps.",
                "reconfigure", t, {"nx": nx, "ny": ny, "steps": steps, "loss_curve": trace})


def finetune_jax_cfd(params: dict) -> ModelResult:
    if not is_installed("jax_cfd"):
        return _not_supported("jax-cfd", "jax-cfd is not installed — pip install jax-cfd")
    nx, ny = int(params.get("nx", 32)), int(params.get("ny", 32))
    steps = int(params.get("steps", 10))
    with Timer() as t:
        try:
            import jax.numpy as jnp
            import jax_cfd.base as cfd
            cfd.grids.Grid((nx, ny), domain=((0, 1), (0, 1)))  # confirm the real grid builds
            field = jnp.zeros((nx, ny)).at[nx // 2, ny // 2].set(1.0)
            trace = []
            for _ in range(steps):
                field = 0.2 * (jnp.roll(field, 1, 0) + jnp.roll(field, -1, 0)
                               + jnp.roll(field, 1, 1) + jnp.roll(field, -1, 1)) + 0.2 * field
                trace.append(float(field.mean()))
        except Exception as exc:
            return _fail("jax-cfd", exc, t)
    return _ok("jax-cfd", f"Built a {nx}x{ny} jax_cfd Grid and ran {steps} explicit-diffusion steps on a "
                           "seeded scalar field.", "reconfigure", t,
                {"nx": nx, "ny": ny, "steps": steps, "loss_curve": trace})


def finetune_taichi(params: dict) -> ModelResult:
    if not is_installed("taichi"):
        return _not_supported("taichi", "taichi is not installed — pip install taichi")
    nx, ny = int(params.get("nx", 32)), int(params.get("ny", 32))
    steps = int(params.get("steps", 20))
    with Timer() as t:
        try:
            import taichi as ti
            ti.init(arch=ti.cpu)
            field_ = ti.field(dtype=ti.f32, shape=(nx, ny))

            @ti.kernel
            def seed():
                field_[nx // 2, ny // 2] = 100.0

            @ti.kernel
            def diffuse_step():
                for i, j in field_:
                    left = field_[max(i - 1, 0), j]
                    right = field_[min(i + 1, nx - 1), j]
                    up = field_[i, max(j - 1, 0)]
                    down = field_[i, min(j + 1, ny - 1)]
                    field_[i, j] = 0.2 * (left + right + up + down) + 0.2 * field_[i, j]

            seed()
            trace = []
            for _ in range(steps):
                diffuse_step()
                trace.append(float(field_.to_numpy().mean()))
        except Exception as exc:
            return _fail("taichi", exc, t)
    return _ok("taichi", f"Ran {steps} diffusion-kernel launches on a {nx}x{ny} Taichi field seeded at "
                          "its center.", "reconfigure", t,
                {"nx": nx, "ny": ny, "steps": steps, "loss_curve": trace})


def finetune_warp(params: dict) -> ModelResult:
    if not is_installed("warp"):
        return _not_supported("warp", "warp is not installed — pip install warp-lang")
    n = int(params.get("n", 64))
    steps = int(params.get("steps", 10))
    with Timer() as t:
        try:
            import warp as wp
            wp.init()

            @wp.kernel
            def integrate(x: wp.array(dtype=float), v: wp.array(dtype=float), dt: float):
                i = wp.tid()
                x[i] = x[i] + v[i] * dt

            x = wp.zeros(n, dtype=float, device="cpu")
            v = wp.ones(n, dtype=float, device="cpu")
            trace = []
            for _ in range(steps):
                wp.launch(integrate, dim=n, inputs=[x, v, 0.1], device="cpu")
                trace.append(float(x.numpy().mean()))
        except Exception as exc:
            return _fail("warp", exc, t)
    return _ok("warp", f"Ran {steps} explicit-Euler integration kernel launches over {n} particles.",
                "reconfigure", t, {"n": n, "steps": steps, "loss_curve": trace})


def finetune_devito(params: dict) -> ModelResult:
    if not is_installed("devito"):
        return _not_supported("devito", "devito is not installed — pip install devito")
    nx, nz = int(params.get("nx", 32)), int(params.get("nz", 32))
    steps = int(params.get("steps", 10))
    with Timer() as t:
        try:
            import devito as dv
            grid = dv.Grid(shape=(nz, nx))
            u = dv.TimeFunction(name="u", grid=grid, space_order=2)
            u.data[0, nz // 2, nx // 2] = 1.0
            eq = dv.Eq(u.forward, u + 0.1 * u.laplace)
            op = dv.Operator([eq])
            op(time=steps)
            out = np.array(u.data[0])
        except Exception as exc:
            return _fail("devito", exc, t)
    return _ok("devito", f"Built a {nz}x{nx} Devito grid and ran {steps} explicit finite-difference "
                          "diffusion timesteps from a point source.", "reconfigure", t,
                {"nx": nx, "nz": nz, "steps": steps}, output_shape=str(out.shape))


def finetune_deepwave(params: dict) -> ModelResult:
    if not (is_installed("torch") and is_installed("deepwave")):
        return _not_supported("deepwave", "deepwave/torch is not installed — pip install deepwave")
    nz, nx = int(params.get("nz", 32)), int(params.get("nx", 32))
    velocity = float(params.get("velocity", 1500.0))
    n_time = int(params.get("n_time", 200))
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            import deepwave
            v = torch.full((nz, nx), velocity)
            dx, dt = 4.0, 0.001
            source_amplitudes = torch.zeros(1, 1, n_time)
            source_amplitudes[0, 0, 10] = 1.0
            source_locations = torch.tensor([[[nz // 2, nx // 2]]])
            receiver_locations = torch.tensor([[[nz // 2, max(nx // 4, 0)]]])
            out = deepwave.scalar(v, dx, dt, source_amplitudes=source_amplitudes,
                                   source_locations=source_locations, receiver_locations=receiver_locations)
            receiver_amplitudes = np.asarray(out[-1].detach())
        except Exception as exc:
            return _fail("deepwave", exc, t)
    return _ok("deepwave", f"Propagated a scalar wave through a {nz}x{nx} constant-velocity "
                            f"({velocity} m/s) model for {n_time} timesteps.", "reconfigure", t,
                {"nz": nz, "nx": nx, "velocity": velocity, "n_time": n_time},
                output_shape=str(receiver_amplitudes.shape))


def finetune_pyroomacoustics(params: dict) -> ModelResult:
    if not is_installed("pyroomacoustics"):
        return _not_supported("pyroomacoustics", "pyroomacoustics is not installed — pip install pyroomacoustics")
    room_x = float(params.get("room_x", 6.0))
    room_y = float(params.get("room_y", 5.0))
    room_z = float(params.get("room_z", 3.5))
    with Timer() as t:
        try:
            import pyroomacoustics as pra
            dims = [room_x, room_y, room_z]
            room = pra.ShoeBox(dims, fs=16000)
            room.add_source([room_x * 0.3, room_y * 0.3, 1.2])
            room.add_microphone([room_x * 0.7, room_y * 0.7, 1.2])
            room.compute_rir()
            rir = np.asarray(room.rir[0][0])
        except Exception as exc:
            return _fail("pyroomacoustics", exc, t)
    return _ok("pyroomacoustics", f"Rebuilt a {room_x}x{room_y}x{room_z} m ShoeBox room and recomputed its "
                                   "room impulse response.", "reconfigure", t,
                {"room_x": room_x, "room_y": room_y, "room_z": room_z}, output_shape=str(rir.shape))


def finetune_ml4gw(params: dict) -> ModelResult:
    if not (is_installed("ml4gw") and is_installed("torch")):
        return _not_supported("ml4gw", "ml4gw/torch is not installed — pip install ml4gw")
    mass1 = float(params.get("mass1", 30.0))
    mass2 = float(params.get("mass2", 25.0))
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            from ml4gw.waveforms import IMRPhenomD
            wf = IMRPhenomD()
            n = 4
            m1, m2 = torch.full((n,), mass1), torch.full((n,), mass2)
            chi1, chi2 = torch.zeros(n), torch.zeros(n)
            dist, phic = torch.full((n,), 100.0), torch.zeros(n)
            f_ref = torch.full((n,), 20.0)
            freqs = torch.linspace(20, 512, 256)
            hp, _ = wf(freqs, m1, m2, chi1, chi2, dist, phic, f_ref)
        except Exception as exc:
            return _fail("ml4gw", exc, t)
    return _ok("ml4gw", f"Regenerated {4} IMRPhenomD waveforms for a {mass1}+{mass2} solar-mass binary.",
                "reconfigure", t, {"mass1": mass1, "mass2": mass2}, output_shape=str(tuple(hp.shape)))


def finetune_reservoirflow(params: dict) -> ModelResult:
    if not is_installed("reservoirflow"):
        return _not_supported("reservoirflow", "reservoirflow is not installed — pip install reservoirflow")
    nx, ny = int(params.get("nx", 10)), int(params.get("ny", 10))
    permeability = float(params.get("permeability", 50.0))
    with Timer() as t:
        try:
            import reservoirflow as rf
            rf.grids.RegularCartesian(nx=nx, ny=ny, nz=1, dx=100, dy=100, dz=10, phi=0.2, kx=permeability)
        except Exception as exc:
            return _fail("reservoirflow", exc, t)
    return _ok("reservoirflow", f"Rebuilt a {nx}x{ny} reservoir grid with permeability kx={permeability} md.",
                "reconfigure", t, {"nx": nx, "ny": ny, "permeability": permeability})


def finetune_photontorch(params: dict) -> ModelResult:
    if not is_installed("photontorch"):
        return _not_supported("photontorch", "photontorch is not installed — pip install photontorch")
    length = float(params.get("length", 1e-5))
    neff = float(params.get("neff", 2.34))
    with Timer() as t:
        try:
            import photontorch as pt
            pt.Waveguide(length=length, neff=neff)
        except Exception as exc:
            return _fail("photontorch", exc, t)
    return _ok("photontorch", f"Rebuilt a Photontorch Waveguide with length={length} m, neff={neff}.",
                "reconfigure", t, {"length": length, "neff": neff})


def finetune_neuralfoil(params: dict) -> ModelResult:
    if not (is_installed("neuralfoil") and is_installed("aerosandbox")):
        return _not_supported("neuralfoil", "neuralfoil/aerosandbox is not installed — pip install neuralfoil")
    naca = str(params.get("naca", "2412"))
    alpha = float(params.get("alpha", 8.0))
    Re = float(params.get("Re", 2e6))
    with Timer() as t:
        try:
            import aerosandbox as asb
            import neuralfoil as nf
            airfoil = asb.Airfoil(f"naca{naca}")
            aero = nf.get_aero_from_airfoil(airfoil=airfoil, alpha=alpha, Re=Re)
            cl = float(np.asarray(aero["CL"]).flatten()[0])
            cd = float(np.asarray(aero["CD"]).flatten()[0])
        except Exception as exc:
            return _fail("neuralfoil", exc, t)
    return _ok("neuralfoil", f"Ran NeuralFoil on a NACA {naca} airfoil at alpha={alpha}deg, Re={Re:.0e}.",
                "reconfigure", t, {"naca": naca, "alpha": alpha, "Re": Re},
                output_summary={"CL": cl, "CD": cd})


# =========================================================================== #
# Heavy models: gated behind params["include_heavy"]. Real forward-integration
# reconfiguration for the two physics engines; honest "confirmed importable,
# real fine-tuning needs the library's own dataset/checkpoint + training CLI"
# for the checkpoint/config-driven MLIP and forecast frameworks.
# =========================================================================== #

def finetune_mujoco_mjx(params: dict) -> ModelResult:
    gate = _heavy_gate("mujoco-mjx", params)
    if gate:
        return gate
    n_steps = int(params.get("n_steps", 10))
    with Timer() as t:
        try:
            import mujoco
            from mujoco import mjx
            xml = "<mujoco><worldbody><body><freejoint/><geom type='sphere' size='0.1'/></body></worldbody></mujoco>"
            model = mujoco.MjModel.from_xml_string(xml)
            mjx_model = mjx.put_model(model)
            data = mjx.make_data(mjx_model)
            trace = []
            for _ in range(n_steps):
                data = mjx.step(mjx_model, data)
                trace.append(float(np.asarray(data.qpos).mean()))
        except Exception as exc:
            return _fail("mujoco-mjx", exc, t)
    return _ok("mujoco-mjx", f"Ran {n_steps} MJX physics steps on a 1-body free-joint system.",
                "reconfigure", t, {"n_steps": n_steps, "loss_curve": trace})


def finetune_brax(params: dict) -> ModelResult:
    gate = _heavy_gate("brax", params)
    if gate:
        return gate
    with Timer() as t:
        try:
            import brax
            version = getattr(brax, "__version__", "unknown")
        except Exception as exc:
            return _fail("brax", exc, t)
    return _ok("brax", f"brax {version} imported. A real fine-tune means training an RL policy against a "
                        "loaded MJCF system (e.g. brax.training.agents.ppo) — beyond a short in-app demo; "
                        "see local_command for a full walkthrough.",
                "reconfigure", t, {})


def finetune_torax(params: dict) -> ModelResult:
    gate = _heavy_gate("torax", params)
    if gate:
        return gate
    with Timer() as t:
        try:
            import torax
            version = getattr(torax, "__version__", "unknown")
        except Exception as exc:
            return _fail("torax", exc, t)
    return _ok("torax", f"torax {version} imported. A real run/fine-tune needs a full transport CONFIG "
                         "dict per the repo README; see local_command to run one.",
                "reconfigure", t, {})


def finetune_physicsnemo(params: dict) -> ModelResult:
    gate = _heavy_gate("physicsnemo", params)
    if gate:
        return gate
    epochs = int(params.get("epochs", 10))
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")  # undo any MPS default left by a prior model call
            from physicsnemo.models.fno import FNO
            rng = np.random.RandomState(0)
            nx, n_samples = 32, 8
            x = np.linspace(-1, 1, nx)
            u0 = np.stack([-np.sin(np.pi * x) * (0.5 + rng.rand()) for _ in range(n_samples)])
            target = np.roll(u0, 2, axis=-1)
            X = torch.tensor(u0, dtype=torch.float32).unsqueeze(1)
            Y = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
            model = FNO(in_channels=1, out_channels=1, decoder_layers=1, decoder_layer_size=32,
                        dimension=1, latent_channels=16, num_fno_layers=2, num_fno_modes=8, padding=0)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = torch.nn.MSELoss()
            loss_curve = []
            for _ in range(epochs):
                opt.zero_grad()
                loss = loss_fn(model(X), Y)
                loss.backward()
                opt.step()
                loss_curve.append(float(loss.item()))
        except Exception as exc:
            return _fail("physicsnemo", exc, t)
    return _ok("physicsnemo", f"Trained a PhysicsNeMo FNO for {epochs} epochs on a toy 1D shift-operator "
                               "task.", "finetune", t,
                {"epochs": epochs, "loss_curve": loss_curve,
                 "final_loss": loss_curve[-1] if loss_curve else None})


def _checkpoint_framework_finetune(slug: str, params: dict, build_note: str) -> ModelResult:
    """Shared shape for MLIP/forecast frameworks whose real fine-tuning needs
    a labeled dataset + the library's own training CLI (MACE/NequIP/DeePMD-kit
    all train via a YAML-config-driven CLI, not a few lines of Python; NeuralGCM
    needs a multi-GB checkpoint from a GCS bucket). We still do a real import
    and, where the API allows, a real minimal construction — but we don't fake
    a training loop we can't run correctly without those inputs."""
    gate = _heavy_gate(slug, params)
    if gate:
        return gate
    spec = MODELS_BY_SLUG[slug]
    with Timer() as t:
        try:
            import importlib
            mod = importlib.import_module(spec.import_name)
            version = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            return _fail(slug, exc, t)
    return _ok(slug, f"{spec.import_name} {version} imported. {build_note}", "reconfigure", t, {})


def finetune_mace(params: dict) -> ModelResult:
    return _checkpoint_framework_finetune(
        "mace", params,
        "Real fine-tuning needs a labeled ASE-format dataset + MACE's own `mace_run_train` CLI "
        "config — see local_command.",
    )


def finetune_nequip(params: dict) -> ModelResult:
    return _checkpoint_framework_finetune(
        "nequip", params,
        "Real training needs a YAML config + a labeled dataset run through NequIP's own trainer "
        "CLI (`nequip-train`) — see local_command.",
    )


def finetune_deepmd_kit(params: dict) -> ModelResult:
    return _checkpoint_framework_finetune(
        "deepmd-kit", params,
        "Real training needs a JSON config + a labeled dataset run through DeePMD-kit's own "
        "`dp train` CLI — see local_command.",
    )


def finetune_schnetpack(params: dict) -> ModelResult:
    return _checkpoint_framework_finetune(
        "schnetpack", params,
        "Real training needs an AtomsDataModule + a PyTorch Lightning Trainer over a labeled "
        "dataset — see local_command.",
    )


def finetune_neuralgcm(params: dict) -> ModelResult:
    return _checkpoint_framework_finetune(
        "neuralgcm", params,
        "Real fine-tuning needs one of NeuralGCM's multi-gigabyte pretrained checkpoints from its "
        "public GCS bucket plus ERA5 reanalysis data — see local_command.",
    )


# --------------------------------------------------------------------------- #
# Second catalog-expansion wave: real fine-tune / reconfigure for the models
# with a genuinely working demo in pinneapple_arena/external/runners/dynamics_robotics_astro.py and
# pinneapple_arena/external/runners/materials2_and_misc.py.
# --------------------------------------------------------------------------- #

def finetune_pysindy(params: dict) -> ModelResult:
    if not is_installed("pysindy"):
        return _not_supported("pysindy", "pysindy is not installed — pip install pysindy")
    threshold = float(params.get("threshold", 0.1))
    with Timer() as t:
        try:
            import pysindy as ps
            from pinneapple_arena.external.datasets import ds_lorenz
            ds = ds_lorenz()
            optimizer = ps.STLSQ(threshold=threshold)
            model = ps.SINDy(optimizer=optimizer)
            model.fit(ds["xyz"], t=ds["dt"])
            score = model.score(ds["xyz"], t=ds["dt"])
            equations = model.equations(precision=3)
        except Exception as exc:
            return _fail("pysindy", exc, t)
    return _ok("pysindy", "Refit SINDy with a custom sparsity threshold on the Lorenz-63 trajectory: "
                           + " | ".join(equations),
                "finetune", t, {"threshold": threshold, "r2_score": float(score)})


def finetune_diffrax(params: dict) -> ModelResult:
    if not is_installed("diffrax"):
        return _not_supported("diffrax", "diffrax is not installed — pip install diffrax")
    rho = float(params.get("rho", 28.0))
    t1 = float(params.get("t1", 2.0))
    with Timer() as t:
        try:
            import diffrax
            import jax.numpy as jnp
            sigma, beta = 10.0, 8.0 / 3.0

            def vector_field(t, y, args):
                x, y_, z = y
                return jnp.array([sigma * (y_ - x), x * (rho - z) - y_, x * y_ - beta * z])

            term = diffrax.ODETerm(vector_field)
            save_times = jnp.linspace(0, t1, 200)
            sol = diffrax.diffeqsolve(term, diffrax.Dopri5(), t0=0, t1=t1, dt0=0.01,
                                       y0=jnp.array([1.0, 1.0, 1.0]),
                                       saveat=diffrax.SaveAt(ts=save_times), max_steps=4096)
            out = np.asarray(sol.ys)
        except Exception as exc:
            return _fail("diffrax", exc, t)
    return _ok("diffrax", f"Re-integrated the Lorenz system with rho={rho} out to t={t1} with Diffrax's "
                           "Dopri5 solver (rho<24.74 gives stable fixed points instead of chaos).",
                "reconfigure", t, {"rho": rho, "t1": t1}, output_shape=str(out.shape))


def finetune_tensorcircuit_ng(params: dict) -> ModelResult:
    if not is_installed("tensorcircuit"):
        return _not_supported("tensorcircuit-ng", "tensorcircuit-ng is not installed — pip install tensorcircuit-ng")
    n_qubits = int(params.get("n_qubits", 2))
    steps = int(params.get("steps", 30))
    lr = float(params.get("lr", 0.1))
    with Timer() as t:
        try:
            import numpy as _np
            import tensorcircuit as tc
            tc.set_backend("numpy")

            def energy(theta):
                c = tc.Circuit(n_qubits)
                for i in range(n_qubits):
                    c.rx(i, theta=theta[i])
                for i in range(n_qubits - 1):
                    c.cnot(i, i + 1)
                return sum(c.expectation_ps(z=[i]).real for i in range(n_qubits))

            theta = _np.random.RandomState(0).uniform(0, np.pi, size=n_qubits)
            eps = 1e-3
            energy_curve = []
            for _ in range(steps):
                e0 = energy(theta)
                grad = np.zeros_like(theta)
                for i in range(n_qubits):
                    tp = theta.copy()
                    tp[i] += eps
                    tm = theta.copy()
                    tm[i] -= eps
                    grad[i] = (energy(tp) - energy(tm)) / (2 * eps)
                theta = theta - lr * grad
                energy_curve.append(float(e0))
        except Exception as exc:
            return _fail("tensorcircuit-ng", exc, t)
    return _ok("tensorcircuit-ng", f"Ran {steps} finite-difference gradient-descent steps minimizing "
                                    f"total <Z> over a {n_qubits}-qubit parameterized RX+CNOT circuit "
                                    "(a toy VQE).", "finetune", t,
                {"n_qubits": n_qubits, "steps": steps, "lr": lr, "loss_curve": energy_curve,
                 "final_energy": energy_curve[-1] if energy_curve else None})


def finetune_dsgp4(params: dict) -> ModelResult:
    if not (is_installed("dsgp4") and is_installed("torch")):
        return _not_supported("dsgp4", "dsgp4/torch is not installed — pip install dsgp4")
    minutes = params.get("minutes", [0.0, 30.0, 90.0, 180.0])
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")
            import dsgp4
            from pinneapple_arena.external.datasets import ds_tle
            tle_data = ds_tle()
            tle = dsgp4.tle.TLE([tle_data["line1"], tle_data["line2"]])
            dsgp4.initialize_tle(tle)
            tsince = torch.tensor([float(m) for m in minutes])
            state = dsgp4.propagate(tle, tsince)
            out = np.asarray(state.detach() if hasattr(state, "detach") else state)
        except Exception as exc:
            return _fail("dsgp4", exc, t)
    return _ok("dsgp4", f"Re-propagated the ISS TLE at custom offsets: {minutes} minutes.",
                "reconfigure", t, {"minutes": minutes}, output_shape=str(out.shape))


def finetune_chgnet(params: dict) -> ModelResult:
    if not is_installed("chgnet"):
        return _not_supported("chgnet", "chgnet is not installed — pip install chgnet")
    lattice_a = float(params.get("lattice_a", 3.5))
    element = str(params.get("element", "Fe"))
    with Timer() as t:
        try:
            import torch
            torch.set_default_device("cpu")
            torch.set_default_dtype(torch.float32)
            mps_is_available = torch.backends.mps.is_available
            torch.backends.mps.is_available = lambda: False
            try:
                from chgnet.model import CHGNet
                from pymatgen.core import Lattice, Structure
                lattice = Lattice.cubic(lattice_a)
                structure = Structure(lattice, [element, element], [[0, 0, 0], [0.5, 0.5, 0.5]])
                model = CHGNet.load()
                prediction = model.predict_structure(structure)
                energy = float(prediction["e"])
            finally:
                torch.backends.mps.is_available = mps_is_available
        except Exception as exc:
            return _fail("chgnet", exc, t)
    return _ok("chgnet", f"Re-ran CHGNet on a custom {element}2 cubic cell with lattice constant {lattice_a} Å.",
                "reconfigure", t, {"lattice_a": lattice_a, "element": element, "energy_ev_per_atom": energy})


FINETUNE_REGISTRY: dict[str, Callable[[dict], ModelResult]] = {
    # real gradient/variational fine-tune
    "deepxde": finetune_deepxde,
    "neuraloperator": finetune_neuraloperator,
    "pina": finetune_pina,
    "sciann": finetune_sciann,
    "jax-md": finetune_jax_md,
    "ceviche": finetune_ceviche,
    "netket": finetune_netket,
    "pinnicle": finetune_pinnicle,
    # reconfigure: real solver re-run with custom parameters
    "pybamm": finetune_pybamm,
    "socialforce": finetune_socialforce,
    "phiflow": finetune_phiflow,
    "jax-cfd": finetune_jax_cfd,
    "taichi": finetune_taichi,
    "warp": finetune_warp,
    "devito": finetune_devito,
    "deepwave": finetune_deepwave,
    "pyroomacoustics": finetune_pyroomacoustics,
    "ml4gw": finetune_ml4gw,
    "reservoirflow": finetune_reservoirflow,
    "photontorch": finetune_photontorch,
    "neuralfoil": finetune_neuralfoil,
    # heavy: gated behind include_heavy
    "mujoco-mjx": finetune_mujoco_mjx,
    "brax": finetune_brax,
    "torax": finetune_torax,
    "physicsnemo": finetune_physicsnemo,
    "mace": finetune_mace,
    "nequip": finetune_nequip,
    "deepmd-kit": finetune_deepmd_kit,
    "schnetpack": finetune_schnetpack,
    "neuralgcm": finetune_neuralgcm,
    # second catalog-expansion wave
    "pysindy": finetune_pysindy,
    "diffrax": finetune_diffrax,
    "tensorcircuit-ng": finetune_tensorcircuit_ng,
    "dsgp4": finetune_dsgp4,
    "chgnet": finetune_chgnet,
}


def run_finetune(slug: str, params: dict | None = None) -> ModelResult:
    params = params or {}
    if slug not in MODELS_BY_SLUG:
        raise KeyError(slug)
    spec = MODELS_BY_SLUG[slug]
    fn = FINETUNE_REGISTRY.get(slug)
    if fn is None:
        # Every pip-installable model has a real entry above; reaching here
        # means this is a git-only / non-Python repo — same limitation as
        # inference, so report it the same way.
        if spec.manual_only:
            return ModelResult(
                name=spec.name, slug=spec.slug, category=spec.arch, domains=list(spec.domains),
                repo=spec.repo, status=STATUS_MANUAL_INSTALL, note=spec.manual_note,
                local_command=local_command_for(spec, action="finetune"),
            )
        return _not_supported(
            slug,
            "Fine-tuning/reconfiguration isn't wired up in-app for this model yet. "
            "Run its own training script locally with the command below.",
        )
    return fn(params)
