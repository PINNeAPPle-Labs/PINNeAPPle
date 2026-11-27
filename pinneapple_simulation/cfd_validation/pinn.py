"""A physics-informed network for the lid-driven cavity, scored with the same ruler as the meshes (#272).

The cavity in :mod:`~pinneapple_simulation.cfd_validation.cavity` is solved by OpenFOAM and compared with Ghia's
tables station by station. This module solves the *same* problem with a network trained on the equations alone — no
solved field is ever shown to it — and then walks through exactly the same comparison: the same 17 published
stations, the two wall stations left out, and the same relative L2, computed by
:func:`~pinneapple_simulation.cfd_validation.cavity.profile_error`. A number measured here is therefore comparable
with the number the mesh table quotes at the same Reynolds number; a network scored against the solver's own field
would not be.

The equations, and what the network is asked to satisfy
------------------------------------------------------
Steady, incompressible, non-dimensional (``U = 1``, ``L = 1``, ``rho = 1``), so ``nu = 1/Re``::

    u u_x + v u_y = -p_x + nu (u_xx + u_yy)
    u v_x + v v_y = -p_y + nu (v_xx + v_yy)
    u_x + v_y = 0

on the unit square, with no-slip on the three fixed walls and ``u = 1, v = 0`` on the lid at ``y = 1``. The residual
is the strong form evaluated at collocation points by automatic differentiation, and the boundary conditions are a
penalty (:attr:`Config.bc_weight`) rather than an ansatz: an ansatz that multiplies the output by a function
vanishing on the walls cannot carry a lid moving at ``u = 1`` along the whole top edge, and the smoothed alternative
would be solving a slightly different problem. The pressure is held to zero mean, which is the gauge the equations
leave free.

The operator over Reynolds number
---------------------------------
:func:`train` with ``reynolds=None`` fits **one** network whose input carries the Reynolds number, so a single object
answers for every ``Re`` in the trained range. It is trained where the physics is defined — the collocation points
draw a Reynolds number over ``[100, 1000]`` at every step — and scored only where Ghia published a table, which is the
point of an operator: the three published columns are three *points* of one field.

Three columns are also a thin test of a claim about a whole interval, so the operator is asked twice more, at Reynolds
numbers no column of the table has (:data:`OFF_SAMPLE`, 250 and 750), and scored there against the *solver* by
:func:`against`: same stations, same exclusion of the walls, same norm, with the `simpleFoam` profile in the place of
the published one. Those two numbers are the difference between an operator over ``Re`` and three networks wearing one
hat, and they are the only measurement in this module that could have been an extrapolation — the range of the input
is fixed at training and asking outside it raises.

Where it stops
--------------
The finding this module exists to produce is the Reynolds number at which each model stops matching the reference
(:func:`stops_matching`), not the average error. A network that is good at ``Re = 100`` and useless at ``Re = 1000``
is the state of the art in this problem and is reported as measured. :attr:`Config` was chosen once, on ``Re = 100``
(where a run is cheapest), and the same configuration is used at every Reynolds number: a per-``Re`` search would
measure the search, and a configuration tuned at the hardest Reynolds number would flatter the easy one.

What is not claimed
-------------------
The network is not a solver. It has no discretisation, no residual norm that says "converged", and no guarantee that
what it returns satisfies the equations it was trained on — the physics residual at the end of training is reported
in :attr:`CavityPINN.history` as the network's own claim about itself, and the profiles at Ghia's stations as the
evidence. A run of this module costs tens of minutes of a CPU where ``simpleFoam`` takes seconds; the value of the
number is that it says how well the *method* does against the same table, not that the method is faster.

The cost of a run is recorded with it (``seconds``, ``steps``, the machine class), so it can be read next to the
solver's wall time in the same table.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace

import numpy as np

try:                                                     # torch is a core dependency, but a reader of the tables
    import torch  # should not need it: the guard keeps this module
except ImportError:                                     # importable for the documentation build and for a
    torch = None                                        # machine that only wants ``results/cavity_pinn.json``.

from .cavity import profile_error
from .reference import GHIA_U, GHIA_V, Profile

__all__ = ["REYNOLDS", "OFF_SAMPLE", "MATCH_TOLERANCES", "PROTOCOL", "Config", "CavityPINN", "train", "profiles",
           "measure", "against", "matches", "stops_matching", "study", "reference_stations", "needs_torch"]

#: The Reynolds numbers the models are trained and scored at: the three Table I columns of Ghia, Ghia and Shin.
REYNOLDS = (100, 400, 1000)

#: The Reynolds numbers the operator is asked at that no column of the table has — the two either side of the middle
#: one, where an operator is worth having and a per-``Re`` network has nothing to say. Scored against the solver's own
#: field at the same stations (there is no published profile here), which is the only reference that exists off the
#: table: a number at these two says whether the network is a surrogate over ``Re`` or a fit of three profiles.
OFF_SAMPLE = (250.0, 750.0)

#: What "matches the reference" means here, as the relative L2 of both centreline profiles — the level a solver
#: reaches on a middling mesh in the same table (5 %), and the level of the finest one (1 %).
MATCH_TOLERANCES = (0.05, 0.01)

#: The comparison a number from this module is taken through, written down next to it so the record says how it was
#: scored without the reader having to read the code. ``tests/test_cfd_validation.py`` holds it against the mesh study.
PROTOCOL = {
    "stations": "the 17 published stations of each profile, as Ghia printed them (Table I)",
    "interpolation": "u at x = 0.5 exactly and v at y = 0.5 exactly: the network evaluates on the centreline, the "
                     "mesh interpolates to it from the two neighbouring columns",
    "norm": "relative L2: the RMS of the difference over the RMS of the published values at the compared stations",
    "excluded": "the two wall stations (the published value there is the boundary condition), and the station "
                "reference.disputes() records as mis-transcribed",
    "comparable_with": "the meshes of results/cavity.json, which go through cavity.profile_error",
}


def needs_torch() -> None:
    """Raise if torch is missing, with the one thing a caller can do about it."""
    if torch is None:
        raise ImportError("a PINN needs torch: pip install torch (the committed measurement in "
                          "examples/cfd/classical_cases/results/cavity_pinn.json needs nothing)")


def reference_stations(reynolds: float) -> dict[str, np.ndarray]:
    """Ghia's own ``y`` and ``x`` stations at ``reynolds``, as the arrays a measurement is taken at.

    The same two lists the OpenFOAM sampling interpolates to
    (:func:`~pinneapple_simulation.cfd_validation.cavity.sample`), which is what makes the two numbers comparable.
    """
    u = GHIA_U[int(round(reynolds))]
    v = GHIA_V[int(round(reynolds))]
    return {"y": np.asarray(u.points, dtype=float), "x": np.asarray(v.points, dtype=float),
            "u": np.asarray(u.values, dtype=float), "v": np.asarray(v.values, dtype=float)}


# =============================================================================================== the network
@dataclass(frozen=True)
class Config:
    """Everything a run is defined by, so a number can be reproduced without reading the code that produced it.

    The defaults are the configuration the committed measurement used. :attr:`fourier` defaults to zero because the
    random Fourier map could not be used as it stood: it multiplies the PDE residual by the square of its own
    frequency, and at ``fourier_scale = 2`` the residual at the first step came out at ``8.5e1`` against ``2.1e-1``
    for the plain network, so the run starts a hundred times further from a solution than the thing it was meant to
    improve. Making it pay needs the output scaled to match, which was not measured — so it is not on by default, and
    the README's PINN section says exactly that rather than claiming a result.
    """

    width: int = 64
    depth: int = 4
    fourier: int = 0                        # random Fourier features on the input; 0 is a plain tanh network
    fourier_scale: float = 3.0
    collocation: int = 2000                 # PDE residual points per step
    boundary: int = 250                     # points per wall
    steps: int = 10000                      # the budget the committed measurement had: 36 minutes of a CPU per model
    lr: float = 2e-3
    resample: int = 200                     # the collocation points are redrawn this often
    pde_weight: float = 1.0
    bc_weight: float = 20.0
    gauge_weight: float = 1.0
    balance: bool = False                   # scale the three loss terms by their gradient norms as training runs
    balance_every: int = 200
    curriculum_from: float = 0.0            # start the physics at this Reynolds number and ramp to the target;
    curriculum: float = 0.5                 # 0 is no ramp, and this is the fraction of the run the ramp takes.
                                            # Measured off: the ramp reaches a nicer field halfway and a worse one
                                            # at the end (the README's PINN section has the numbers)
    seed: int = 0
    checkpoint: str = ""                    # where to save and resume a long run; "" trains without touching disk
    dtype: str = "float32"                  # the residual is a second derivative of a small network: float32 is
                                            # enough for a field measured to 1e-3, and twice the throughput

    def to_dict(self) -> dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "checkpoint"}


if torch is not None:                                    # the class body needs torch to exist

    class _MLP(torch.nn.Module):
        """A tanh perceptron, with an optional frozen random-Fourier map on the input."""

        def __init__(self, n_in: int, config: Config) -> None:
            super().__init__()
            gen = torch.Generator().manual_seed(int(config.seed))
            d = n_in
            self.register_buffer("fourier", None)      # registered as None so the map is a buffer when it exists and
            if config.fourier:                         # the module still moves to a device or a dtype as one piece
                self.register_buffer("fourier", torch.randn(n_in, config.fourier, generator=gen) * config.fourier_scale)
                d = 2 * config.fourier
            layers: list[torch.nn.Module] = []
            for _ in range(int(config.depth)):
                layers += [torch.nn.Linear(d, int(config.width)), torch.nn.Tanh()]
                d = int(config.width)
            layers += [torch.nn.Linear(d, 3)]                       # u, v, p
            self.net = torch.nn.Sequential(*layers)
            for m in self.net:
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight, generator=gen)   # the generator, not the global RNG: with
                    torch.nn.init.zeros_(m.bias)                            # the global one ``seed`` does not
                    #                                                       reproduce the initial weights, and a
                    #                                                       configuration that does not reproduce
                    #                                                       its own starting point is not a record

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            if self.fourier is not None:
                q = 2.0 * np.pi * z @ self.fourier
                z = torch.cat([torch.sin(q), torch.cos(q)], dim=-1)
            return self.net(z)

    class CavityPINN(torch.nn.Module):
        """``(x, y) -> (u, v, p)``, or ``(x, y, Re) -> (u, v, p)`` when the network carries the parameter.

        ``parametric`` is what makes it an operator: with it the Reynolds number is an input, and
        :meth:`velocity` refuses a number outside the trained range and a single-``Re`` network refuses any number but
        its own — an extrapolation reported as a result is the one failure mode that would make the measurement
        worthless.
        """

        def __init__(self, config: Config, reynolds: float | None = None, *,
                     reynolds_range: tuple[float, float] = (100.0, 1000.0)) -> None:
            super().__init__()
            needs_torch()
            self.config = config
            self.parametric = reynolds is None
            self.reynolds = None if reynolds is None else float(reynolds)
            self.reynolds_range = (float(min(reynolds_range)), float(max(reynolds_range)))
            self.net = _MLP(2 + int(self.parametric), config)
            self.net.to(getattr(torch, config.dtype))
            self.history: list[dict[str, float]] = []
            self.seconds = 0.0

        # ------------------------------------------------------------------ the input and the forward pass
        def _input(self, x: torch.Tensor, y: torch.Tensor, reynolds: float | None = None) -> torch.Tensor:
            """The network's input. The Reynolds number is an argument and not only :attr:`reynolds`, so that an
            operator can be *asked* at the ``Re`` of a measurement without its state being moved under it."""
            if not self.parametric:
                return torch.cat([x, y], dim=-1)
            re = float(self.reynolds if reynolds is None else reynolds)
            lo, hi = float(np.log10(self.reynolds_range[0])), float(np.log10(self.reynolds_range[1]))
            t = (float(np.log10(re)) - lo) / (hi - lo)
            return torch.cat([x, y, torch.full_like(x, t)], dim=-1)

        def forward(self, x: torch.Tensor, y: torch.Tensor, reynolds: float | None = None) -> torch.Tensor:
            return self.net(self._input(x, y, reynolds))

        # ------------------------------------------------------------------ the answer, for a caller
        @torch.no_grad()
        def velocity(self, x: np.ndarray | float, y: np.ndarray | float, reynolds: float | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
            """``u`` and ``v`` at the points ``(x, y)``, in float64 whatever the network was trained in."""
            if self.parametric:
                if reynolds is None:
                    raise ValueError("this is a parametric network: give the Reynolds number to evaluate it at")
                if not self.reynolds_range[0] <= float(reynolds) <= self.reynolds_range[1]:
                    raise ValueError(
                        f"Re = {float(reynolds):g} is outside the trained range [{self.reynolds_range[0]:g}, "
                        f"{self.reynolds_range[1]:g}]: the network has no answer there, and asking for one would "
                        f"be an extrapolation reported as a result")
            elif reynolds is not None and abs(float(reynolds) - float(self.reynolds or 0.0)) > 1e-9:
                raise ValueError(f"this network was trained at Re = {float(self.reynolds):g} and has no answer at "
                                 f"Re = {float(reynolds):g}")
            xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            real = getattr(torch, self.config.dtype)         # the dtype the weights are in: a float64 network
            xt = torch.tensor(xa.ravel(), dtype=real).reshape(-1, 1)   # fed float32 input raises inside the
            yt = torch.tensor(ya.ravel(), dtype=real).reshape(-1, 1)   # first matmul, not here
            out = self.forward(xt, yt, reynolds)
            return out[:, 0].double().numpy(), out[:, 1].double().numpy()

        def to_dict(self) -> dict[str, object]:
            return {"parametric": self.parametric, "reynolds": self.reynolds,
                    "reynolds_range": list(self.reynolds_range) if self.parametric else None,
                    "config": self.config.to_dict(), "seconds": round(self.seconds, 1),
                    "parameters": int(sum(p.numel() for p in self.parameters())),
                    "history": self.history}

else:                                                    # a machine with no torch still reads the module's data
    class CavityPINN:                                    # type: ignore[no-redef]
        """Placeholder: without torch there is no network. :func:`train` raises with the reason."""

        def __init__(self, *a: object, **k: object) -> None:
            needs_torch()


# =============================================================================================== the physics
def _residual(model: CavityPINN, x: torch.Tensor, y: torch.Tensor, nu: float) -> torch.Tensor:
    """The three strong-form residuals as one ``(n, 3)`` tensor: x-momentum, y-momentum, continuity."""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    out = model(x, y)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    ones = torch.ones_like(u)
    ux, uy = torch.autograd.grad(u, (x, y), ones, create_graph=True)
    vx, vy = torch.autograd.grad(v, (x, y), ones, create_graph=True)
    px, py = torch.autograd.grad(p, (x, y), ones, create_graph=True)
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, torch.ones_like(uy), create_graph=True)[0]
    vxx = torch.autograd.grad(vx, x, torch.ones_like(vx), create_graph=True)[0]
    vyy = torch.autograd.grad(vy, y, torch.ones_like(vy), create_graph=True)[0]
    return torch.stack([u * ux + v * uy + px - nu * (uxx + uyy),
                        u * vx + v * vy + py - nu * (vxx + vyy),
                        ux + vy], dim=-1)


def _wall_points(n: int, gen: torch.Generator, dtype: str = "float32"
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(x, y, u)`` on the four walls: ``v = 0`` everywhere, and ``u = 1`` on the lid at ``y = 1`` only."""
    real = getattr(torch, dtype)                          # a float64 network fed float32 samples raises in the
    t = torch.rand(n, 1, generator=gen, dtype=real)       # first matmul, which is why the dtype is threaded here
    z, o = torch.zeros(n, 1, dtype=real), torch.ones(n, 1, dtype=real)
    return torch.cat([t, z, o, t]), torch.cat([z, t, t, o]), torch.cat([z, z, z, o])


def train(reynolds: float | None = None, *, config: Config | None = None,
          reynolds_range: tuple[float, float] = (100.0, 1000.0), on_step: object = None) -> CavityPINN:
    """Train one network: at a single ``reynolds``, or — with ``reynolds=None`` — the operator over ``reynolds_range``.

    ``on_step(model, step, losses)`` is called every ``config.steps // 10`` steps if given, which is how the runner
    prints a training curve; it is not part of the measurement. ``reynolds_range`` only matters for the parametric
    network, whose input is ``log10 Re`` mapped onto ``[-1, 1]`` over that range and whose collocation points draw a
    fresh Reynolds number from it every step.
    """
    needs_torch()
    cfg = config or Config()
    model = CavityPINN(cfg, reynolds, reynolds_range=reynolds_range)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr / 200)
    gen = torch.Generator().manual_seed(int(cfg.seed) + 1)
    lo, hi = float(np.log10(reynolds_range[0])), float(np.log10(reynolds_range[1]))

    def draw_re(step: int) -> float:
        """The Reynolds number this step's residual is taken at.

        The operator draws a fresh one over the range every step, which is what makes it an operator. A single-``Re``
        network uses its own — unless :attr:`Config.curriculum_from` asked for the ramp, in which case the first
        ``Config.curriculum`` of the run is spent sliding from that easier Reynolds number to the target.
        """
        if reynolds is None:
            return float(10.0 ** (lo + (hi - lo) * float(torch.rand(1, generator=gen))))
        if not cfg.curriculum_from or float(reynolds) <= 0 or cfg.curriculum_from <= 0:
            return float(reynolds)
        span = max(1.0, float(cfg.curriculum) * int(cfg.steps))
        t = min(1.0, step / span)
        start_log = float(np.log10(cfg.curriculum_from))
        return float(10.0 ** (start_log + t * (float(np.log10(reynolds)) - start_log)))
    def set_re(value: float) -> None:
        model.reynolds = value

    real = getattr(torch, cfg.dtype)
    bx, by, bu = _wall_points(int(cfg.boundary), gen, cfg.dtype)
    x = torch.rand(int(cfg.collocation), 1, generator=gen, dtype=real)
    y = torch.rand(int(cfg.collocation), 1, generator=gen, dtype=real)
    every = max(1, int(cfg.steps) // 10)
    w = np.ones(3)                            # the weight of each loss term, changed only by ``balance``
    t0 = time.time()
    start = 0
    if cfg.checkpoint and os.path.exists(cfg.checkpoint):        # a run that was interrupted continues from its
        blob = torch.load(cfg.checkpoint, map_location="cpu")    # last save instead of starting over: on a shared
        model.load_state_dict(blob["model"])                     # machine a 20 000-step training is exactly what a
        opt.load_state_dict(blob["optimizer"])                   # session boundary kills
        sched.load_state_dict(blob["scheduler"])
        w = np.asarray(blob.get("weights", w), dtype=float)
        model.history = list(blob.get("history", []))
        start = int(blob["step"]) + 1
        t0 -= float(blob.get("seconds", 0.0) or 0.0)
        print(f"resuming from step {start - 1}", flush=True)
    for step in range(start, int(cfg.steps)):
        if cfg.resample and step and step % int(cfg.resample) == 0:
            x = torch.rand(int(cfg.collocation), 1, generator=gen, dtype=real)
            y = torch.rand(int(cfg.collocation), 1, generator=gen, dtype=real)
        nu = 1.0 / draw_re(step)
        if model.parametric:
            set_re(1.0 / nu)
        r = _residual(model, x, y, nu)
        m_pde = (r ** 2).mean(dim=0).sum()
        out = model(bx, by)
        m_bc = ((out[:, 0:1] - bu) ** 2).mean() + (out[:, 1:2] ** 2).mean()
        m_g = out[:, 2:3].mean() ** 2
        if cfg.balance and step % int(cfg.balance_every) == 0:
            w = _gradient_weights(model, (cfg.pde_weight * m_pde, cfg.bc_weight * m_bc, cfg.gauge_weight * m_g), w)
        loss = (w[0] * cfg.pde_weight * m_pde + w[1] * cfg.bc_weight * m_bc + w[2] * cfg.gauge_weight * m_g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if cfg.checkpoint and step and (step % every == 0 or step == int(cfg.steps) - 1):
            # the last step is saved as well as the round ones: the save a resumed run wants is the one that makes the
            # next call do nothing, and with only the round saves a 20 000-step model could never be picked up at 20 000
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "weights": [float(v) for v in w], "history": model.history, "step": step,
                        "seconds": time.time() - t0}, cfg.checkpoint)
        if step % every == 0 or step == int(cfg.steps) - 1:
            model.history.append({"step": float(step), "pde": float(m_pde.detach()), "bc": float(m_bc.detach()),
                                  "gauge": float(m_g.detach()), "total": float(loss.detach()),
                                  "weights": [float(v) for v in w],
                                  "reynolds": float(model.reynolds or 0.0), "seconds": round(time.time() - t0, 1)})
            if callable(on_step):
                on_step(model, step, model.history[-1])
    model.seconds = time.time() - t0
    return model


def _gradient_weights(model: CavityPINN, terms: tuple, current: np.ndarray) -> np.ndarray:
    """Balance the three loss terms so the largest gradient norm sets the scale, smoothed against the last setting.

    A penalty formulation with fixed weights spends the whole optimisation on whichever term has the largest gradient
    at the start — here the continuity/pressure pair — and the two momentum equations learn nothing (measured: ``v``
    stays at zero for thousands of steps). Scaling each term by the inverse of its own gradient norm is the fix of
    Wang, Teng and Perdikaris (2021), *Understanding and mitigating gradient pathologies in physics-informed neural
    networks*, and it is what makes the three equations move together. The smoothing (0.9/0.1) keeps the weight from
    jumping when one step's norm happens to be small.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    norms = []
    for term in terms:
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        norms.append(float(sum(float((g ** 2).sum()) for g in grads if g is not None)) ** 0.5 + 1e-30)
    target = np.asarray([max(norms) / n for n in norms])
    return 0.9 * np.asarray(current) + 0.1 * target


# =============================================================================================== the measurement
def profiles(model: CavityPINN, reynolds: float | None = None) -> dict[str, np.ndarray | float]:
    """The network's ``u`` along ``x = 0.5`` and ``v`` along ``y = 0.5``, at Ghia's own stations.

    ``u`` is asked for at ``x = 0.5`` exactly and ``v`` at ``y = 0.5`` exactly, which the mesh cannot do (it
    interpolates to the centreline from the two neighbouring columns, :func:`cavity.sample`); what has to match for
    the two errors to be comparable is the *stations* and the *norm*, and both do.
    """
    re = float(reynolds if reynolds is not None else (model.reynolds or 0.0))
    st = reference_stations(re)
    u, _ = model.velocity(np.full_like(st["y"], 0.5), st["y"], re)
    _, v = model.velocity(st["x"], np.full_like(st["x"], 0.5), re)
    return {"reynolds": re, "y": st["y"], "u": u, "x": st["x"], "v": v}


def measure(model: CavityPINN, reynolds: float | None = None) -> dict[str, object]:
    """One model at one Reynolds number, as the record the case's JSON holds: the stations, the values, the errors.

    The errors are :func:`cavity.profile_error` of the *published* profile against what the network returned, which
    is the identical call the mesh study makes on a solved field — same stations, same exclusion of the two wall
    stations, same RMS of the table in the denominator.
    """
    got = profiles(model, reynolds)
    re = int(round(float(got["reynolds"])))
    errs = {"u": profile_error(GHIA_U[re], np.asarray(got["u"])),
            "v": profile_error(GHIA_V[re], np.asarray(got["v"]))}
    return {"reynolds": float(got["reynolds"]),
            "y": [float(v) for v in got["y"]], "u": [float(v) for v in got["u"]],
            "x": [float(v) for v in got["x"]], "v": [float(v) for v in got["v"]],
            "errors": errs}


#: What the off-table rows are scored against, written into every one of them.
_SOLVER_REFERENCE = ("the case solved by simpleFoam at the same Reynolds number, on the finest mesh of the ladder in "
                     "results/cavity.json (linearUpwind, residual 1e-6), sampled at Ghia's stations by cavity.sample")


def against(model: CavityPINN, reynolds: float, solved: dict[str, np.ndarray],
            *, solver: str = "simpleFoam") -> dict[str, object]:
    """The model at a Reynolds number **no published column has**, scored against a solver's field at the stations.

    ``solved`` is what :func:`cavity.sample` returns for a run at ``reynolds`` — ``y`` and ``u`` along ``x = 0.5``,
    ``x`` and ``v`` along ``y = 0.5`` — and it is the reference here because Ghia publishes nothing at this ``Re``:
    the alternative is to leave the operator untested away from the three columns it is asked to reproduce, which is
    exactly the claim that needs evidence. The comparison is the same one the table gets — the same stations, the
    same exclusion of the two wall stations, the same :func:`cavity.profile_error` — with the solved profile in the
    place of the published one, so the number reads on the same scale. The rows carry both the model's values and the
    solver's, so the error can be recomputed from the record without the mesh being on disk.
    """
    re = float(reynolds)
    u_model, _ = model.velocity(np.full_like(solved["y"], 0.5, dtype=float), solved["y"], re)
    _, v_model = model.velocity(solved["x"], np.full_like(solved["x"], 0.5, dtype=float), re)
    ref_u = Profile(solver, "u", "y", re, tuple(float(p) for p in solved["y"]),
                    tuple(float(v) for v in solved["u"]), _SOLVER_REFERENCE)
    ref_v = Profile(solver, "v", "x", re, tuple(float(p) for p in solved["x"]),
                    tuple(float(v) for v in solved["v"]), _SOLVER_REFERENCE)
    return {"reynolds": re, "reference": solver, "citation": _SOLVER_REFERENCE,
            "y": [float(v) for v in solved["y"]], "u": [float(v) for v in u_model],
            "x": [float(v) for v in solved["x"]], "v": [float(v) for v in v_model],
            "reference_u": [float(v) for v in solved["u"]], "reference_v": [float(v) for v in solved["v"]],
            "errors": {"u": profile_error(ref_u, np.asarray(u_model)),
                       "v": profile_error(ref_v, np.asarray(v_model))}}


def matches(row: dict[str, object], tolerance: float) -> bool:
    """Whether one measured row matches the reference at ``tolerance``: **both** profiles have to clear it."""
    errors = row["errors"]
    return bool(max(float(errors["u"]["l2_relative"]), float(errors["v"]["l2_relative"])) <= tolerance)  # type: ignore[index]


def stops_matching(rows: list[dict[str, object]], tolerance: float) -> dict[str, object]:
    """Where a model stops matching the reference: the last ``Re`` it clears and the first it does not.

    The rows are the measured ones, sorted by Reynolds number; the answer is the *bracket* between two measured
    numbers and never an interpolation, because the reference exists at the published stations only. A model that
    fails at the lowest measured ``Re`` has no bracket, and says so.
    """
    ordered = sorted(rows, key=lambda r: float(r["reynolds"]))
    ok = [float(r["reynolds"]) for r in ordered if matches(r, tolerance)]
    bad = [float(r["reynolds"]) for r in ordered if not matches(r, tolerance)]
    out: dict[str, object] = {"tolerance": float(tolerance), "matches_at": ok, "fails_at": bad}
    if not ok:
        out["stops"] = "below the lowest measured Reynolds number"
    elif not bad:
        out["stops"] = "above the highest measured Reynolds number"
    else:
        out["stops"] = f"between Re = {ok[-1]:g} and Re = {min(bad):g}"
    out["bracket"] = [ok[-1], min(bad)] if ok and bad else None
    return out


def study(reynolds: tuple[int, ...] = REYNOLDS, *, parametric: bool = False, config: Config | None = None,
          on_step: object = None, probe: tuple[float, ...] = (),
          solved: object = None) -> dict[str, object]:
    """Train (one network per ``Re``, or the operator) and measure every model against the published profiles.

    Returns the record the case's ``results/cavity_pinn.json`` holds: what was trained, at what cost, the profiles it
    produced at Ghia's stations, the errors those profiles give, and — the finding — the Reynolds number each model
    stops matching the table at.

    ``probe`` is a set of Reynolds numbers **not** in the table, and ``solved`` the function that returns the solver's
    field at one of them (:func:`cavity.sample` at :data:`OFF_SAMPLE`); the operator is then also scored there, which
    is the measurement that separates an operator from three networks wearing one hat. Only the operator can be asked:
    a network trained at a single ``Re`` refuses another by construction, and that refusal is the point of it.
    """
    cfg = config or Config()
    rows: list[dict[str, object]] = []
    models: list[dict[str, object]] = []
    off: list[dict[str, object]] = []
    if parametric:
        model = train(None, config=_for(cfg, "operator", reynolds), on_step=on_step,
                      reynolds_range=(float(min(reynolds)), float(max(reynolds))))
        models.append(model.to_dict())
        for re in reynolds:
            rows.append(measure(model, re))
        if probe:
            if not callable(solved):
                raise ValueError("a probe Reynolds number needs the solved field to score it against: pass "
                                 "solved=<callable returning cavity.sample's dict>")
            for re in probe:
                off.append(against(model, float(re), solved(float(re))))
    else:
        for re in reynolds:
            model = train(re, config=_for(cfg, "single", re), on_step=on_step)
            models.append(model.to_dict())
            rows.append(measure(model, re))
    return {"what": "parametric operator over Re" if parametric else "one network per Reynolds number",
            "models": models, "rows": rows, "off_sample": off,
            "stops_matching": {f"{t:g}": stops_matching(rows, t) for t in MATCH_TOLERANCES}}


def _for(cfg: Config, mode: str, reynolds: object) -> Config:
    """``cfg`` with its own checkpoint path, so an interrupted study resumes each model where it stopped.

    One path for the whole study would have the operator try to load the weights of a network with a different input
    width, which is a crash in the middle of a long run rather than a resume.
    """
    if not cfg.checkpoint:
        return cfg
    tag = "operator" if mode == "operator" else f"re{int(round(float(reynolds)))}"
    return replace(cfg, checkpoint=f"{cfg.checkpoint}.{tag}.pt")
