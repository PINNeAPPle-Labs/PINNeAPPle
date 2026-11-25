"""One problem, three training backends: ``problem.backend("pytorch" | "jax" | "physicsnemo")`` (#75).

:class:`~pinneapple_physics.physical_problem.PhysicalProblem` describes a problem once; this module trains it. The
same object -- the same coordinates, the same PDE, the same conditions, the same domain -- is fitted by a network on
whichever engine was asked for, and what comes out is read the same way in all three cases::

    import pinneapple as pp

    problem = pp.PhysicalProblem.from_preset("heat_1d")          # or a benchmark case, or your own
    train = problem.backend("jax", epochs=300, seed=0)
    report = train.fit()                                         # a FitReport: loss history, seconds, parameters
    y = train.predict(X)                                         # a Prediction (a torch tensor, one row per point)
    train.save("runs/heat_1d_jax.npz")                           # the artifact each engine writes
    reloaded = pinneapple_neural.model_provider.load(...)        # the same door for all three

What each backend is
--------------------
* **pytorch** -- the registry architecture (``modified_mlp`` by default) trained by :func:`pinneapple_physics.solve`,
  i.e. the library's own PINN path, through the compiled residual. Nothing new is computed here: the backend wraps it
  so that the other two can be compared against it on the same protocol.
* **jax** -- a tanh multilayer perceptron whose derivatives come from :mod:`jax` (``jacfwd`` twice for the Laplacian),
  trained with Adam written in JAX. It builds its own residual for the PDE kinds in :data:`JAX_KINDS`, using the same
  loss weighting the torch compiler uses (``pde`` 1, ``bc`` 10, ``ic`` 10, times each condition's own ``weight``).
* **physicsnemo** -- a :class:`physicsnemo.Module` trained by the library's own PINN loop (the module is an
  ``nn.Module``, so the compiled residual accepts it unchanged) and saved as an ``.mdlus`` checkpoint. That is what
  lets ``ModelProvider``'s ``physicsnemo`` door read it back.

Honesty rules
-------------
* **A backend that cannot run here says so, naming the package** (:class:`BackendNotInstalled`), before any training
  starts. ``available()`` answers the same question without raising.
* **The two engines do not build the same network.** PyTorch trains the registry's gated highway MLP with Fourier
  features; JAX trains a plain tanh MLP of the same width and depth. What the two runs share is the problem, the
  protocol (epochs, collocation points, seeds) and the closed-form reference they are scored against -- not the
  weights. Reporting them side by side is a comparison of *engines*; it is not a claim that a JAX run reproduces a
  torch run number for number, and the report keeps the two losses apart.
* **A kind or condition the JAX residual does not build is refused by name** (:class:`UnsupportedProblem`), listing
  what it does support, rather than trained with a residual that silently omits a term.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pinneapple_neural.physics_model import (
    FitReport,
    PhysicsModel,
    Prediction,
    UncertaintyNotAvailable,
)

__all__ = [
    "BACKENDS",
    "JAX_CONDITIONS",
    "JAX_KINDS",
    "BackendNotInstalled",
    "JaxProblemBackend",
    "PhysicsNeMoProblemBackend",
    "ProblemBackend",
    "TorchProblemBackend",
    "UnsupportedProblem",
    "available_backends",
    "backends",
    "jax_forward",
    "jax_function_ref",
    "problem_backend",
]


class BackendNotInstalled(RuntimeError):
    """The backend exists but the package it trains with is not installed (the message names the package)."""


class UnsupportedProblem(RuntimeError):
    """The backend cannot build this problem's residual, and the message names the term it will not fake."""


#: The backends ``problem.backend()`` accepts, in the order the issue names them.
BACKENDS: tuple[str, ...] = ("pytorch", "jax", "physicsnemo")

#: PDE kinds the JAX residual is written for. Everything else is refused by name, never approximated.
JAX_KINDS: tuple[str, ...] = ("heat_equation", "poisson", "laplace", "burgers", "inviscid_burgers")

#: Conditions the JAX residual turns into a loss term (``neumann``/``robin``/``interface`` need a boundary normal).
JAX_CONDITIONS: tuple[str, ...] = ("dirichlet", "initial", "data")

#: The weights :func:`pinneapple_physics.pinn_solver.compiler.compile_problem` applies by default, kept identical here
#: so the two engines minimise the same objective.
W_PDE, W_BC, W_IC, W_DATA = 1.0, 10.0, 10.0, 1.0


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _installed(name: str) -> tuple[bool, str]:
    """``(usable, detail)`` for a distribution: a missing package is reported, never raised from inside a train."""
    import importlib.util

    if importlib.util.find_spec(name) is None:
        return False, f"{name} is not installed (pip install {name})"
    return True, f"{name} {_package_version(name)}".strip()


def _n_params_numpy(tree: Mapping[str, Any]) -> int:
    return int(sum(np.asarray(v).size for v in tree.values()))


# --------------------------------------------------------------------------------------------- the shared face
class ProblemBackend(PhysicsModel):
    """A problem bound to one engine: ``fit`` trains it, ``predict`` runs it, ``save`` writes the artifact.

    It is a :class:`~pinneapple_neural.physics_model.PhysicsModel` -- ``family="pinn"`` for all three backends, so
    the interface, the report and the comparison keep one shape and only :attr:`backend_name` differs. ``predict``
    takes the same array ``Solution.predict`` takes, ``(N, len(problem.coords))``, and returns a ``Prediction`` whose
    ``mean`` is ``(N, len(problem.fields))``.
    """

    family = "pinn"
    #: The engine this instance trains on.
    backend_name: str = ""
    #: The distribution whose version is recorded, and whose absence is a named refusal.
    package: str = ""
    #: The suffix ``save`` writes.
    suffix: str = ""

    def __init__(self, problem: Any, *, name: str = ""):
        super().__init__(None, name=name or f"{getattr(problem, 'name', '?')} ({self.backend_name})",
                         backend=self.backend_name)
        self.problem = problem
        self.history: list[dict[str, float]] = []
        self.report: FitReport | None = None

    # -- identity ------------------------------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} backend={self.backend_name!r}>"

    def available(self) -> tuple[bool, str]:
        """Whether this backend can train here at all, and what is missing when it cannot."""
        if not self.package:
            return True, ""
        return _installed(self.package)

    def describe(self) -> dict[str, Any]:
        ok, detail = self.available()
        return {"backend": self.backend_name, "available": ok, "detail": detail,
                "package": self.package, "artifact": self.suffix, "problem": getattr(self.problem, "name", ""),
                "coords": list(getattr(self.problem, "coords", ())),
                "fields": list(getattr(self.problem, "fields", ())),
                "trainable": self.capabilities()}

    def capabilities(self) -> dict[str, str]:
        ok, detail = self.available()
        return {"fit": "yes" if ok else f"no: {detail}", "predict": "yes" if ok else f"no: {detail}",
                "uncertainty": "no: a PINN trained on the problem's residual carries no sampler; fit members and "
                               "compare their spread instead",
                "export": f"yes: save() writes {self.suffix or 'an artifact'} that model_provider.load reads back"}

    # -- the calls -----------------------------------------------------------------------------------------
    def fit(self, data: Any = None, **options: Any) -> FitReport:
        """Train on the problem's own residual and conditions. ``data`` is refused: the problem carries them."""
        raise NotImplementedError

    def predict(self, x: Any, **options: Any) -> Prediction:
        """``x`` of shape ``(N, len(problem.coords))``; returns a ``Prediction`` (batch first, one column per field)."""
        raise NotImplementedError

    def uncertainty(self, x: Any, *, method: str = "mc_dropout", **options: Any) -> Prediction:
        raise UncertaintyNotAvailable(
            f"{self.name}: a PINN trained on a problem's residual has no sampler behind it, so there is no spread to "
            "report here. Fit the same problem twice (different seeds or n_collocation) and compare the two models "
            "on the points you care about.")

    def predict_numpy(self, x: Any) -> np.ndarray:
        """The same prediction as a numpy array, which is what ``Solution.predict`` and ``pp.compare`` take."""
        return np.asarray(self.predict(x).mean.detach().cpu().numpy(), dtype=np.float64)

    def save(self, path: str | Path) -> Path:
        """Write the artifact this engine reloads: ``.pt`` (torch checkpoint), ``.npz`` (JAX), ``.mdlus``."""
        raise NotImplementedError

    def export(self, path: str | Path, *, fmt: str = "native", example: Any = None) -> Path:
        """Only the engine's own artifact is written here. TorchScript/ONNX are ``PhysicsModel.export``'s job on a
        model that has one, and this backend does not pretend to lower a JAX pytree into either of them."""
        if fmt not in ("native", "artifact"):
            raise ValueError(f"{self.name}: fmt={fmt!r} is not written here; use fmt='native' (which save() also "
                             f"writes, as {self.suffix or 'the engine artifact'})")
        return self.save(path)

    def to_solution(self):
        """A :class:`pinneapple_physics.solving.Solution` for this trained model, so ``pp.compare`` can score it
        against a closed form or another method without a special case."""
        from pinneapple_physics.solving import Solution

        if self.report is None:
            raise RuntimeError(f"{self.name}: call fit() before to_solution()")
        return Solution(self.problem, f"pinn:{self.backend_name}", self.predict_numpy, self.report.seconds,
                        history={"loss": [e.get("train_total", float('nan')) for e in self.report.history]},
                        info={"backend": self.backend_name, "n_parameters": self.report.n_params,
                              "epochs": self.report.epochs}, kind="neural")

    # -- helpers -------------------------------------------------------------------------------------------
    def _check_problem(self) -> None:
        """The problems this module trains are the ones with a PDE, a box and an engine that can read both."""
        problem = self.problem
        if getattr(problem, "pde", None) is None:
            raise UnsupportedProblem(f"{self.name}: the problem declares no PDE (problem.pde is None), so there is "
                                     "nothing to fit a physics residual to")
        if not problem.coords or not problem.fields:
            raise UnsupportedProblem(f"{self.name}: the problem declares coords {problem.coords} and fields "
                                     f"{problem.fields}; both are needed to build a network")
        issues = problem.validate() if hasattr(problem, "validate") else []
        if issues:
            raise UnsupportedProblem(f"problem '{problem.name}' is not consistent:\n  - " + "\n  - ".join(issues))
        missing = sorted(set(problem.coords) - set(problem.domain_bounds))
        if missing:
            raise UnsupportedProblem(
                f"{self.name}: the problem has no bounds for {missing}; training a PINN on this engine needs a box "
                f"(domain_bounds for every coordinate), not a geometry with tag selectors")

    def _report(self, *, epochs: int, seconds: float, n_params: int, notes: str = "") -> FitReport:
        self.report = FitReport(family=self.family, method="gradient", epochs=int(epochs), seconds=float(seconds),
                                n_params=int(n_params), history=list(self.history), notes=notes)
        return self.report


# --------------------------------------------------------------------------------------------- pytorch
class TorchProblemBackend(ProblemBackend):
    """The library's own PINN path (:func:`pinneapple_physics.solve` with ``method="pinn"``).

    It trains the registry architecture on the compiled residual -- the same call ``pp.solve(problem, "pinn")``
    makes -- so this backend is the control the other two are compared against, not a second implementation of it.
    """

    backend_name = "pytorch"
    package = "torch"
    suffix = ".pt"

    def __init__(self, problem: Any, *, architecture: str = "modified_mlp", **options: Any):
        super().__init__(problem)
        self.architecture = str(architecture)
        self.options = dict(options)

    def fit(self, data: Any = None, *, epochs: int = 300, n_collocation: int = 512, lr: float = 1e-3, seed: int = 0,
            hidden_dim: int = 32, n_layers: int = 3, device: str = "cpu", **options: Any) -> FitReport:
        """Train through :func:`pinneapple_physics.solve` with the problem's compiled residual."""
        if data is not None:
            raise TypeError(f"{self.name}: pass data as a 'data' condition on the problem, not to fit(); a PINN on "
                            "this backend minimises the problem's residual and conditions")
        self._check_problem()
        from pinneapple_physics.solving import solve

        ok, detail = self.available()
        if not ok:
            raise BackendNotInstalled(f"cannot train on pytorch: {detail}")
        t0 = time.perf_counter()
        solution = solve(self.problem, "pinn", epochs=int(epochs), n_collocation=int(n_collocation), lr=float(lr),
                         seed=int(seed), device=device, architecture=self.architecture, hidden_dim=int(hidden_dim),
                         n_layers=int(n_layers))
        seconds = time.perf_counter() - t0
        self.model = solution.model
        self.history = [{"train_total": float(v)} for v in _history_values(solution.history)]
        return self._report(epochs=epochs, seconds=seconds, n_params=int(solution.info.get("n_parameters", 0)),
                            notes=f"pinneapple.pinn, architecture {self.architecture}")

    def predict(self, x: Any, **options: Any) -> Prediction:
        if self.model is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        array = np.asarray(x, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"{self.name}: x must be (N, {len(self.problem.coords)}), got {np.shape(x)}")
        with torch.no_grad():
            out = self.model(torch.as_tensor(array))
        y = getattr(out, "y", out)
        return Prediction(mean=torch.as_tensor(np.asarray(y.detach().cpu().numpy()), dtype=torch.float32))

    def save(self, path: str | Path) -> Path:
        """A ``save_checkpoint`` file (class name + constructor arguments), which is what the ``pytorch`` door reads."""
        if self.model is None:
            raise RuntimeError(f"{self.name}: call fit() before save()")
        save = getattr(self.model, "save_checkpoint", None)
        if save is None:                                # a bare nn.Module: no constructor to rebuild from
            raise NotImplementedError(
                f"{self.name}: {type(self.model).__name__} is not a library model, so it carries no constructor "
                "arguments for the pytorch door to rebuild it from; save the state_dict and rebuild it yourself")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return Path(save(str(path), self._metadata()))

    def _metadata(self) -> dict[str, Any]:
        return {"pinneapple_problem": getattr(self.problem, "name", ""),
                "pinneapple_fingerprint": self.problem.fingerprint() if hasattr(self.problem, "fingerprint") else "",
                "pinneapple_backend": self.backend_name,
                "pinneapple_coords": list(self.problem.coords),
                "pinneapple_fields": list(self.problem.fields)}


def _history_values(history: Any) -> list[float]:
    """The ``total`` (else ``loss``) series out of whatever a solve returned."""
    if isinstance(history, Mapping):
        for key in ("total", "loss", "train_total"):
            if key in history:
                history = history[key]
                break
    try:
        return [float(v) for v in history]
    except (TypeError, ValueError):
        return []


# --------------------------------------------------------------------------------------------- jax
class JaxProblemBackend(ProblemBackend):
    """A tanh MLP trained in JAX on a residual this module builds itself (see :data:`JAX_KINDS`).

    The parameters are kept as a flat dictionary of arrays (``layer0_w``, ``layer0_b``, ...), which is exactly what
    the ``jax`` provider of :mod:`pinneapple_neural.model_provider` reads: an ``.npz`` with a ``function`` entry
    naming :func:`jax_forward`. Saving and reloading therefore goes through the same door as the other two engines
    instead of a JAX-specific reload path.
    """

    backend_name = "jax"
    package = "jax"
    suffix = ".npz"

    def __init__(self, problem: Any, **options: Any):
        super().__init__(problem)
        self.options = dict(options)
        self._params: dict[str, Any] | None = None

    @property
    def model(self) -> Any:
        return self._params

    @model.setter
    def model(self, value: Any) -> None:
        self._params = dict(value) if value is not None else None

    # -- the network ---------------------------------------------------------------------------------------
    def fit(self, data: Any = None, *, epochs: int = 300, n_collocation: int = 512, lr: float = 1e-3, seed: int = 0,
            hidden_dim: int = 32, n_layers: int = 3, n_fourier: int = 32, sigma: float = 0.5,
            **options: Any) -> FitReport:
        """Train on the problem's residual with Adam, ``epochs`` steps of ``n_collocation`` fresh points.

        ``n_fourier``/``sigma`` shape the fixed Fourier embedding in front of the tanh stack. ``sigma=0.5`` is the
        default because the registry's ``1.0`` puts most of the band above the frequency of the benchmark cases'
        solutions: on case 02, at the same budget, it measured 0.074 in-domain relative L2 against 0.25 with 1.0.
        """
        if data is not None:
            raise TypeError(f"{self.name}: pass data as a 'data' condition on the problem, not to fit()")
        self._check_problem()
        ok, detail = self.available()
        if not ok:
            raise BackendNotInstalled(f"cannot train on jax: {detail}")
        kind = str(self.problem.pde.kind)
        if kind not in JAX_KINDS:
            raise UnsupportedProblem(
                f"{self.name}: the residual for PDE kind {kind!r} is not written in this module. Kinds it builds: "
                f"{list(JAX_KINDS)}. The pytorch and physicsnemo backends go through the library's compiled residual, "
                f"which covers every kind the compiler knows")
        bad = [f"{c.name} ({c.kind})" for c in self.problem.conditions if c.kind not in JAX_CONDITIONS]
        if bad:
            raise UnsupportedProblem(
                f"{self.name}: conditions {bad} need a boundary normal (or an interface pair), which this residual "
                f"does not build. It turns {list(JAX_CONDITIONS)} conditions into loss terms")

        import jax
        import jax.numpy as jnp

        rng = np.random.default_rng(int(seed))
        params = _jax_init_params(jax.random.PRNGKey(int(seed)), in_dim=len(self.problem.coords),
                                  out_dim=len(self.problem.fields), width=int(hidden_dim), depth=int(n_layers),
                                  n_fourier=int(n_fourier), sigma=float(sigma))
        batches = _condition_batches(self.problem, n=max(64, int(n_collocation) // 4), rng=rng)
        residual = _jax_residual(self.problem)

        def loss_fn(params: dict[str, Any], x_col: Any):
            res = residual(params, x_col)
            l_pde = jnp.mean(res ** 2)
            terms = {"pde": l_pde}
            total = W_PDE * l_pde
            for name, (x_c, y_c, weight, cond_kind) in batches.items():
                pred = _jax_mlp(params, x_c)
                term = jnp.mean((pred - y_c) ** 2)
                terms[name] = term
                total = total + (_cond_weight(cond_kind) * weight) * term
            return total, terms

        state = _jax_adam_init(params)
        lo = np.array([self.problem.domain_bounds[c][0] for c in self.problem.coords], dtype=np.float64)
        hi = np.array([self.problem.domain_bounds[c][1] for c in self.problem.coords], dtype=np.float64)

        @jax.jit
        def step(params, state, x_col, step_size):
            (total, terms), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x_col)
            params, state = _jax_adam_apply(params, grads, state, step_size)
            return params, state, total, terms

        self.history = []
        t0 = time.perf_counter()
        for epoch in range(int(epochs)):
            x_col = jnp.asarray(lo + (hi - lo) * rng.random((int(n_collocation), len(lo))), dtype=jnp.float32)
            step_size = _adam_step_size(float(lr), epoch + 1)
            params, state, total, terms = step(params, state, x_col, step_size)
            self.history.append({"train_total": float(total),
                                 **{f"train_{k}": float(v) for k, v in terms.items()}})
        seconds = time.perf_counter() - t0
        self._params = {k: np.asarray(v) for k, v in params.items()}
        return self._report(epochs=epochs, seconds=seconds, n_params=_n_params_numpy(self._params),
                            notes=f"jax {jax.__version__}, Fourier({int(n_fourier)}) + tanh MLP "
                                  f"{int(hidden_dim)} x {int(n_layers)}, "
                                  f"residual built in pinneapple_neural.problem_backends")

    def predict(self, x: Any, **options: Any) -> Prediction:
        if self._params is None:
            raise RuntimeError(f"{self.name}: call fit() (or load()) before predict()")
        array = np.asarray(x, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != len(self.problem.coords):
            raise ValueError(f"{self.name}: x must be (N, {len(self.problem.coords)}), got {array.shape}")
        y = np.asarray(jax_forward(self._params, array))
        return Prediction(mean=torch.as_tensor(y, dtype=torch.float32))

    def save(self, path: str | Path) -> Path:
        """An ``.npz`` with one entry per parameter array plus ``function``, which is the ``jax`` door's format."""
        if self._params is None:
            raise RuntimeError(f"{self.name}: call fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, function=np.asarray(jax_function_ref()), **{k: np.asarray(v) for k, v in self._params.items()})
        return path


def jax_function_ref() -> str:
    """The ``function`` entry a JAX artifact carries: ``module:attribute`` naming :func:`jax_forward`."""
    return f"{__name__}:jax_forward"


def jax_forward(params: Mapping[str, Any], x: Any) -> Any:
    """``(params, x) -> y`` for a saved JAX artifact: the callable the ``jax`` provider runs with the loaded pytree."""
    import jax.numpy as jnp

    return _jax_mlp(params, jnp.asarray(x, dtype=jnp.float32))


def _jax_mlp(params: Mapping[str, Any], x: Any) -> Any:
    """A tanh MLP whose layers are named ``layer{i}_w`` / ``layer{i}_b``, from the last one back to the first.

    When the parameters carry ``embed_b`` the input goes through the same random Fourier embedding the registry's
    ``modified_mlp`` uses -- ``[cos(2 pi x B), sin(2 pi x B)]`` with ``B`` fixed -- which is what lets a plain tanh
    stack fit an oscillatory field at the benchmark budget. One function reads both layouts, so a saved artifact
    reloads through the ``jax`` provider without the provider knowing which one it wrote.
    """
    import jax.numpy as jnp

    n = sum(1 for k in params if str(k).endswith("_w") and not str(k).startswith("embed"))
    if n == 0:
        raise ValueError(f"no layer weights in the parameter dictionary (keys: {sorted(params)})")
    h = x
    if "embed_b" in params:
        proj = h @ params["embed_b"]
        h = jnp.concatenate([jnp.cos(2.0 * jnp.pi * proj), jnp.sin(2.0 * jnp.pi * proj)], axis=-1)
    for i in range(n - 1):
        h = jnp.tanh(h @ params[f"layer{i}_w"] + params[f"layer{i}_b"])
    return h @ params[f"layer{n - 1}_w"] + params[f"layer{n - 1}_b"]


def _jax_init_params(key: Any, *, in_dim: int, out_dim: int, width: int, depth: int, n_fourier: int = 32,
                     sigma: float = 0.5) -> dict[str, Any]:
    """LeCun-scaled weights and zero biases, one split of the key per layer (the ``jax_pinn`` backend's scheme).

    With ``n_fourier > 0`` the input first goes through a fixed random Fourier projection (``embed_b``), matching
    ``pinneapple_neural.architectures.modified_mlp.FourierFeatureEmbedding``.
    """
    import jax
    import jax.numpy as jnp

    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    keys = jax.random.split(key, depth + 2)
    params: dict[str, Any] = {}
    d_in = int(in_dim)
    if int(n_fourier) > 0:
        params["embed_b"] = jax.random.normal(keys[0], (d_in, int(n_fourier)), dtype=jnp.float32) * float(sigma)
        d_in = 2 * int(n_fourier)
    for i in range(depth):
        k = keys[i + 1]
        params[f"layer{i}_w"] = jax.random.normal(k, (d_in, width), dtype=jnp.float32) * jnp.sqrt(2.0 / d_in)
        params[f"layer{i}_b"] = jnp.zeros((width,), dtype=jnp.float32)
        d_in = width
    params[f"layer{depth}_w"] = jax.random.normal(keys[-1], (d_in, out_dim),
                                                  dtype=jnp.float32) * jnp.sqrt(2.0 / d_in)
    params[f"layer{depth}_b"] = jnp.zeros((out_dim,), dtype=jnp.float32)
    return params


def _jax_adam_init(params: Mapping[str, Any]) -> dict[str, Any]:
    import jax.numpy as jnp

    return {"m": {k: jnp.zeros_like(v) for k, v in params.items()},
            "v": {k: jnp.zeros_like(v) for k, v in params.items()}}


def _jax_adam_apply(params, grads, state, step_size, beta1: float = 0.9, beta2: float = 0.999,
                    eps: float = 1e-8):
    """Adam with the bias correction folded into ``step_size`` by the caller (so no step counter is traced)."""
    m = {k: beta1 * state["m"][k] + (1.0 - beta1) * grads[k] for k in params}
    v = {k: beta2 * state["v"][k] + (1.0 - beta2) * grads[k] ** 2 for k in params}
    new = {k: params[k] - step_size * m[k] / (v[k] ** 0.5 + eps) for k in params}
    return new, {"m": m, "v": v}


def _adam_step_size(lr: float, step: int, beta1: float = 0.9, beta2: float = 0.999) -> float:
    """``lr * sqrt(1 - beta2^t) / (1 - beta1^t)``: the Adam bias correction, applied to the step and not the moments."""
    return lr * (1.0 - beta2 ** step) ** 0.5 / (1.0 - beta1 ** step)


def _jax_residual(problem: Any) -> Callable[[Mapping[str, Any], Any], Any]:
    """The PDE residual of ``problem`` as a function ``(params, X) -> (N,)``.

    The derivatives are the network's own, by forward-mode differentiation (:func:`jax.jacfwd` twice for a second
    derivative), which is what makes this a physics residual and not a fitted field. The forms match the compiler's
    branches in ``pinneapple_physics/pinn_solver/compiler/compile.py`` term for term.
    """
    coords = tuple(problem.coords)
    kind = str(problem.pde.kind)
    p = dict(problem.pde.params or {})
    has_t = "t" in coords
    t_index = coords.index("t") if has_t else None
    spatial = [i for i, c in enumerate(coords) if c != "t"]

    import jax
    import jax.numpy as jnp

    def _terms(params: Mapping[str, Any], X: Any):
        """``(y, jac, hess)`` of the network: ``(N,F)``, ``(N,F,D)`` and ``(N,F,D,D)``."""
        def one(z):
            return _jax_mlp(params, z[None, :])[0]

        jac = jax.vmap(jax.jacfwd(one))(X)
        hess = jax.vmap(jax.jacfwd(jax.jacfwd(one)))(X)
        return _jax_mlp(params, X), jac, hess

    if kind == "heat_equation":
        if not has_t:
            raise UnsupportedProblem("heat_equation needs a time coordinate 't'")
        alpha = float(p.get("alpha", p.get("kappa", 1e-3)))

        def residual(params: Mapping[str, Any], X: Any):
            _y, jac, hess = _terms(params, X)
            T_t = jac[:, 0, t_index]
            lap = jnp.zeros_like(T_t)
            for i in spatial:
                lap = lap + hess[:, 0, i, i]
            return T_t - alpha * lap
        return residual

    if kind in ("poisson", "laplace"):
        f_value = float(p.get("f", 0.0)) if kind == "poisson" else 0.0

        def residual(params: Mapping[str, Any], X: Any):
            _y, _jac, hess = _terms(params, X)
            lap = jnp.zeros(X.shape[0], dtype=jnp.float32)
            for i in spatial:
                lap = lap + hess[:, 0, i, i]
            return lap - f_value
        return residual

    if kind in ("burgers", "inviscid_burgers"):
        if not has_t or len(spatial) != 1:
            raise UnsupportedProblem(f"{kind} is written for one spatial coordinate plus 't', got {coords}")
        nu = float(p.get("nu", 0.0)) if kind == "burgers" else 0.0
        x_index = spatial[0]

        def residual(params: Mapping[str, Any], X: Any):
            y, jac, hess = _terms(params, X)
            u = y[:, 0]
            u_t = jac[:, 0, t_index]
            u_x = jac[:, 0, x_index]
            u_xx = hess[:, 0, x_index, x_index]
            return u_t + u * u_x - nu * u_xx
        return residual

    raise UnsupportedProblem(f"no residual is written for PDE kind {kind!r}; kinds: {list(JAX_KINDS)}")


def _cond_weight(kind: str) -> float:
    """The loss weight a condition's own ``weight`` multiplies: the compiler's ``w_bc``/``w_ic``/``w_data``."""
    return {"dirichlet": W_BC, "initial": W_IC}.get(kind, W_DATA)


def _condition_batches(problem: Any, *, n: int, rng: np.random.Generator) -> dict[str, tuple[Any, Any, float, str]]:
    """``{condition name: (x, y, weight, kind)}`` sampled once, so the compiled JAX step keeps its shapes.

    Points come from :func:`pinneapple_physics.pde_environment.condition_sampling.sample_condition_points`, the
    sampler ``solve_pde`` itself uses (it draws inside the box *and* on every face, because a boundary selector
    almost never accepts a point drawn inside the volume). A condition that selects no point is an error and not a
    silently missing term.
    """
    import jax.numpy as jnp

    from pinneapple_physics.pde_environment.condition_sampling import sample_condition_points

    batches: dict[str, tuple[Any, Any, float, str]] = {}
    for cond in problem.conditions:
        x = sample_condition_points(lambda X, _c=cond: _c.mask(X, {}), problem.domain_bounds, problem.coords,
                                    int(n), rng)
        if x.shape[0] == 0:
            raise UnsupportedProblem(
                f"condition {cond.name!r} ({cond.kind}) selects no point on the box {dict(problem.domain_bounds)} "
                f"or its faces; check its selector")
        y = cond.values(x, {}) if callable(getattr(cond, "values", None)) else np.zeros((x.shape[0], 1), np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        if y.shape[1] != len(problem.fields):
            full = np.zeros((y.shape[0], len(problem.fields)), dtype=np.float32)
            cols = [list(problem.fields).index(f) for f in cond.fields if f in problem.fields]
            if len(cols) == y.shape[1]:
                full[:, cols] = y
            y = full
        batches[cond.name] = (jnp.asarray(x, dtype=jnp.float32), jnp.asarray(y, dtype=jnp.float32),
                              float(getattr(cond, "weight", 1.0) or 1.0), str(cond.kind))
    return batches


# --------------------------------------------------------------------------------------------- physicsnemo
try:                                                        # pragma: no cover - exercised only where it is installed
    from physicsnemo.core import Module as _PhysicsNeMoModule
except Exception:                                           # noqa: BLE001 - an optional dependency, never an import error
    _PhysicsNeMoModule = None

if _PhysicsNeMoModule is not None:
    class PhysicsNeMoPINN(_PhysicsNeMoModule):               # type: ignore[misc, valid-type]
        """A PhysicsNeMo container around the library's own PINN architecture.

        It is a ``physicsnemo.Module`` so that ``.mdlus`` (:meth:`save`) and ``physicsnemo.Module.from_checkpoint``
        carry it, and a plain ``torch.nn.Module`` so that the library's own compiled residual (:func:`solve_pde`)
        trains it unchanged. The network inside is the registry architecture -- the same one the PyTorch backend
        trains -- because a plain tanh MLP of the same width scores an order of magnitude worse on the benchmark
        cases at the protocol's budget (measured, see the issue comment): the difference would then be the
        architecture and not PhysicsNeMo.

        Only ``in_dim``/``out_dim``/``width``/``depth``/``architecture`` are recorded, so ``from_checkpoint``
        rebuilds the same module from ``args.json`` and loads the weights into it.
        """

        def __init__(self, in_dim: int = 3, out_dim: int = 1, width: int = 32, depth: int = 3,
                     architecture: str = "modified_mlp"):
            super().__init__(meta=None)
            import pinneapple_neural.architectures  # noqa: F401 - registers the model zoo
            from pinneapple_neural.architectures.registry import ModelRegistry

            self.net = ModelRegistry.build(str(architecture), in_dim=int(in_dim), out_dim=int(out_dim),
                                           hidden_dim=int(width), n_layers=int(depth))
            self.in_dim, self.out_dim = int(in_dim), int(out_dim)
            self.width, self.depth = int(width), int(depth)
            self.architecture = str(architecture)

        def forward(self, x: torch.Tensor) -> Any:
            return self.net(x)


class PhysicsNeMoProblemBackend(ProblemBackend):
    """The problem trained on a :class:`physicsnemo.Module` and saved as ``.mdlus``.

    The physics is not PhysicsNeMo's: the residual is the same compiled one the PyTorch backend uses, because a
    PhysicsNeMo module *is* a torch module. What PhysicsNeMo owns here is the model container and the checkpoint
    format -- and that is enough for the ``physicsnemo`` door of ``ModelProvider`` to read the result back.
    """

    backend_name = "physicsnemo"
    package = "physicsnemo"
    suffix = ".mdlus"

    def __init__(self, problem: Any, *, architecture: str = "modified_mlp", **options: Any):
        super().__init__(problem)
        self.architecture = str(architecture)
        self.options = dict(options)

    def fit(self, data: Any = None, *, epochs: int = 300, n_collocation: int = 512, lr: float = 1e-3, seed: int = 0,
            hidden_dim: int = 32, n_layers: int = 3, device: str = "cpu", **options: Any) -> FitReport:
        """Train the PhysicsNeMo module through :func:`solve_pde` -- the library's own PINN loop."""
        if data is not None:
            raise TypeError(f"{self.name}: pass data as a 'data' condition on the problem, not to fit()")
        self._check_problem()
        ok, detail = self.available()
        if not ok:
            raise BackendNotInstalled(f"cannot train on physicsnemo: {detail}")
        if _PhysicsNeMoModule is None:
            raise BackendNotInstalled(f"cannot train on physicsnemo: {detail}")
        import physicsnemo

        from pinneapple_physics import solve_pde

        model = PhysicsNeMoPINN(in_dim=len(self.problem.coords), out_dim=len(self.problem.fields),
                                width=int(hidden_dim), depth=int(n_layers), architecture=self.architecture)
        spec = self.problem.to_pde_spec()
        t0 = time.perf_counter()
        out = solve_pde(spec, model, epochs=int(epochs), device=device, lr=float(lr), n_collocation=int(n_collocation),
                        seed=int(seed))
        seconds = time.perf_counter() - t0
        self.model = out["model"].eval()
        self.history = [{"train_total": float(v)} for v in _history_values(out.get("history"))]
        return self._report(epochs=epochs, seconds=seconds,
                            n_params=int(sum(p.numel() for p in self.model.parameters())),
                            notes=f"physicsnemo {physicsnemo.__version__} module, residual from "
                                  f"pinneapple_physics.solve_pde")

    def predict(self, x: Any, **options: Any) -> Prediction:
        if self.model is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        array = np.asarray(x, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != len(self.problem.coords):
            raise ValueError(f"{self.name}: x must be (N, {len(self.problem.coords)}), got {array.shape}")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(torch.as_tensor(array))
        finally:
            self.model.train(was_training)
        return Prediction(mean=torch.as_tensor(_first_tensor(out), dtype=torch.float32))

    def save(self, path: str | Path) -> Path:
        """The ``.mdlus`` checkpoint: ``Module.save`` writes the graph, the weights and the constructor arguments."""
        if self.model is None:
            raise RuntimeError(f"{self.name}: call fit() before save()")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        return path


def _first_tensor(out: Any) -> np.ndarray:
    """The array inside whatever the module returned (a tensor, a ``ModelOutput``-like ``.y``, a mapping, a tuple)."""
    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy()
    y = getattr(out, "y", None)
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy()
    if isinstance(out, Mapping) and out:
        return _first_tensor(next(iter(out.values())))
    if isinstance(out, (list, tuple)) and out:
        return _first_tensor(out[0])
    return np.asarray(out)


# --------------------------------------------------------------------------------------------- the factory
_BACKENDS: dict[str, Callable[..., ProblemBackend]] = {
    "pytorch": TorchProblemBackend,
    "jax": JaxProblemBackend,
    "physicsnemo": PhysicsNeMoProblemBackend,
}


def problem_backend(problem: Any, backend: str = "pytorch", **options: Any) -> ProblemBackend:
    """The trainer for ``problem`` on ``backend`` -- what ``PhysicalProblem.backend()`` returns.

    ``options`` are the backend's own (``architecture=`` for pytorch, nothing for the other two yet) and are passed
    to the constructor; the training options (``epochs``, ``n_collocation``, ``lr``, ``seed``, ...) go to ``fit``.
    An unknown backend is refused by name, with the ones registered.
    """
    key = str(backend).strip().lower()
    if key not in _BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; registered: {sorted(_BACKENDS)}")
    return _BACKENDS[key](problem, **options)


def backends() -> list[str]:
    """The backend names ``problem.backend()`` accepts."""
    return sorted(_BACKENDS)


def available_backends() -> dict[str, dict[str, Any]]:
    """What each backend says about itself here: usable or not, with the reason, and what it writes."""
    out: dict[str, dict[str, Any]] = {}
    for name in backends():
        cls = _BACKENDS[name]
        ok, detail = _installed(cls.package) if cls.package else (True, "")
        out[name] = {"available": ok, "detail": detail, "artifact": cls.suffix}
    return out
