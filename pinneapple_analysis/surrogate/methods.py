"""Surrogate candidates: which methods apply to a dataset, and how each is trained under a ``ComputeBudget``.

==========  =====================================================  ===========================================
id           method                                                 applies to
==========  =====================================================  ===========================================
ridge        quadratic-polynomial ridge regression (scikit-learn)   scalars, fields (classical baseline)
gp           Gaussian process, Matern-5/2 ARD + noise; fields on    up to ``GP_MAX_N`` training samples
             POD coefficients (scikit-learn)
gbt          gradient-boosted trees (XGBoost if installed, else     scalars
             scikit-learn HistGradientBoosting)
mlp          multilayer perceptron (PyTorch)                        scalars, fields (all grid points as outputs)
fno          Fourier neural operator, parameters as constant        fields on a 1-D or 2-D uniform grid
             input channels (``pinneapple_neural``)
deeponet     DeepONet, branch = parameters, trunk = coordinates     fields
             (``pinneapple_neural``)
pinn         physics-informed MLP u(parameters, coordinates):       fields with a declared PDE (``pde.py``)
             data loss + PDE residual
==========  =====================================================  ===========================================

Every method sees the same training samples, normalisation (training split only) and validation samples. The
iterative methods (mlp, fno, deeponet, pinn) train until the same ``ComputeBudget`` is spent
(``train_within_budget``) and keep their best validation checkpoint out of ten evaluations, whose time is not
charged; the one-shot fits (ridge, gp, gbt) are fitted once and report the time they used, which can be below the
budget (they are never given more).
"""
from __future__ import annotations

import copy
import io
import math
import pickle
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_analysis.cost.equal_budget import (
    ComputeBudget,
    cosine_lr,
    set_lr,
    train_within_budget,
)

from .pde import PDEDeclaration

__all__ = ["Candidate", "Fitted", "NonNegative", "TrainContext", "CANDIDATES", "candidate_ids", "applicable", "fit_candidate",
           "GP_MAX_N"]

GP_MAX_N = 2000
EVAL_AT = tuple(np.linspace(0.1, 0.9, 9))


@dataclass
class TrainContext:
    """What a method gets: raw arrays (``Y`` flattened to (n, p)), training-split scalers, the field layout."""
    Xtr: np.ndarray
    Ytr: np.ndarray
    Xva: np.ndarray
    Yva: np.ndarray
    budget: ComputeBudget
    seed: int = 0
    grid_shape: tuple[int, ...] = ()
    coords: np.ndarray | None = None
    pde: PDEDeclaration | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.xm, self.xs = self.Xtr.mean(0), self.Xtr.std(0) + 1e-12
        self.ym, self.ys = self.Ytr.mean(0), self.Ytr.std(0) + 1e-12
        if self.grid_shape:          # pointwise scalers can be ~0 at fixed boundary values: floor them
            self.ys = np.maximum(self.ys, 1e-3 * float(self.Ytr.std()) + 1e-12)

    def xn(self, X):
        return (np.asarray(X, float) - self.xm) / self.xs


class Fitted:
    """A trained candidate: ``predict(X_raw) -> Y_raw`` with ``Y`` flattened (n, p)."""
    method: str = ""
    n_params: int = 0
    info: dict[str, Any]

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def size_bytes(self) -> int:
        buf = io.BytesIO()
        pickle.dump(self, buf)
        return buf.tell()


class NonNegative(Fitted):
    """A candidate trained on the square root of the outputs in ``mask`` (a boolean over the flattened outputs, e.g.
    a von Mises stress channel): it predicts their square, so they are never negative."""

    def __init__(self, inner: Fitted, mask: np.ndarray):
        self.inner, self.mask = inner, np.asarray(mask, bool)
        self.method, self.n_params, self.info = inner.method, inner.n_params, inner.info

    @staticmethod
    def forward_target(Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """What the inner model is trained on: ``sqrt`` of the masked outputs."""
        Y = np.array(Y, float, copy=True)
        Y[:, mask] = np.sqrt(np.maximum(Y[:, mask], 0.0))
        return Y

    def predict(self, X):
        Y = np.array(self.inner.predict(X), float, copy=True)
        Y[:, self.mask] = Y[:, self.mask] ** 2
        return Y


# ---------------------------------------------------------------------------------------------- one-shot fits
class _Sk(Fitted):
    def __init__(self, method, models, ctx: TrainContext, basis=None):
        self.method, self.models, self.basis = method, models, basis
        self.xm, self.xs, self.ym, self.ys = ctx.xm, ctx.xs, ctx.ym, ctx.ys
        self.info = {}

    def predict(self, X):
        Z = (np.asarray(X, float) - self.xm) / self.xs
        if len(self.models) == 1:
            out = np.asarray(self.models[0].predict(Z)).reshape(len(Z), -1)
        else:
            out = np.stack([m.predict(Z) for m in self.models], 1)
        if self.basis is not None:
            mean, V = self.basis
            out = out @ V + mean
        return out * self.ys + self.ym


def _pod(Yn: np.ndarray, energy: float = 0.999, max_modes: int = 24):
    mean = Yn.mean(0)
    U, s, Vt = np.linalg.svd(Yn - mean, full_matrices=False)
    e = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-30)
    r = int(min(max_modes, np.searchsorted(e, energy) + 1, len(s)))
    V = Vt[:r]
    return mean, V, (Yn - mean) @ V.T, float(e[r - 1])


def _fit_ridge(ctx: TrainContext) -> Fitted:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    deg = 2 if ctx.Xtr.shape[1] <= 30 else 1
    m = make_pipeline(PolynomialFeatures(deg), RidgeCV(alphas=np.logspace(-6, 2, 17)))
    m.fit(ctx.xn(ctx.Xtr), (ctx.Ytr - ctx.ym) / ctx.ys)
    f = _Sk("ridge", [m], ctx)
    f.n_params = int(np.size(m[-1].coef_) + np.size(m[-1].intercept_))
    f.info = {"degree": deg, "alpha": float(m[-1].alpha_)}
    return f


def _fit_gp(ctx: TrainContext) -> Fitted:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    d = ctx.Xtr.shape[1]
    Z, Yn = ctx.xn(ctx.Xtr), (ctx.Ytr - ctx.ym) / ctx.ys
    basis, energy = None, None
    if ctx.grid_shape:
        mean, V, Yn, energy = _pod(Yn)
        basis = (mean, V)
    models = []
    for j in range(Yn.shape[1]):
        k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(d), (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-4, (1e-9, 1e-1))
        gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=1, random_state=ctx.seed)
        with warnings.catch_warnings():         # a hyper-parameter at its bound is not an error
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(Z, Yn[:, j])
        models.append(gp)
    f = _Sk("gp", models, ctx, basis)
    f.n_params = int(sum(m.alpha_.size for m in models) + Z.size)
    f.info = {"kernel": str(models[0].kernel_)[:200], "outputs_modelled": len(models)}
    if energy is not None:
        f.info.update(pod_modes=len(models), pod_energy=round(energy, 6))
    return f


def _fit_gbt(ctx: TrainContext) -> Fitted:
    Z, Zv = ctx.xn(ctx.Xtr), ctx.xn(ctx.Xva)
    Yn, Yv = (ctx.Ytr - ctx.ym) / ctx.ys, (ctx.Yva - ctx.ym) / ctx.ys
    models, engine, nodes = [], "", 0
    try:
        import xgboost as xgb
        engine = f"xgboost {xgb.__version__}"
        for j in range(Yn.shape[1]):
            m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.9,
                                 random_state=ctx.seed, early_stopping_rounds=40 if len(Zv) else None, n_jobs=1)
            m.fit(Z, Yn[:, j], eval_set=[(Zv, Yv[:, j])] if len(Zv) else None, verbose=False)
            models.append(m)
            nodes += int(m.get_booster().trees_to_dataframe().shape[0])
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        engine = "scikit-learn HistGradientBoosting"
        for j in range(Yn.shape[1]):
            m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, early_stopping=True,
                                              random_state=ctx.seed)
            m.fit(Z, Yn[:, j])
            models.append(m)
            nodes += int(sum(p.nodes.size for pred in getattr(m, "_predictors", []) for p in pred))
    f = _Sk("gbt", models, ctx)
    f.n_params, f.info = nodes, {"engine": engine}
    return f


# ---------------------------------------------------------------------------------------------- torch methods
class _Torch(Fitted):
    """A torch module wrapped with its scalers; ``forward_fn(module, Zx) -> Yn (n, p)``."""

    def __init__(self, method, module, ctx: TrainContext, forward_fn: Callable, extra=None):
        self.method, self.module, self.forward_fn = method, module.eval(), forward_fn
        self.xm, self.xs, self.ym, self.ys = ctx.xm, ctx.xs, ctx.ym, ctx.ys
        self.extra = extra or {}
        self.n_params = int(sum(p.numel() for p in module.parameters()))
        self.info = {}

    def predict(self, X):
        import torch
        Z = torch.as_tensor((np.asarray(X, float) - self.xm) / self.xs, dtype=torch.float32)
        with torch.no_grad():
            out = []
            for i in range(0, len(Z), 256):
                out.append(self.forward_fn(self.module, Z[i:i + 256], self.extra).numpy().astype(np.float64))
        Yn = np.concatenate(out) if out else np.zeros((0, len(self.ym)))
        return Yn * self.ys + self.ym

    def __getstate__(self):
        s = dict(self.__dict__)
        s["module"] = copy.deepcopy(self.module).cpu()
        return s


def _mlp(i, o, hidden=64, depth=3, act=None):
    import torch.nn as nn
    act = act or nn.SiLU
    layers, w = [], i
    for _ in range(depth):
        layers += [nn.Linear(w, hidden), act()]
        w = hidden
    layers.append(nn.Linear(w, o))
    return nn.Sequential(*layers)


def _train(ctx: TrainContext, module, loss_on: Callable, val_loss: Callable, lr: float, batch: int):
    """Adam with a budget-aware cosine schedule; best validation checkpoint out of ``EVAL_AT`` + the end."""
    import torch
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1000 + ctx.seed)
    n = len(ctx.Xtr)
    best = {"loss": math.inf, "state": copy.deepcopy(module.state_dict()), "step": 0}
    counter = {"steps": 0}

    def step(progress):
        set_lr(opt, cosine_lr(progress, lr))
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        loss = loss_on(idx, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        counter["steps"] += 1
        return loss.detach()

    def evaluate():
        module.eval()
        with torch.no_grad():
            v = float(val_loss())
        module.train()
        if v < best["loss"]:
            best.update(loss=v, state=copy.deepcopy(module.state_dict()), step=counter["steps"])
        return {"val_loss": v}

    module.train()
    run = train_within_budget(step, ctx.budget, evaluate=evaluate, eval_at=EVAL_AT, count_first_step=False,
                              log_every_s=1e9)
    module.load_state_dict(best["state"])
    module.eval()
    return run, best


def _torch_info(run, best) -> dict[str, Any]:
    return {"steps": run.steps, "train_wall_s": round(run.wall_s, 3), "stopped_by": run.stopped_by,
            "best_val_loss": best["loss"], "best_step": best["step"]}


def _tensors(ctx: TrainContext):
    import torch
    f = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    return (f(ctx.xn(ctx.Xtr)), f((ctx.Ytr - ctx.ym) / ctx.ys), f(ctx.xn(ctx.Xva)) if len(ctx.Xva) else None,
            f((ctx.Yva - ctx.ym) / ctx.ys) if len(ctx.Xva) else None)


def _val_or_train(Zv, Yv, Z, Y):
    return (Zv, Yv) if Zv is not None else (Z, Y)


def _mlp_forward(m, z, extra):
    return m(z)


def _fit_mlp(ctx: TrainContext) -> Fitted:
    import torch
    torch.manual_seed(ctx.seed)
    Z, Y, Zv, Yv = _tensors(ctx)
    o = ctx.options
    net = _mlp(Z.shape[1], Y.shape[1], o.get("hidden", 64), o.get("depth", 3))
    Zv, Yv = _val_or_train(Zv, Yv, Z, Y)
    run, best = _train(ctx, net, lambda idx, g: ((net(Z[idx]) - Y[idx]) ** 2).mean(),
                       lambda: ((net(Zv) - Yv) ** 2).mean(), o.get("lr", 3e-3), o.get("batch", 64))
    f = _Torch("mlp", net, ctx, _mlp_forward)
    f.info = _torch_info(run, best)
    return f


def _fno_forward(m, z, extra):
    shape = extra["shape"]
    x = z.reshape(len(z), -1, *([1] * len(shape))).expand(len(z), z.shape[1], *shape)
    return m(x.contiguous()).y.reshape(len(z), -1)


def _fit_fno(ctx: TrainContext) -> Fitted:
    import torch

    from pinneapple_neural.architectures.neural_operators.fno import FNO2d, FourierNeuralOperator
    torch.manual_seed(ctx.seed)
    Z, Y, Zv, Yv = _tensors(ctx)
    shape, d, o = ctx.grid_shape, Z.shape[1], ctx.options
    w = o.get("width", 24)
    if len(shape) == 1:
        net = FourierNeuralOperator(d, 1, width=w, modes=min(12, shape[0] // 2), layers=4, use_grid=True)
    else:
        net = FNO2d(d, 1, width=w, modes1=min(8, shape[0] // 2), modes2=min(8, shape[1] // 2 + 1), layers=4,
                    use_grid=True)
    extra = {"shape": shape}
    Zv, Yv = _val_or_train(Zv, Yv, Z, Y)
    run, best = _train(ctx, net, lambda idx, g: ((_fno_forward(net, Z[idx], extra) - Y[idx]) ** 2).mean(),
                       lambda: ((_fno_forward(net, Zv, extra) - Yv) ** 2).mean(), o.get("lr", 2e-3),
                       o.get("batch", 16))
    f = _Torch("fno", net, ctx, _fno_forward, extra)
    f.info = _torch_info(run, best)
    return f


def _deeponet_forward(m, z, extra):
    return m(z, extra["trunk"]).y.reshape(len(z), -1)


def _unit_coords(coords: np.ndarray) -> np.ndarray:
    lo, hi = coords.min(0), coords.max(0)
    return (coords - lo) / np.where(hi > lo, hi - lo, 1.0)


def _fit_deeponet(ctx: TrainContext) -> Fitted:
    import torch

    from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet
    torch.manual_seed(ctx.seed)
    Z, Y, Zv, Yv = _tensors(ctx)
    o = ctx.options
    net = DeepONet(Z.shape[1], ctx.coords.shape[1], 1, hidden=o.get("hidden", 64), modes=o.get("modes", 32),
                   depth=o.get("depth", 3))
    extra = {"trunk": torch.as_tensor(_unit_coords(ctx.coords), dtype=torch.float32)}
    Zv, Yv = _val_or_train(Zv, Yv, Z, Y)
    run, best = _train(ctx, net, lambda idx, g: ((_deeponet_forward(net, Z[idx], extra) - Y[idx]) ** 2).mean(),
                       lambda: ((_deeponet_forward(net, Zv, extra) - Yv) ** 2).mean(), o.get("lr", 2e-3),
                       o.get("batch", 32))
    f = _Torch("deeponet", net, ctx, _deeponet_forward, extra)
    f.info = _torch_info(run, best)
    return f


def _pinn_forward(m, z, extra):
    import torch
    c = extra["c_unit"]
    B, N = len(z), len(c)
    inp = torch.cat([z[:, None, :].expand(B, N, z.shape[1]), c[None].expand(B, N, c.shape[1])], -1)
    u = m(inp.reshape(B * N, -1)).reshape(B, N)
    # the net predicts the field in global units (mean mu, scale sd); convert to the pointwise-normalised targets
    return (u * extra["sd"] + extra["mu"] - extra["ym"]) / extra["ys"]


def _fit_pinn(ctx: TrainContext) -> Fitted:
    import torch
    torch.manual_seed(ctx.seed)
    pde = ctx.pde
    if pde is None:
        raise ValueError("pinn needs a declared PDE")
    o = ctx.options
    d, nd = ctx.Xtr.shape[1], ctx.coords.shape[1]
    net = _mlp(d + nd, 1, o.get("hidden", 64), o.get("depth", 4), act=torch.nn.Tanh)
    lo, hi = ctx.coords.min(0), ctx.coords.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    f32 = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32)  # noqa: E731
    mu, sd = float(ctx.Ytr.mean()), float(ctx.Ytr.std() + 1e-12)
    extra = {"c_unit": f32(_unit_coords(ctx.coords)), "mu": mu, "sd": sd, "ym": f32(ctx.ym), "ys": f32(ctx.ys)}
    Xtr_p, xm, xs = f32(ctx.Xtr), f32(ctx.xm), f32(ctx.xs)
    Zt = (Xtr_p - xm) / xs
    Yt = f32(ctx.Ytr)                                   # (n, N) physical
    C_unit = extra["c_unit"]
    lo_t, span_t = f32(lo), f32(span)
    res_scale = sd / float(span.min())                  # u / length: the residual's natural magnitude
    w_pde = float(o.get("pde_weight", 1.0))
    n_data, n_col = int(o.get("n_data", 512)), int(o.get("n_colloc", 512))
    n, N = Yt.shape

    def u_phys(params_p, coords_p):
        z = (params_p - xm) / xs
        cu = (coords_p - lo_t) / span_t
        return net(torch.cat([z, cu], 1)) * sd + mu

    def loss_on(idx, g):
        s = torch.randint(0, n, (n_data,), generator=g)
        p = torch.randint(0, N, (n_data,), generator=g)
        pred = net(torch.cat([Zt[s], C_unit[p]], 1)) * sd + mu
        l_data = (((pred[:, 0] - Yt[s, p]) / sd) ** 2).mean()
        sc = torch.randint(0, n, (n_col,), generator=g)
        cc = (lo_t + span_t * torch.rand(n_col, nd, generator=g)).requires_grad_(True)
        r = pde.residual(u_phys(Xtr_p[sc], cc), cc, Xtr_p[sc])
        return l_data + w_pde * ((r / res_scale) ** 2).mean()

    Zv, Yv = (f32(ctx.xn(ctx.Xva)), f32((ctx.Yva - ctx.ym) / ctx.ys)) if len(ctx.Xva) else \
        (Zt, (Yt - extra["ym"]) / extra["ys"])
    run, best = _train(ctx, net, loss_on, lambda: ((_pinn_forward(net, Zv, extra) - Yv) ** 2).mean(),
                       o.get("lr", 2e-3), 1)
    f = _Torch("pinn", net, ctx, _pinn_forward, extra)
    f.info = {**_torch_info(run, best), "pde": pde.to_dict(), "pde_weight": w_pde,
              "residual_scale": res_scale}
    return f


# ---------------------------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class Candidate:
    id: str
    title: str
    family: str                       # "classical ML", "neural network", "neural operator", "physics-informed"
    iterative: bool
    fit: Callable[[TrainContext], Fitted]
    check: Callable[[str, int, tuple, bool], str | None]   # (kind, n_train, grid_shape, has_pde) -> reason or None


def _ok(*_):
    return None


def _gp_check(kind, n, shape, pde):
    return f"{n} training samples: a Gaussian process is O(n^3); limit {GP_MAX_N}" if n > GP_MAX_N else None


def _scalar_only(kind, n, shape, pde):
    return "fields: one tree ensemble per grid point is not offered (use gp on POD modes or an operator)" \
        if kind == "field" else None


def _field_only(name):
    def chk(kind, n, shape, pde):
        if kind != "field":
            return f"{name} maps parameters to a field; the outputs here are scalars"
        if name == "FNO" and len(shape) not in (1, 2):
            return f"FNO here supports 1-D and 2-D uniform grids (grid shape {shape})"
        return None
    return chk


def _pinn_check(kind, n, shape, pde):
    if kind != "field":
        return "a PINN needs a field over coordinates (the outputs here are scalars)"
    if not pde:
        return "no PDE declared: without the equation there is no physics loss"
    return None


CANDIDATES: dict[str, Candidate] = {c.id: c for c in [
    Candidate("ridge", "Polynomial ridge", "classical ML", False, _fit_ridge, _ok),
    Candidate("gp", "Gaussian process", "classical ML", False, _fit_gp, _gp_check),
    Candidate("gbt", "Gradient-boosted trees", "classical ML", False, _fit_gbt, _scalar_only),
    Candidate("mlp", "MLP", "neural network", True, _fit_mlp, _ok),
    Candidate("fno", "FNO", "neural operator", True, _fit_fno, _field_only("FNO")),
    Candidate("deeponet", "DeepONet", "neural operator", True, _fit_deeponet, _field_only("DeepONet")),
    Candidate("pinn", "PINN", "physics-informed", True, _fit_pinn, _pinn_check),
]}


def candidate_ids() -> list[str]:
    return list(CANDIDATES)


def applicable(kind: str, n_train: int, grid_shape: tuple = (), has_pde: bool = False) -> dict[str, str | None]:
    """``{method: None if applicable else the reason it is not}`` for every registered candidate."""
    return {k: c.check(kind, n_train, tuple(grid_shape), has_pde) for k, c in CANDIDATES.items()}


def fit_candidate(method: str, ctx: TrainContext) -> tuple[Fitted, float]:
    """Train ``method``; returns the fitted model and the wall time of the fit (one-shot fits: the whole fit)."""
    c = CANDIDATES[method]
    t0 = time.perf_counter()
    f = c.fit(ctx)
    return f, time.perf_counter() - t0
