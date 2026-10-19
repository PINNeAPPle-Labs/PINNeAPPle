"""Reduced-order model study: which ROM, at which rank, for which error and speed, measured on held-out snapshots.

``rom_study(snaps)`` takes a :class:`pinneapple_data.snapshots.SnapshotSet` (time series or parametric results on one
mesh), holds out test snapshots (later times, interleaved times or unseen parameter points), fits every method at
several ranks on the training snapshots only and reports, per method and rank:

* the relative L2 error on the test snapshots (mean, max, per snapshot) and the POD projection error at that rank
  (the floor no method using that basis can beat);
* the fit time and the inference time per snapshot (``pinneapple_analysis.cost.time_per_query``) and, when the
  full-order model's time per snapshot is known, the speed-up;
* the singular values and cumulative energy of the training snapshots.

Methods (all from ``pinneapple_neural.architectures.rom``):

``pod_rbf``   POD + radial-basis interpolation of the coefficients in the parameters (or in time)
``pod_gpr``   POD + one Gaussian process per coefficient (parametric data)
``nearest``   the nearest training snapshot (baseline every surrogate must beat; parametric data)
``persistence`` the latest training snapshot at or before the time (the baseline of a forecast)
``dmd``       dynamic mode decomposition: a linear map between consecutive snapshots (uniform time step); exact
              (uncentred) DMD, so the mean field is one of its ``rank`` modes
``opinf``     operator inference: a quadratic map of the POD coefficients, ridge regularisation chosen on the end
              of the training window (uniform time step)
``neural``    a neural surrogate trained within the same wall-clock budget
              (``pinneapple_analysis.cost.train_within_budget``): parameters -> field (parametric data) or a residual
              MLP stepping the POD coefficients (time series)

Time-stepping methods predict a time from the latest training snapshot at or before it (forecast: the last
training snapshot; interleaved: the snapshot just before), so the test error is a real prediction, never a fit.

Every fitted model is a :class:`ReducedModel`: ``predict(times=...)`` or ``predict(params=...)``, and
``to_bytes()`` / ``ReducedModel.from_bytes`` exchange it as a plain ``.npz`` (arrays + a JSON manifest, no pickle).
"""
from __future__ import annotations

import io
import json
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

__all__ = ["ReducedModel", "ROMStudyResult", "rom_study", "fit_reduced_model", "split_snapshots", "default_ranks",
           "field_scales", "METHODS"]

METHODS = {"time": ("persistence", "pod_rbf", "dmd", "opinf", "neural"),
           "parametric": ("nearest", "pod_rbf", "pod_gpr", "neural")}
LABELS = {"persistence": "Persistence (latest snapshot)", "pod_rbf": "POD + RBF", "pod_gpr": "POD + GPR", "nearest": "Nearest snapshot", "dmd": "DMD",
          "opinf": "Operator Inference", "neural": "Neural surrogate"}
RANKLESS = {"nearest", "neural", "persistence"}
FORMAT = "pinneapple-rom/1"


# ------------------------------------------------------------------------------------------------ helpers
def _uniform_dt(t: np.ndarray) -> float | None:
    if t is None or len(t) < 3:
        return None
    d = np.diff(t)
    dt = float(np.median(d))
    if dt <= 0:
        return None
    k = d / dt
    return dt if np.allclose(k, np.round(k), atol=1e-6) and k.min() >= 1 - 1e-6 else None


def _rel(pred: np.ndarray, true: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    den = np.linalg.norm(true if ref is None else true - ref, axis=1)
    return np.linalg.norm(pred - true, axis=1) / np.maximum(den, 1e-300)


def _mlp(in_dim: int, out_dim: int, width: int, depth: int):
    import pinneapple_neural.architectures  # noqa: F401  (registers the models)
    from pinneapple_neural.architectures.registry import ModelRegistry
    return ModelRegistry.build("bench_mlp", in_dim=in_dim, out_dim=out_dim, width=width, depth=depth).double()


def _state_arrays(module, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}{k}": v.detach().cpu().numpy() for k, v in module.state_dict().items()}


def _load_state(module, arrays: dict[str, np.ndarray], prefix: str):
    import torch
    sd = {k[len(prefix):]: torch.as_tensor(np.asarray(v)) for k, v in arrays.items() if k.startswith(prefix)}
    module.load_state_dict(sd)
    return module


def default_ranks(n_train: int, n_values: int, sv: np.ndarray | None = None, max_rank: int = 64) -> list[int]:
    """1, 2, 4, 8, ... up to the data's rank, plus the ranks holding 99 %, 99.9 % and 99.99 % of the energy."""
    top = max(1, min(n_train - 1, n_values, max_rank))
    ranks = {r for r in (1, 2, 4, 8, 16, 32, 64) if r <= top}
    if sv is not None and len(sv):
        e = np.cumsum(sv ** 2) / max(float((sv ** 2).sum()), 1e-300)
        for lvl in (0.99, 0.999, 0.9999):
            ranks.add(int(min(top, np.searchsorted(e, lvl) + 1)))
    ranks.add(top)
    return sorted(ranks)


# ------------------------------------------------------------------------------------------------ split
@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray
    how: str

    def to_dict(self):
        return {"how": self.how, "n_train": int(len(self.train)), "n_test": int(len(self.test)),
                "train": self.train.tolist(), "test": self.test.tolist()}


def split_snapshots(snaps, *, how: str = "auto", test_fraction: float = 0.2, seed: int = 0,
                    test_idx: Sequence[int] | None = None) -> Split:
    """Held-out snapshots. Time series: ``forecast`` (the last ``test_fraction`` of the times, default) or
    ``interleaved`` (every k-th time inside the window). Parametric: ``random`` (seeded). ``test_idx`` overrides."""
    n = snaps.n
    if n < 4:
        raise ValueError(f"{n} snapshots: a ROM study needs at least 4 (training and held-out)")
    if test_idx is not None:
        te = np.unique(np.asarray(test_idx, int))
        if te.min() < 0 or te.max() >= n or len(te) >= n - 1:
            raise ValueError("test_idx must leave at least two training snapshots")
        return Split(np.setdiff1d(np.arange(n), te), te, "given")
    k = max(1, int(round(test_fraction * n)))
    k = min(k, n - 3)
    if snaps.kind == "time":
        how = "forecast" if how in ("auto", "random") else how
        if how == "forecast":
            return Split(np.arange(n - k), np.arange(n - k, n), how)
        if how == "interleaved":
            step = max(2, int(round(n / k)))
            te = np.arange(step // 2 + 1, n - 1, step)[:k]
            return Split(np.setdiff1d(np.arange(n), te), te, how)
        raise ValueError("time series split: forecast or interleaved")
    rng = np.random.default_rng(seed)
    te = np.sort(rng.choice(n, size=k, replace=False))
    return Split(np.setdiff1d(np.arange(n), te), te, "random")


# ------------------------------------------------------------------------------------------------ the model
@dataclass
class ReducedModel:
    """A fitted ROM: ``predict`` and a portable ``.npz`` form (``to_bytes`` / ``from_bytes``)."""
    method: str
    mode: str                                  # "time" | "parametric"
    rank: int | None
    arrays: dict[str, np.ndarray]
    config: dict[str, Any]
    fit_s: float = 0.0
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- prediction
    def predict(self, times=None, params=None) -> np.ndarray:
        """Snapshots (n, n_values) at ``times`` (time series) or ``params`` (n, p) (parametric results)."""
        out = self._predict(times, params)
        sc = self.arrays.get("value_scale")
        return out if sc is None else out * sc

    def _predict(self, times, params) -> np.ndarray:
        if self.mode == "parametric":
            if params is None:
                raise ValueError("this ROM takes params")
            P = np.atleast_2d(np.asarray(params, float))
            if P.shape[1] != len(self.config["param_names"]):
                P = P.reshape(-1, len(self.config["param_names"]))
            if self.config.get("log_params"):
                P = P.copy()
                P[:, self.config["log_params"]] = np.log(P[:, self.config["log_params"]])
            return self._predict_inputs(P)
        if times is None:
            raise ValueError("this ROM takes times")
        t = np.atleast_1d(np.asarray(times, float))
        if self.method in ("pod_rbf", "pod_gpr", "nearest"):
            return self._predict_inputs(t[:, None])
        if self.method == "persistence":
            at = self.arrays["anchor_t"]
            j = np.clip(np.searchsorted(at, t + 1e-9, side="right") - 1, 0, len(at) - 1)
            return self.arrays["X"][j].copy()
        return self._rollout(t)

    def _ppod(self):
        if "ppod" not in self._cache:
            from .parametric import ParametricPOD
            self._cache["ppod"] = ParametricPOD.from_arrays({k[5:]: v for k, v in self.arrays.items()
                                                             if k.startswith("ppod.")}, self.config["ppod"])
        return self._cache["ppod"]

    def _net(self):
        if "net" not in self._cache:
            c = self.config["mlp"]
            self._cache["net"] = _load_state(_mlp(c["in_dim"], c["out_dim"], c["width"], c["depth"]), self.arrays,
                                             "mlp.").eval()
        return self._cache["net"]

    def _predict_inputs(self, P: np.ndarray) -> np.ndarray:
        if self.method in ("pod_rbf", "pod_gpr", "nearest"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return self._ppod().predict(P)
        import torch
        a = self.arrays
        x = torch.as_tensor((P - a["lo"]) / a["span"], dtype=torch.float64)
        with torch.no_grad():
            y = self._net()(x).y.numpy()
        return y * a["y_scale"] + a["y_mean"]

    def _step_fn(self) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        """(latent step a_k -> a_{k+1}, decode latent -> snapshots)."""
        a = self.arrays
        basis, mean = a["basis"], a["mean"]
        decode = lambda z: z @ basis.T + mean  # noqa: E731
        if self.method == "dmd":
            A = a["A"]
            return (lambda z: z @ A.T), decode
        import torch
        if self.method == "opinf":
            if "oi" not in self._cache:
                from .opinf import OperatorInference
                oi = OperatorInference(**self.config["opinf"]).double()
                _load_state(oi, a, "oi.")
                oi._fitted = True
                self._cache["oi"] = oi
            oi = self._cache["oi"]
            return (lambda z: oi.rollout(torch.as_tensor(z), 1)[:, 1].numpy()), decode
        net, mu, s = self._net(), a["z_mean"], a["z_scale"]

        def step(z):
            with torch.no_grad():
                return z + s * net(torch.as_tensor((z - mu) / s)).y.numpy()
        return step, decode

    def _rollout(self, t: np.ndarray) -> np.ndarray:
        a = self.arrays
        at, anchors, dt = a["anchor_t"], a["anchors"], float(self.config["dt"])
        j = np.clip(np.searchsorted(at, t + 1e-9 * dt, side="right") - 1, 0, len(at) - 1)
        k = np.round((t - at[j]) / dt).astype(int)
        if (k < 0).any():
            raise ValueError(f"times before the first training snapshot (t = {at[0]:g}) cannot be predicted by "
                             f"time stepping")
        step, decode = self._step_fn()
        uj = np.unique(j)
        z = anchors[uj].astype(float)
        K = int(k.max())
        traj = [z]
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(K):
                z = step(z)
                traj.append(z)
        traj = np.stack(traj, 1)                                   # (B, K+1, r)
        pos = np.searchsorted(uj, j)
        return decode(traj[pos, k])

    # ---------------------------------------------------------------- exchange
    def manifest(self) -> dict[str, Any]:
        return {"format": FORMAT, "method": self.method, "label": LABELS.get(self.method, self.method),
                "mode": self.mode, "rank": self.rank, "fit_s": self.fit_s, **{k: v for k, v in self.config.items()
                                                                                if k not in ("ppod", "opinf", "mlp")},
                "config": {k: self.config[k] for k in ("ppod", "opinf", "mlp") if k in self.config},
                "n_floats": int(sum(np.asarray(v).size for v in self.arrays.values()))}

    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, __manifest__=np.array(json.dumps(self.manifest())),
                            **{k: np.asarray(v) for k, v in self.arrays.items()})
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, b: bytes) -> ReducedModel:
        z = np.load(io.BytesIO(b), allow_pickle=False)
        m = json.loads(str(z["__manifest__"].item()))
        if m.get("format") != FORMAT:
            raise ValueError(f"not a PINNeAPPle ROM file (format {m.get('format')!r})")
        cfg = {k: v for k, v in m.items() if k not in ("format", "method", "label", "mode", "rank", "fit_s", "config",
                                                       "n_floats")}
        cfg.update(m.get("config", {}))
        return cls(m["method"], m["mode"], m["rank"], {k: z[k] for k in z.files if k != "__manifest__"}, cfg,
                   float(m.get("fit_s", 0.0)))

    # ---------------------------------------------------------------- ONNX (for the model gateway)
    ONNX_METHODS = ("pod_rbf", "nearest", "neural", "dmd")

    def onnx_spec(self) -> dict[str, Any]:
        """Inputs, outputs (one column per value of the snapshot) and validity bounds, in the form
        ``pinneapple_registry.ModelGateway.publish(kind="onnx", spec=...)`` takes."""
        v = self.config["validity"]
        if self.mode == "parametric":
            names = list(self.config["param_names"])
            bounds = {n: [lo, hi] for n, lo, hi in zip(names, v["lo"], v["hi"], strict=True)}
        else:
            names = ["t"]
            bounds = {"t": [v["t_min"], v["t_max"]]}
        outs = []
        for f in self.config["fields"]:
            c = int(f["components"])
            outs += [f"{f['name']}[{i}]" for i in range(f["size"])] if c == 1 else \
                [f"{f['name']}[{i},{j}]" for i in range(f["size"] // c) for j in range(c)]
        return {"inputs": names, "outputs": outs, "validity_bounds": bounds,
                "method": f"rom:{self.method}" + ("" if self.rank is None else f":r{self.rank}")}

    def to_onnx(self) -> bytes:
        """The ROM as an ONNX graph ``x (n, inputs) -> snapshot (n, n_values)`` (float32). Available for
        ``pod_rbf``, ``nearest``, the parametric ``neural`` and ``dmd`` (closed form from the last training
        snapshot: valid forward in time from it). Needs ``onnx``."""
        import onnx
        import torch
        if self.method not in self.ONNX_METHODS or (self.method == "neural" and self.mode == "time"):
            raise NotImplementedError(f"{LABELS.get(self.method, self.method)} ({self.mode}) has no ONNX export; "
                                      f"use the .npz file (ReducedModel.from_bytes)")
        mod = _OnnxROM(self).float().eval()
        d = len(self.onnx_spec()["inputs"])
        buf = io.BytesIO()
        kw = dict(input_names=["x"], output_names=["y"], opset_version=17, dynamic_axes={"x": {0: "n"}, "y": {0: "n"}},
                  do_constant_folding=True)
        try:
            torch.onnx.export(mod, (torch.zeros(3, d),), buf, dynamo=False, **kw)
        except TypeError:                                              # torch < 2.5 has no dynamo flag
            torch.onnx.export(mod, (torch.zeros(3, d),), buf, **kw)
        m = onnx.load_from_string(buf.getvalue())
        m.producer_name, m.producer_version = "pinneapple", FORMAT
        onnx.checker.check_model(m)
        return m.SerializeToString()

    def in_range(self, times=None, params=None) -> np.ndarray:
        """True where a query lies inside the training range (time window or parameter box)."""
        v = self.config["validity"]
        if self.mode == "parametric":
            P = np.atleast_2d(np.asarray(params, float))
            lo, hi = np.asarray(v["lo"]), np.asarray(v["hi"])
            return ((P >= lo - 1e-12) & (P <= hi + 1e-12)).all(1)
        t = np.atleast_1d(np.asarray(times, float))
        return (t >= v["t_min"] - 1e-12) & (t <= v["t_max"] + 1e-12)


class _OnnxROM(torch.nn.Module):
    """Torch twin of ``ReducedModel.predict`` used only to trace the ONNX graph."""

    def __init__(self, rm: ReducedModel):
        super().__init__()
        self.method = rm.method
        T = lambda a: torch.as_tensor(np.asarray(a, float))  # noqa: E731
        a = rm.arrays
        self.register_buffer("value_scale", T(a.get("value_scale", np.ones(rm.config["n_values"]))))
        lp = rm.config.get("log_params") or []
        mask = np.zeros(len(rm.config.get("param_names", ["t"])), bool)
        mask[lp] = True
        self.has_log = bool(lp)
        self.register_buffer("log_mask", torch.as_tensor(mask))
        if rm.method in ("pod_rbf", "nearest"):
            pp = rm._ppod()
            if pp.normalize:
                raise NotImplementedError("the amplitude split has no ONNX export")
            self.register_buffer("lo", T(pp._lo))
            self.register_buffer("span", T(pp._span))
            self.register_buffer("basis", T(pp.basis_.T))
            self.register_buffer("mean", T(pp.mean_))
            if rm.method == "pod_rbf":
                r = pp._model
                if r.kernel != "thin_plate_spline" or float(r.epsilon) != 1.0:
                    raise NotImplementedError("only the thin-plate-spline RBF has an ONNX export")
                self.register_buffer("centres", T(r.y))
                self.register_buffer("shift", T(r._shift))
                self.register_buffer("scale", T(r._scale))
                self.register_buffer("coeffs", T(r._coeffs))
                self.register_buffer("powers", T(r.powers))
            else:
                self.register_buffer("centres", T(pp._P))
                self.register_buffer("A", T(pp._A))
        elif rm.method == "neural":
            import copy
            self.net = copy.deepcopy(rm._net())                     # .float() must not touch the ROM's net
            for k in ("lo", "span", "y_mean", "y_scale"):
                self.register_buffer(k, T(a[k]))
        else:                                                        # dmd: z_k = W diag(lambda^k) W^-1 z_anchor
            lam, W = np.linalg.eig(a["A"])
            c = np.linalg.solve(W, a["anchors"][-1])
            self.register_buffer("t0", T(a["anchor_t"][-1]))
            self.register_buffer("dt", T(rm.config["dt"]))
            self.register_buffer("logmag", T(np.log(np.maximum(np.abs(lam), 1e-30))))
            self.register_buffer("theta", T(np.angle(lam)))
            self.register_buffer("cr", T(c.real))
            self.register_buffer("ci", T(c.imag))
            self.register_buffer("Wr", T(W.real.T))
            self.register_buffer("Wi", T(W.imag.T))
            self.register_buffer("basis", T(a["basis"].T))
            self.register_buffer("mean", T(a["mean"]))

    def forward(self, x):
        if self.has_log:
            x = torch.where(self.log_mask, torch.log(x.abs() + 1e-30), x)
        return self._forward(x) * self.value_scale

    def _forward(self, x):
        if self.method == "dmd":
            k = (x[:, :1] - self.t0) / self.dt
            mag = torch.exp(k * self.logmag)
            cs, sn = torch.cos(k * self.theta), torch.sin(k * self.theta)
            vr = mag * (self.cr * cs - self.ci * sn)
            vi = mag * (self.cr * sn + self.ci * cs)
            return (vr @ self.Wr - vi @ self.Wi) @ self.basis + self.mean
        xs = (x - self.lo) / self.span
        if self.method == "neural":
            return self.net(xs).y * self.y_scale + self.y_mean
        d2 = ((xs[:, None, :] - self.centres[None]) ** 2).sum(-1)
        if self.method == "nearest":
            A = self.A[torch.argmin(d2, dim=1)]
        else:
            ker = 0.5 * d2 * torch.log(d2 + 1e-30)                    # r^2 log r
            xh = (xs - self.shift) / self.scale
            poly = torch.exp(torch.log(xh[:, None, :].abs() + 1e-30) * self.powers[None]) * \
                torch.where((xh[:, None, :] < 0) & (torch.remainder(self.powers[None], 2) == 1), -1.0, 1.0)
            poly = torch.where(self.powers[None] == 0, torch.ones_like(poly), poly).prod(-1)
            A = torch.cat([ker, poly], 1) @ self.coeffs
        return A @ self.basis + self.mean


# ------------------------------------------------------------------------------------------------ fitting
def _base_config(snaps, train, mode) -> dict[str, Any]:
    cfg: dict[str, Any] = {"n_values": snaps.n_values, "fields": [{k: f[k] for k in ("name", "where", "components",
                                                                                    "offset", "size")}
                                                                   for f in snaps.fields]}
    if mode == "parametric":
        P = snaps.params[train]
        cfg.update(param_names=list(snaps.param_names),
                   validity={"lo": P.min(0).tolist(), "hi": P.max(0).tolist()})
    else:
        t = snaps.times[train]
        cfg.update(validity={"t_min": float(t.min()), "t_max": float(t.max())})
    return cfg


def _anchors(snaps, train, basis, mean):
    X = snaps.X[train]
    return {"anchors": (X - mean) @ basis, "anchor_t": snaps.times[train].astype(float)}


def _pairs(t: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.round(np.diff(t) / dt).astype(int)
    i = np.flatnonzero(k == 1)
    return i, i + 1


def field_scales(snaps, train: Sequence[int]) -> np.ndarray:
    """One positive scale per value: the RMS of each field over the training snapshots, so fields in different units
    (a displacement in mm and a stress in MPa) weigh the same in the POD."""
    X = snaps.X[np.asarray(train, int)]
    out = np.ones(snaps.n_values)
    for f in snaps.fields:
        sl = slice(f["offset"], f["offset"] + f["size"])
        out[sl] = max(float(np.sqrt(np.mean(X[:, sl] ** 2))), 1e-300)
    return out


def fit_reduced_model(snaps, train: Sequence[int], method: str, rank: int | None = None, *,
                      budget_s: float = 5.0, seed: int = 0, mode: str | None = None,
                      scale: str | np.ndarray | None = "auto", log_params: Sequence[str] = (),
                      amplitude_split: bool = False) -> ReducedModel:
    """Fit one ROM on the snapshots ``train`` (indices). ``rank``: number of POD modes (ignored by ``nearest`` and
    the parametric ``neural``). ``budget_s``: wall-clock budget of the neural surrogate. ``scale``: ``"auto"``
    (each field divided by its training RMS when there are several fields), ``"fields"``, ``None`` or a vector.
    ``log_params``: parameters regressed in log scale (positive, spanning a range: a thickness, a Reynolds number).
    ``amplitude_split``: POD of the shape and RBF of the log amplitude (``ParametricPOD(normalize=True)``), for
    responses whose size spans orders of magnitude (no ONNX export)."""
    if isinstance(scale, str) and (scale == "fields" or (scale == "auto" and len(snaps.fields) > 1)):
        scale = field_scales(snaps, train)
    elif isinstance(scale, str):
        scale = None
    if scale is not None:
        from dataclasses import replace
        sc = np.asarray(scale, float)
        m = fit_reduced_model(replace(snaps, X=snaps.X / sc), train, method, rank, budget_s=budget_s, seed=seed,
                              mode=mode, scale=None, log_params=log_params, amplitude_split=amplitude_split)
        m.arrays["value_scale"] = sc
        return m
    if log_params:
        from dataclasses import replace
        idx = [snaps.param_names.index(n) for n in log_params]
        if snaps.params is None or (snaps.params[:, idx] <= 0).any():
            raise ValueError(f"log-scale parameters must be positive: {list(log_params)}")
        P = snaps.params.copy()
        P[:, idx] = np.log(P[:, idx])
        m = fit_reduced_model(replace(snaps, params=P), train, method, rank, budget_s=budget_s, seed=seed, mode=mode,
                              scale=None, amplitude_split=amplitude_split)
        m.config["log_params"] = idx
        m.config["validity"] = _base_config(snaps, np.asarray(train, int), mode or snaps.kind)["validity"]
        return m
    import torch
    mode = mode or snaps.kind
    train = np.asarray(train, int)
    X = snaps.X[train]
    cfg = _base_config(snaps, train, mode)
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    if method in ("pod_rbf", "pod_gpr", "nearest"):
        from .parametric import ParametricPOD
        if mode == "parametric":
            P = snaps.params[train]
        else:
            P = snaps.times[train][:, None]
            cfg["param_names"] = ["t"]
        reg = {"pod_rbf": "rbf", "pod_gpr": "gpr", "nearest": "nearest"}[method]
        r = len(train) if method == "nearest" else int(rank)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ppod = ParametricPOD(r=r, energy=None, regressor=reg, seed=seed,
                                 normalize=bool(amplitude_split and method != "nearest")).fit(P, X)
        cfg["ppod"] = ppod.config()
        arrays = {f"ppod.{k}": v for k, v in ppod.to_arrays().items()}
        rank = None if method == "nearest" else ppod.rank
        m = ReducedModel(method, mode, rank, arrays, cfg)
        m._cache["ppod"] = ppod
    elif method == "persistence":
        if mode != "time":
            raise ValueError("persistence is a baseline of time series")
        m = ReducedModel(method, mode, None, {"X": X, "anchor_t": snaps.times[train].astype(float)}, cfg)
    elif method == "dmd":
        dt = _uniform_dt(snaps.times[train]) if mode == "time" else None
        if dt is None:
            raise ValueError("DMD needs a time series on a uniform time step")
        from .dmd import DynamicModeDecomposition
        i0, i1 = _pairs(snaps.times[train], dt)
        r = int(min(rank, len(i0)))
        d = DynamicModeDecomposition(r=r).fit_pairs(torch.as_tensor(X[i0]), torch.as_tensor(X[i1]), center=False)
        basis, mean = d.basis_.numpy(), d.mean_.numpy()[0]
        cfg["dt"] = dt
        arrays = {"basis": basis, "mean": mean, "A": d.A_.numpy(), **_anchors(snaps, train, basis, mean)}
        m = ReducedModel(method, mode, r, arrays, cfg)
    elif method == "opinf":
        dt = _uniform_dt(snaps.times[train]) if mode == "time" else None
        if dt is None:
            raise ValueError("Operator Inference needs a time series on a uniform time step")
        m = _fit_opinf(snaps, train, int(rank), dt, cfg)
    elif method == "neural":
        m = _fit_neural(snaps, train, rank, cfg, mode, budget_s, seed)
    else:
        raise ValueError(f"unknown method {method!r}; methods: {sorted(LABELS)}")
    m.fit_s = time.perf_counter() - t0 if "fit_s" not in m.config else m.config.pop("fit_s")
    return m


def _pod(X: np.ndarray, r: int):
    mean = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    r = int(min(r, len(S)))
    return Vt[:r].T, mean, S


def _fit_opinf(snaps, train, rank, dt, cfg) -> ReducedModel:
    import torch

    from .opinf import OperatorInference
    t = snaps.times[train]
    basis, mean, _ = _pod(snaps.X[train], rank)
    r = basis.shape[1]
    Z = (snaps.X[train] - mean) @ basis
    cfg["dt"] = dt

    def fit(idx, lq):
        i0, i1 = _pairs(t[idx], dt)
        a = torch.as_tensor(np.stack([Z[idx][i0], Z[idx][i1]], 1))
        kw = dict(r=r, use_quadratic=True, l2_linear=1e-6, l2_quad=lq, scale=True)
        return OperatorInference(**kw).fit(a), kw

    def model(oi, kw, idx):
        arrays = {"basis": basis, "mean": mean, "anchors": Z[idx], "anchor_t": t[idx].astype(float),
                  **_state_arrays(oi, "oi.")}
        mm = ReducedModel("opinf", "time", r, arrays, dict(cfg, opinf=kw))
        mm._cache["oi"] = oi
        return mm

    # quadratic regularisation chosen on the end of the training window (a forecast inside the training data)
    n_fit = max(3, int(0.75 * len(train)))
    best = None
    if len(train) - n_fit >= 1:
        idx = np.arange(n_fit)
        val = np.arange(n_fit, len(train))
        scale = np.linalg.norm(Z, axis=1).mean() + 1e-300
        for lq in (1e-4, 1e-2, 1e0, 1e2):
            try:
                oi, kw = fit(idx, lq)
                pred = model(oi, kw, idx)._rollout(t[val])
                e = float(np.mean(np.linalg.norm((pred - mean) @ basis - Z[val], axis=1)) / scale)
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                e = np.inf
            if np.isfinite(e) and (best is None or e < best[0]):
                best = (e, lq)
    lq = best[1] if best else 1e0
    oi, kw = fit(np.arange(len(train)), lq)
    return model(oi, kw, np.arange(len(train)))


def _fit_neural(snaps, train, rank, cfg, mode, budget_s, seed) -> ReducedModel:
    import torch

    from pinneapple_analysis.cost import ComputeBudget, cosine_lr, set_lr, train_within_budget
    X = snaps.X[train]
    width, depth = 64, 3
    if mode == "parametric":
        P = snaps.params[train]
        lo, span = P.min(0), np.maximum(np.ptp(P, 0), 1e-30)
        y_mean = X.mean(0)
        y_scale = float(np.std(X - y_mean)) or 1.0
        xin = torch.as_tensor((P - lo) / span)
        yt = torch.as_tensor((X - y_mean) / y_scale)
        net = _mlp(P.shape[1], X.shape[1], width, depth)
        arrays_extra = {"lo": lo, "span": span, "y_mean": y_mean, "y_scale": np.array(y_scale)}
        cfg_extra: dict[str, Any] = {}
        rank = None
    else:
        dt = _uniform_dt(snaps.times[train])
        if dt is None:
            raise ValueError("the neural time stepper needs a uniform time step")
        basis, mean, _ = _pod(X, int(rank or 8))
        Z = (X - mean) @ basis
        i0, i1 = _pairs(snaps.times[train], dt)
        mu, s = Z.mean(0), np.maximum(Z.std(0), 1e-12)
        xin = torch.as_tensor((Z[i0] - mu) / s)
        yt = torch.as_tensor((Z[i1] - Z[i0]) / s)
        net = _mlp(basis.shape[1], basis.shape[1], width, depth)
        arrays_extra = {"basis": basis, "mean": mean, "z_mean": mu, "z_scale": s,
                        **_anchors(snaps, train, basis, mean)}
        cfg_extra = {"dt": dt}
        rank = basis.shape[1]
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)

    def step(progress):
        set_lr(opt, cosine_lr(progress, 3e-3))
        opt.zero_grad()
        loss = torch.mean((net(xin).y - yt) ** 2)
        loss.backward()
        opt.step()
        return float(loss.detach())
    run = train_within_budget(step, ComputeBudget(wall_s=float(budget_s)), count_first_step=False, log_every_s=max(1.0, budget_s / 10))
    net.eval()
    arrays = {**_state_arrays(net, "mlp."), **arrays_extra}
    c = dict(cfg, **cfg_extra, mlp={"in_dim": int(xin.shape[1]), "out_dim": int(yt.shape[1]), "width": width,
                                   "depth": depth},
             budget={"wall_s": float(budget_s), "steps": run.steps, "used_s": run.wall_s, "stopped_by": run.stopped_by,
                     "final_loss": run.history[-1].get("loss") if run.history else None})
    c["fit_s"] = run.wall_s
    m = ReducedModel("neural", mode, rank, arrays, c)
    m._cache["net"] = net
    return m


# ------------------------------------------------------------------------------------------------ the study
@dataclass
class ROMStudyResult:
    mode: str
    split: Split
    singular_values: np.ndarray
    rows: list[dict[str, Any]]
    projection: list[dict[str, Any]]
    skipped: list[dict[str, str]]
    budget_s: float
    hf_seconds: float | None
    metric: str
    models: dict[tuple[str, int | None], ReducedModel] = field(default_factory=dict, repr=False)
    test_predictions: dict[tuple[str, int | None], np.ndarray] = field(default_factory=dict, repr=False)

    def best(self, tolerance: float | None = None) -> dict[str, Any] | None:
        """The lowest test error; with ``tolerance``, the fastest model whose mean test error is below it."""
        ok = [r for r in self.rows if np.isfinite(r["error_mean"])]
        if not ok:
            return None
        if tolerance is not None:
            under = [r for r in ok if r["error_mean"] <= tolerance]
            if under:
                return min(under, key=lambda r: r["infer_s"])
        return min(ok, key=lambda r: r["error_mean"])

    def pareto(self) -> list[dict[str, Any]]:
        """Models no other model beats on both test error and inference time."""
        ok = sorted((r for r in self.rows if np.isfinite(r["error_mean"])), key=lambda r: (r["infer_s"], r["error_mean"]))
        front, best = [], np.inf
        for r in ok:
            if r["error_mean"] < best:
                front.append(r)
                best = r["error_mean"]
        return front

    def to_dict(self, tolerance: float | None = None) -> dict[str, Any]:
        sv = self.singular_values
        e = np.cumsum(sv ** 2) / max(float((sv ** 2).sum()), 1e-300) if len(sv) else sv
        b = self.best(tolerance)
        return {"mode": self.mode, "split": self.split.to_dict(), "metric": self.metric,
                "singular_values": sv.tolist(), "energy": e.tolist(), "rows": self.rows,
                "projection": self.projection, "skipped": self.skipped, "budget_s": self.budget_s,
                "hf_seconds": self.hf_seconds, "tolerance": tolerance,
                "best": None if b is None else {k: b[k] for k in ("method", "rank", "error_mean", "infer_s")},
                "pareto": [{k: r[k] for k in ("method", "rank", "error_mean", "infer_s")} for r in self.pareto()]}


def rom_study(snaps, *, methods: Sequence[str] | None = None, ranks: Sequence[int] | None = None,
              split: str = "auto", test_fraction: float = 0.2, test_idx: Sequence[int] | None = None,
              budget_s: float = 5.0, seed: int = 0, hf_seconds: float | None = None, max_rank: int = 64,
              timing_repeats: int = 5, scale: str | None = "auto", log_params: Sequence[str] = (),
              amplitude_split: bool = False,
              progress: Callable[[str], None] | None = None) -> ROMStudyResult:
    """Fit every method at every rank on the training snapshots and score it on the held-out ones (see the module
    docstring). ``hf_seconds``: wall time of the full-order model per snapshot (default: ``snaps.meta``).
    ``scale``: with several fields (``"auto"``) each is divided by its training RMS before the POD, and the error is
    measured on the scaled snapshot (every field counts); the per-field errors are reported as well."""
    import torch

    from pinneapple_analysis.cost import time_per_query
    torch.zeros(2, 2, dtype=torch.float64) @ torch.zeros(2, 2, dtype=torch.float64)   # first-call overhead, not a fit
    mode = snaps.kind
    sp = split_snapshots(snaps, how=split, test_fraction=test_fraction, seed=seed, test_idx=test_idx)
    sc = field_scales(snaps, sp.train) if (scale == "fields" or (scale == "auto" and len(snaps.fields) > 1)) else None
    Xtr, Xte = snaps.X[sp.train], snaps.X[sp.test]
    if sc is not None:
        Xtr, Xte = Xtr / sc, Xte / sc
    mean = Xtr.mean(0)
    _, S, Vt = np.linalg.svd(Xtr - mean, full_matrices=False)
    ranks = sorted({int(r) for r in ranks}) if ranks else default_ranks(len(sp.train), snaps.n_values, S, max_rank)
    ranks = [r for r in ranks if 1 <= r <= len(S)] or [1]
    methods = list(methods or METHODS[mode])
    hf = hf_seconds if hf_seconds is not None else snaps.meta.get("hf_seconds")
    # time series: the error relative to the fluctuation about the training mean (the mean alone scores ~0 otherwise)
    ref = mean if mode == "time" else None
    metric = "relative L2 of the fluctuation about the training mean" if mode == "time" else "relative L2"
    if sc is not None:
        metric += " (each field scaled by its training RMS)"
    inputs = {"times": snaps.times[sp.test]} if mode == "time" else {"params": snaps.params[sp.test]}
    proj = []
    Xc = Xte - mean
    for r in ranks:
        V = Vt[:r].T
        P = (Xc @ V) @ V.T + mean
        e = _rel(P, Xte, ref)
        proj.append({"rank": r, "error_mean": float(e.mean()), "error_max": float(e.max()),
                     "energy": float((S[:r] ** 2).sum() / max((S ** 2).sum(), 1e-300))})
    rows, skipped, models, preds = [], [], {}, {}
    dt = _uniform_dt(snaps.times[sp.train]) if mode == "time" else None
    for meth in methods:
        if meth not in LABELS:
            skipped.append({"method": meth, "reason": "unknown method"})
            continue
        if meth in ("dmd", "opinf") and dt is None:
            skipped.append({"method": meth, "reason": "needs a time series on a uniform time step"})
            continue
        if meth in ("pod_gpr", "nearest") and mode == "time":
            skipped.append({"method": meth, "reason": "used for parametric data"})
            continue
        if meth == "persistence" and mode != "time":
            skipped.append({"method": meth, "reason": "a baseline of time series"})
            continue
        rs = [None] if (meth in RANKLESS and not (meth == "neural" and mode == "time")) else \
            ([min(max(ranks), 16)] if meth == "neural" else ranks)
        for r in rs:
            if progress:
                progress(f"{LABELS[meth]}{'' if r is None else f', rank {r}'}")
            try:
                m = fit_reduced_model(snaps, sp.train, meth, r, budget_s=budget_s, seed=seed, mode=mode,
                                      scale=sc, log_params=log_params if mode == "parametric" else (),
                                      amplitude_split=amplitude_split)
                with np.errstate(all="ignore"):
                    pred = m.predict(**inputs)
                    pred_s = pred if sc is None else pred / sc
                tq = time_per_query(lambda m=m: m.predict(**inputs), repeats=timing_repeats, warmup=1,
                                    n_per_call=len(sp.test))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as ex:
                skipped.append({"method": meth, "rank": r, "reason": str(ex)[:200]})
                continue
            finite = bool(np.isfinite(pred).all())
            e = _rel(pred_s, Xte, ref) if finite else np.full(len(sp.test), np.inf)
            per_field = {}
            for f in snaps.fields:
                sl = slice(f["offset"], f["offset"] + f["size"])
                ef = _rel(pred_s[:, sl], Xte[:, sl], None if ref is None else ref[sl]) if finite else [np.inf]
                per_field[f["name"]] = float(np.mean(ef))
            row = {"method": meth, "label": LABELS[meth], "rank": m.rank, "error_mean": float(e.mean()),
                   "error_max": float(e.max()), "error_median": float(np.median(e)), "errors": e.tolist(),
                   "fit_s": float(m.fit_s), "infer_s": float(tq["median_s"]),
                   "n_floats": m.manifest()["n_floats"], "diverged": not finite, "field_errors": per_field,
                   "speedup": (float(hf / tq["median_s"]) if hf and tq["median_s"] > 0 else None)}
            if meth == "neural":
                row["budget"] = m.config.get("budget")
            rows.append(row)
            models[(meth, m.rank)] = m
            preds[(meth, m.rank)] = pred
    return ROMStudyResult(mode, sp, S, rows, proj, skipped, float(budget_s), hf, metric, models, preds)

