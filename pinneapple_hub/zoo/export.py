"""ONNX export of zoo models and their per-query trust checks.

``export_onnx(name, path)`` writes one ONNX graph from the model card's physical inputs to its physical outputs:
pre-processing (CST encoding, log features, normalisation) and post-processing (denormalisation, exp, ensemble mean
and spread, conformal bound) are inside the graph, so any ONNX runtime gives the same numbers as ``predict``.
With ``checks=True`` the graph takes a second input, ``query``, and also returns the values of the model's query
checks (:mod:`pinneapple_hub.zoo.checks`) for those rows. :func:`onnx_parity` measures the largest difference
between a graph and the PyTorch model on the zoo's held-out and independent-reference inputs.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import numpy as np

from ._models import ZooModel

OPSET = 17
PARITY_TOL = 1e-5


# ====================================================================== predictor modules (physical in -> out)
def predictor_module(zm: ZooModel):
    """``torch.nn.Module`` mapping the card's inputs (n, d) to a tuple of outputs (see :func:`output_names`)."""
    import torch
    from torch import nn

    if zm.adapter == "airfoil_mlp":
        from pinneapple_design.aero.geometry import BOUNDS

        class Airfoil(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = zm.module
                self.register_buffer("lo", torch.tensor(BOUNDS[:, 0], dtype=torch.float32))
                self.register_buffer("span", torch.tensor(BOUNDS[:, 1] - BOUNDS[:, 0], dtype=torch.float32))
                self.register_buffer("ym", torch.tensor(zm.config["y_mean"], dtype=torch.float32))
                self.register_buffer("ys", torch.tensor(zm.config["y_std"], dtype=torch.float32))

            def forward(self, x):
                s = (x[:, 0:6] - self.lo) / self.span * 2 - 1
                a = x[:, 6:7]
                z = torch.cat([s, (a - 6) / 8, torch.sin(a * (math.pi / 180)) * 4], 1)
                y = self.net(z) * self.ys + self.ym                                  # (K, n, 3): Cl, log Cd, Cm
                y = torch.cat([y[..., 0:1], torch.exp(y[..., 1:2]), y[..., 2:3]], 2)
                m = y.mean(0)
                sd = torch.sqrt(((y - m) ** 2).mean(0))                              # population std, as numpy
                ld = y[..., 0] / y[..., 1]                                           # (K, n)
                ldm = ld.mean(0)
                lds = torch.sqrt(((ld - ldm) ** 2).mean(0))
                return m, sd, torch.stack([ldm, lds], 1)
        return Airfoil()

    if zm.adapter in ("heatsink_forced", "heatsink_natural"):
        c = zm.config
        forced = zm.adapter == "heatsink_forced"

        class Heatsink(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = zm.module
                for k in ("x_mean", "x_std", "y_mean", "y_std"):
                    self.register_buffer(k, torch.tensor(c[k], dtype=torch.float32))
                self.q95 = float(math.exp(c["conformal_log_q"]["95"]))

            def forward(self, x):
                W, D, tb, N, t, H, k, sw, sd, Ta = (x[:, i:i + 1] for i in range(10))
                Q = x[:, -1:]
                gap = (W - N * t) / torch.clamp(N - 1, min=1.0)
                f = [torch.log(W), torch.log(D), torch.log(tb), torch.log(torch.clamp(gap, min=1e-3)),
                     torch.log(t), torch.log(H), torch.log(N), torch.log(k), torch.log(sw / W), torch.log(sd / D), Ta]
                f = torch.cat([*f, *([torch.log(x[:, 10:11])] if forced else []), torch.log(Q)], 1)
                y = self.net((f - self.x_mean) / self.x_std)
                y = (y.y if hasattr(y, "y") else y) * self.y_std + self.y_mean
                r = torch.exp(y[:, 0:1])
                if not forced:                                       # natural convection: no pressure drop
                    return (torch.cat([r, r * self.q95], 1),)
                return (torch.cat([r, torch.exp(y[:, 1:2]), r * self.q95], 1),)
        return Heatsink()

    if zm.adapter == "rover_contact":
        c = zm.config

        class Rover(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = zm.module
                for k in ("x_lo", "x_hi", "y_lo", "y_hi"):
                    self.register_buffer(k, torch.tensor(c[k], dtype=torch.float32))

            def forward(self, x):
                xn = 2.0 * (x - self.x_lo) / (self.x_hi - self.x_lo) - 1.0
                return ((self.net(xn) + 1.0) * (self.y_hi - self.y_lo) / 2.0 + self.y_lo,)
        return Rover()
    raise NotImplementedError(f"no ONNX export for adapter {zm.adapter!r}")


OUTPUTS = {
    "airfoil_mlp": {"mean": ["Cl", "Cd", "Cm"], "std": ["Cl_std", "Cd_std", "Cm_std"], "ld": ["LD", "LD_std"]},
    "heatsink_forced": {"y": ["r_base_k_w", "pressure_drop_pa", "r_base_upper95_k_w"]},
    "heatsink_natural": {"y": ["r_base_k_w", "r_base_upper95_k_w"]},
    "rover_contact": {"y": ["Fx_N", "Fz_N", "My_Nm"]},
}


def output_names(zm: ZooModel) -> dict[str, list[str]]:
    """ONNX output -> the ``predict`` keys of its columns."""
    return OUTPUTS[zm.adapter]


def as_graph_outputs(zm: ZooModel, pred: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``predict`` output laid out like the graph's outputs."""
    return {o: np.stack([np.asarray(pred[k], float) for k in cols], 1) for o, cols in output_names(zm).items()}


def output_scale(zm: ZooModel) -> dict[str, np.ndarray]:
    """Per column, the scale of the parity tolerance: NaN = relative to the value (positive, log-modelled outputs:
    Cd, R, dP; L/D relative to max(|L/D|, y_std(Cl) / Cd), the error a Cl within tolerance implies), otherwise the
    network's standardisation scale."""
    if zm.adapter == "airfoil_mlp":
        s = np.array([zm.config["y_std"][0], np.nan, zm.config["y_std"][2]])
        return {"mean": s, "std": s, "ld": np.array([np.nan, np.nan])}
    if zm.adapter in ("heatsink_forced", "heatsink_natural"):
        return {"y": np.full(len(OUTPUTS[zm.adapter]["y"]), np.nan)}
    lo, hi = np.asarray(zm.config["y_lo"]), np.asarray(zm.config["y_hi"])
    return {"y": (hi - lo) / 2}


def parity_error(zm: ZooModel, ref: dict[str, np.ndarray], got: dict[str, np.ndarray]) -> float:
    """Largest |got - ref| / scale over every output column (see :func:`output_scale`). Spreads (``std``, ``LD_std``)
    are measured relative to the quantity they spread around."""
    scale, worst = output_scale(zm), 0.0
    for o, r in ref.items():
        g = np.asarray(got[o], float)
        s = np.broadcast_to(scale[o], r.shape).copy()
        rel = np.abs(ref["mean"] if o == "std" else r)
        if o == "ld":
            # L/D = Cl / Cd crosses zero near alpha -2; there the error a Cl within tolerance implies is
            # (Cl scale) / Cd, so L/D is held to the same standard as Cl: scale max(|L/D|, y_std(Cl) / Cd)
            floor = zm.config["y_std"][0] / np.abs(ref["mean"][:, 1])
            rel = np.maximum(np.abs(np.stack([r[:, 0], r[:, 0]], 1)), floor[:, None])
        s = np.where(np.isnan(s), rel, s)
        worst = max(worst, float(np.max(np.abs(g - r) / np.maximum(s, 1e-12))))
    return worst


# ====================================================================== export
def _export(module, args, input_names, output_names_, dynamic) -> bytes:
    import onnx
    import torch
    buf = io.BytesIO()
    kw = dict(input_names=input_names, output_names=output_names_, opset_version=OPSET, dynamic_axes=dynamic,
              do_constant_folding=True)
    try:
        torch.onnx.export(module.eval(), args, buf, dynamo=False, **kw)
    except TypeError:                                                  # torch < 2.5 has no dynamo flag
        torch.onnx.export(module.eval(), args, buf, **kw)
    m = onnx.load_from_string(buf.getvalue())
    m.producer_name, m.producer_version = "pinneapple", "1"
    onnx.checker.check_model(m)
    return m.SerializeToString()


def export_bytes(zm: ZooModel, *, checks: bool = False) -> bytes:
    """The ONNX graph of a loaded zoo model (see :func:`export_onnx`)."""
    import torch
    pred = predictor_module(zm)
    outs = list(output_names(zm))
    d = len(zm.input_names)
    if not checks:
        return _export(pred, (torch.zeros(4, d),), ["x"], outs, {"x": {0: "n"}, **{o: {0: "n"} for o in outs}})
    from .checks import checks_module
    qc = checks_module(zm, pred)

    class WithChecks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pred, self.qc = pred, qc

        def forward(self, x, query):
            return (*self.pred(x), *self.qc(query))

    names = outs + ["checks", "aux"]
    dyn = {"x": {0: "n"}, "query": {0: "m"}, **{o: {0: "n"} for o in outs}, "checks": {0: "m"}, "aux": {0: "m"}}
    return _export(WithChecks(), (torch.zeros(4, d), torch.zeros(2, d)), ["x", "query"], names, dyn)


def export_onnx(name: str | ZooModel, path: str | Path, *, checks: bool = False, check_parity: bool = True,
                offline: bool | None = None) -> dict[str, Any]:
    """Export a zoo model to ONNX at ``path``. The graph maps the card's inputs (input ``x``, shape (n, d)) to its
    outputs; with ``checks=True`` it also takes ``query`` (m, d) and returns the query-check values (``checks``,
    ``aux``; see :func:`pinneapple_hub.zoo.checks.check_spec`). Unless ``check_parity=False`` the export is refused
    (``ValueError``) when the predictions differ from the PyTorch model by more than 1e-5 (:func:`onnx_parity`) or a
    query-check status differs from Python away from its limit (:func:`checks_parity`).
    Returns ``{"path", "bytes", "sha256", "parity", "inputs", "outputs"}``."""
    import hashlib

    from . import load
    zm = load(name, offline=offline) if isinstance(name, str) else name
    data = export_bytes(zm, checks=checks)
    info = {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "inputs": ["x", "query"] if checks else ["x"], "input_columns": zm.input_names,
            "outputs": {**output_names(zm), **({"checks": "query checks", "aux": "check details"} if checks else {})},
            "opset": OPSET}
    if check_parity:
        info["parity"] = onnx_parity(zm, data)
        if info["parity"] > PARITY_TOL:
            raise ValueError(f"{zm.name}: ONNX differs from PyTorch by {info['parity']:.2e} (> {PARITY_TOL:g})")
        if checks:
            info["checks_parity"] = checks_parity(zm, data)
            if info["checks_parity"]["status_flips"]:
                raise ValueError(f"{zm.name}: ONNX query checks disagree with Python: {info['checks_parity']}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return info


# ====================================================================== parity
def parity_inputs(zm: ZooModel, *, n_random: int = 500, seed: int = 7) -> np.ndarray:
    """The model's held-out and independent-reference inputs (every evaluation split) plus random points inside the
    validity box (when the box describes every input)."""
    X = np.concatenate([zm.reference(s, offline=True)[0] for s in zm.eval_splits])
    b = zm.bounds()
    if n_random and all(k in b for k in zm.input_names):
        lo = np.array([b[k][0] for k in zm.input_names])
        hi = np.array([b[k][1] for k in zm.input_names])
        X = np.concatenate([X, lo + np.random.default_rng(seed).random((n_random, len(lo))) * (hi - lo)])
    return X


def run_onnx(data: bytes | str | Path, X: np.ndarray, query: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Run a graph in onnxruntime (CPU, one thread) and return its outputs by name (float64)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    src = Path(data).read_bytes() if isinstance(data, (str, Path)) else data
    s = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    feeds = {"x": np.asarray(X, np.float32)}
    if any(i.name == "query" for i in s.get_inputs()):
        feeds["query"] = np.asarray(X if query is None else query, np.float32)
    res = s.run(None, feeds)
    return {o.name: np.asarray(r, np.float64) for o, r in zip(s.get_outputs(), res, strict=True)}


def onnx_parity(name: str | ZooModel, onnx_model: bytes | str | Path, X: np.ndarray | None = None, *,
                offline: bool | None = None) -> float:
    """Largest scaled difference (:func:`parity_error`) between an exported graph's predictions and
    ``ZooModel.predict`` on ``X`` (default :func:`parity_inputs`)."""
    from . import load
    zm = load(name, offline=offline) if isinstance(name, str) else name
    X = parity_inputs(zm) if X is None else np.asarray(X, float)
    got = run_onnx(onnx_model, X)
    ref = as_graph_outputs(zm, zm._forward(X))
    return parity_error(zm, ref, {k: got[k] for k in ref})


CHECK_TOL = 1e-3


def checks_parity(name: str | ZooModel, onnx_model: bytes | str | Path, X: np.ndarray | None = None, *,
                  offline: bool | None = None) -> dict[str, float]:
    """Compare the query checks of a graph exported with ``checks=True`` with
    :func:`pinneapple_hub.zoo.checks.check_values`: the largest value difference (relative to max(|value|, 1)) and
    the number of rows where a status differs although the value is farther than ``CHECK_TOL`` from its limit
    (float32 finite differences can flip a status only right at a limit)."""
    from . import load
    from .checks import check_spec, check_values, compare
    zm = load(name, offline=offline) if isinstance(name, str) else name
    X = parity_inputs(zm) if X is None else np.asarray(X, float)
    got = run_onnx(onnx_model, X)
    v, aux = check_values(zm, X)
    diff = 0.0
    for a, b in ((v, got["checks"]), (aux, got["aux"])):
        fin = np.isfinite(a)
        diff = max(diff, float(np.max(np.abs(b[fin] - a[fin]) / np.maximum(np.abs(a[fin]), 1.0))))
    flips = 0
    for c, col in zip(check_spec(zm)["checks"], range(v.shape[1]), strict=False):
        lims = c["limit"] if c["op"] == "in" else [c["limit"]]
        for i in range(len(X)):
            if compare(v[i, col], c["op"], c["limit"]) != compare(got["checks"][i, col], c["op"], c["limit"]):
                if min(abs(v[i, col] - lim) for lim in lims) > CHECK_TOL:
                    flips += 1
    return {"max_value_diff": diff, "status_flips": flips}


# ====================================================================== safetensors
def export_safetensors(name: str | ZooModel, path: str | Path, *, offline: bool | None = None) -> dict[str, Any]:
    """Write the model's weights (the tensors of ``weights.pt``) as safetensors at ``path``; the file's metadata
    names the model, its version and adapter. Returns ``{"path", "bytes", "sha256", "tensors"}``."""
    import hashlib

    from safetensors.torch import save_file

    from . import load
    zm = load(name, offline=offline) if isinstance(name, str) else name
    state = {k: v.detach().cpu().contiguous().clone() for k, v in zm.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata={"format": "pt", "model": zm.name, "version": str(zm.card.version),
                                          "adapter": zm.adapter, "library": "pinneapple"})
    data = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "tensors": len(state)}


def load_safetensors(name: str | ZooModel, path: str | Path, *, offline: bool | None = None) -> ZooModel:
    """The zoo model ``name`` (config and card from the zoo) with its weights read from a safetensors file."""
    from safetensors.torch import load_file

    from . import load
    from ._models import copy_model
    zm = copy_model(load(name, offline=offline) if isinstance(name, str) else name)
    zm.module.load_state_dict(load_file(str(path)))
    zm.module.eval()
    return zm
