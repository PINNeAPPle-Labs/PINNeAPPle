"""A trained surrogate packaged for use: predict with the OOD gate, save / load, export, model card.

``SurrogateModel.save(dir)`` writes ``model.pkl`` (the fitted candidate, pickled) and ``card.json`` (inputs,
outputs, validity bounds, OOD gate, held-out metrics, limitations, trust report, dataset hash, the model file's
sha256). ``load`` verifies that hash before unpickling. Unpickling runs code: load only models you produced
(the gateway stores the ones it publishes; it never unpickles an upload).

``export(path, "torchscript" | "onnx")`` writes a self-contained graph of the PyTorch candidates (mlp, fno,
deeponet, pinn) that takes raw inputs ``(n, d)`` and returns raw outputs ``(n, p)`` (normalisation included); the
scikit-learn and XGBoost candidates have no such export here.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .limits import OODGate
from .methods import Fitted, NonNegative, _Torch

__all__ = ["SurrogateModel", "MODEL_KIND"]

MODEL_KIND = "pinneapple.surrogate"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class SurrogateModel:
    def __init__(self, fitted: Fitted, gate: OODGate, *, input_names: Sequence[str], output_names: Sequence[str],
                 grid_shape: Sequence[int] = (), coords: np.ndarray | None = None, coord_names: Sequence[str] = (),
                 card: Mapping[str, Any] | None = None):
        self.fitted, self.gate = fitted, gate
        self.input_names, self.output_names = list(input_names), list(output_names)
        self.grid_shape = tuple(int(s) for s in grid_shape)
        self.coords = None if coords is None else np.asarray(coords, float)
        self.coord_names = list(coord_names)
        self.card: dict[str, Any] = dict(card or {})
        self.card.setdefault("kind", MODEL_KIND)
        self.card.setdefault("created", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.card.update(method=fitted.method, inputs=self.input_names, outputs=self.output_names,
                         grid_shape=list(self.grid_shape), coord_names=self.coord_names,
                         validity_bounds=gate.bounds(), ood_gate=gate.to_dict(), n_params=int(fitted.n_params))

    @property
    def kind(self) -> str:
        return "field" if self.grid_shape else "scalar"

    # ------------------------------------------------------------------------------------------ predict
    def as_array(self, inputs: Mapping[str, Any] | Sequence | np.ndarray) -> np.ndarray:
        """Inputs as ``{name: value or list}`` (broadcast) or an ``(n, d)`` array in ``input_names`` order."""
        if isinstance(inputs, Mapping):
            missing = [k for k in self.input_names if k not in inputs]
            if missing:
                raise ValueError(f"missing inputs: {missing} (the model needs {self.input_names})")
            cols = np.broadcast_arrays(*[np.atleast_1d(np.asarray(inputs[k], float)) for k in self.input_names])
            return np.stack([c.ravel() for c in cols], 1)
        X = np.atleast_2d(np.asarray(inputs, float))
        if X.shape[1] != len(self.input_names):
            raise ValueError(f"inputs have {X.shape[1]} columns; the model needs {len(self.input_names)}")
        return X

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        Y = self.fitted.predict(np.asarray(X, float))
        return Y.reshape(len(Y), *self.grid_shape) if self.grid_shape else Y

    def predict(self, inputs, *, ood_policy: str = "warn") -> dict[str, Any]:
        """``{"outputs": {name: list}, "ood": [per-row assessment], "n": rows}``. ``ood_policy``: ``"warn"``
        predicts every row and flags the out-of-distribution ones; ``"refuse"`` returns ``None`` for them;
        ``"ignore"`` skips the gate (rows are still assessed and reported)."""
        if ood_policy not in ("warn", "refuse", "ignore"):
            raise ValueError("ood_policy must be warn, refuse or ignore")
        X = self.as_array(inputs)
        flags = self.gate.assess(X)
        ok = np.array([not f["ood"] for f in flags]) if ood_policy == "refuse" else np.ones(len(X), bool)
        Y = np.full((len(X), *(self.grid_shape or (len(self.output_names),))), np.nan)
        if ok.any():
            Y[ok] = self.predict_array(X[ok])
        if self.grid_shape:
            outs = {self.output_names[0]: [y.tolist() if k else None for y, k in zip(Y, ok, strict=True)]}
        else:
            outs = {n: [float(v) if k else None for v, k in zip(Y[:, j], ok, strict=True)]
                    for j, n in enumerate(self.output_names)}
        return {"outputs": outs, "ood": flags, "n": int(len(X)), "refused": int((~ok).sum()),
                "ood_policy": ood_policy}

    # ------------------------------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "model.pkl", "wb") as f:
            pickle.dump({"fitted": self.fitted, "gate": self.gate, "coords": self.coords}, f)
        self.card["model_sha256"] = _sha(d / "model.pkl")
        (d / "card.json").write_text(json.dumps(self.card, indent=1, default=_json_default))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> SurrogateModel:
        d = Path(directory)
        card = json.loads((d / "card.json").read_text())
        if card.get("kind") != MODEL_KIND:
            raise ValueError(f"{d} is not a {MODEL_KIND} model (kind {card.get('kind')!r})")
        if card.get("model_sha256") != _sha(d / "model.pkl"):
            raise ValueError(f"{d / 'model.pkl'} does not match the sha256 recorded in its card")
        with open(d / "model.pkl", "rb") as f:
            obj = pickle.load(f)  # noqa: S301 (verified hash, written by save)
        return cls(obj["fitted"], obj["gate"], input_names=card["inputs"], output_names=card["outputs"],
                   grid_shape=card.get("grid_shape") or (), coords=obj["coords"], coord_names=card.get("coord_names", ()),
                   card=card)

    def model_ref(self, *, name: str, version: str, uri: str = "") -> dict[str, Any]:
        """The fields a ModelRef carries: what the model is, where it is, what it may be used for."""
        c = self.card
        return {"name": name, "version": version, "kind": MODEL_KIND, "uri": uri, "method": c.get("method"),
                "inputs": self.input_names, "outputs": self.output_names, "grid_shape": list(self.grid_shape),
                "validity_bounds": c.get("validity_bounds"), "metrics": c.get("metrics", {}),
                "dataset_sha256": c.get("dataset_sha256"), "sha256": c.get("model_sha256"),
                "trust": (c.get("trust") or {}).get("decision"), "created": c.get("created")}

    # ------------------------------------------------------------------------------------------ export
    def export(self, path: str | Path, fmt: str = "torchscript") -> Path:
        import torch
        f, mask = self.fitted, None
        if isinstance(f, NonNegative):
            f, mask = f.inner, f.mask
        if not isinstance(f, _Torch):
            raise NotImplementedError(f"{f.method}: only the PyTorch candidates (mlp, fno, deeponet, pinn) export "
                                      "to TorchScript / ONNX here")

        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.module = f.module
                t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32)  # noqa: E731
                self.register_buffer("xm", t(f.xm))
                self.register_buffer("xs", t(f.xs))
                self.register_buffer("ym", t(f.ym))
                self.register_buffer("ys", t(f.ys))
                self.extra = f.extra
                self.register_buffer("sq", torch.as_tensor(np.zeros(len(f.ym), bool) if mask is None else mask))

            def forward(self, x):
                y = f.forward_fn(self.module, (x - self.xm) / self.xs, self.extra) * self.ys + self.ym
                return torch.where(self.sq, y * y, y)               # non-negative outputs: the square

        w = Wrapped().eval()
        ex = torch.as_tensor(self.gate.lo[None] * 0.5 + self.gate.hi[None] * 0.5, dtype=torch.float32)
        path = Path(path)
        if fmt == "torchscript":
            import warnings
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)          # jit.trace still works; it is deprecated
                torch.jit.trace(w, ex, check_trace=False).save(str(path))
        elif fmt == "onnx":
            torch.onnx.export(w, (ex,), str(path), input_names=["inputs"], output_names=["outputs"],
                              dynamic_axes={"inputs": {0: "n"}, "outputs": {0: "n"}}, dynamo=False)
        else:
            raise ValueError("fmt must be torchscript or onnx")
        return path


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)
