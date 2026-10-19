"""One model interface for uncertainty and sensitivity studies: ``predict(X) -> (n, m)`` with input/output names,
units and validity range, whatever the model is.

- :meth:`StudyModel.from_zoo`: a ``pp.zoo`` model (torch); exposes the physical-inputs torch module for automatic
  differentiation, the ensemble spread when the model has one, its validity range and its reference dataset splits;
- :meth:`StudyModel.from_onnx`: an ONNX graph (no Python code from the user is run): one ``(n, d)`` float input,
  outputs concatenated in graph order; needs ``onnxruntime``;
- :meth:`StudyModel.from_function`: any vectorised Python callable (the built-in benchmark functions, a closed-form
  model, a solver wrapper);
- :func:`builtin`: the analytic benchmarks of :mod:`pinneapple_analysis.sensitivity.testfunctions`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StudyModel:
    name: str
    kind: str
    input_names: list[str]
    output_names: list[str]
    predict_fn: Callable[[np.ndarray], np.ndarray] = field(repr=False)
    units: dict[str, str] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    description: str = ""
    torch_fn: Callable | None = field(default=None, repr=False)
    std_fn: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False)
    reference_dataset: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def d(self) -> int:
        return len(self.input_names)

    def predict(self, X, outputs: Sequence[str] | None = None, batch: int = 50000) -> np.ndarray:
        """Outputs ``(n, k)`` for inputs ``(n, d)`` (columns in ``input_names`` order)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.d:
            raise ValueError(f"{self.name} takes {self.d} inputs ({', '.join(self.input_names)}); got {X.shape[1]}")
        parts = [np.asarray(self.predict_fn(X[i:i + batch]), dtype=float) for i in range(0, X.shape[0], batch)]
        Y = np.concatenate([p[:, None] if p.ndim == 1 else p for p in parts]) if parts else np.empty((0, 0))
        if outputs is None:
            return Y
        return Y[:, [self.output_index(o) for o in outputs]]

    def output_index(self, name: str) -> int:
        if name not in self.output_names:
            raise ValueError(f"{self.name} has no output {name!r}; outputs: {', '.join(self.output_names)}")
        return self.output_names.index(name)

    def std(self, X, outputs: Sequence[str] | None = None) -> np.ndarray | None:
        """Ensemble spread per output, or ``None`` when the model has none."""
        if self.std_fn is None:
            return None
        S = np.asarray(self.std_fn(np.atleast_2d(np.asarray(X, dtype=float))), dtype=float)
        S = S[:, None] if S.ndim == 1 else S
        return S if outputs is None else S[:, [self.output_index(o) for o in outputs]]

    def reference(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Inputs and true outputs of a split of the model's reference dataset (zoo models)."""
        if not self.reference_dataset:
            raise ValueError(f"{self.name} has no reference dataset")
        from pinneapple_hub.zoo import datasets
        ds = datasets.load(self.reference_dataset, split=split)
        X = np.column_stack([np.asarray(ds[c], dtype=float) for c in self.input_names])
        Y = np.column_stack([np.asarray(ds[c], dtype=float) if c in ds.columns else np.full(len(X), np.nan)
                             for c in self.output_names])
        return X, Y

    def info(self) -> dict:
        return {"name": self.name, "kind": self.kind, "inputs": self.input_names, "outputs": self.output_names,
                "units": self.units, "bounds": {k: list(v) for k, v in self.bounds.items()},
                "description": self.description, "differentiable": self.torch_fn is not None,
                "ensemble_spread": self.std_fn is not None, "reference_dataset": self.reference_dataset,
                **({"metadata": self.metadata} if self.metadata else {})}

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_function(cls, fn: Callable[[np.ndarray], np.ndarray], input_names: Sequence[str],
                      output_names: Sequence[str], name: str = "function", **kw) -> StudyModel:
        return cls(name=name, kind="function", input_names=list(input_names), output_names=list(output_names),
                   predict_fn=fn, **kw)

    @classmethod
    def from_zoo(cls, name: str, outputs: Sequence[str] | None = None) -> StudyModel:
        from pinneapple_hub import zoo
        zm = zoo.load(name)
        card = zm.card
        outs = list(outputs or zm.output_names)
        names = list(zm.input_names)

        def as_inputs(X):
            return {n: X[:, i] for i, n in enumerate(names)}

        def predict(X):
            out = zm.predict(**as_inputs(X), warn=False)
            return np.column_stack([np.asarray(out[o], dtype=float) for o in outs])

        lineage = card.data_lineage or {}
        probe = {}
        if lineage.get("dataset"):
            from pinneapple_hub.zoo import datasets
            ds = datasets.load(lineage["dataset"], split="test")
            probe = zm.predict(**{n: np.asarray(ds[n][:1], dtype=float) for n in names}, warn=False)
        std_fn = None
        if outs and all(f"{o}_std" in probe for o in outs):
            def std_fn(X):
                out = zm.predict(**as_inputs(X), warn=False)
                return np.column_stack([np.asarray(out[f"{o}_std"], dtype=float) for o in outs])

        torch_fn = None
        try:
            from pinneapple_hub.zoo.export import output_names as graph_outputs
            from pinneapple_hub.zoo.export import predictor_module
            module = predictor_module(zm).eval()
            first = list(graph_outputs(zm).values())[0]
            cols = [first.index(o) for o in outs] if all(o in first for o in outs) else None
            if cols is not None:
                def torch_fn(x):
                    return module(x)[0][:, cols]
        except (NotImplementedError, ImportError, KeyError):
            torch_fn = None
        units = {d["name"]: d.get("unit", "") for d in list(card.inputs) + list(card.outputs)}
        return cls(name=name, kind="zoo", input_names=names, output_names=outs, predict_fn=predict, units=units,
                   bounds={k: tuple(v) for k, v in zm.bounds().items()}, description=card.task,
                   torch_fn=torch_fn, std_fn=std_fn, reference_dataset=lineage.get("dataset"),
                   metadata={"version": card.version, "validation_metrics": card.validation_metrics,
                             "known_limitations": list(card.known_limitations),
                             "validity_range": card.validity_range, "reference_source": card.reference_source})

    @classmethod
    def from_onnx(cls, data: bytes, input_names: Sequence[str] | None = None,
                  output_names: Sequence[str] | None = None, name: str = "onnx model") -> StudyModel:
        """An ONNX graph with one float input of shape ``(n, d)``. Names default to the graph's metadata
        (``input_names`` / ``output_names`` comma-separated entries) or ``x0..``/``y0..``."""
        try:
            import onnxruntime as ort
        except ImportError as e:                                  # pragma: no cover - depends on the install
            raise ImportError("ONNX models need onnxruntime: pip install onnxruntime") from e
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(bytes(data), so, providers=["CPUExecutionProvider"])
        ins = sess.get_inputs()
        if len(ins) != 1:
            raise ValueError(f"the graph has {len(ins)} inputs; one (n, d) input is supported")
        shape = ins[0].shape
        d = shape[1] if len(shape) == 2 and isinstance(shape[1], int) else None
        meta = sess.get_modelmeta().custom_metadata_map or {}
        in_names = list(input_names or [s.strip() for s in meta.get("input_names", "").split(",") if s.strip()])
        if not in_names:
            if d is None:
                raise ValueError("the graph's input width is dynamic: give the input names")
            in_names = [f"x{i}" for i in range(d)]
        if d is not None and len(in_names) != d:
            raise ValueError(f"{len(in_names)} input names for a graph with {d} inputs")
        dtype = np.float64 if "double" in ins[0].type else np.float32
        feed = ins[0].name

        def predict(X):
            res = sess.run(None, {feed: np.asarray(X, dtype=dtype)})
            return np.column_stack([np.asarray(r, dtype=float).reshape(len(X), -1) for r in res])

        n_out = predict(np.zeros((1, len(in_names)))).shape[1]
        out_names = list(output_names or [s.strip() for s in meta.get("output_names", "").split(",") if s.strip()])
        if not out_names:
            out_names = [o.name for o in sess.get_outputs()] if len(sess.get_outputs()) == n_out else \
                [f"y{j}" for j in range(n_out)]
        if len(out_names) != n_out:
            raise ValueError(f"{len(out_names)} output names for a graph with {n_out} outputs")
        return cls(name=name, kind="onnx", input_names=in_names, output_names=out_names, predict_fn=predict,
                   description="ONNX graph (onnxruntime, CPU)")


def builtin(name: str) -> StudyModel:
    """``"ishigami"`` or ``"sobol_g"`` (6 inputs, a = 0, 1, 4.5, 9, 99, 99), with their exact Sobol indices in
    ``metadata["exact"]``."""
    from pinneapple_analysis.sensitivity import testfunctions as tf
    if name == "ishigami":
        ex = tf.ishigami_indices()
        return StudyModel.from_function(tf.ishigami, ["x1", "x2", "x3"], ["y"], name="ishigami",
                                        bounds={f"x{i}": (-np.pi, np.pi) for i in (1, 2, 3)},
                                        description="Ishigami & Homma (1990): sin x1 + 7 sin^2 x2 + 0.1 x3^4 sin x1",
                                        metadata={"exact": {"S1": ex["S1"].tolist(), "ST": ex["ST"].tolist(),
                                                            "S2": {"x1,x3": ex["S2"][(0, 2)]}, "V": ex["V"]},
                                                  "space": tf.ishigami_space()})
    if name == "sobol_g":
        a = [0.0, 1.0, 4.5, 9.0, 99.0, 99.0]
        ex = tf.sobol_g_indices(a)
        return StudyModel.from_function(lambda X: tf.sobol_g(X, a), [f"x{i + 1}" for i in range(6)], ["y"],
                                        name="sobol_g", bounds={f"x{i + 1}": (0.0, 1.0) for i in range(6)},
                                        description="Sobol' G function, a = 0, 1, 4.5, 9, 99, 99",
                                        metadata={"exact": {"S1": ex["S1"].tolist(), "ST": ex["ST"].tolist(),
                                                            "V": ex["V"]}, "a": a,
                                                  "space": tf.sobol_g_space(6)})
    raise ValueError(f"unknown built-in model {name!r}; use 'ishigami' or 'sobol_g'")


def columns(table: Mapping[str, Sequence] | object, names: Sequence[str]) -> np.ndarray:
    """Stack the named columns of a table (dict of arrays or a DataFrame) into ``(n, k)`` floats."""
    missing = [n for n in names if n not in (table.columns if hasattr(table, "columns") else table)]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    return np.column_stack([np.asarray(table[n], dtype=float) for n in names])


__all__ = ["StudyModel", "builtin", "columns"]
