"""SciREX adapter with explicit layout, units, checkpoint provenance and lazy imports.

SciREX 1.0 public models are FNO, GINO and Transolver. Training here is a
minimal supervised JAX loop; physics-only objectives require a separate,
problem-specific residual and verification before promotion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
from typing import Any
import numpy as np

_MODELS = {"fno": "FNO", "gino": "GINO", "transolver": "Transolver"}
_LAYOUTS = {"fno": "channels_first", "gino": "points_last", "transolver": "points_last"}

@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    parameters: dict[str, Any]
    seed: int = 0
    def __post_init__(self):
        if self.architecture not in _MODELS:
            raise ValueError(f"Unsupported SciREX architecture: {self.architecture}")

@dataclass(frozen=True)
class FieldSpec:
    layout: str
    channels: int
    units: tuple[str, ...]
    coordinate_units: str = "m"
    spatial_dim: int = 2
    def __post_init__(self):
        if self.layout not in ("channels_first", "points_last"):
            raise ValueError("Unsupported layout")
        if self.channels <= 0 or len(self.units) != self.channels:
            raise ValueError("Each output channel needs an explicit physical unit")
        if self.spatial_dim not in (1, 2, 3) or not self.coordinate_units:
            raise ValueError("Invalid geometry metadata")

def available_models() -> tuple[str, ...]:
    return tuple(_MODELS)

def validate_field(array: Any, field: FieldSpec, *, coordinates: Any = None) -> np.ndarray:
    x = np.asarray(array)
    if not np.issubdtype(x.dtype, np.number) or not np.all(np.isfinite(x)):
        raise ValueError("Field must be numeric and finite")
    if field.layout == "channels_first":
        if x.ndim != field.spatial_dim + 2 or x.shape[1] != field.channels:
            raise ValueError("Expected (batch, channels, *spatial)")
    elif x.ndim != 3 or x.shape[-1] != field.channels:
        raise ValueError("Expected (batch, points, channels)")
    if coordinates is not None:
        c = np.asarray(coordinates)
        if not np.all(np.isfinite(c)) or c.shape[-1] != field.spatial_dim:
            raise ValueError("Coordinates must be finite and match spatial_dim")
        if field.layout == "points_last" and c.shape[-2] != x.shape[-2]:
            raise ValueError("Coordinate and field point counts differ")
    return x.astype(np.float32, copy=False)

class ScirexAdapter:
    def __init__(self, spec: ModelSpec, field: FieldSpec):
        if field.layout != _LAYOUTS[spec.architecture]:
            raise ValueError(f"{spec.architecture} expects {_LAYOUTS[spec.architecture]}")
        self.spec, self.field = spec, field
        self.model = None
        self.params = None
        self.normalization = None

    @staticmethod
    def _dependencies():
        try:
            import jax
            import jax.numpy as jnp
            from scirex import models
        except ImportError as exc:
            raise ImportError("Install SciREX and JAX in an isolated worker: pip install scirex") from exc
        return jax, jnp, models

    def initialize(self, *example_inputs: Any):
        jax, jnp, models = self._dependencies()
        self.model = getattr(models, _MODELS[self.spec.architecture])(**self.spec.parameters)
        inputs = tuple(jnp.asarray(x) for x in example_inputs)
        self.params = self.model.init(jax.random.PRNGKey(self.spec.seed), *inputs)
        return self

    def fit_normalizer(self, training_data: Any):
        """Fit only on training data; callers must split BEFORE calling."""
        x = validate_field(training_data, self.field)
        axes = tuple(i for i in range(x.ndim) if i != (1 if self.field.layout == "channels_first" else x.ndim - 1))
        mean, std = x.mean(axis=axes, keepdims=True), x.std(axis=axes, keepdims=True)
        self.normalization = {"mean": mean, "std": np.maximum(std, 1e-8)}
        return self

    def normalize(self, values: Any) -> np.ndarray:
        x = validate_field(values, self.field)
        if self.normalization is None:
            raise RuntimeError("Fit normalization on the training split first")
        return (x - self.normalization["mean"]) / self.normalization["std"]

    def predict(self, *inputs: Any) -> np.ndarray:
        if self.model is None or self.params is None:
            raise RuntimeError("Initialize or load a model first")
        jax, jnp, _ = self._dependencies()
        return np.asarray(jax.jit(self.model.apply)(self.params, *(jnp.asarray(x) for x in inputs)))

    def train_supervised(self, inputs: Any, targets: Any, *, epochs: int = 10, learning_rate: float = 1e-3):
        """Minimal FNO supervised loop; GINO/Transolver need geometry-aware batches."""
        if self.spec.architecture != "fno":
            raise NotImplementedError("Geometry-aware GINO/Transolver training requires a dedicated batch adapter")
        if epochs < 1 or learning_rate <= 0:
            raise ValueError("Invalid training hyperparameters")
        import optax
        from scirex.training import adamw_cx
        jax, jnp, _ = self._dependencies()
        x = jnp.asarray(inputs)
        y = jnp.asarray(validate_field(targets, self.field))
        if x.shape[0] != y.shape[0]:
            raise ValueError("Input/target batch mismatch")
        if self.model is None:
            self.initialize(x[:1])
        optimizer = adamw_cx(learning_rate=learning_rate)
        state = optimizer.init(self.params)
        def step(params, state):
            def loss_fn(p):
                pred = self.model.apply(p, x)
                return jnp.mean((pred - y) ** 2)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_state = optimizer.update(grads, state, params)
            return optax.apply_updates(params, updates), new_state, loss
        step = jax.jit(step)
        history = []
        for _ in range(epochs):
            self.params, state, loss = step(self.params, state)
            history.append(float(loss))
        return history

    def save(self, directory: str | Path, *, dataset_id: str, source_revision: str | None = None):
        if self.model is None or self.params is None:
            raise RuntimeError("No initialized model")
        dest = Path(directory)
        dest.mkdir(parents=True, exist_ok=True)
        self.model.save_checkpoint(dest, "model", self.params)
        manifest = {
            "architecture": self.spec.architecture,
            "parameters": self.spec.parameters,
            "seed": self.spec.seed,
            "field": asdict(self.field),
            "dataset_id": dataset_id,
            "source_revision": source_revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "versions": {name: metadata.version(name) for name in ("scirex", "jax", "flax")},
            "normalization": None if self.normalization is None else {
                k: v.tolist() for k, v in self.normalization.items()
            },
            "verification_status": "not_run",
        }
        (dest / "pinneapple_manifest.json").write_text(json.dumps(manifest, indent=2))
        return dest
