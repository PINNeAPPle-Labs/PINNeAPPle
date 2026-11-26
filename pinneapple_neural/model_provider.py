"""One interface over the doors a trained model reaches the library through (#1477).

A model arrives as a PyTorch checkpoint, a TorchScript file, an ONNX graph, a JAX pytree, a PhysicsNeMo ``.mdlus``
checkpoint, a Hugging Face repo, or a model this library built itself. Each door has its own loading convention and its
own way of reporting failure, and a saved file rarely says which door it came through — so a comparison between "the
same model on two runtimes" is a guess.

This module gives them the same three calls::

    from pinneapple_neural.model_provider import load

    m = load("runs/beam/model.pt")                 # whichever door it came through
    m.provenance                                   # {"provider": "pytorch", "provider_version": "2.15.0", ...}
    y = m.predict(x)                               # a numpy array in, a numpy array out
    report = m.evaluate({"x": x, "y": y_expected})  # reuses pinneapple_analysis.validation
    m.export("onnx", "beam.onnx")                  # or "torchscript", or the artifact directory format

Honesty rules
-------------
* **A model that is not exportable is refused with the reason, not exported into something that loads and computes
  something else.** The ONNX path refuses in three named cases: the exporter raises (the message names the operator),
  PyTorch's own tracer warns that it is tracing *one branch* of data-dependent control flow (the warning names the
  conversion), or the reloaded graph disagrees with the eager model on a fixture input. The last two are the ones a
  naive ``torch.onnx.export`` turns into a silently wrong file.
* **A missing backend is a refusal that names the package**, never a ``ModuleNotFoundError`` from inside a load
  (``ProviderUnavailable``).
* **The provider is recorded with the model.** :class:`Provenance` carries the provider, the version of the package
  that ran it, the reference and, for a local file, its SHA-256. Two models that differ only in provider are then
  visible as such in a comparison instead of being inferred.

What this is not: it is not :class:`pinneapple_neural.physics_model.PhysicsModel`, which is the contract over the
model *families* in this library (fit, predict, uncertainty, export). Where a loaded object is one of those families,
:meth:`LoadedModel.physics_model` hands it over — the two interfaces meet there and nowhere else.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ARTIFACT_SCHEMA",
    "ExportNotSupported",
    "LoadedModel",
    "ModelNotLoadable",
    "ModelProvider",
    "Provenance",
    "ProviderUnavailable",
    "available_providers",
    "detect",
    "get_provider",
    "load",
    "providers",
    "register_provider",
]

#: The manifest of the artifact directory format (``export("artifact", dir)``).
ARTIFACT_SCHEMA = "pinneapple.model_artifact/1"
_MANIFEST = "pinneapple_model.json"


class ModelNotLoadable(RuntimeError):
    """No provider could open the reference, and the message says what was tried."""


class ExportNotSupported(RuntimeError):
    """The model cannot be exported to the requested target, and the message names what got in the way."""

    def __init__(self, message: str, *, target: str = "", operation: str = ""):
        self.target = target
        self.operation = operation                  # the operator, conversion or branch that refused
        super().__init__(message)


class ProviderUnavailable(RuntimeError):
    """The provider exists but the package it needs is not installed (the message names the package)."""


# -------------------------------------------------------------------------------------------------- records
@dataclass
class Provenance:
    """Where a loaded model came from and what is running it."""

    provider: str
    provider_version: str = ""
    ref: str = ""
    framework: str = ""
    sha256: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "provider_version": self.provider_version, "ref": self.ref,
                "framework": self.framework, "sha256": self.sha256, **self.extra}

    def summary(self) -> str:
        bit = f"{self.provider} {self.provider_version}".strip()
        return f"{bit} <- {self.ref}" + (f" ({self.framework})" if self.framework else "")


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


# -------------------------------------------------------------------------------------------------- the providers
class ModelProvider(ABC):
    """One door. ``load`` returns the object and its provenance; ``predict`` and ``export`` work on that object.

    ``available()`` says whether this provider can run here at all (a missing package is reported, not raised from a
    load), so a caller can list the providers before choosing one.
    """

    name: str = ""
    #: The distribution whose version goes into the provenance.
    package: str = ""

    def available(self) -> tuple[bool, str]:
        if not self.package:
            return True, ""
        version = _package_version(self.package)
        import importlib.util

        if importlib.util.find_spec(self.package) is None:
            return False, f"{self.package} is not installed (pip install {self.package})"
        return True, f"{self.package} {version}".strip()

    @property
    def version(self) -> str:
        return _package_version(self.package) if self.package else ""

    @abstractmethod
    def can_load(self, ref: str | Path) -> bool:
        """Whether this provider recognises the reference (checked in :data:`_DETECTION_ORDER` order)."""

    @abstractmethod
    def load(self, ref: str | Path, **options: Any) -> tuple[Any, Provenance]:
        """The model object and its provenance."""

    @abstractmethod
    def predict(self, model: Any, x: Any) -> np.ndarray:
        """Run the model on neutral inputs (a numpy array or a mapping of arrays) and return the first output."""

    def export(self, model: Any, target: str, path: Path, *, example: Any = None) -> Path:
        raise ExportNotSupported(f"the {self.name} provider cannot export to {target!r}: it has no exporter for that "
                                 f"target (targets it can produce: {sorted(self.targets)})", target=target)

    #: Targets this provider can write, in the words ``export`` accepts.
    targets: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        ok, detail = self.available()
        return {"provider": self.name, "available": ok, "detail": detail, "targets": list(self.targets)}


# -- PyTorch ------------------------------------------------------------------------------------------------------
class PyTorchProvider(ModelProvider):
    """A checkpoint written by ``save_checkpoint`` (``class_name`` + ``model_kwargs`` since #941) or a pickled
    ``nn.Module``."""

    name = "pytorch"
    package = "torch"
    targets = ("onnx", "torchscript", "artifact")

    def can_load(self, ref: str | Path) -> bool:
        p = Path(ref)
        return p.is_file() and p.suffix.lower() in (".pt", ".pth", ".ckpt", ".bin") and not _is_torchscript(p)

    def load(self, ref: str | Path, **options: Any) -> tuple[Any, Provenance]:
        import torch

        from pinneapple_neural.architectures.base import load_checkpoint

        path = Path(ref)
        obj = load_checkpoint(path, **options)
        obj.eval()
        return obj, Provenance(provider=self.name, provider_version=torch.__version__, ref=str(path),
                               framework=f"pytorch {torch.__version__}", sha256=_sha256(path))

    def predict(self, model: Any, x: Any) -> np.ndarray:
        import torch

        tensor = torch.from_numpy(_as_float_array(x)) if not isinstance(x, Mapping) else {
            k: torch.from_numpy(_as_float_array(v)) for k, v in x.items()}
        with torch.no_grad():
            out = model(tensor)
        return _first_array(out)

    def export(self, model: Any, target: str, path: Path, *, example: Any = None) -> Path:
        if target == "onnx":
            return _onnx_export(model, path, example)
        if target == "torchscript":
            return _torchscript_export(model, path, example)
        raise ExportNotSupported(f"the {self.name} provider does not write {target!r}", target=target)


class TorchScriptProvider(ModelProvider):
    """A ``torch.jit`` archive: portable, no Python class needed to reload it."""

    name = "torchscript"
    package = "torch"
    targets = ("artifact",)

    def can_load(self, ref: str | Path) -> bool:
        p = Path(ref)
        return p.is_file() and p.suffix.lower() in (".pt", ".pth", ".ts") and _is_torchscript(p)

    def load(self, ref: str | Path, **options: Any) -> tuple[Any, Provenance]:
        import torch

        path = Path(ref)
        obj = torch.jit.load(str(path), map_location="cpu")
        obj.eval()
        return obj, Provenance(provider=self.name, provider_version=torch.__version__, ref=str(path),
                               framework=f"torchscript {torch.__version__}", sha256=_sha256(path))

    def predict(self, model: Any, x: Any) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = model(torch.from_numpy(_as_float_array(x)))
        return _first_array(out)

    def export(self, model: Any, target: str, path: Path, *, example: Any = None) -> Path:
        raise ExportNotSupported(
            "a TorchScript archive cannot be exported again: it has already been lowered to a graph, and the Python "
            "module it came from is not in the file (export the original model instead)", target=target)


class ONNXProvider(ModelProvider):
    """An ``.onnx`` graph through ``onnxruntime``. The portable end of the road: it runs, and it cannot be exported
    further from here."""

    name = "onnx"
    package = "onnxruntime"
    targets = ()

    def can_load(self, ref: str | Path) -> bool:
        return Path(ref).suffix.lower() == ".onnx"

    def load(self, ref: str | Path, **options: Any) -> tuple[Any, Provenance]:
        ok, why = self.available()
        if not ok:
            raise ProviderUnavailable(f"cannot load {ref}: {why}")
        import onnxruntime

        path = Path(ref)
        session = onnxruntime.InferenceSession(str(path), providers=options.get("providers") or
                                               ["CPUExecutionProvider"])
        self._inputs = [i.name for i in session.get_inputs()]
        return session, Provenance(provider=self.name, provider_version=onnxruntime.__version__, ref=str(path),
                                   framework=f"onnxruntime {onnxruntime.__version__}", sha256=_sha256(path),
                                   extra={"inputs": self._inputs})

    def predict(self, model: Any, x: Any) -> np.ndarray:
        names = [i.name for i in model.get_inputs()]
        array = _as_float_array(x)
        feed = {names[0]: array}
        outputs = model.run(None, feed)
        return np.asarray(outputs[0])

    def export(self, model: Any, target: str, path: Path, *, example: Any = None) -> Path:
        raise ExportNotSupported(
            f"an ONNX graph cannot be exported to {target!r} from here: ONNX Runtime runs the graph and the module "
            f"that produced it is not in the file. Export the original model to the target you need", target=target)


class JAXProvider(ModelProvider):
    """A JAX pytree saved next to the function that uses it: an ``.npz`` with ``params`` and a ``function`` entry
    naming ``module:fn``."""

    name = "jax"
    package = "jax"
    targets = ()

    def can_load(self, ref: str | Path) -> bool:
        return Path(ref).suffix.lower() in (".npz", ".jax")

    def load(self, ref: str | Path, **options: Any) -> tuple[Any, Provenance]:
        ok, why = self.available()
        if not ok:
            raise ProviderUnavailable(f"cannot load {ref}: {why}")
        import importlib

        import jax

        path = Path(ref)
        with np.load(path, allow_pickle=True) as saved:
            params = {k: np.asarray(v) for k, v in saved.items() if k != "function"}
            target = str(saved["function"]) if "function" in saved else str(options.get("function") or "")
        if not target:
            raise ModelNotLoadable(f"{path} has no `function` entry naming the callable to run with its parameters")
        module_name, _, attr = target.partition(":")
        fn = getattr(importlib.import_module(module_name), attr or "forward", None)
        if fn is None:
            raise ModelNotLoadable(f"{path} names {target!r}, which does not resolve to a callable")
        model = {"params": jax.tree_util.tree_map(lambda a: jax.numpy.asarray(a), params), "fn": fn,
                 "jitted": jax.jit(fn)}
        return model, Provenance(provider=self.name, provider_version=jax.__version__, ref=str(path),
                                 framework=f"jax {jax.__version__}", sha256=_sha256(path))

    def predict(self, model: Any, x: Any) -> np.ndarray:
        import jax.numpy as jnp

        out = model["jitted"](model["params"], jnp.asarray(_as_float_array(x)))
        return np.asarray(out)


class PhysicsNeMoProvider(ModelProvider):
    """A PhysicsNeMo ``.mdlus`` checkpoint (``physicsnemo.Module.from_checkpoint``); the object it returns is a torch
    module, so prediction and the torch exports work on it."""

    name = "physicsnemo"
    package = "physicsnemo"
    targets = ("onnx", "torchscript", "artifact")

    def can_load(self, ref: str | Path) -> bool:
        return Path(ref).suffix.lower() in (".mdlus", ".mdu")

    def load(self, ref: str | Path, **options: Any) -> tuple[Any, Provenance]:
        ok, why = self.available()
        if not ok:
            raise ProviderUnavailable(f"cannot load {ref}: {why}")
        import physicsnemo
        import torch

        path = Path(ref)
        obj = physicsnemo.Module.from_checkpoint(path, override_args=options.get("override_args") or None)
        obj.eval()
        return obj, Provenance(provider=self.name, provider_version=physicsnemo.__version__, ref=str(path),
                               framework=f"physicsnemo {physicsnemo.__version__} on torch {torch.__version__}",
                               sha256=_sha256(path))

    def predict(self, model: Any, x: Any) -> np.ndarray:
        return PyTorchProvider().predict(model, x)

    def export(self, model: Any, target: str, path: Path, *, example: Any = None) -> Path:
        return PyTorchProvider().export(model, target, path, example=example)


class HuggingFaceProvider(ModelProvider):
    """A Hugging Face repo, or a local mirror of one: the layout ``pinneapple_hub.push_to_hub`` writes
    (``pinneapple_config.json`` + ``pytorch_model.bin``), rebuilt through ``ModelRegistry``.

    A mirror directory is read without the network, which is how the tests exercise this provider. A bare
    ``"org/name"`` goes to ``huggingface_hub`` and needs the network and the ``hub`` extra.
    """

    name = "huggingface"
    package = "huggingface_hub"
    targets = ("onnx", "torchscript", "artifact")
    _CONFIG = "pinneapple_config.json"
    _WEIGHTS = "pytorch_model.bin"

    def can_load(self, ref: str | Path) -> bool:
        p = Path(ref)
        if p.is_dir():
            return (p / self._CONFIG).is_file() and (p / self._WEIGHTS).is_file()
        return False

    def load(self, ref: str | Path, **options: Any) -> tuple[Any, Provenance]:
        import torch

        import pinneapple_neural.architectures  # noqa: F401 - registers the model zoo
        from pinneapple_neural.architectures.registry import ModelRegistry

        p = Path(ref)
        if p.is_dir() and (p / self._CONFIG).is_file():
            config = json.loads((p / self._CONFIG).read_text(encoding="utf-8"))
            weights = p / self._WEIGHTS
            origin = f"local mirror {p}"
        else:                                            # a repo id: org/name
            if not self.available()[0]:
                raise ProviderUnavailable(f"cannot load {ref}: {self.available()[1]}")
            from pinneapple_hub.hub import _CONFIG_FILENAME, _WEIGHTS_FILENAME, _require_hub

            hub = _require_hub()
            revision = str(options.get("revision") or "main")
            config = json.loads(Path(hub.hf_hub_download(str(ref), _CONFIG_FILENAME,
                                                         revision=revision)).read_text(encoding="utf-8"))
            weights = Path(hub.hf_hub_download(str(ref), _WEIGHTS_FILENAME, revision=revision))
            origin = f"hub {ref}@{revision}"
        model = ModelRegistry.build(config["architecture"], **(config["architecture_config"] or {}))
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        model.eval()
        return model, Provenance(provider=self.name, provider_version=_package_version("huggingface_hub"),
                                 ref=str(ref), framework=f"pytorch {torch.__version__}",
                                 sha256=_sha256(weights) if weights.is_file() else "",
                                 extra={"origin": origin, "architecture": config["architecture"]})

    def predict(self, model: Any, x: Any) -> np.ndarray:
        return PyTorchProvider().predict(model, x)

    def export(self, model: Any, target: str, path: Path, *, example: Any = None) -> Path:
        return PyTorchProvider().export(model, target, path, example=example)


# -------------------------------------------------------------------------------------------------- the registry
PROVIDERS: dict[str, ModelProvider] = {}


def register_provider(provider: ModelProvider) -> ModelProvider:
    """Add a provider (a third-party format, a plugin's own loader)."""
    if not isinstance(provider, ModelProvider) or not provider.name:
        raise TypeError("a provider must be a ModelProvider with a name")
    PROVIDERS[provider.name] = provider
    return provider


def providers() -> list[str]:
    return sorted(PROVIDERS)


def get_provider(name: str) -> ModelProvider:
    key = str(name).strip().lower()
    if key not in PROVIDERS:
        raise ModelNotLoadable(f"unknown provider {name!r}; known: {providers()}")
    return PROVIDERS[key]


def available_providers() -> dict[str, dict[str, Any]]:
    """What each provider says about itself here: usable or not, with the reason and the targets it writes."""
    return {name: PROVIDERS[name].describe() for name in providers()}


for _provider in (PyTorchProvider(), TorchScriptProvider(), ONNXProvider(), JAXProvider(), PhysicsNeMoProvider(),
                  HuggingFaceProvider()):
    register_provider(_provider)

#: The order :func:`detect` tries, before asking any provider. Ordered so that a file whose extension is shared
#: (``.pt`` is both a torch checkpoint and a TorchScript archive) is settled by inspecting it, not by guessing.
_DETECTION_ORDER = ("pytorch", "torchscript", "onnx", "physicsnemo", "jax", "huggingface")


def detect(ref: str | Path) -> ModelProvider:
    """The provider that recognises ``ref``, or :class:`ModelNotLoadable` naming what was tried."""
    tried: list[str] = []
    for name in _DETECTION_ORDER:
        provider = PROVIDERS[name]
        try:
            if provider.can_load(ref):
                return provider
        except OSError as exc:                          # an unreadable file is not a match, and not a crash
            tried.append(f"{name} ({exc})")
    for name in providers():
        if name not in _DETECTION_ORDER and PROVIDERS[name].can_load(ref):
            return PROVIDERS[name]
    raise ModelNotLoadable(f"no provider recognises {str(ref)!r}; tried {list(_DETECTION_ORDER)}. A local file needs "
                           f"one of the suffixes .pt/.pth/.ckpt/.bin (torch), .onnx, .mdlus (PhysicsNeMo), "
                           f".npz (JAX), or a directory with pinneapple_config.json (a hub mirror). "
                           f"{('Problems: ' + '; '.join(tried)) if tried else ''}")


def _is_torchscript(path: Path) -> bool:
    import torch

    try:
        torch.jit.load(str(path), map_location="cpu")
        return True
    except Exception:                                   # noqa: BLE001 - not an archive is the answer, not an error
        return False


def _first_array(out: Any) -> np.ndarray:
    import torch

    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy()
    # the library's own families return a dataclass (``ModelOutput(y=..., losses=..., extras=...)``) whose ``.y`` is
    # the tensor: without this branch ``np.asarray`` turned it into a 0-d object array holding the dataclass, so a
    # checkpoint of a registry model loaded through this door predicted a Python object rather than a field.
    y = getattr(out, "y", None)
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy()
    if isinstance(out, Mapping):
        first = next(iter(out.values()))
        return _first_array(first)
    if isinstance(out, (list, tuple)):
        return _first_array(out[0])
    return np.asarray(out)


# -------------------------------------------------------------------------------------------------- the loaded model
class LoadedModel:
    """A model with its provenance: the three calls of the issue, whatever produced the file.

    ``predict`` takes a numpy array (or a mapping of them) and returns the first output as a numpy array; ``evaluate``
    reuses :func:`pinneapple_analysis.validation.validate`; ``export`` writes the requested target or refuses with the
    reason. ``model`` is the underlying object (a torch module, an ONNX session, a dict of JAX parameters).
    """

    def __init__(self, model: Any, provider: ModelProvider, provenance: Provenance):
        self.model = model
        self.provider = provider
        self.provenance = provenance

    # -- the interesting part --------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return type(self.model).__name__ if not isinstance(self.model, Mapping) else "jax pytree"

    def __repr__(self) -> str:
        return f"LoadedModel({self.name}, {self.provenance.summary()})"

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "provenance": self.provenance.to_dict(),
                "targets": list(self.provider.targets)}

    @property
    def physics_model(self) -> Any:
        """This model as a :class:`~pinneapple_neural.physics_model.PhysicsModel`, when it belongs to one of the
        families that interface covers (a PINN, an operator, a GNN, a ROM, a hybrid).

        That is where the two interfaces meet: the door loads it, the family interface fits, predicts with an
        uncertainty and exports it. An object of no family is refused by ``as_physics_model`` naming what it is,
        rather than wrapped in an adapter that would have nothing to call.
        """
        from pinneapple_neural.physics_model import as_physics_model

        return as_physics_model(self.model)

    def predict(self, x: Any) -> np.ndarray:
        return self.provider.predict(self.model, x)

    def as_callable(self, inputs: Sequence[str] = (), outputs: Sequence[str] = ("y",)) -> Callable[[Mapping], dict]:
        """A function ``{input name: array} -> {output name: array}``, which is what ``pp.validate`` wants for a
        tabular model. The inputs are stacked along the last axis in the order given."""
        order = list(inputs)

        def call(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
            keys = order or [k for k in payload if k not in outputs]
            arrays = [_as_float_array(payload[name]) for name in keys]
            stacked = arrays[0] if len(arrays) == 1 else np.stack(arrays, axis=-1)
            got = self.predict(stacked)
            names = list(outputs) or ["y"]
            if got.ndim == 1 or len(names) == 1:
                return {names[0]: got}
            return {n: got[..., i] for i, n in enumerate(names)}

        return call

    def evaluate(self, dataset: Mapping[str, Any], *, inputs: Sequence[str] = (), outputs: Sequence[str] = ("y",),
                 level: str = "quick", valid_range: Mapping[str, Sequence[float]] | None = None,
                 seed: int = 0) -> Any:
        """Compare the model with a dataset through :func:`pinneapple_analysis.validation.validate` and return its
        report. ``dataset`` maps input names and target names to arrays; ``outputs`` says which are the targets."""
        from pinneapple_analysis.validation.validate import validate

        data = {"test": {k: np.asarray(v) for k, v in dataset.items()}}
        spec = {"name": f"{self.name} ({self.provenance.provider})", "outputs": list(outputs),
                "valid_range": dict(valid_range or {})}
        return validate(self.as_callable(inputs, outputs), problem=spec, data=data, level=level, seed=seed)

    def export(self, target: str, path: str | Path, *, example: Any = None) -> Path:
        """Write ``target`` (``onnx``, ``torchscript`` or ``artifact``) to ``path``.

        ``example`` is the input the exporter traces: a numpy array. Without one, the model's own first linear layer
        is used to build a plausible shape, and the exported file says so in its metadata — an export without a real
        fixture input is a claim about the shapes, not about the model.
        """
        p = Path(path)
        if target == "artifact":
            return self._export_artifact(p, example)
        if target == "onnx" and example is None:
            raise ExportNotSupported("ONNX export needs an example input: the file is a recording of one trace, so "
                                     "pass the input the model will actually see", target="onnx",
                                     operation="missing example input")
        return self.provider.export(self.model, target, p, example=example)

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _export_artifact(self, directory: Path, example: Any) -> Path:
        """The directory format: the graph plus the manifest that says which provider reads it back."""
        directory.mkdir(parents=True, exist_ok=True)
        inner = directory / "model.pt"
        example_tensor = example
        if self.provenance.provider in ("pytorch", "physicsnemo", "huggingface"):
            _torchscript_export(self.model, inner, example_tensor)
            provider = "torchscript"
        elif self.provenance.provider == "torchscript":
            inner.write_bytes(Path(self.provenance.ref).read_bytes())
            provider = "torchscript"
        else:
            raise ExportNotSupported(f"the {self.provenance.provider} provider cannot be written to the artifact "
                                     f"directory format: only a model that is a torch module (so it can be lowered to "
                                     f"TorchScript) or an existing archive can", target="artifact")
        manifest = {"schema": ARTIFACT_SCHEMA, "provider": provider, "file": inner.name,
                    "source_provenance": self.provenance.to_dict(), "model": self.name,
                    "example_shape": list(getattr(example_tensor, "shape", []) or [])}
        (directory / _MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return directory


def load(ref: str | Path, *, provider: str | None = None, **options: Any) -> LoadedModel:
    """Load a model from anywhere: a local artifact, a hub id, or a path a provider recognises.

    With ``provider=`` the reference is handed to that provider directly (and an unknown one is refused by name).
    Without it, the provider is detected — including the artifact directory format, whose manifest names the provider
    that wrote it.
    """
    p = Path(ref) if isinstance(ref, str) and Path(ref).exists() else ref
    directory = Path(p) if isinstance(p, (str, Path)) else None
    if provider is None and directory is not None and directory.is_dir() and (directory / _MANIFEST).is_file():
        manifest = json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))
        if manifest.get("schema") != ARTIFACT_SCHEMA:
            raise ModelNotLoadable(f"{directory / _MANIFEST}: schema {manifest.get('schema')!r} is not "
                                   f"{ARTIFACT_SCHEMA!r}")
        chosen = get_provider(str(manifest.get("provider") or ""))
        model, prov = chosen.load(directory / str(manifest["file"]), **options)
        prov.extra["artifact"] = str(directory)
        prov.extra["exported_from"] = manifest.get("source_provenance", {})
        return LoadedModel(model, chosen, prov)
    chosen = get_provider(provider) if provider else detect(p)
    model, prov = chosen.load(p, **options)
    return LoadedModel(model, chosen, prov)


# -------------------------------------------------------------------------------------------------- the exporters
def _torchscript_export(model: Any, path: Path, example: Any = None) -> Path:
    """``torch.jit.trace`` when there is an example, ``torch.jit.script`` otherwise; the error names the reason."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    try:
        if example is not None:
            scripted = torch.jit.trace(model, example)
        else:
            scripted = torch.jit.script(model)
    except Exception as exc:                            # noqa: BLE001 - the reason is what a caller needs
        raise ExportNotSupported(f"TorchScript export refused: {type(exc).__name__}: {exc}", target="torchscript",
                                 operation=type(exc).__name__) from exc
    scripted.save(str(path))
    return path


def _onnx_export(model: Any, path: Path, example: Any = None) -> Path:
    """Export to ONNX, refusing when the artifact would compute something other than the model.

    Three refusals, each naming what got in the way: the exporter raises; the tracer warns that it is tracing one
    branch of control flow that depends on the data (the warning names the conversion, e.g. "Converting a tensor to a
    Python boolean"); or the reloaded graph disagrees with the eager model on the example input.
    """
    import torch

    if example is None:
        raise ExportNotSupported("ONNX export needs an example input: without one, the traced shapes are a claim "
                                 "about the model rather than a recording of it", target="onnx",
                                 operation="missing example input")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = example if isinstance(example, torch.Tensor) else torch.from_numpy(_as_float_array(example))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            torch.onnx.export(model, example, str(path), opset_version=17, dynamo=False,
                              input_names=["input"], output_names=["output"],
                              dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        except Exception as exc:                        # noqa: BLE001 - torch's message names the operator
            path.unlink(missing_ok=True)
            raise ExportNotSupported(f"ONNX export refused: {type(exc).__name__}: {exc}", target="onnx",
                                     operation=_operator_of(str(exc))) from exc
    traced = [str(w.message).replace("\n", " ") for w in caught if issubclass(w.category, torch.jit.TracerWarning)]
    if traced:
        path.unlink(missing_ok=True)
        raise ExportNotSupported(
            f"ONNX export refused: the graph would not be this model. PyTorch traced one path through the code, so "
            f"the file would compute what the branch taken at export time computes, not what the model computes. "
            f"Tracer warning: {traced[0]}", target="onnx", operation=_operator_of(traced[0]))
    _verify_onnx(model, path, example)
    return path


def _verify_onnx(model: Any, path: Path, example: Any) -> None:
    """Run the file and the model on the same input; a mismatch is a refusal, with the difference named."""
    ok, _why = ONNXProvider().available()
    if not ok:                                          # cannot verify here: the tracer check above still held
        return
    import torch

    session = ONNXProvider().load(path)[0]
    with torch.no_grad():
        eager = _first_array(model(example))
    got = ONNXProvider().predict(session, example.numpy())
    if got.shape != eager.shape:
        path.unlink(missing_ok=True)
        raise ExportNotSupported(f"ONNX export refused: the graph returns {got.shape}, the model returns "
                                 f"{eager.shape} for the same input", target="onnx", operation="shape mismatch")
    worst = float(np.max(np.abs(got - eager)))
    if worst > 1e-3:
        path.unlink(missing_ok=True)
        raise ExportNotSupported(f"ONNX export refused: the graph does not reproduce the model (largest difference "
                                 f"{worst:.3g} on the example input), which is what a traced branch or an op with no "
                                 f"ONNX equivalent looks like", target="onnx", operation="numerical mismatch")


def _operator_of(message: str) -> str:
    """The operator or conversion named in a torch message, for the refusal to report."""
    for marker in ("Converting a tensor to a Python", "Iterating over a tensor", "Couldn't export Python operator",
                   "is not supported", "Unsupported", "aten::"):
        if marker in message:
            start = max(0, message.find(marker) - 40)
            return message[start:start + 120].strip()
    return message.strip()[:120]
