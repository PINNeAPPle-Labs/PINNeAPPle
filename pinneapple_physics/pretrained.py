"""Bring your own foundation model: wrap a pre-trained model as an ``Expert`` of :class:`PhysicsEnsemble` and let the
same residual, measurement and horizon checks decide when to trust it.

Nobody will train a better global model than the large pre-trained ones; the value is in knowing **when** to trust
them and **where** they leave their domain. This module puts one next to the models trained here:

* :data:`REGISTRY` knows the open models worth wrapping — the forecasts published in WeatherBench2 (IFS HRES,
  Pangu-Weather, GraphCast, NeuralGCM), Prithvi for satellite imagery, SAM 2 and Depth-Anything for vision — with
  the source, the grid and variables each was trained on, and the **licence**, with the place and date it was read
  (``Licence.verified``) or an explicit "not stated upstream" (``NOASSERTION``);
* **adapters** load the weights and map the query to the model's input and back to the common output:
  :class:`WeatherBench2Adapter` (the published archive), :class:`TorchCheckpointAdapter` (a checkpoint of a network
  trained here or elsewhere), :class:`ArrayArchiveAdapter` (a ``.npz`` of fields: the offline path) and
  :class:`HuggingFaceAdapter` (weights from the Hub). An adapter that cannot get its weights raises
  :class:`PretrainedUnavailable` saying what is missing and how to point it at a local copy: it never silently
  pretends to have run, and :class:`PhysicsEnsemble` then counts that expert as failed for the case;
* the **cost estimate** per call (seconds, the unit ``cost_weight`` uses) comes from the adapter's payload and
  latency or from a per-step time measured on the checkpoint, and the provenance says which of the two it was;
* :func:`from_pretrained_expert` returns an :class:`~pinneapple_physics.ensemble.Expert` whose ``provenance``
  carries the model, the licence and the cost method, and :func:`licence_report` collects them for a whole ensemble.

Weather queries are ``{"i0": <index into the Era5Store>, "steps": int, "variable": "z500"}`` and the prediction is
``(steps + 1, lat, lon)`` in physical units — exactly what :mod:`pinneapple_physics.weather.evaluate` scores, so an
expert built here can be compared case by case with the published skill curves. The physics checks that say where
each model leaves its domain are in :mod:`pinneapple_physics.domain_checks`.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .ensemble import Expert

__all__ = ["Licence", "PretrainedModel", "REGISTRY", "register", "spec", "known_models",
           "PretrainedUnavailable", "UnknownPretrainedModel", "LicenceError",
           "Adapter", "WeatherBench2Adapter", "TorchCheckpointAdapter", "ArrayArchiveAdapter", "HuggingFaceAdapter",
           "from_pretrained_expert", "provenance_of", "licence_report"]

#: Measured 2026-11-26 on the machine that ran the demo, reading WeatherBench2 over HTTPS: one request costs about
#: 1.5 s of latency (12 leads of one level of IFS HRES took 1.58 s) and the payload arrives at ~0.8 MB/s (a month of
#: ERA5, 25 MB, took 30.9 s chunk by chunk). Both are the declared default of the network adapters' cost estimate and
#: are replaced by the measured wall time the ensemble keeps for the experts whose cost it does not know.
NET_LATENCY_S = 1.5
NET_MB_S = 0.8


class PretrainedUnavailable(RuntimeError):
    """The model could not be loaded (no weights, no network, no optional dependency). The message says what is
    missing and how to point the adapter at a local copy; nothing is guessed in its place."""


class UnknownPretrainedModel(KeyError):
    """The name is not in :data:`REGISTRY` and the caller gave no licence record for it."""


class LicenceError(ValueError):
    """The licence is missing, or it does not allow the use the caller declared."""


@dataclass(frozen=True)
class Licence:
    """What the model's licence allows, with the evidence: ``verified`` says where and when it was read
    (``"2026-11-26: huggingface.co/api/models/<id>, cardData.license"``), and is empty when the upstream states
    nothing — in that case ``spdx`` is ``"NOASSERTION"`` and the record says so instead of guessing."""
    spdx: str
    url: str = ""
    commercial_use: bool | None = None
    redistribution: bool | None = None
    verified: str = ""
    note: str = ""

    @property
    def known(self) -> bool:
        return bool(self.spdx) and self.spdx.upper() != "NOASSERTION"

    def to_dict(self) -> dict[str, Any]:
        return {"spdx": self.spdx, "url": self.url, "commercial_use": self.commercial_use,
                "redistribution": self.redistribution, "verified": self.verified, "note": self.note}


@dataclass(frozen=True)
class PretrainedModel:
    """A model this repository knows how to wrap: where the weights are, what they were trained on (its envelope),
    what it costs and what its licence allows."""
    name: str
    family: str                        # "weather", "satellite", "vision"
    source: str
    licence: Licence
    envelope: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "family": self.family, "source": self.source,
                "licence": self.licence.to_dict(), "envelope": dict(self.envelope), "notes": self.notes}


_WB2 = "https://weatherbench2.readthedocs.io/en/latest/data.html"
_HF = "https://huggingface.co/"

#: The registry. ``verified`` dates are when the licence was read from the source by this repository, not when the
#: model was released: a licence can change, so a commercial user re-reads it.
REGISTRY: dict[str, PretrainedModel] = {
    "ifs-hres": PretrainedModel(
        "ifs-hres", "weather", "gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr",
        Licence("NOASSERTION", _WB2, None, None, "",
                "ECMWF IFS HRES forecasts republished by WeatherBench2; the bucket's object metadata states no "
                "licence (checked 2026-11-26), and ECMWF's own terms apply to the meteorological data"),
        {"grid": "64x32 equiangular", "variables": ["z500", "t850", "t2m", "msl", "u10", "tp6"],
         "leads_h": "6 to 240, 6-hourly", "inits": "00 and 12 UTC"}),
    "pangu-weather": PretrainedModel(
        "pangu-weather", "weather", "gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr",
        Licence("NOASSERTION", _WB2, None, None, "",
                "published Pangu-Weather forecasts (Huawei) republished by WeatherBench2; the bucket states no licence "
                "(checked 2026-11-26) — the vendor's terms and the EC Copernicus data licence apply"),
        {"grid": "64x32 equiangular", "variables": ["z500", "t850", "t2m", "msl", "u10", "tp6"],
         "leads_h": "6 to 240, 6-hourly", "inits": "00 and 12 UTC"}),
    "graphcast": PretrainedModel(
        "graphcast", "weather", "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours"
                                 "-1440x721.zarr",
        Licence("NOASSERTION", "https://github.com/google-deepmind/graphcast", False, None, "",
                "DeepMind's GraphCast weights are released for non-commercial use only; the archive here is on "
                "WeatherBench2's 1440x721 grid, not the 64x32 one, so this adapter needs a regridder"),
        {"grid": "1440x721", "variables": ["z500", "t850", "t2m", "msl", "u10"],
         "leads_h": "6 to 240, 6-hourly", "inits": "00 and 12 UTC"},
        "listed for the licence and the envelope; not wired into the demo (regridding to 64x32 first)"),
    "neuralgcm": PretrainedModel(
        "neuralgcm", "weather",
        "gs://weatherbench2/datasets/neuralgcm_deterministic/2020-64x32_equiangular_conservative.zarr",
        Licence("NOASSERTION", "https://github.com/google-research/neuralgcm", None, None, "",
                "NeuralGCM deterministic forecasts in WeatherBench2; no licence stated in the bucket metadata "
                "(checked 2026-11-26), the code repository is Apache-2.0"),
        {"grid": "64x32 equiangular", "variables": ["z500", "t850", "t2m", "msl", "u10", "tp6"],
         "leads_h": "6 to 240, 6-hourly", "inits": "00 and 12 UTC"}),
    "pinneapple-weather": PretrainedModel(
        "pinneapple-weather", "weather", "examples/weather_forecasting/checkpoints/phase1_step15000_fp16.pt",
        Licence("Apache-2.0", "https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/LICENSE", True, True,
                "2026-11-26: this repository's LICENSE file (Apache License 2.0)"),
        {"grid": "64x32 equiangular", "variables": "all 25 channels of pinneapple_physics.weather.data",
         "leads_h": "6-hourly autoregressive rollout, no fixed horizon",
         "note": "phase-1 checkpoint: one 6-hour step, trained 2011-2017"}),
    "prithvi-100m": PretrainedModel(
        "prithvi-100m", "satellite", "ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
        Licence("Apache-2.0", _HF + "ibm-nasa-geospatial/Prithvi-EO-1.0-100M", True, True,
                "2026-11-26: huggingface.co/api/models/ibm-nasa-geospatial/Prithvi-EO-1.0-100M, cardData.license"),
        {"modality": "HLS / Sentinel-2 multispectral 224x224 patches", "bands": ["B02", "B03", "B04", "B8A",
                                                                                 "B11", "B12"]},
        "NASA/IBM geospatial foundation model: a ViT backbone with task heads"),
    "sam2.1": PretrainedModel(
        "sam2.1", "vision", "facebook/sam2.1-hiera-large",
        Licence("Apache-2.0", _HF + "facebook/sam2.1-hiera-large", True, True,
                "2026-11-26: huggingface.co/api/models/facebook/sam2.1-hiera-large, cardData.license"),
        {"modality": "RGB video and images", "output": "masks with temporal propagation"},
        "promptable segmentation; the video predictor is what a temporal-continuity check is for"),
    "depth-anything-v2-large": PretrainedModel(
        "depth-anything-v2-large", "vision", "depth-anything/Depth-Anything-V2-Large",
        Licence("CC-BY-NC-4.0", _HF + "depth-anything/Depth-Anything-V2-Large", False, True,
                "2026-11-26: huggingface.co/api/models/depth-anything/Depth-Anything-V2-Large, cardData.license"),
        {"modality": "RGB images", "output": "relative monocular depth"},
        "non-commercial: the licence record is why the check exists"),
}


def register(model: PretrainedModel) -> PretrainedModel:
    """Add or replace an entry of :data:`REGISTRY` (a new pre-trained model, or a local mirror of a known one)."""
    if not model.licence.spdx:
        raise LicenceError(f"{model.name}: a registry entry needs a licence (use 'NOASSERTION' with a note when the "
                           "source states none)")
    REGISTRY[model.name] = model
    return model


def spec(name: str) -> PretrainedModel:
    try:
        return REGISTRY[name]
    except KeyError:
        raise UnknownPretrainedModel(
            f"{name!r} is not a known pre-trained model; known: {sorted(REGISTRY)}. Register yours with "
            f"pretrained.register(PretrainedModel(..., licence=Licence(...))) or pass licence= to "
            f"from_pretrained_expert") from None


def known_models(family: str | None = None) -> list[str]:
    return sorted(n for n, m in REGISTRY.items() if family is None or m.family == family)


# ------------------------------------------------------------------------------------------------ adapters

@dataclass
class Adapter:
    """Loads a pre-trained model and maps a query to its input and its output back to the common shape.

    ``predict(query)`` returns the prediction in the same shape and units as the other experts of the ensemble.
    ``cost(query)`` returns the estimated cost of that call in **seconds** (the unit the ensemble's ``cost_weight``
    expects), and ``cost_method`` says how it was obtained. ``open()`` loads the weights and raises
    :class:`PretrainedUnavailable` with a message that says what is missing.
    """
    name: str = ""
    family: str = "generic"

    def bind(self, model: PretrainedModel | None) -> Adapter:
        self.model = model
        return self

    def open(self) -> Adapter:
        return self

    def predict(self, query: Any) -> np.ndarray:
        raise NotImplementedError

    def cost(self, query: Any = None) -> float:
        """Estimated seconds for one call (a declared constant when the adapter cannot do better)."""
        return 1.0

    @property
    def cost_method(self) -> str:
        return "declared constant: every call costs 1 (arbitrary unit)"

    def describe(self) -> dict[str, Any]:
        return {"adapter": type(self).__name__, "model": self.name, "family": self.family}


def _channel(fields: np.ndarray, channels: list[str], variable: str | None, who: str) -> np.ndarray:
    """One variable out of a (..., channel, lat, lon) forecast, and a message that says what to do when it is absent."""
    if variable is None:
        raise ValueError(f"{who}: the query needs 'variable' (one of {channels[:6]}...): the published archives carry "
                         f"a handful of channels, not the whole state")
    if variable not in channels:
        raise PretrainedUnavailable(f"{who} has no channel {variable!r}; it provides {channels}")
    y = np.asarray(fields)[:, channels.index(variable)]
    if not np.all(np.isfinite(y)):
        raise PretrainedUnavailable(f"{who}: no values for {variable!r} at this initial time (the archive only has "
                                    f"00 and 12 UTC initialisations)")
    return y


class WeatherBench2Adapter(Adapter):
    """The forecasts published in WeatherBench2 (IFS HRES, Pangu-Weather, Keisler, NeuralGCM) as one expert, read
    through :class:`pinneapple_physics.weather.evaluate.PublishedForecasts` (the same loader the skill curves use).

    ``forecast`` injects the source instead of opening the archive (a test, or a local mirror with the same
    ``fn(i0, steps) -> (steps + 1, channel, lat, lon)`` contract); without it the archive is opened on first use and
    a missing one is reported with the reason :class:`PublishedForecasts` recorded.
    """
    family = "weather"

    def __init__(self, store: Any, name: str, forecast: Callable | None = None,
                 latency_s: float = NET_LATENCY_S, throughput_mb_s: float = NET_MB_S):
        super().__init__(name=name, family="weather")
        self.store, self._forecast, self.latency_s, self.mb_s = store, forecast, float(latency_s), float(throughput_mb_s)
        self._pub = None
        self._opened = forecast is not None

    def open(self) -> WeatherBench2Adapter:
        if self._opened:
            return self
        from .weather.evaluate import WB2_NAMES, PublishedForecasts

        self._pub = PublishedForecasts(self.store, names=[self.name])
        if self.name not in self._pub.ds:
            why = self._pub.errors.get(self.name, "the archive is not in WeatherBench2's published baselines")
            raise PretrainedUnavailable(
                f"{self.name}: the published forecasts could not be opened ({why}). They live in the public bucket "
                f"{self.model.source if self.model else 'gs://weatherbench2'}: check the network, or install gcsfs, "
                f"or download the subset once and pass forecast=<a local copy with the same contract>")
        self._forecast, self._opened = self._pub(self.name), True
        self._wb2 = WB2_NAMES
        return self

    def forecast(self, query: Any) -> np.ndarray:
        if not self._opened:
            self.open()
        i0, steps = int(query["i0"]), int(query["steps"])
        f = self._forecast(i0, steps)
        if f is None:
            raise PretrainedUnavailable(
                f"{self.name}: no published forecast for the initialisation {self.store.times[i0]} (WeatherBench2 "
                f"has only the 00 and 12 UTC ones, and only for the years each model covers)")
        return np.asarray(f)

    def predict(self, query: Any) -> np.ndarray:
        return _channel(self.forecast(query), list(self.store.channels), query.get("variable"), self.name)

    def cost(self, query: Any = None) -> float:
        """Latency plus payload: one field per lead and per channel the archive provides (the request fetches the
        whole set, so asking one variable does not make it cheaper)."""
        steps = int((query or {}).get("steps", 40))
        groups = 6                                     # WB2_NAMES: 5 surface/single-level + z and t at a level
        cells = int(np.prod(self.store.state.shape[-2:]))
        payload_mb = steps * groups * cells * 4 / 1e6
        return self.latency_s + payload_mb / max(self.mb_s, 1e-9)

    @property
    def cost_method(self) -> str:
        return (f"estimate: {self.latency_s:.1f} s of request latency + payload at {self.mb_s:.1f} MB/s "
                f"(both measured 2026-11-26 over HTTPS to storage.googleapis.com)")

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"grid": "from the store", "archive": getattr(self.model, "source", "")}


class TorchCheckpointAdapter(Adapter):
    """A ``torch.nn.Module`` checkpoint with the arithmetic of :mod:`pinneapple_physics.weather.train`
    (:class:`~pinneapple_physics.weather.train.Forecaster`): the network trained here, or any checkpoint saved with
    the same ``{"state_dict", "config"}`` layout. The prediction is rolled out autoregressively from the two states
    before ``i0`` and de-normalised, so it comes out in the same units as the published forecasts.

    ``per_step_seconds``: measured seconds per 6-hour step (the cost estimate); ``None`` uses the declared default
    and refines it with every call, which ``cost_method`` reports.
    """
    family = "weather"
    DEFAULT_STEP_S = 0.5                                # declared; the first call replaces it (see cost_method)

    def __init__(self, store: Any, checkpoint: str, name: str = "pinneapple-weather",
                 per_step_seconds: float | None = None, weights: Callable | None = None):
        super().__init__(name=name, family="weather")
        self.store, self.checkpoint, self._step_s = store, str(checkpoint), per_step_seconds
        self._weights, self._fc, self._measured = weights, None, per_step_seconds is not None

    def open(self) -> TorchCheckpointAdapter:
        if self._fc is not None:
            return self
        if self._weights is not None:                  # injected: a Forecaster, or anything with .rollout
            self._fc = self._weights
            return self
        try:
            from .weather.train import Forecaster
        except ImportError as exc:                      # torch missing
            raise PretrainedUnavailable(f"{self.name}: needs torch ({exc}); install it or pass weights=<a loaded "
                                        f"Forecaster>") from exc
        try:
            self._fc = Forecaster.load(self.checkpoint, self.store)
        except FileNotFoundError as exc:
            raise PretrainedUnavailable(f"{self.name}: no checkpoint at {self.checkpoint}; train one with "
                                        f"pinneapple_physics.weather.train.train or pass weights=") from exc
        return self

    def predict(self, query: Any) -> np.ndarray:
        if self._fc is None:
            self.open()
        i0, steps = int(query["i0"]), int(query["steps"])
        if i0 < 1:
            raise PretrainedUnavailable(f"{self.name}: the rollout starts from the two states before i0, so i0 >= 1 "
                                        f"(got {i0})")
        t0 = time.perf_counter()
        x = self.store.state[i0 - 1:i0 + 1]             # the model maps two states to the next one
        f = self._fc.rollout(np.asarray(x[1], np.float32), np.asarray(x[0], np.float32), self.store.times[i0],
                             steps)[0]
        if self._step_s is None or not self._measured:
            per = (time.perf_counter() - t0) / max(steps, 1)
            self._step_s = per if self._step_s is None else 0.5 * self._step_s + 0.5 * per
            self._measured = True
        f = self.store.denorm(np.asarray(f, np.float32))
        return _channel(f, list(self.store.channels), query.get("variable"), self.name)

    def cost(self, query: Any = None) -> float:
        return float(max(int((query or {}).get("steps", 40)), 1)) * float(self._step_s or self.DEFAULT_STEP_S)

    @property
    def cost_method(self) -> str:
        if self._measured:
            return f"measured: {self._step_s:.2f} s per 6-hour step on this machine (first call of the run)"
        return (f"declared: {self.DEFAULT_STEP_S:.2f} s per 6-hour step until the first call measures it "
                f"(replaced by the measured time from then on)")

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"checkpoint": self.checkpoint}


class ArrayArchiveAdapter(Adapter):
    """A local ``.npz`` archive of fields laid out like a published forecast: ``fields`` (n_init, steps + 1,
    channel, lat, lon) in physical units, ``times`` (datetime64 of the initialisations) and ``channels`` (str).

    This is the offline path: a subset of WeatherBench2 saved once (or a model's output), scored with the same code
    and the same checks as the archive it came from, on a machine with no network.
    """
    family = "weather"

    def __init__(self, path: str, store: Any = None, name: str = "archive"):
        super().__init__(name=name, family="weather")
        self.path, self.store = str(path), store
        self._a = None

    def open(self) -> ArrayArchiveAdapter:
        if self._a is not None:
            return self
        try:
            z = np.load(self.path)
        except FileNotFoundError as exc:
            raise PretrainedUnavailable(f"{self.name}: no archive at {self.path} (save one with "
                                        f"ArrayArchiveAdapter.save)") from exc
        self._a = {"fields": z["fields"], "times": z["times"], "channels": [str(c) for c in z["channels"]]}
        return self

    @staticmethod
    def save(path: str, fields: np.ndarray, times: np.ndarray, channels: list[str]) -> str:
        """Write the archive an :class:`ArrayArchiveAdapter` reads (``fields`` in physical units)."""
        np.savez_compressed(path, fields=np.asarray(fields, np.float32), times=np.asarray(times, "datetime64[ns]"),
                            channels=np.array(channels, dtype=object).astype(str))
        return str(path)

    def _index(self, query: Any) -> int:
        i0 = int(query["i0"])
        if self.store is None:
            return i0
        t = np.datetime64(self.store.times[i0], "ns")
        hit = np.nonzero(self._a["times"] == t)[0]
        if not hit.size:
            raise PretrainedUnavailable(f"{self.name}: the archive has no initialisation at {t}")
        return int(hit[0])

    def predict(self, query: Any) -> np.ndarray:
        if self._a is None:
            self.open()
        steps = int(query["steps"])
        f = self._a["fields"][self._index(query)]
        if steps + 1 > f.shape[0]:
            raise PretrainedUnavailable(f"{self.name}: the archive holds {f.shape[0] - 1} leads, {steps} asked")
        return _channel(f[:steps + 1], self._a["channels"], query.get("variable"), self.name)

    def cost(self, query: Any = None) -> float:
        return 0.01                                     # memory-mapped local read

    @property
    def cost_method(self) -> str:
        return "declared: a local array read costs 0.01 s"

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"archive": self.path}


class HuggingFaceAdapter(Adapter):
    """Weights from the Hugging Face Hub (Prithvi, SAM 2, Depth-Anything...): ``to_input(query)`` builds the model
    input, ``from_output(out, query)`` maps it back to the common shape, and ``predict`` runs under ``no_grad``.

    The model is loaded with ``transformers`` (or ``timm`` for Prithvi's ViT backbone) on first use.
    ``local_path`` skips the Hub entirely: the only way to run without network, and the reason the failure message
    names it. A missing package, a gated repository or no network all raise :class:`PretrainedUnavailable` with the
    specific cause: an adapter that cannot load never returns a prediction.
    """

    def __init__(self, repo_id: str, name: str = "", to_input: Callable | None = None,
                 from_output: Callable | None = None, family: str = "vision", local_path: str | None = None,
                 loader: Callable | None = None, cost_s: float = 0.1):
        super().__init__(name=name or repo_id, family=family)
        self.repo_id, self.local_path, self.cost_s = repo_id, local_path, float(cost_s)
        self.to_input, self.from_output, self._loader, self._model = to_input, from_output, loader, None

    def open(self) -> HuggingFaceAdapter:
        if self._model is not None:
            return self
        if self._loader is not None:
            self._model = self._loader(self)
            return self
        source = self.local_path or self.repo_id
        try:
            import transformers
        except ImportError as exc:
            raise PretrainedUnavailable(
                f"{self.name}: needs the 'transformers' package to load {source} ({exc}). Install it "
                f"(pip install transformers) or download the weights once and pass local_path=<the folder>: this "
                f"adapter does not run without the weights") from exc
        try:
            self._model = transformers.AutoModel.from_pretrained(source, local_files_only=bool(self.local_path))
        except Exception as exc:  # noqa: BLE001 - the cause is what the caller needs to see
            raise PretrainedUnavailable(
                f"{self.name}: could not load {source} ({type(exc).__name__}: {exc}). If there is no network, or the "
                f"repository is gated, download it once (huggingface-cli download {self.repo_id}) and pass "
                f"local_path=<the folder>") from exc
        return self

    def predict(self, query: Any) -> np.ndarray:
        if self._model is None:
            self.open()
        try:
            import torch
        except ImportError as exc:
            raise PretrainedUnavailable(f"{self.name}: needs torch ({exc})") from exc
        x = self.to_input(query) if self.to_input is not None else query
        self._model.eval()
        with torch.no_grad():
            out = self._model(x)
        y = out.y if hasattr(out, "y") else out
        y = self.from_output(y, query) if self.from_output is not None else y
        return np.asarray(y.detach().cpu().numpy() if hasattr(y, "detach") else y, dtype=float)

    def cost(self, query: Any = None) -> float:
        return self.cost_s

    @property
    def cost_method(self) -> str:
        return "declared: a forward pass was not measured on this machine"

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"repo_id": self.repo_id, "local_path": self.local_path}


# ------------------------------------------------------------------------------------------------ the entry point

def from_pretrained_expert(name: str, adapter: Adapter | None = None, *, licence: Licence | None = None,
                           cost: float | Callable | None = None, kind: str = "", commercial: bool = False,
                           require_verified: bool = False, lazy: bool = False) -> Expert:
    """Wrap a pre-trained model as an :class:`~pinneapple_physics.ensemble.Expert`, with its licence in the provenance.

    ``name`` is a key of :data:`REGISTRY` (its licence and envelope are picked up from there) or any name given
    together with ``licence`` — the way to wrap a model this repository does not know without losing the record.
    ``adapter`` does the loading and the mapping; without one the expert is built for a model whose prediction is the
    query itself mapped by an adapter bound later (not useful on its own, so the common case gives one).

    ``commercial``: the caller declares a commercial use, and a licence that forbids it raises :class:`LicenceError`
    instead of being ignored; ``require_verified`` does the same for a licence nobody read at the source
    (``NOASSERTION``). ``cost`` overrides the adapter's estimate; ``lazy`` defers :meth:`Adapter.open` to the first
    prediction, so building an ensemble does not download anything.
    """
    model = REGISTRY.get(name)
    if model is None and licence is None:
        spec(name)                                       # noqa: B018 - raises UnknownPretrainedModel with the names
    lic = licence or model.licence
    if not lic.spdx:
        raise LicenceError(f"{name}: the licence record is empty; say what it is, or 'NOASSERTION' with a note, "
                           f"before the model is used")
    if commercial and lic.commercial_use is False:
        raise LicenceError(f"{name}: {lic.spdx} does not allow commercial use ({lic.url}); this expert cannot be "
                           f"part of a commercial deployment")
    if require_verified and not lic.verified:
        raise LicenceError(f"{name}: the licence ({lic.spdx}) was not read at the source; set Licence(verified=...) "
                           f"or pass require_verified=False")
    if adapter is None:
        raise TypeError("from_pretrained_expert needs an adapter: WeatherBench2Adapter, TorchCheckpointAdapter, "
                        "ArrayArchiveAdapter or HuggingFaceAdapter")
    adapter.bind(model)
    if not lazy:
        adapter.open()
    prov = {"pretrained": model.to_dict() if model is not None else {"name": name},
            "licence": lic.to_dict(), "adapter": adapter.describe(),
            "cost": {"estimated_seconds": None if cost is None else float(cost) if not callable(cost) else None,
                     "method": adapter.cost_method},
            "wrapped_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return Expert(adapter.name or name, lambda q: adapter.predict(q), cost if cost is not None else adapter.cost,
                  kind or (model.family if model is not None else adapter.family), prov)


def provenance_of(expert: Expert) -> dict[str, Any]:
    """The provenance record of one expert (empty for an expert that declares none)."""
    return dict(expert.provenance or {})


def licence_report(expert_or_ensemble: Any) -> dict[str, Any]:
    """``{expert: licence record}`` plus the two lists a deployment asks about: the experts whose licence nobody read
    at the source (``unverified``) and those that forbid commercial use (``non_commercial``)."""
    experts = getattr(expert_or_ensemble, "experts", expert_or_ensemble)
    if isinstance(experts, Expert):
        experts = [experts]
    out, unverified, non_commercial = {}, [], []
    for e in experts:
        p = e.provenance or {}
        lic = p.get("licence")
        if lic is None:
            continue
        out[e.name] = lic
        if not lic.get("verified"):
            unverified.append(e.name)
        if lic.get("commercial_use") is False:
            non_commercial.append(e.name)
    return {"licences": out, "unverified": unverified, "non_commercial": non_commercial}
