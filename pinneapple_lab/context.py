"""What an experiment sees while it runs: parameters, and places to put everything it produces."""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import time
from typing import Any

import numpy as np

from .dataset import DatasetWriter


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return v


class RunContext:
    """Handed to ``Experiment.run``. Everything recorded here lands in the run folder and in the index.

    Folder layout::

        run.json         id, experiment, version, params, status, stages, timings, environment, error
        metrics.json     {name: value} (final) and metrics_log.jsonl (with steps)
        validation.json  [{name, passed, value, reference, tolerance, detail}]
        inputs/ outputs/ arrays (.npy), tables / dicts (.json), files
        figures/         PNG, GIF
        datasets/NAME/   shards of samples + card.json (see ``DatasetWriter``)
        log.txt
    """

    def __init__(self, run_dir: str, params: dict[str, Any], experiment: str, run_id: str, seed: int | None = None):
        self.dir = run_dir
        self.params = dict(params)
        self.experiment = experiment
        self.run_id = run_id
        self.metrics: dict[str, Any] = {}
        self.checks: list[dict[str, Any]] = []
        self.stages: list[dict[str, Any]] = []
        self.datasets: dict[str, DatasetWriter] = {}
        self.files: dict[str, list[str]] = {"inputs": [], "outputs": [], "figures": []}
        s = seed if seed is not None else int(self.params.get("seed", 0))
        self.rng = np.random.default_rng(s)
        for sub in ("inputs", "outputs", "figures", "datasets"):
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
        self._log = open(os.path.join(run_dir, "log.txt"), "a")
        self._mlog = open(os.path.join(run_dir, "metrics_log.jsonl"), "a")

    # -- paths -----------------------------------------------------------
    def path(self, *parts: str) -> str:
        p = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    # -- log and stages --------------------------------------------------
    def log(self, msg: str) -> None:
        self._log.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self._log.flush()

    @contextlib.contextmanager
    def stage(self, name: str):
        """``with ctx.stage("train"): ...`` records start, end and duration (shown in the status)."""
        rec = {"name": name, "started": time.time(), "status": "running"}
        self.stages.append(rec)
        self._write_progress()
        self.log(f"stage {name} started")
        try:
            yield
            rec["status"] = "done"
        except BaseException:
            rec["status"] = "failed"
            raise
        finally:
            rec["seconds"] = round(time.time() - rec["started"], 3)
            self._write_progress()
            self.log(f"stage {name} {rec['status']} ({rec['seconds']} s)")

    def _write_progress(self) -> None:
        with open(os.path.join(self.dir, "progress.json"), "w") as f:
            json.dump({"stages": self.stages, "updated": time.time()}, f, indent=1)

    # -- inputs, outputs, figures ----------------------------------------
    def _save(self, kind: str, name: str, value: Any) -> str:
        if isinstance(value, str) and os.path.isfile(value):
            dst = self.path(kind, os.path.basename(value) if "." not in name else name)
            shutil.copy(value, dst)
        elif isinstance(value, np.ndarray) or (hasattr(value, "__array__") and not isinstance(value, (dict, list))):
            dst = self.path(kind, name if name.endswith(".npy") else name + ".npy")
            np.save(dst, np.asarray(value))
        else:
            dst = self.path(kind, name if name.endswith(".json") else name + ".json")
            with open(dst, "w") as f:
                json.dump(_jsonable(value), f, indent=1)
        self.files[kind].append(os.path.relpath(dst, self.dir))
        return dst

    def input(self, name: str, value: Any) -> str:
        """Record an input (array -> .npy, dict/list/number -> .json, existing file path -> copied)."""
        return self._save("inputs", name, value)

    def output(self, name: str, value: Any) -> str:
        return self._save("outputs", name, value)

    def figure(self, name: str, fig=None, *, dpi: int = 110) -> str:
        """Save a matplotlib figure (default: the current one) as figures/NAME.png and close it."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = fig or plt.gcf()
        dst = self.path("figures", name if name.endswith(".png") else name + ".png")
        fig.savefig(dst, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        self.files["figures"].append(os.path.relpath(dst, self.dir))
        return dst

    def figure_file(self, path: str, name: str | None = None) -> str:
        """Copy an existing image / GIF / video file into figures/."""
        dst = self.path("figures", name or os.path.basename(path))
        shutil.copy(path, dst)
        self.files["figures"].append(os.path.relpath(dst, self.dir))
        return dst

    def image(self, name: str, array: np.ndarray) -> str:
        """Save an (H, W) or (H, W, 3) array in [0, 1] or uint8 as figures/NAME.png."""
        from PIL import Image
        a = np.asarray(array)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        dst = self.path("figures", name if name.endswith(".png") else name + ".png")
        Image.fromarray(a).save(dst)
        self.files["figures"].append(os.path.relpath(dst, self.dir))
        return dst

    def gif(self, name: str, frames, duration_ms: int = 120) -> str:
        """Save frames (list of (H, W[, 3]) arrays or PIL images) as figures/NAME.gif."""
        from PIL import Image
        ims = []
        for fr in frames:
            if isinstance(fr, Image.Image):
                ims.append(fr.convert("P", palette=Image.ADAPTIVE))
            else:
                a = np.asarray(fr)
                if a.dtype != np.uint8:
                    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
                ims.append(Image.fromarray(a).convert("P", palette=Image.ADAPTIVE))
        dst = self.path("figures", name if name.endswith(".gif") else name + ".gif")
        ims[0].save(dst, save_all=True, append_images=ims[1:], duration=duration_ms, loop=0, optimize=True)
        self.files["figures"].append(os.path.relpath(dst, self.dir))
        return dst

    # -- metrics and validation ------------------------------------------
    def metric(self, name: str, value: Any, step: int | None = None) -> None:
        """Record a metric. With ``step`` it is also logged as a curve; the last value is the final metric."""
        v = _jsonable(value)
        self.metrics[name] = v
        self._mlog.write(json.dumps({"name": name, "value": v, "step": step, "t": time.time()}) + "\n")
        self._mlog.flush()

    def check(self, name: str, passed: Any = None, *, value: Any = None, reference: Any = None,
              rtol: float | None = None, atol: float | None = None, max: float | None = None,
              min: float | None = None, detail: str = "") -> bool:
        """Record a validation check. Either pass a boolean, or a value with a reference and tolerances
        (|value - reference| <= atol + rtol |reference|), or bounds (``max``/``min``).
        A failed check does not stop the run; the run ends with status ``failed_validation``."""
        if passed is None:
            ok = True
            v = float(value)
            if reference is not None:
                tol = (atol or 0.0) + (rtol or 0.0) * abs(float(reference))
                ok &= abs(v - float(reference)) <= tol
            if max is not None:
                ok &= v <= max
            if min is not None:
                ok &= v >= min
            passed = ok
        rec = {"name": name, "passed": bool(passed), "value": _jsonable(value), "reference": _jsonable(reference),
               "rtol": rtol, "atol": atol, "max": max, "min": min, "detail": detail}
        self.checks.append(rec)
        self.log(f"check {name}: {'PASS' if passed else 'FAIL'} {detail}")
        return bool(passed)

    # -- datasets ----------------------------------------------------------
    def dataset(self, name: str, *, description: str = "", units: dict[str, str] | None = None,
                shard_size: int = 256, license: str = "") -> DatasetWriter:
        """A dataset of samples (dicts of arrays and scalars) produced by this run, written in shards."""
        if name not in self.datasets:
            self.datasets[name] = DatasetWriter(
                os.path.join(self.dir, "datasets", name), name=name, description=description, units=units or {},
                shard_size=shard_size, license=license,
                provenance={"experiment": self.experiment, "run_id": self.run_id, "params": self.params})
        return self.datasets[name]

    def close(self) -> None:
        for ds in self.datasets.values():
            ds.close()
        self._log.close()
        self._mlog.close()
