"""``pp.zoo``: pre-trained, verified PINNeAPPle models in one line -- load, predict, evaluate against the stored
reference, validate (physics checks + validity range), benchmark and fine-tune with a held-out guard.

    import pinneapple as pp
    pp.zoo.list()                                       # ['airfoil_cst_rans_mlp', 'heatsink_forced_mlp', ...]
    card = pp.zoo.info("airfoil_cst_rans_mlp")          # ModelCard: task, inputs/outputs + units, validity range,
                                                        # metrics vs reference, training data, licence, citation
    m = pp.zoo.load("airfoil_cst_rans_mlp")
    m.predict(T0=0.17, T1=0.15, T2=0.13, M0=0.02, M1=0.05, M2=0.03, alpha_deg=4.0)
    pp.zoo.evaluate("airfoil_cst_rans_mlp")             # recomputes the card's metrics on the stored reference
    pp.zoo.validate("airfoil_cst_rans_mlp").decisao     # TrustReport: APPROVED | REVIEW | REJECT
    print(pp.zoo.catalogue())                           # every model and dataset: metrics, size, Hub link
    print(pp.zoo.anchors.catalogue())                   # the same models as a named family, one per problem
    res = pp.zoo.finetune("airfoil_cst_rans_mlp", (X_new, Y_new), save_to="my_airfoil")  # refused if worse

Only models that passed the zoo's gate ship (see ``_catalog.py`` and ``docs/dev/model_dataset_inventory.md``).
Weights and datasets are listed in ``manifest.json`` with SHA-256 and versions; small files are bundled, the rest is
downloaded lazily from the Hugging Face Hub (``PINNeAPPle-Labs``). ``pp.datasets`` is :mod:`pinneapple_hub.zoo.datasets`.
"""
from __future__ import annotations

import builtins
import hashlib
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_hub.model_card import ModelCard

from . import _artifacts as art
from . import anchors, datasets
from ._artifacts import (
    ArtifactNotPublishedError,
    ArtifactUnavailableError,
    ChecksumMismatchError,
    ZooError,
    cache_dir,
)
from ._models import (
    OutOfRangeWarning,
    SupersededModelWarning,
    ZooModel,
    copy_model,
    fit,
    load_dir,
    load_model,
)

__all__ = ["list", "info", "load", "versions", "predict", "evaluate", "validate", "benchmark", "finetune", "status",
           "catalogue", "export_onnx", "onnx_parity", "query_checks", "check_spec", "export_web",
           "load_dir", "datasets", "anchors", "ZooModel", "FinetuneResult", "FinetuneRejectedError",
           "OutOfRangeWarning", "SupersededModelWarning",
           "ZooError", "ArtifactNotPublishedError", "ArtifactUnavailableError", "ChecksumMismatchError", "cache_dir"]


def export_onnx(name: ModelRef, path, *, checks: bool = False, check_parity: bool = True,
                offline: bool | None = None) -> dict[str, Any]:
    """Export a model to ONNX (pre/post-processing included; parity with PyTorch checked to 1e-5). See
    :func:`pinneapple_hub.zoo.export.export_onnx`."""
    from .export import export_onnx as _export
    return _export(name, path, checks=checks, check_parity=check_parity, offline=offline)


def onnx_parity(name: ModelRef, onnx_model, X=None, *, offline: bool | None = None) -> float:
    """Largest scaled difference between an ONNX graph and the PyTorch model (:mod:`pinneapple_hub.zoo.export`)."""
    from .export import onnx_parity as _parity
    return _parity(name, onnx_model, X, offline=offline)


def query_checks(name: ModelRef, inputs=None, *, offline: bool | None = None, **named) -> builtins.list:
    """Per-query trust checks and verdict for each input row (:mod:`pinneapple_hub.zoo.checks`)."""
    from .checks import query_checks as _qc
    return _qc(name, named if inputs is None else inputs, offline=offline)


def check_spec(name: ModelRef, *, offline: bool | None = None) -> dict[str, Any]:
    """The declarative spec of a model's query checks (:func:`pinneapple_hub.zoo.checks.check_spec`)."""
    from .checks import check_spec as _spec
    return _spec(_model(name, offline))


def export_web(name: str, directory, *, offline: bool | None = None) -> dict[str, Any]:
    """ONNX graph with query checks + JSON card/spec for the browser (:mod:`pinneapple_hub.zoo.web`)."""
    from .web import export_web as _web
    return _web(name, directory, offline=offline)


class FinetuneRejectedError(ZooError):
    """The fine-tuned model is worse than the original on the original test set (pass ``force=True`` to save it)."""


ModelRef = str | ZooModel


def _model(m: ModelRef, offline: bool | None = None) -> ZooModel:
    return m if isinstance(m, ZooModel) else load(m, offline=offline)


def list(task: str | None = None) -> builtins.list:  # noqa: A001 - the public name is pp.zoo.list()
    """Names of the shipped models (optionally only those whose task or tags mention ``task``)."""
    items = art.manifest()["models"]
    names = sorted(items)
    if task:
        t = task.lower()
        names = [n for n in names if t in items[n].get("task", "").lower()
                 or any(t in g.lower() for g in items[n].get("tags", []))]
    return names


def info(name: str, *, offline: bool | None = None, version: str | None = None) -> ModelCard:
    """The model card (``ModelCard``): task, inputs/outputs with units, validity range, metrics against the
    reference, training data, licence, citation and the stored trust report. ``print(card.to_markdown())``."""
    return ModelCard.load(str(art.fetch("models", name, "model_card.json", offline=offline, version=version)))


def load(name: str, *, offline: bool | None = None, version: str | None = None) -> ZooModel:
    """A ready-to-use predictor (weights verified against the manifest's SHA-256). ``version`` loads a superseded
    release (see :func:`versions`; a ``SupersededModelWarning`` says why it was superseded)."""
    return load_model(name, offline=offline, version=version)


def versions(name: str) -> builtins.list[dict[str, str]]:
    """Every release of a model, newest first, with the reason each older one was superseded."""
    return art.versions("models", name)


def predict(name: ModelRef, inputs=None, **named) -> dict[str, np.ndarray]:
    """``load(name).predict(...)`` in one call."""
    return _model(name).predict(inputs, **named)


def catalogue() -> str:
    """Markdown page of every model and dataset: metrics, trust decision, size, Hugging Face link
    (``docs/guides/zoo_catalogue.md`` is this page)."""
    from .catalogue import markdown
    return markdown()


def status(name: str) -> dict[str, str]:
    """Where each file of a model resolves from right now (bundled | cached | remote), without downloading."""
    return art.status("models", name)


# ---------------------------------------------------------------------------------------------- evaluation
def _cases(m: ZooModel, data) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(data, tuple) and len(data) == 2:
        return m.as_array(data[0]), np.asarray(data[1], float)
    if isinstance(data, Mapping):
        X = m.as_array({k: data[k] for k in m.input_names})
        Y = np.stack([np.asarray(data[k], float) if k in data else np.full(len(X), np.nan)
                      for k in m.reference_outputs], 1)
        return X, Y
    raise TypeError("cases must be (inputs, outputs) arrays, a dataset split or a mapping of named columns")


def _compare(new: dict[str, float], stored: dict[str, float], rtol: float) -> tuple[bool, float]:
    common = [k for k in new if k in stored and not k.endswith("/n")]
    if not common:
        return False, float("nan")
    d = max(abs(new[k] - stored[k]) / max(abs(stored[k]), 1e-12) for k in common)
    return bool(d <= rtol), float(d)


def evaluate(name: ModelRef, dataset=None, *, split: str | None = None, offline: bool | None = None,
             rtol: float | None = None) -> dict[str, Any]:
    """Metrics of a model against reference cases.

    ``dataset``: None (the model's own reference splits, e.g. held-out test + NACA reference for the airfoil),
    a dataset name (with ``split``), a ``DatasetSplit``, an ``(inputs, outputs)`` pair or a mapping of columns.
    With the stored reference, the result also says whether the card's metrics are reproduced (``matches_card``)."""
    m = _model(name, offline)
    if dataset is None:
        splits = (split,) if split else m.eval_splits
        metrics = m.evaluate_splits(splits, offline=offline)
        tol = rtol if rtol is not None else m.spec.get("card_tolerance", 1e-3)
        ok, diff = _compare(metrics, m.card.validation_metrics, tol)
        return {"model": m.name, "version": m.card.version, "reference": m.card.reference_source,
                "metrics": metrics, "card_metrics": {k: v for k, v in m.card.validation_metrics.items()
                                                     if k.split("/")[0] in splits},
                "matches_card": ok, "max_rel_diff_vs_card": diff, "tolerance": tol}
    if isinstance(dataset, str):
        dataset = datasets.load(dataset, split=split or "test", offline=offline)
    X, Y = _cases(m, dataset)
    inside = m.in_range(X)
    return {"model": m.name, "version": m.card.version, "metrics": m.evaluate_xy(X, Y),
            "metrics_in_range": m.evaluate_xy(X[inside], Y[inside]) if 0 < inside.sum() < len(X) else None,
            "fraction_out_of_range": float(1 - inside.mean()) if len(X) else 0.0}


def validate(name: ModelRef, inputs=None, *, offline: bool | None = None):
    """Trust report (``pinneapple_analysis.trust.TrustReport``) of a model: its card metrics reproduced on the stored
    reference, the acceptance limits of the zoo gate, the model's physics checks and -- if ``inputs`` are given --
    whether they are inside the validity range (an out-of-range warning is raised for those that are not)."""
    from pinneapple_analysis.trust.trust_report import (
        TrustReport,
        checagem_booleana,
        checagem_erro_relativo,
    )
    m = _model(name, offline)
    ev = evaluate(m, offline=offline)
    checks = [checagem_erro_relativo("card metrics reproduced on the stored reference (max relative difference)",
                                     ev["max_rel_diff_vs_card"], ev["tolerance"], fonte=m.card.reference_source[:120],
                                     critica=True)]
    for key, limit in m.spec.get("acceptance", {}).items():
        checks.append(checagem_erro_relativo(f"held-out {key} within the zoo acceptance limit",
                                             ev["metrics"].get(key), limit, fonte="independent reference",
                                             critica=True))
    checks += m.physics_checks()
    details: dict[str, Any] = {"metrics": ev["metrics"]}
    if inputs is not None:
        X = m.as_array(inputs)
        inside = m.in_range(X)
        frac = float(1 - inside.mean()) if len(X) else 0.0
        details["fraction_out_of_range"] = frac
        if frac > 0:
            warnings.warn(f"{m.name}: {int((~inside).sum())} of {len(X)} inputs are outside the validity range",
                          OutOfRangeWarning, stacklevel=2)
        checks.append(checagem_booleana("query inputs inside the validity range", frac == 0.0,
                                        fonte="model card validity_range", critica=False))
    return TrustReport(checks, model=f"{m.name} v{m.card.version}", details=details)


# ---------------------------------------------------------------------------------------------- fine-tuning
@dataclass
class FinetuneResult:
    model: ZooModel
    original: str
    before: dict[str, float]           # guard metrics of the original on the original test set
    after: dict[str, float]            # same metrics after fine-tuning
    accepted: bool                     # no guard metric worse than before by more than ``tolerance``
    tolerance: float
    history: builtins.list = field(default_factory=builtins.list)
    new_data_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    saved_to: str | None = None

    def save(self, directory: str | Path, *, force: bool = False) -> Path:
        """Save the fine-tuned model; refused (:class:`FinetuneRejectedError`) if it is worse than the original on
        the original test set, unless ``force``."""
        if not self.accepted and not force:
            raise FinetuneRejectedError(_reject_msg(self))
        p = self.model.save(directory)
        self.saved_to = str(p)
        return p

    def summary(self) -> str:
        rows = [f"{k}: {self.before[k]:.5g} -> {self.after[k]:.5g}" for k in self.before]
        return (f"fine-tune of {self.original}: {'accepted' if self.accepted else 'REJECTED'} "
                f"(tolerance {self.tolerance:.1%})\n  " + "\n  ".join(rows))


def _reject_msg(r: FinetuneResult) -> str:
    worse = [f"{k} {r.before[k]:.5g} -> {r.after[k]:.5g}" for k in r.before
             if r.after[k] > r.before[k] * (1 + r.tolerance)]
    return (f"the fine-tuned {r.original} is worse than the original on the original test set ({'; '.join(worse)}); "
            "not saved. Pass force=True to save it anyway (the card records the regression).")


def finetune(name: ModelRef, new_data, *, epochs: int = 200, lr: float = 1e-4, batch: int = 256,
             weight_decay: float = 0.0, tolerance: float = 0.02, seed: int = 0, force: bool = False,
             save_to: str | Path | None = None, registry: str | Any | None = None,
             replay: bool = True, new_test=None, offline: bool | None = None, progress=None) -> FinetuneResult:
    """Fine-tune a copy of a zoo model on ``new_data`` (``(inputs, outputs)``, a dataset split or a mapping of
    columns; NaN outputs are ignored) and check it on the model's ORIGINAL held-out test set.

    The fine-tuned model is ``accepted`` when none of its guard metrics (the card's primary held-out errors) got worse
    than the original's by more than ``tolerance`` (relative). With ``save_to`` (a directory) or ``registry`` (an
    ``ArtifactRegistry``, a ``ModelStore`` or a registry root path), a rejected model raises
    :class:`FinetuneRejectedError` and nothing is written, unless ``force=True``. ``new_test``: optional held-out
    cases of the new data, reported in ``new_data_metrics``. ``replay`` (default True) mixes the model's original
    training split into the fine-tuning set, the usual guard against forgetting what the model already knew."""
    base = _model(name, offline)
    guard = base.spec.get("guard_metrics") or [f"test/{k}" for k in base.card.validation_metrics][:1]
    splits = sorted({g.split("/")[0] for g in guard})
    before_all = base.evaluate_splits(splits, offline=offline)
    X, Y = _cases(base, new_data)
    Xf, Yf = X, Y
    if replay:
        tr = datasets.load(base.card.data_lineage["dataset"], split="train", offline=offline)
        keep = base.replay_mask(tr)
        Xr, Yr = tr.xy(base.input_names, base.reference_outputs)
        Xf, Yf = np.concatenate([X, Xr[keep]]), np.concatenate([Y, Yr[keep]])
    m = copy_model(base)
    hist = fit(m, Xf, Yf, epochs=epochs, lr=lr, batch=batch, weight_decay=weight_decay, seed=seed, progress=progress)
    after_all = m.evaluate_splits(splits, offline=offline)
    before = {k: before_all[k] for k in guard}
    after = {k: after_all[k] for k in guard}
    accepted = all(after[k] <= before[k] * (1 + tolerance) for k in guard)
    nd = {"train": m.evaluate_xy(X, Y)}
    if new_test is not None:
        Xt, Yt = _cases(base, new_test)
        nd["test"] = m.evaluate_xy(Xt, Yt)
        nd["test_original_model"] = base.evaluate_xy(Xt, Yt)
    # the card of the copy describes what it now is
    tag = hashlib.sha256(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(Y).tobytes()).hexdigest()[:10]
    c = m.card
    c.version = f"{base.card.version}+ft.{tag}"
    c.name = f"{base.name}-finetuned"
    c.validation_metrics = {**m.evaluate_splits(m.eval_splits, offline=offline)}
    c.data_lineage = {**c.data_lineage, "finetuned_from": f"{base.name} v{base.card.version}",
                      "finetune_data_sha256_10": tag, "finetune_rows": int(len(X)),
                      "finetune": {"epochs": epochs, "lr": lr, "batch": batch, "seed": seed, "replay": replay},
                      "guard": {"before": before, "after": after, "accepted": accepted, "tolerance": tolerance,
                                "forced": bool(force and not accepted)}}
    c.trust_report, c.trust_override = {}, {}
    m.finetuned_from = base.name
    res = FinetuneResult(m, base.name, before, after, accepted, tolerance, hist, nd)
    if save_to is not None or registry is not None:
        if not accepted and not force:
            raise FinetuneRejectedError(_reject_msg(res))
        if save_to is not None:
            res.save(save_to, force=force)
        if registry is not None:
            res.saved_to = _save_to_registry(res, registry)
    return res


def _save_to_registry(res: FinetuneResult, registry) -> str:
    from pinneapple_registry import ArtifactRegistry, ModelStore
    store = registry if isinstance(registry, ModelStore) else (
        registry.models if isinstance(registry, ArtifactRegistry) else ArtifactRegistry(str(registry)).models)
    meta = {"zoo_adapter": res.model.adapter, "config": res.model.config, "model_card": res.model.card.to_dict(),
            "metrics": res.model.card.validation_metrics, "guard": res.model.card.data_lineage["guard"]}
    v = store.save(res.model.card.name, res.model.module, metadata=json.loads(json.dumps(meta, default=float)))
    return f"{store.root}/{res.model.card.name}/{v}"


# ---------------------------------------------------------------------------------------------- benchmark
def benchmark(name: ModelRef, *, budget_s: float = 60.0, seeds: int = 1, baselines: Sequence[str] = ("retrain",),
              out_dir: str | Path = "zoo_benchmark", force: bool = False, log=print, **kwargs):
    """Benchmark a zoo model with ``pp.benchmark``: the shipped model against the same architecture retrained from
    scratch on the same training split under ``budget_s`` seconds (CPU), on the same held-out test set, with
    ``pp.benchmark``'s gates, report and trust card. Returns its ``BenchmarkResult``. Needs
    ``pinneapple_analysis.benchmark``."""
    from ._bench import run_benchmark
    return run_benchmark(_model(name), budget_s=budget_s, seeds=seeds, baselines=baselines, out_dir=out_dir,
                         force=force, log=log, **kwargs)
