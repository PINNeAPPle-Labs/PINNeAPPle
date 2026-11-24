"""One graph over the four registry stores (issue #156, phase 3 of the physical data graph).

``pinneapple_registry`` keeps problems, datasets, models and experiments in four stores that do not know about each
other: every record is filed under a ``problem_id`` and nothing points from a model back to the simulations it was
trained on. This module writes the edges the stores do record as one :class:`~pinneapple_data.lineage.graph.Lineage`,
so a prediction can be traced to the problem, the geometry, the simulation runs, the dataset and model versions and
the validation record that stand behind it.

Nothing is inferred that a store does not say. Which dataset a model was trained on is a free-form key in its
metadata (``datasets``, ``dataset``, ``scenarios``, ``trained_on``); when the key is absent the model is linked to the
problem it is filed under and to nothing else, the node says so, and ``Lineage.trace()`` reports the dataset as
missing. The same holds for the geometry and for the validation record: the answer says what was not recorded instead
of filling the hole.

    from pinneapple_registry import ArtifactRegistry
    from pinneapple_data.lineage import from_registry

    reg = ArtifactRegistry("artifacts")
    g = from_registry(reg, "poisson_2d", prediction="PRED-001")
    answer = g.trace("PRED-001")
    answer["missing"]                          # ['geometry'] when nobody recorded the geometry
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .graph import Artifact, Lineage

__all__ = ["from_registry", "DATASET_KEYS", "RUN_KEYS", "GEOMETRY_KEYS"]

# metadata keys a caller may use to name what a model was trained on, in the order they are tried
DATASET_KEYS = ("datasets", "dataset", "dataset_id", "scenarios", "scenario_ids", "trained_on")
# metadata keys a caller may use to name the run that produced a model version
RUN_KEYS = ("run_id", "experiment_run", "run")
# metadata keys a dataset may use to name the geometry it was generated on
GEOMETRY_KEYS = ("geometry", "geometry_id", "cad", "mesh")
# keys copied from the stores' metadata into the node parameters, so the graph carries the numbers
MODEL_KEYS = ("metrics", "stage", "n_params", "width")
DATASET_KEYS_KEPT = ("field_names", "stats", "n_samples")


def _names(value: Any) -> List[str]:
    """One metadata value as a list of names: a string, a list of them, or the keys of a dict."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(k) for k in value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return [str(value)]


def _json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _store(registry: Any) -> Any:
    from pinneapple_registry import ArtifactRegistry          # local: pinneapple_data does not depend on it

    return registry if not isinstance(registry, str) else ArtifactRegistry(registry)


def _problem_nodes(registry: Any, problem_id: str) -> tuple[List[Artifact], Optional[str]]:
    """One node per stored version of the problem, and the id of the newest (the one the other records link to)."""
    out: List[Artifact] = []
    newest: Optional[str] = None
    for version in registry.problems.versions(problem_id):
        folder = f"{registry.problems.root}/{problem_id}/{version}"
        preset = _json(f"{folder}/preset.json")
        snapshot = _json(f"{folder}/snapshot.json")
        out.append(Artifact(id=f"problem:{problem_id}@{version}", kind="problem", name=problem_id, version=version,
                            item=problem_id, origin=registry.problems.root,
                            file=f"{folder}/preset.json" if preset else (f"{folder}/snapshot.json" if snapshot else None),
                            parameters={**snapshot, **preset}))
        newest = out[-1].id
    return out, newest


def _dataset_nodes(registry: Any, problem_id: str, problem: Optional[str]) -> tuple[List[Artifact], Dict[str, str]]:
    """One node per stored dataset — one simulation run's worth of arrays — plus the geometry each one names.

    Returns the nodes and the lookup from a name a model metadata may use (``scenario`` or ``problem/scenario``) to
    the node id.
    """
    nodes: List[Artifact] = []
    index: Dict[str, str] = {}
    for scenario in registry.datasets.list_scenarios(problem_id):
        meta = _json(f"{registry.datasets.root}/{problem_id}/{scenario}/metadata.json")
        version = str(meta.get("version") or meta.get("created_at") or scenario)
        inputs = [problem] if problem else []
        geometry = next((meta[k] for k in GEOMETRY_KEYS if meta.get(k)), None)
        if geometry:
            inputs.append(f"geometry:{geometry}")
            nodes.append(Artifact(id=f"geometry:{geometry}", kind="geometry", name=str(geometry),
                                  item=str(geometry), origin=registry.datasets.root))
        node = Artifact(id=f"dataset:{problem_id}/{scenario}", kind="dataset", name=scenario, version=version,
                        item=f"{problem_id}/{scenario}", origin=registry.datasets.root,
                        timestamp=meta.get("created_at"), inputs=inputs,
                        parameters={k: meta[k] for k in DATASET_KEYS_KEPT if k in meta},
                        notes="" if meta.get("version") else "the store records no dataset version: the creation "
                                                            "timestamp stands in for it")
        nodes.append(node)
        index[scenario] = node.id
        index[f"{problem_id}/{scenario}"] = node.id
    return nodes, index


def _model_nodes(registry: Any, problem_id: str, datasets: Dict[str, str], problem: Optional[str],
                 experiments: Dict[str, str]) -> List[Artifact]:
    """One node per stored model version, linked to the datasets its metadata names and to its training run."""
    out: List[Artifact] = []
    for version in registry.models.versions(problem_id):
        meta = _json(f"{registry.models.root}/{problem_id}/{version}/metadata.json") or {}
        named: List[str] = []
        for key in DATASET_KEYS:
            named += _names(meta.get(key))
        inputs = list(dict.fromkeys(datasets[n] for n in named if n in datasets))
        run = next((meta[k] for k in RUN_KEYS if meta.get(k) is not None), None)
        if run is not None and f"{problem_id}#{run}" in experiments:
            inputs.append(experiments[f"{problem_id}#{run}"])
        if not any(i.startswith("dataset:") for i in inputs):
            if problem:
                inputs.append(problem)
            where = "the model metadata names no dataset: the dataset version behind it is unknown"
        else:
            where = ""
        out.append(Artifact(id=f"model:{problem_id}@{version}", kind="model", name=problem_id, version=version,
                            item=f"{problem_id}@{version}", origin=registry.models.root,
                            timestamp=meta.get("created_at") or meta.get("saved_at"),
                            parameters={k: meta[k] for k in MODEL_KEYS if k in meta},
                            inputs=list(dict.fromkeys(inputs)), notes=where))
    return out


def _verdict(report: Dict[str, Any]) -> str:
    """One word for the outcome, from whichever shape the report was stored in.

    A ``ValidationReport.to_dict()`` carries ``passed``; a trust report carries ``status``; a hand-written record may
    carry ``verdict`` or ``grade``. Nothing is invented when none of them is there: the word stays empty, and the
    node says so instead of showing a blank that reads as fine.
    """
    for key in ("status", "verdict", "grade"):
        if report.get(key):
            return str(report[key])
    if "passed" in report:
        return "PASS" if report["passed"] else "FAIL"
    return ""


def _validation_nodes(registry: Any, problem_id: str, models: List[Artifact]) -> List[Artifact]:
    """The validation record stored with each model version: its trust report, or its ``validation`` block.

    Both shapes are read because both are written in the library: ``ModelStore.save(trust_report=...)`` stores a
    ``TrustReport`` (or its ``to_dict()``) under ``trust_report``, and a ``ValidationReport`` — the physics checks of
    ``pinneapple_analysis.validation`` — under the same key or under ``validation``. The node keeps the verdict and
    the counts of either, so a trace says which record backed the version and how it came out.
    """
    out: List[Artifact] = []
    for model in models:
        meta = _json(f"{registry.models.root}/{problem_id}/{model.version}/metadata.json") or {}
        report = meta.get("trust_report") or meta.get("validation")
        if not report:
            continue
        report = report if isinstance(report, dict) else (
            report.to_dict() if hasattr(report, "to_dict") else {"raw": str(report)})
        counts: Dict[str, Any] = {k: report[k] for k in ("n_passed", "n_failed", "n_not_run") if k in report}
        if isinstance(report.get("checks"), list):
            counts["checks"] = len(report["checks"])
        params: Dict[str, Any] = {}
        verdict = _verdict(report)
        if verdict:
            params["verdict"] = verdict
        params.update(counts)
        out.append(Artifact(id=f"validation:{problem_id}@{model.version}", kind="validation",
                            name=f"validation of {model.version}", version=model.version,
                            item=f"{problem_id}@{model.version}", origin=registry.models.root,
                            timestamp=report.get("timestamp") or None, inputs=[model.id], parameters=params,
                            notes=None if verdict else "the stored validation record carries no verdict"))
    return out


def _experiment_nodes(registry: Any, problem_id: str, problem: Optional[str]) -> tuple[List[Artifact], Dict[str, str]]:
    """One node per training run in the experiment store, and the lookup ``problem#run`` -> node id."""
    nodes: List[Artifact] = []
    index: Dict[str, str] = {}
    try:
        stored = registry.experiments.list_runs(problem_id)
    except Exception:                                       # noqa: BLE001 - a store that cannot answer has no runs
        return nodes, index
    for run in stored:
        params = {k: run[k] for k in ("status", "final_metrics") if run.get(k)}
        node = Artifact(id=f"experiment:{problem_id}#{run.get('id')}", kind="experiment",
                        name=f"run {run.get('id')}", version=str(run.get("id")), item=f"{problem_id}#{run.get('id')}",
                        origin=getattr(registry.experiments, "db_path", None), timestamp=run.get("started_at"),
                        parameters=params, inputs=[problem] if problem else [])
        nodes.append(node)
        index[f"{problem_id}#{run.get('id')}"] = node.id
    return nodes, index


def from_registry(registry: Any, problem_id: str, *, prediction: Optional[str] = None,
                  model_version: Optional[str] = None) -> Lineage:
    """Build the lineage of one problem from the four stores: the problem versions, the datasets (the simulation runs)
    and the geometry they name, the model versions with the datasets and the run they came from, and the validation
    record of each model version.

    ``registry`` is an ``ArtifactRegistry`` or the path of one. ``prediction`` adds a prediction node linked to
    ``model_version``, or to the newest stored model version when that is not given; ``trace()`` on it answers the
    question of issue #156: problem, geometry, simulation runs, dataset version, model version and validation record.
    """
    reg = _store(registry)
    problems, newest = _problem_nodes(reg, problem_id)
    experiments, experiment_index = _experiment_nodes(reg, problem_id, newest)
    datasets, dataset_index = _dataset_nodes(reg, problem_id, newest)
    models = _model_nodes(reg, problem_id, dataset_index, newest, experiment_index)
    validations = _validation_nodes(reg, problem_id, models)

    artifacts: List[Artifact] = problems + experiments + datasets + models + validations

    if prediction:
        source = next((m for m in models if model_version and m.version == model_version), None)
        if source is None and models:
            source = models[-1]                            # versions() lists oldest first
        artifacts.append(Artifact(
            id=str(prediction), kind="prediction", name=str(prediction), item=str(prediction),
            version=source.version if source else None, inputs=[source.id] if source else [],
            notes="" if source else "no model version is stored for this problem: the prediction has no source"))

    return Lineage(artifacts, project=problem_id)
