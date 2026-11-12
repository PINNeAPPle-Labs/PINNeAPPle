"""One AirfRANS submission to FluidsBench, from the public release to a validated result package.

The campaign is four stages, each usable alone and each recording what it did:

``prepare``   extract the public release case by case into the feature store (needs the AirfRANS/VTK stack);
``train``     fit the coordinate network on one split of the release;
``predict``   run it over the official test cases and pack the participant entry;
``submit``    run the benchmark's own validate → evaluate → package → verify → validate chain, then re-check the
              package with :mod:`pinneapple_arena.fluidsbench` and recompute the domain metrics independently.

::

    from pinneapple_simulation.pipelines.airfrans_benchmark import CampaignSpec, TrainingSpec, run_campaign

    spec = CampaignSpec(dataset_root="…/Dataset", support_root="…/support/airfrans", checkout="…/fluidsbench-submission")
    result = run_campaign(spec, "work/airfrans-full", submission_id="my-model-full-v1", split_id="full")
    print(result["package"]["metric_values"]["overall_score"])

Nothing here scores anything: the metric values in the result come from the benchmark's evaluator, and the campaign's
own numbers only exist to check them.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import data
from . import entry as entry_stage
from . import predict as predict_stage
from . import submit as submit_stage
from . import train as train_stage
from .train import TrainingSpec

__all__ = ["CampaignSpec", "STAGES", "run_campaign", "campaign_from_dict"]

STAGES = ("prepare", "train", "predict", "submit")


@dataclass(frozen=True)
class CampaignSpec:
    """Everything a campaign is defined by: the three paths it reads, and the run it makes."""
    dataset_root: str
    support_root: str
    checkout: str
    training: TrainingSpec = field(default_factory=TrainingSpec)
    python: str = ""                        # interpreter of the participant environment; defaults to this one
    model_name: str = "PINNeAPPle coordinate network"
    model_type: str = "Coordinate network (point-wise MLP)"
    hardware: str = ""
    training_note: str = ""
    submitter: dict[str, str] = field(default_factory=dict)
    extract_parallel: int = 4

    def support_manifest(self) -> str:
        return os.path.join(self.support_root, "installed", "native", "manifest.json")

    def support_tables(self) -> str:
        return os.path.join(self.support_root, "installed", "native")

    def split_file(self, split_id: str) -> str:
        return os.path.join(self.checkout, "benchmark-specs", "airfrans", "splits", f"{split_id}.json")


def campaign_from_dict(payload: dict[str, Any]) -> CampaignSpec:
    """Build a spec from a plain mapping (the form an API or a JSON config carries)."""
    training = payload.pop("training", {}) or {}
    known = {f for f in TrainingSpec.__dataclass_fields__}
    return CampaignSpec(training=TrainingSpec(**{k: v for k, v in training.items() if k in known}), **payload)


def run_campaign(spec: CampaignSpec, workdir: str, *, submission_id: str, split_id: str = "full",
                 stages: Sequence[str] = STAGES, store_dir: str | None = None,
                 progress=None) -> dict[str, Any]:
    """Run the campaign's stages in order and return the record (also written to ``<workdir>/campaign.json``).

    ``store_dir`` is the extracted feature store: by default ``<workdir>/store``, and worth pointing at a shared one,
    because the extraction is the expensive stage and every campaign on the same release reads the same arrays."""
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; the campaign has {list(STAGES)}")
    if split_id not in entry_stage.CASE_SET_FOR:
        raise ValueError(f"{split_id!r} is not a split this campaign knows: {sorted(entry_stage.CASE_SET_FOR)}")
    workdir = os.path.abspath(workdir)
    store = os.path.abspath(store_dir) if store_dir else os.path.join(workdir, "store")
    series = submission_id.rsplit("-v", 1)[0]
    run_dir = os.path.join(workdir, "runs", series)
    entry_dir = os.path.join(workdir, "entry")
    os.makedirs(workdir, exist_ok=True)
    result: dict[str, Any] = {"submission_id": submission_id, "split_id": split_id, "workdir": workdir,
                              "stages": {}, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def note(stage: str, payload: dict[str, Any]) -> None:
        result["stages"][stage] = payload
        if progress:
            progress(stage, payload)

    if "prepare" in stages:
        manifest = data.manifest_splits(spec.dataset_root)
        summary = data.extract_cases(spec.dataset_root, store, cases=data.case_ids_for(manifest),
                                     parallel=spec.extract_parallel)
        if summary["failures"]:
            raise RuntimeError(f"extraction failed for {len(summary['failures'])} case(s): "
                               f"{summary['failures'][:3]}")
        note("prepare", summary)

    manifest = data.manifest_splits(spec.dataset_root)
    if "train" in stages:
        record = train_stage.train(store, manifest, run_dir, spec.training)
        note("train", {k: v for k, v in record.items() if k != "history"})
    training_record_path = os.path.join(run_dir, "training.json")
    if not os.path.exists(training_record_path):
        raise FileNotFoundError(f"no training record at {training_record_path}: run the 'train' stage first")
    with open(training_record_path) as handle:
        training_record = json.load(handle)
    weights = os.path.join(run_dir, "model.pt")

    test_key = {"full": "full_test", "scarce": "full_test", "aoa_extrapolation": "aoa_test"}[split_id]
    if "predict" in stages:
        prediction = predict_stage.predict_cases(store, weights, sorted(manifest[test_key]), entry_dir)
        note("predict", prediction)
    with open(os.path.join(entry_dir, "predict.json")) as handle:
        prediction = json.load(handle)

    if "submit" in stages:
        support_manifest = spec.support_manifest()
        written = entry_stage.write_entry(
            entry_dir, submission_id=submission_id, series_id=series, split_id=split_id,
            support_root=spec.support_tables(), training=training_record, prediction=prediction,
            training_hours=training_record["seconds"] / 3600.0, model_name=spec.model_name,
            model_type=spec.model_type, hardware=spec.hardware, physics_weight=spec.training.physics_weight,
            training_note=spec.training_note, split_file=spec.split_file(split_id),
            checkpoint_path=weights, submitter=spec.submitter)
        note("entry", {"cases": written["cases"], "package_config": written["package_config"]})

        toolchain = submit_stage.OfficialToolchain(spec.checkout, python=spec.python or sys.executable)
        dataset_root, support_root = spec.dataset_root, spec.support_root
        evidence = {"steps": []}
        for call in (
                lambda: toolchain.validate_entry(entry_dir, dataset_root, support_root),
                lambda: toolchain.evaluate_entry(entry_dir, dataset_root, support_root,
                                                 os.path.join(spec.checkout, "work", series)),
                lambda: toolchain.package(os.path.join(spec.checkout, "work", series, "evaluation"),
                                          written["package_config"], written["discretization_cases"],
                                          os.path.join(spec.checkout, "submissions", "airfrans", submission_id),
                                          checkpoint=weights),
                lambda: toolchain.verify_package(os.path.join(spec.checkout, "submissions", "airfrans", submission_id)),
                lambda: toolchain.contributor_validator(os.path.join(spec.checkout, "submissions", "airfrans",
                                                                    submission_id))):
            evidence["steps"].append(call().as_dict())
        package_dir = os.path.join(spec.checkout, "submissions", "airfrans", submission_id)
        evidence["adapter_validation"] = toolchain.adapter_validation(package_dir)
        cross = submit_stage.cross_check_domain_metrics(entry_dir, package_dir, support_manifest,
                                                        spec.support_tables(),
                                                        case_set=entry_stage.CASE_SET_FOR[split_id])
        with open(os.path.join(package_dir, "submission.json")) as handle:
            submission_json = json.load(handle)
        evidence["package"] = {"path": package_dir, "metric_values": submission_json["metric_values"],
                               "checkpoints": submission_json["methodology"]["checkpoints"]}
        evidence["cross_check"] = cross
        note("submit", evidence)

    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(os.path.join(workdir, "campaign.json"), "w") as handle:
        json.dump(result, handle, indent=1)
    return result

