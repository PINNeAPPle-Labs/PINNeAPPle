"""
"Probe" action for the dataset catalog: load one real sample of one subset
and report its fields/shapes — the dataset equivalent of a model's /infer.

Same honesty contract as model results (core/result.py): a probe is `ok`
only if the real upstream loader returned a real sample. Otherwise it says
why (`not_installed`, `skipped_heavy` when neither a local copy nor remote
streaming is available, `failed` with the real exception) and hands back the
exact commands to get the data locally.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

from pinneapple_arena.external.installer import is_installed
from pinneapple_arena.external.result import (
    STATUS_NOT_INSTALLED,
    STATUS_SKIPPED_HEAVY,
    ModelResult,
    Timer,
    failed,
    ok,
    summarize_array,
)
from pinneapple_arena.external.schema import DatasetSpec


def _load_the_well(base_path: str, subset: str, split: str) -> tuple[int, dict]:
    from the_well.data import WellDataset
    dataset = WellDataset(well_base_path=base_path, well_dataset_name=subset, well_split_name=split)
    return len(dataset), dataset[0]


# slug -> loader(base_path, subset, split) -> (n_samples, first_sample)
LOADERS: dict[str, Callable[[str, str, str], tuple[int, Any]]] = {
    'the-well': _load_the_well,
}


def local_commands(spec: DatasetSpec, subset: str, split: str) -> str:
    steps = []
    if spec.pip_package:
        steps.append(f"pip install {spec.pip_package}")
    if spec.download_command:
        steps.append(spec.download_command.format(subset=subset, split=split))
    else:
        steps.append(f"git clone {spec.repo}")
    return " && ".join(steps)


def _describe_sample(sample: Any) -> tuple[dict, dict]:
    """Return ({field: "shape dtype"}, {field: summary stats}) for every
    array-like field of one sample."""
    shapes, stats = {}, {}
    items = sample.items() if isinstance(sample, dict) else [("sample", sample)]
    for key, value in items:
        if not hasattr(value, "shape"):
            continue
        shapes[key] = f"{tuple(value.shape)} {getattr(value, 'dtype', '')}".strip()
        try:
            stats[key] = summarize_array(np.asarray(value, dtype=np.float64))
        except (TypeError, ValueError):
            pass
    return shapes, stats


def probe_dataset(spec: DatasetSpec, subset: str, split: str, stream: bool = False,
                  dataset_root: str | None = None) -> ModelResult:
    """Load sample 0 of ``subset``/``split``. ``dataset_root``: a local download (default: the
    ``EXTERNAL_CATALOG_DATASET_ROOT`` environment variable); otherwise ``stream=True`` streams one sample."""
    sub = next(s for s in spec.subsets if s.name == subset)
    meta = dict(name=f"{spec.name} / {subset}", slug=spec.slug, category="dataset",
                domains=list(sub.domains), repo=spec.repo)
    dataset_info = {"name": f"{subset} [{split}]", "description": sub.desc}
    commands = local_commands(spec, subset, split)

    if spec.import_name and not is_installed(spec.import_name):
        return ModelResult(**meta, status=STATUS_NOT_INSTALLED, dataset_name=dataset_info["name"],
                           dataset_description=sub.desc,
                           note=f"Loader not installed. Try: pip install {spec.pip_package}",
                           local_command=commands)

    local_root = dataset_root or os.environ.get("EXTERNAL_CATALOG_DATASET_ROOT") or None
    if local_root and os.path.isdir(local_root):
        base_path, source = local_root, f"local copy at {local_root}"
    elif stream and spec.stream_base_path:
        missing = [pkg for pkg in spec.stream_requires if not is_installed(pkg)]
        if missing:
            return ModelResult(**meta, status=STATUS_NOT_INSTALLED, dataset_name=dataset_info["name"],
                               dataset_description=sub.desc,
                               note=f"Remote streaming needs: pip install {' '.join(missing)}",
                               local_command=commands)
        base_path, source = spec.stream_base_path, f"remote stream from {spec.stream_base_path}"
    else:
        return ModelResult(**meta, status=STATUS_SKIPPED_HEAVY, dataset_name=dataset_info["name"],
                           dataset_description=sub.desc,
                           note=("No local copy (EXTERNAL_CATALOG_DATASET_ROOT is unset) and remote streaming "
                                 "was not requested — rerun with stream=true to fetch one sample"
                                 + (f" from {spec.stream_base_path}" if spec.stream_base_path else "")
                                 + ", or download the subset locally."),
                           local_command=commands)

    loader = LOADERS[spec.slug]
    with Timer() as t:
        try:
            n_samples, sample = loader(base_path, subset, split)
        except Exception as exc:  # noqa: BLE001
            result = failed(**meta, exc=exc, duration=t.elapsed, dataset=dataset_info)
            result.local_command = commands
            return result

    shapes, stats = _describe_sample(sample)
    return ok(**meta, dataset=dataset_info, duration=t.elapsed,
              output_shape=", ".join(f"{k}: {v}" for k, v in shapes.items()) or None,
              output_summary=stats or None,
              metrics={"n_samples": n_samples, "fields": shapes},
              note=f"Loaded sample 0 of {n_samples} via the real {spec.pip_package or spec.name} loader ({source}).")
