"""External model catalog: references to third-party physics-ML libraries and the glue that runs them.

Every entry (`ModelSpec`) is a reference: name, organisation, repository URL, licence (with where it was read),
citation and install instructions. No third-party code is copied here: each runner in
``pinneapple_arena.external.runners`` imports the real library at run time, builds a minimal instance, runs one
forward pass or simulation on a small synthetic input (``demo_datasets``) and reports a ``ModelResult`` whose
``status`` says what really happened:

========================== ===============================================================================
``ok``                     the library was imported and a real call ran; the output is genuine
``not_installed``          pip-installable, not installed here (the result carries the pip command)
``skipped_heavy``          GPU / heavy dependency, run only with ``include_heavy=True``
``manual_install``         no pip package (git-only, Julia, C++); the result carries the clone/build steps
``failed``                 it ran and raised; the exception is in ``error``
``finetune_not_supported`` no fine-tune / reconfigure routine for this library
========================== ===============================================================================

A result is never invented. Datasets (The Well) have their own catalog (``DATASETS``) and a ``probe`` action that
loads one real sample through the dataset's own loader.

    from pinneapple_arena import external as ext

    ext.MODELS_BY_SLUG["deepxde"].license            # 'LGPL-2.1' (read from the repository's LICENSE file)
    r = ext.call_infer("deepxde")                     # ModelResult; status 'not_installed' without deepxde
    ext.run_benchmark(ext.select_slugs(category="pinn"))

Command line: ``python -m pinneapple_arena.external --list`` / ``--model deepxde --action infer``.
"""
from __future__ import annotations

import os

# Some catalog libraries call torch.set_default_device("mps") on Apple Silicon as a side effect; with this
# fallback an op they leave on MPS runs on the CPU instead of raising (the finetune runners also reset the
# default device to "cpu" per call). setdefault: an operator who wants MPS can still override it.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pinneapple_arena.external.arch_meta import ARCH_META  # noqa: E402
from pinneapple_arena.external.benchmark import run_benchmark, run_compare, select_slugs  # noqa: E402
from pinneapple_arena.external.dataset_probe import probe_dataset  # noqa: E402
from pinneapple_arena.external.datasets_data import DATASETS, DATASETS_BY_SLUG  # noqa: E402
from pinneapple_arena.external.domain_meta import DOMAIN_META  # noqa: E402
from pinneapple_arena.external.finetune import run_finetune  # noqa: E402
from pinneapple_arena.external.installer import is_installed  # noqa: E402
from pinneapple_arena.external.local_command import local_command_for, pip_install_command  # noqa: E402
from pinneapple_arena.external.models_data import MODELS, MODELS_BY_SLUG  # noqa: E402
from pinneapple_arena.external.result import ModelResult  # noqa: E402
from pinneapple_arena.external.runners.registry import RUNNERS, call_infer, get_dataset_preview  # noqa: E402
from pinneapple_arena.external.schema import DatasetSpec, ModelSpec  # noqa: E402
from pinneapple_arena.external.status import install_state  # noqa: E402


def licence_counts() -> dict[str, int]:
    """Number of catalog entries per licence (``"none found"`` when the repository has no licence file)."""
    counts: dict[str, int] = {}
    for m in MODELS:
        key = m.license or "none found"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


__all__ = [
    "ARCH_META", "DATASETS", "DATASETS_BY_SLUG", "DOMAIN_META", "MODELS", "MODELS_BY_SLUG", "RUNNERS",
    "DatasetSpec", "ModelResult", "ModelSpec",
    "call_infer", "get_dataset_preview", "install_state", "is_installed", "licence_counts", "local_command_for",
    "pip_install_command", "probe_dataset", "run_benchmark", "run_compare", "run_finetune", "select_slugs",
]
