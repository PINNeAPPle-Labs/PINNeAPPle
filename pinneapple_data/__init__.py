try:
    from .stl_import import STLMesh, load_stl, load_stl_bytes
except Exception:
    pass

try:
    from .physical_sample import PhysicalSample
except Exception:
    pass

try:
    from .zarr_iterable import ZarrUPDIterable
    from .zarr_store import UPDZarrStore
except Exception:
    pass

try:
    from .collate import collate_pinn_batches, collate_upd_supervised, move_batch_to_device
except Exception:
    pass

try:
    from .collocation import CollocationConfig, CollocationSampler
except Exception:
    pass

try:
    from .active_learning import (
        ActiveLearningConfig,
        AdaptiveCollocationTrainer,
        CombinedAL,
        ResidualBasedAL,
        VarianceBasedAL,
    )
except Exception:
    pass

try:
    from .transforms import MinMaxScaler, Normalizer, StandardScaler
except Exception:
    pass

try:
    from .upd_types import Batch, ConditionSpec, SamplingSpec, UPDItem
except Exception:
    pass

try:
    from .splits import SplitSpec, split_indices
except Exception:
    pass

try:
    from .adapters.pinn_batch_builders import (
        PINNBatch,
        build_from_bundle,
        build_from_real_data,
        build_from_solver,
    )
except Exception:
    pass

from . import datasets, schema
from .datasets import (
    DatasetInfo,
    DatasetRegistry,
    dataset_ids,
    dataset_info,
    list_datasets,
    load_dataset,
)
from .schema import AxisSpec, Field, PhysicsDataBundle, run_gates

__all__ = [
    # STL
    "STLMesh",
    "load_stl",
    "load_stl_bytes",
    # Physical sample
    "PhysicalSample",
    # UPD / Zarr pipeline
    "UPDZarrStore",
    "ZarrUPDIterable",
    # Collate
    "collate_upd_supervised",
    "collate_pinn_batches",
    "move_batch_to_device",
    # Collocation
    "CollocationSampler",
    "CollocationConfig",
    # Active learning
    "ActiveLearningConfig",
    "ResidualBasedAL",
    "VarianceBasedAL",
    "CombinedAL",
    "AdaptiveCollocationTrainer",
    # Transforms / normalizers
    "Normalizer",
    "StandardScaler",
    "MinMaxScaler",
    # UPD types
    "UPDItem",
    "ConditionSpec",
    "SamplingSpec",
    "Batch",
    # Splits
    "SplitSpec",
    "split_indices",
    # PINN batch builders
    "PINNBatch",
    "build_from_bundle",
    "build_from_solver",
    "build_from_real_data",
    # Datasets
    "datasets",
    "DatasetInfo",
    "DatasetRegistry",
    "load_dataset",
    "list_datasets",
    "dataset_info",
    "dataset_ids",
    # Universal physics data schema (INT-001) and its gates (INT-005)
    "schema",
    "AxisSpec",
    "Field",
    "PhysicsDataBundle",
    "run_gates",
]
