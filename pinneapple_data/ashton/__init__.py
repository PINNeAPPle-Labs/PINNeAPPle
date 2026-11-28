"""The Ashton-family automotive CFD datasets (AhmedML, DrivAerML, WindsorML) in one reader, one benchmark and one
out-of-distribution experiment.

Neil Ashton and co-authors published three open, CC BY-SA 4.0, road-vehicle CFD datasets in the same on-disk
layout -- 500 Ahmed bodies (arXiv:2407.20801), 500 morphed DrivAer notchbacks (arXiv:2408.11969) and 355 Windsor
bodies (arXiv:2407.19320) -- so one loader serves all three and a model can be trained on one geometry family and
tested on another. That last property is what this package is built around: it is the only public automotive CFD
corpus where an out-of-distribution test across *bodies* is a real experiment rather than a recount of the same
family's variation.

Four pieces:

* :mod:`~pinneapple_data.ashton.catalog` -- the three datasets: the paper, the licence, the credit, the layout,
  the reference quantities, and which file holds what;
* :mod:`~pinneapple_data.ashton.loader` -- open a dataset on disk, read one case, integrate its drag from the
  surface pressure, subsample its volume field by byte range. A dataset that is not there raises with the URL
  and the call that fetches it, not with a ``KeyError``;
* :mod:`~pinneapple_data.ashton.benchmark` -- the comparison table: published baselines as *declared* numbers
  with their source, measured numbers where a case is on disk, and an empty column that says what it needs;
* :mod:`~pinneapple_data.ashton.trust` -- train on one family, walk towards the other, and measure whether the
  embedding distance and the conformal interval move before the error does.

The datasets are terabytes (2 TB, 31 TB, 8 TB) and need the network, so nothing here downloads anything unless
asked, and the tests run on a synthetic fixture in the same layout (:mod:`~pinneapple_data.ashton.fixture`).

    >>> from pinneapple_data.ashton import open_dataset, benchmark_table
    >>> data = open_dataset("drivaerml", root="data/ashton/drivaerml")     # reads nothing yet
    >>> data.licence, data.attribution[:40]
    ('CC-BY-SA-4.0', 'Ashton, N., Mockett, C., Fuchs, M., Flies')

What this package does **not** do: it does not ship the data, it does not train a surrogate on the 31 TB, and it
does not claim a number it did not measure or could not source. Each of the three is stated where it matters --
in :mod:`pinneapple_data.ashton.benchmark` for the numbers and in ``docs/guides/ashton_datasets.md`` for the whole
picture, including the one place the issue that asked for this was wrong (the AB-UPT paper reports DrivAerML and
AhmedML, not all three).
"""
from __future__ import annotations

from .benchmark import (
                        BASELINES,
                        BenchmarkRow,
                        BenchmarkTable,
                        benchmark_table,
                        constant_field_baseline,
                        declared,
                        mae,
                        mse,
                        relative_l2,
                        surface_metrics,
)
from .catalog import (
                        ATTRIBUTION,
                        DATASETS,
                        LICENCE,
                        LICENCE_URL,
                        AshtonDataset,
                        attribution,
                        dataset,
                        dataset_ids,
                        layout_notes,
                        licences,
)
from .loader import (
                        AshtonData,
                        Case,
                        CaseNotPresent,
                        MeasuredForces,
                        case_bundle,
                        fetch,
                        list_arrays,
                        open_dataset,
                        read_force_csv,
                        read_forces,
                        read_parameters,
                        resolve_field,
                        resolve_file,
                        volume_subsample,
)
from .trust import CrossFamilyResult, cross_family_experiment, run_on_fixtures
from .vtk_xml import Surface, VtkFormatError, face_area_normals, read_surface, write_surface

__all__ = [
    # catalogue
    "DATASETS", "LICENCE", "LICENCE_URL", "ATTRIBUTION", "AshtonDataset", "dataset", "dataset_ids", "attribution",
    "licences", "layout_notes",
    # reading a case
    "open_dataset", "AshtonData", "Case", "CaseNotPresent", "MeasuredForces", "fetch", "resolve_file",
    "resolve_field", "read_force_csv", "read_forces", "read_parameters", "list_arrays", "volume_subsample",
    "case_bundle",
    # surfaces
    "Surface", "VtkFormatError", "read_surface", "write_surface", "face_area_normals",
    # benchmark
    "BASELINES", "declared", "benchmark_table", "BenchmarkTable", "BenchmarkRow", "constant_field_baseline",
    "surface_metrics", "relative_l2", "mse", "mae",
    # cross-family trust
    "cross_family_experiment", "run_on_fixtures", "CrossFamilyResult",
]
