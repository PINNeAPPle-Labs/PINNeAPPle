"""The dataset catalogue: what external physics data exists, how it is reached, and what reading it takes.

The registry in :mod:`pinneapple_data.datasets.registry` answers "which arrays can I get in this process" -- it is
a name to loader table. This module answers the question a person actually asks first: *which public dataset
covers this physics, in what format, under what licence, and is there an adapter here that reads it?*

Two fields carry the honesty of the catalogue:

* ``adapter`` -- ``None`` where this library has no reader. The entry then names the formats and the homepage, so
  the next step is a decision, not a discovery. A catalogue that listed ten datasets with no way to tell which
  ones work is a reading list, not an integration;
* ``license`` + ``license_verified`` -- a licence read off the project's own distribution page is marked verified;
  anything taken from memory or from a repository's ``LICENSE`` file without checking which artefact it covers is
  marked unverified, because "the code is MIT" says nothing about the 15 TB of data beside it.

    >>> from pinneapple_data.datasets.catalog import CATALOG, entries, implemented
    >>> [e.id for e in entries(priority="P0")]
    ['cfdbench', 'nasa_tmr', 'pdebench', 'spe10', 'the_well']
    >>> [e.id for e in implemented()]           # the ones with a working reader in this library
    ['cfdbench', 'flowbench', 'pdebench', 'spe10', 'the_well']
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Literal

__all__ = ["CatalogEntry", "CATALOG", "entries", "get", "implemented", "missing", "redistributable", "to_json"]

Priority = Literal["P0", "P1", "P2"]


@dataclass(frozen=True)
class CatalogEntry:
    """One external dataset: where it is, what it holds, what it may be used for, and what reads it."""

    id: str
    title: str
    domain: str
    priority: Priority
    formats: tuple[str, ...]
    homepage: str
    license: str
    license_verified: bool = False
    access: Literal["open", "registration", "restricted"] = "open"
    adapter: str | None = None
    use: tuple[str, ...] = ()
    size: str = ""
    citation: str = ""
    notes: str = ""
    related: tuple[str, ...] = _dc_field(default_factory=tuple)

    @property
    def readable(self) -> bool:
        """Whether this library can turn the dataset into a bundle today."""
        return self.adapter is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id, "title": self.title, "domain": self.domain, "priority": self.priority,
            "formats": list(self.formats), "homepage": self.homepage, "license": self.license,
            "license_verified": self.license_verified, "access": self.access, "adapter": self.adapter,
            "readable": self.readable, "use": list(self.use), "size": self.size, "citation": self.citation,
            "notes": self.notes, "related": list(self.related),
        }


_BENCHMARK = "PDE benchmarks and operator learning"
_CFD = "CFD and turbulence"
_GEO = "subsurface and reservoir"

CATALOG: dict[str, CatalogEntry] = {e.id: e for e in (
    CatalogEntry(
        id="pdebench", title="PDEBench: 1D/2D/3D PDE benchmark suite", domain=_BENCHMARK, priority="P0",
        formats=("HDF5",), homepage="https://github.com/pdebench/PDEBench",
        license="CC BY 4.0 (data, verified on the archive record); code MIT per the project",
        license_verified=True, access="open", adapter="pinneapple_data.datasets.pdebench",
        use=("forward PINN/FNO training", "inverse problems", "operator benchmarks"),
        size="1.3 GB (2D Darcy, smallest case) to 88 GB (2D/3D CFD); 336 files",
        citation="Takamoto et al. (2022), PDEBench: An Extensive Benchmark for Scientific Machine Learning, "
                 "NeurIPS Datasets and Benchmarks; data DOI 10.18419/darus-2986",
        notes="Two on-disk layouts are handled (root 'tensor' arrays and per-sample groups); the archive's "
              "coordinate arrays are one step longer than its fields in the 1D solvers, which the adapter aligns "
              "and records.",
        related=("pdearena",)),
    CatalogEntry(
        id="the_well", title="The Well: 15 TB of diverse physics simulations", domain=_BENCHMARK, priority="P0",
        formats=("HDF5", "Zarr"), homepage="https://polymathic-ai.org/the_well/",
        license="code BSD-3-Clause (verified); data licence per scenario, unverified",
        license_verified=False, access="open", adapter="pinneapple_data.datasets.the_well",
        use=("operator generalisation", "multi-physics surrogates", "temporal forecasting"),
        size="~15 TB over 16+ scenarios; 2.8-21 GB per file",
        citation="Ohana et al. (2024), The Well: a Large-Scale Collection of Diverse Physics Simulations for "
                 "Machine Learning, NeurIPS Datasets and Benchmarks",
        notes="The adapter requires the licence explicitly: the package's BSD-3-Clause covers its code, not the "
              "data.",
        related=("pdebench",)),
    CatalogEntry(
        id="cfdbench", title="CFDBench: incompressible 2D CFD benchmark", domain=_CFD, priority="P0",
        formats=("NumPy",), homepage="https://huggingface.co/datasets/chen-yingfa/CFDBench",
        license="Apache-2.0 (the dataset card's licence field, which is what covers the data; the repository's own "
                "LICENSE file covers its code)",
        license_verified=True, access="open", adapter="pinneapple_data.datasets.cfdbench",
        use=("CFD surrogates", "generalisation across geometry and Reynolds"),
        size="~2.3 GB in four zips (cavity 786 MB, dam 1.35 GB, tube 213 MB, cylinder split by its three subsets)",
        citation="Luo, Chen & Zhang (2023), CFDBench: A Large-Scale Benchmark for Machine Learning Methods in Fluid "
                 "Dynamics, arXiv:2310.05963",
        notes="Read from the archive: cavity/tube/dam/cylinder, each with bc/geo/prop subsets, one directory per case "
              "holding u.npy and v.npy shaped (time, 64, 64), a mask.npy where the case has one, and case.json with "
              "the case's parameters. The archive states no units -- the reader records that as unknown and takes "
              "units={...} when the caller knows them.",),
    CatalogEntry(
        id="spe10", title="SPE10: heterogeneous reservoir model", domain=_GEO, priority="P0",
        formats=("text",), homepage="https://www.spe.org/web/csp/datasets/set02.htm",
        license="not stated: the SPE page links the archive (por_perm_case2a.zip) without terms",
        license_verified=False, access="open", adapter="pinneapple_data.datasets.spe10",
        use=("Darcy flow", "FNO/PINN permeability upscaling"),
        size="77 MB (spe_perm.dat 58 MB, spe_phi.dat 19 MB)",
        citation="Christie & Blunt (2001), Tenth SPE Comparative Solution Project, SPE Reservoir Evaluation",
        notes="The fine grid of model 2 (60 x 220 x 85 cells, 20 x 10 x 2 ft, Tarbert over Upper Ness) read from the "
              "archive itself: six values per line, three permeability components per cell, i varying fastest -- "
              "checked against the model's own layer structure, which flips at the Tarbert/Upper Ness boundary. The "
              "licence is the caller's: the download requires no registration but states no terms."),
    CatalogEntry(
        id="nasa_tmr", title="NASA Turbulence Modeling Resource", domain=_CFD, priority="P0",
        formats=("CGNS", "Plot3D", "text"), homepage="https://turbmodels.larc.nasa.gov/",
        license="public domain (US Government work), unverified per case", access="open", adapter=None,
        use=("numerical verification", "turbulence model validation"),
        citation="NASA Langley Research Center, Turbulence Modeling Resource",
        notes="Verification cases with published reference quantities (flat plate, bump, cylinder) -- the natural "
              "reference half of a benchmark. No reader here: on 2026-11-13 the legacy host answered every path with "
              "the same HTML page (its validation files are no longer served) and the nasa.gov landing page the old "
              "URLs redirect to is the generic site shell, so there is no file whose layout could be checked.",
        related=("nrel_wind",)),
    CatalogEntry(
        id="pdearena", title="PDEArena: PDE operator learning benchmark", domain=_BENCHMARK, priority="P1",
        formats=("HDF5", "NumPy"), homepage="https://github.com/microsoft/pdearena",
        license="MIT (code), unverified", access="open", adapter=None,
        use=("operator benchmark", "time-dependent PDEs"),
        notes="Shares the 1D/2D PDE families of PDEBench; a cross-check rather than a second format."),
    CatalogEntry(
        id="flowbench", title="FlowBench: complex geometry flows", domain=_CFD, priority="P1",
        formats=("NumPy", "npz"), homepage="https://huggingface.co/datasets/BGLab/FlowBench",
        license="CC-BY-NC-4.0 (stated on the dataset card and in the paper; non-commercial)",
        license_verified=True, access="open", adapter="pinneapple_data.datasets.flowbench",
        use=("neural simulators on complex geometry", "geometry generalisation"),
        size="over 10 000 samples; the FPO families are ~1.4 GB per sample",
        citation="Tali et al. (2024), FlowBench: A Large Scale Benchmark for Flow Simulation over Complex Geometries, "
                 "arXiv:2409.18032",
        notes="Read from the archive: the LDC families store X/Y tensors shaped (n_samples, n_channels, height, width) "
              "whose channel lists are the paper's Table 12, and an FPO sample stores (n_time, height, width, 3) with "
              "u, v, p last plus an input_geometry.npz. The repository the catalogue used to name "
              "(github.com/erizmr/FlowBench) is gone; the data now lives on HuggingFace. NC: verified is not "
              "redistributable in a product."),
    CatalogEntry(
        id="jhtdb", title="Johns Hopkins Turbulence Database", domain=_CFD, priority="P1",
        formats=("web service", "HDF5"), homepage="https://turbulence.pha.jhu.edu/",
        license="terms of use, registration required", access="registration", adapter=None,
        use=("DNS reconstruction", "extrapolation", "subgrid modelling"),
        citation="Li et al. (2008), A public turbulence database cluster and applications, J. Turbulence",
        notes="Access is an authenticated web service (cutout requests), not a file: an adapter is an HTTP client, "
              "not a reader."),
    CatalogEntry(
        id="norne", title="Norne: full-field reservoir model", domain=_GEO, priority="P1",
        formats=("GRDECL", "EGRID", "INIT"), homepage="https://github.com/OPM/opm-data",
        license="unverified (Open Porous Media data terms)", access="open", adapter=None,
        use=("data assimilation", "production forecasting"),
        citation="Equinor / Open Porous Media, Norne benchmark case"),
    CatalogEntry(
        id="nrel_wind", title="NREL Wind Integration National Dataset (WIND Toolkit)", domain="weather and energy",
        priority="P2", formats=("HDF5", "CSV"), homepage="https://www.nrel.gov/grid/wind-toolkit.html",
        license="requires API key, terms unverified", access="registration", adapter=None,
        use=("wind forecasting", "digital twins"),
        notes="Time series at fixed points, not fields: the digital-twin side (INT-012) rather than a PDE case."),
)}


def entries(*, priority: str | None = None, domain: str | None = None,
            readable: bool | None = None) -> list[CatalogEntry]:
    """The catalogue, filtered and sorted by priority then id."""
    out = list(CATALOG.values())
    if priority is not None:
        out = [e for e in out if e.priority == priority]
    if domain is not None:
        out = [e for e in out if e.domain == domain]
    if readable is not None:
        out = [e for e in out if e.readable is readable]
    return sorted(out, key=lambda e: (e.priority, e.id))


def get(dataset_id: str) -> CatalogEntry:
    try:
        return CATALOG[dataset_id]
    except KeyError:
        raise KeyError(f"'{dataset_id}' is not catalogued; known: {sorted(CATALOG)}") from None


def implemented() -> list[CatalogEntry]:
    """The catalogued datasets this library can actually read into a bundle today."""
    return entries(readable=True)


def missing(priority: str | None = None) -> list[CatalogEntry]:
    """The catalogued datasets with no reader here -- the backlog, written down."""
    return entries(priority=priority, readable=False)


def redistributable(dataset_id: str) -> bool:
    """Whether the dataset's licence is *known* to allow redistribution.

    ``False`` is not a legal opinion: it means the licence has not been verified here, which is the only safe
    answer to give before someone ships the data inside a product. ``True`` is not a commercial clearance either:
    a non-commercial licence (FlowBench's CC-BY-NC-4.0) grants redistribution and still forbids the product, so the
    entry's ``license`` text is what the decision is made on."""
    return get(dataset_id).license_verified


def to_json(entries_: Iterable[CatalogEntry] | None = None) -> list[dict[str, object]]:
    """The catalogue as JSON-ready dicts (the payload the Engineering API serves)."""
    return [e.to_dict() for e in (entries_ if entries_ is not None else entries())]
