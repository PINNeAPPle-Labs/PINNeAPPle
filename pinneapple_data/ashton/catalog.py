"""The three Ashton-family automotive CFD datasets: what they are, where they are, and under what licence.

Neil Ashton (AWS, now NVIDIA) and co-authors published three open road-vehicle CFD datasets built for machine
learning, in the same on-disk layout, under the same licence (CC BY-SA 4.0), each a parametric family of one
reference body run with a scale-resolving method:

* **AhmedML** -- 500 Ahmed-body variants, hybrid RANS-LES in OpenFOAM v2212, ~20 M cells, 2 TB,
  arXiv:2407.20801;
* **DrivAerML** -- 500 morphed DrivAer notchbacks (484 published), hybrid RANS-LES, 140-160 M cells, 31 TB,
  arXiv:2408.11969;
* **WindsorML** -- 355 Windsor bodies (350 run folders), GPU wall-modelled LES with a Cartesian immersed
  boundary, more than 280 M cells, 8 TB, arXiv:2407.19320 (NeurIPS 2024 Datasets and Benchmarks).

"Same layout" is the property this package exists for: every case is a directory ``run_<i>/`` holding a geometry
(``*.stl``, sometimes ``*.step``/``*.stp``), the time-averaged surface field (``boundary_<i>.vtp`` for the
OpenFOAM runs, ``boundary_<i>.vtu`` for WindsorML), an array of the surface quadrature weights the dataset
publishes (``boundary_cell_area_<i>.npy`` per native polygon, ``boundary_dual_area_<i>.npy`` per point), the
time-averaged force and moment coefficients (``force_mom_<i>.csv``), the design parameters
(``geo_parameters_<i>.csv``) and, optionally, the volumetric field (``volume_<i>.vtu``, 5.6 GB to 44 GB per
case). One reader (:mod:`pinneapple_data.ashton.loader`) serves all three; only the surface kind and the
quadrature differ, and both are declared here rather than sniffed at read time.

Nothing in this module downloads anything by itself. :func:`pinneapple_data.ashton.loader.fetch` streams a
single case file through :func:`pinneapple_data.datasets.download.fetch_cached`, so a 66 KB forces CSV costs
66 KB and a 44 GB volume file costs what it costs -- and the caller decides which one to ask for.

**Licence and credit.** The three datasets are CC BY-SA 4.0 (the ``license`` field of each dataset card, and
their ``LICENSE.txt``). Share-alike travels with the data: a derived dataset redistributed from these has to
carry the same licence, which for a surrogate trained on it matters to whoever ships the surrogate. The
attribution each dataset asks for is its paper, quoted in :data:`DATASETS` and returned by
:func:`attribution`; :func:`pinneapple_data.ashton.loader.open_dataset` carries both in the object it returns,
so a bundle made from a case cannot lose them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["LICENCE", "LICENCE_URL", "ATTRIBUTION", "AshtonDataset", "DATASETS", "dataset", "dataset_ids",
           "attribution", "licences", "layout_notes"]

#: The licence of all three datasets: the ``license`` field of each Hugging Face card.
LICENCE = "CC-BY-SA-4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

#: The credit the datasets ask for, one line per dataset, quoted from its card ("How to cite this dataset").
ATTRIBUTION: dict[str, str] = {
    "ahmedml": "Ashton, N., Maddix, D., Gundry, S. and Shabestari, P. (2024), AhmedML: High-Fidelity Computational "
               "Fluid Dynamics dataset for incompressible, low-speed bluff body aerodynamics, arXiv:2407.20801",
    "drivaerml": "Ashton, N., Mockett, C., Fuchs, M., Fliessbach, L., Hetmann, H., Knacke, T., Schonwald, N., "
                 "Skaperdas, V., Fotiadis, G., Walle, A., Hupertz, B. and Maddix, D. (2024), DrivAerML: High-Fidelity "
                 "Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics, arXiv:2408.11969",
    "windsorml": "Ashton, N., Angel, J., Ghate, A., Kenway, G., Long Wong, M., Kiris, C., Walle, A., Maddix, D. and "
                 "Page, G. (2024), WindsorML: High-Fidelity Computational Fluid Dynamics dataset for automotive "
                 "aerodynamics, arXiv:2407.19320 (NeurIPS 2024 Datasets and Benchmarks)",
}

SurfaceKind = Literal["polydata", "unstructuredgrid"]
QuadratureKind = Literal["cell", "point"]


@dataclass(frozen=True)
class AshtonDataset:
    """One of the three datasets, as everything a reader needs to find and to credit it.

    The file names are stored as format strings with ``{i}``, which is how the cards spell them; a case that
    spells the index differently is resolved by :func:`pinneapple_data.ashton.loader.resolve_file`, not by a
    second table.
    """

    id: str
    name: str
    repo: str                          # Hugging Face dataset repo
    paper: str                         # arXiv id of the dataset paper
    attribution: str
    runs: tuple[int, ...] = ()         # the run indices the cards list
    withdrawn: tuple[int, ...] = ()    # runs the card says are not in the dataset
    #: The surface field file and the VTK container it is written in.
    surface: str = "run_{i}/boundary_{i}.vtp"
    surface_kind: SurfaceKind = "polydata"
    #: The quadrature the dataset publishes: an area per native polygon (cell) or a weight per point.
    areas: str = "run_{i}/boundary_cell_area_{i}.npy"
    quadrature: QuadratureKind = "cell"
    #: The time-averaged force and moment coefficients, and the same with a per-case reference area.
    forces: str = "run_{i}/force_mom_{i}.csv"
    forces_varref: str = "run_{i}/force_mom_varref_{i}.csv"
    parameters: str = "run_{i}/geo_parameters_{i}.csv"
    geometry: str = "run_{i}/{body}_{i}.stl"
    volume: str = "run_{i}/volume_{i}.vtu"
    #: The columns the card says ``forces`` holds. Read as written; a column that is not there is absent, not
    #: renamed.
    force_columns: tuple[str, ...] = ()
    #: The freestream the coefficients are normalised with, from the dataset paper (see each field's docstring
    #: in :data:`FREESTREAM_SOURCE`). ``a_ref`` is the constant reference area of ``force_mom_i.csv``.
    velocity: float = 0.0              # m/s
    density: float = 0.0               # kg/m^3
    a_ref: float = 0.0                 # m^2
    l_ref: float = 0.0                 # m
    #: The candidate names of the surface fields, resolved case-insensitively against what the file holds.
    fields: dict[str, tuple[str, ...]] = field(default_factory=dict)
    size: str = ""
    cells: str = ""
    notes: str = ""

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}"

    @property
    def resolve_url(self) -> str:
        return f"{self.url}/resolve/main"

    @property
    def licence(self) -> str:
        return LICENCE

    @property
    def n_cases(self) -> int:
        """The cases the dataset publishes: the runs the card lists, minus the ones it says are withdrawn."""
        return len([i for i in self.runs if i not in self.withdrawn])

    def run_dir(self, run: int) -> str:
        return f"run_{int(run)}"

    def file(self, which: str, run: int) -> str:
        """The path of one file of one case, relative to the dataset root: ``file("surface", 7)``."""
        if which not in _FILES:
            raise KeyError(f"'{which}' is not one of {sorted(_FILES)}")
        template = getattr(self, _FILES[which])
        return template.format(i=int(run), body=self.id.removesuffix("ml"))

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "name": self.name, "repo": self.repo, "url": self.url, "paper": self.paper,
                "licence": self.licence, "licence_url": LICENCE_URL, "attribution": self.attribution,
                "cases": self.n_cases, "size": self.size, "cells": self.cells, "surface": self.surface,
                "surface_kind": self.surface_kind, "quadrature": self.quadrature,
                "force_columns": list(self.force_columns), "velocity": self.velocity, "density": self.density,
                "a_ref": self.a_ref, "l_ref": self.l_ref, "notes": self.notes}


_FILES = {"surface": "surface", "areas": "areas", "forces": "forces", "forces_varref": "forces_varref",
          "parameters": "parameters", "geometry": "geometry", "volume": "volume"}

#: Where each number in :class:`AshtonDataset` comes from, so a reader does not have to trust the docstring.
FREESTREAM_SOURCE = {
    "ahmedml": "U_inf = 1 m/s and rho = 1 kg/m^3 with nu = 3.75e-7 (Re = 7.68e5 on the body height), A_ref = 0.112 "
               "m^2 held constant across variants in force_mom_i.csv: Ashton et al. 2024, Section 'Boundary "
               "conditions' and Eq. 35",
    "drivaerml": "U_inf = 38.889 m/s, rho = 1 kg/m^3, A_ref = 2.17 m^2 and L_ref = 2.78618 m for "
                 "force_mom_constref_i.csv (A_ref 1.779-2.636 m^2 and L_ref 2.636-3.035 m for the per-case "
                 "force_mom_i.csv): Ashton et al. 2024, Table 3",
    "windsorml": "U_ref = 40 m/s, rho_ref = 1.31 kg/m^3, p_ref = 37330.4 Pa, A_ref = 0.112 m^2, L_ref = 0.6375 m: "
                 "Ashton et al. 2024, Table 5",
}

DATASETS: dict[str, AshtonDataset] = {d.id: d for d in (
    AshtonDataset(
        id="ahmedml", name="AhmedML", repo="neashton/ahmedml", paper="arXiv:2407.20801",
        attribution=ATTRIBUTION["ahmedml"], runs=tuple(range(1, 501)), withdrawn=(),
        surface="run_{i}/boundary_{i}.vtp", surface_kind="polydata",
        areas="run_{i}/boundary_cell_area_{i}.npy", quadrature="cell",
        force_columns=("cd", "cl"), velocity=1.0, density=1.0, a_ref=0.112,
        fields={"cp": ("Cp", "Pressure_Coefficient", "cpMean", "CpMean"), "pressure": ("pMean", "p", "Pressure"),
                "wall_shear": ("wallShearStress", "WallShearStress", "wallShearStressMean"),
                "y_plus": ("yPlus", "yPlusMean", "y")},
        size="2 TB (500 cases; the surface file is 82 MB, the volume 5.6 GB)", cells="~1.1 M surface, ~20 M volume",
        notes="OpenFOAM v2212 hybrid RANS-LES, ~80 convective time units at dt U/L = 6e-4. The card spells the "
              "geometry parameters file 'geo_parameters_1.csv' for every run, which the file resolver accepts "
              "alongside 'geo_parameters_<i>.csv'. Run 500 has no geo_parameters file.",
        # the geometry file name uses the body prefix, not the dataset id
        geometry="run_{i}/ahmed_{i}.stl", volume="run_{i}/volume_{i}.vtu"),
    AshtonDataset(
        id="drivaerml", name="DrivAerML", repo="neashton/drivaerml", paper="arXiv:2408.11969",
        attribution=ATTRIBUTION["drivaerml"], runs=tuple(range(1, 501)),
        withdrawn=(167, 211, 218, 221, 248, 282, 291, 295, 316, 325, 329, 364, 370, 376, 403, 473),
        surface="run_{i}/boundary_{i}.vtp", surface_kind="polydata",
        areas="run_{i}/boundary_cell_area_{i}.npy", quadrature="cell",
        force_columns=("cd", "cl", "clf", "clr"), velocity=38.889, density=1.0, a_ref=2.17, l_ref=2.78618,
        fields={"cp": ("Cp", "CpMean", "Pressure_Coefficient"), "pressure": ("pMean", "p", "Pressure"),
                "wall_shear": ("wallShearStress", "WallShearStress"), "y_plus": ("yPlus", "yPlusMean")},
        size="31 TB (484 public cases; the surface file is ~612 MB, the volume 44 GB)",
        cells="~8.8 M surface, ~160 M volume",
        notes="OpenFOAM v2212, morphed DrivAer notchback, 16 of the 500 runs are not published (the card lists "
              "them) and 15 more are kept blind. volume_i.vtu is split into two parts on the Hub and has to be "
              "concatenated; the reader here never opens the volume unless asked for a subsample.",
        geometry="run_{i}/drivaer_{i}.stl"),
    AshtonDataset(
        id="windsorml", name="WindsorML", repo="neashton/windsorml", paper="arXiv:2407.19320",
        attribution=ATTRIBUTION["windsorml"], runs=tuple(range(0, 350)), withdrawn=(),
        surface="run_{i}/boundary_{i}.vtu", surface_kind="unstructuredgrid",
        areas="run_{i}/boundary_dual_area_{i}.npy", quadrature="point",
        force_columns=("cd", "cs", "cl", "cmy"), velocity=40.0, density=1.31, a_ref=0.112, l_ref=0.6375,
        fields={"cp": ("Cp", "Pressure_Coefficient", "cp"), "pressure": ("p", "Pressure", "pressureAvg"),
                "skin_friction": ("Cf", "SkinFrictionCoefficient", "cf"),
                "wall_shear": ("wallShearStress", "WallShearStress"), "y_plus": ("yPlus", "y")},
        size="8 TB (350 run folders over 355 design ids; the surface file is ~500 MB, the volume 20 GB)",
        cells=">280 M volume cells",
        notes="Volcano Platforms GPU wall-modelled LES, Cartesian immersed boundary, and so a different surface "
              "container (.vtu) and a different quadrature (a weight per point, not an area per polygon) from the "
              "two OpenFOAM datasets -- the one place the shared layout is not identical. Splits v2 exclude six "
              "duplicate geometries and five unavailable runs from the 355 design ids."),
)}

for _d in DATASETS.values():
    if _d.id not in ATTRIBUTION or _d.id not in FREESTREAM_SOURCE:
        raise RuntimeError(f"{_d.id} has no attribution or no source for its reference quantities")


def dataset_ids() -> list[str]:
    return list(DATASETS)


def dataset(name: str) -> AshtonDataset:
    """The dataset by id (``"ahmedml"``), case-insensitively; raises with the list when it is not one of them."""
    key = str(name).strip().lower()
    if key not in DATASETS:
        raise KeyError(f"{name!r} is not one of {sorted(DATASETS)}: these are the three Ashton-family datasets")
    return DATASETS[key]


def attribution(name: str) -> str:
    """The credit the dataset asks for: its paper, as its card spells it."""
    return dataset(name).attribution


def licences() -> dict[str, str]:
    """Dataset id to licence, which is the same string for all three and is not a detail to be assumed."""
    return {key: DATASETS[key].licence for key in DATASETS}


def layout_notes() -> str:
    """The on-disk layout, as the three cards describe it, for a reader deciding what to download."""
    return """run_<i>/                        one case (i is the run index; WindsorML runs are 0..349)
    boundary_<i>.vtp            surface field, VTK XML PolyData        (AhmedML, DrivAerML)
    boundary_<i>.vtu            surface field, VTK XML UnstructuredGrid (WindsorML)
    boundary_cell_area_<i>.npy  one area per native polygon, float32 <f4  (AhmedML, DrivAerML)
    boundary_dual_area_<i>.npy  one quadrature weight per point, float32   (WindsorML)
    force_mom_<i>.csv           time-averaged coefficients, constant reference area
    force_mom_varref_<i>.csv    the same, reference area of the case
    geo_parameters_<i>.csv      the design parameters of the case
    <body>_<i>.stl              the surface geometry (and .step/.stp for DrivAerML and WindsorML)
    volume_<i>.vtu              the volumetric field, 5.6 GB to 44 GB per case
force_mom_all.csv, geo_parameters_all.csv, splits/manifest.json   the aggregate tables and the splits
"""
