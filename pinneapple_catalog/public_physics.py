"""Source discovery without treating public data as validated PINN benchmarks.

Curated from matteus-dev1/10-Bases-publicas-de-dados-de-fisica (2026-11-14).
Links are leads, not verified download endpoints or licences. Existing licensed
PINNeAPPle resource records take precedence and must still pass their gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from .resources import get_resource

SOURCE_REPOSITORY = "https://github.com/matteus-dev1/10-Bases-publicas-de-dados-de-fisica"


@dataclass(frozen=True)
class PhysicsSource:
    id: str
    name: str
    url: str
    domain: str
    data_kind: str
    existing_resource: str | None = None
    role: str = "discovery_only"
    license_status: str = "unverified"

    def to_dict(self) -> dict:
        item = asdict(self)
        item["source_repository"] = SOURCE_REPOSITORY
        if self.existing_resource:
            try:
                resource = get_resource(self.existing_resource)
            except KeyError:
                item["existing_resource_status"] = "missing"
            else:
                item["existing_resource_status"] = "catalogued"
                item["resource_commercial"] = resource.commercial
                item["resource_license"] = resource.license
                item["license_status"] = "see_existing_resource"
        return item


SOURCES = (
    PhysicsSource("the_well", "The Well", "https://github.com/PolymathicAI/the_well",
                  "multiphysics", "simulation_fields", "the_well"),
    PhysicsSource("higgs", "HIGGS", "https://archive.ics.uci.edu/dataset/280/higgs",
                  "particle_physics", "classification"),
    PhysicsSource("magic", "MAGIC Gamma Telescope", "https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope",
                  "astrophysics", "classification"),
    PhysicsSource("sdss17", "Stellar Classification SDSS17",
                  "https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17",
                  "astronomy", "classification"),
    PhysicsSource("exoplanet_archive", "NASA Exoplanet Archive",
                  "https://exoplanetarchive.ipac.caltech.edu", "astrophysics", "tabular_observations"),
    PhysicsSource("cern_open_data", "CERN Open Data", "https://opendata.cern.ch",
                  "particle_physics", "experimental_events"),
    PhysicsSource("gaia_dr3", "Gaia DR3", "https://gea.esac.esa.int/archive/",
                  "astrophysics", "astrometry"),
    PhysicsSource("gwosc", "GWOSC", "https://gwosc.org", "gravitation", "time_series"),
    PhysicsSource("materials_project", "Materials Project",
                  "https://next-gen.materialsproject.org", "materials", "atomistic_dft", "materials_project"),
    PhysicsSource("nist_asd", "NIST Atomic Spectra Database",
                  "https://physics.nist.gov/asd", "atomic_physics", "spectral_lines"),
)


def list_public_physics_sources(*, domain: str | None = None,
                                data_kind: str | None = None) -> list[dict]:
    """List metadata only: no downloading, no automatic licence approval."""
    result = []
    for source in SOURCES:
        if domain and source.domain != domain:
            continue
        if data_kind and source.data_kind != data_kind:
            continue
        parsed = urlsplit(source.url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(f"invalid curated URL for {source.id}")
        result.append(source.to_dict())
    return result
