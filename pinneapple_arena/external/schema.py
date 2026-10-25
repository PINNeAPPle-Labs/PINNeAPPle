"""
Data model for one catalog entry (`ModelSpec`) plus the physical-domain and
architecture-family metadata used to group/label entries in both the API
and the frontend.

`pinneapple_arena.external.models_data.MODELS` is the single source of truth
for the external catalog: the catalog view, the model-detail
view, the inference/fine-tune runners and the benchmark module all read from
here. There is no separate copy of this data embedded in the frontend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    slug: str                      # stable machine key, e.g. "deepxde" — used in URLs/API
    name: str                      # display name
    org: str                       # author / organization
    arch: str                      # architecture family: pinn, operator, diffsim, rigidbody, graph, domain
    domains: tuple[str, ...]       # physical domain keys, see DOMAIN_META
    desc: str
    tags: tuple[str, ...]
    repo: str                      # GitHub (or equivalent) URL
    ref: str                       # citation string

    # --- runtime/capability metadata (drives what the app can actually do) ---
    pip_package: str | None = None    # pip package name if installable, else None
    import_name: str | None = None    # module name to `import` for the install check
    heavy: bool = False                  # GPU/heavy dependency — gated behind --include-heavy
    manual_only: bool = False            # not pip-installable at all (git-only / non-Python / non-commercial weights)
    manual_note: str | None = None    # human instructions when manual_only is True
    dataset_fn: str | None = None     # name of the ds_* generator in pinneapple_arena.external.demo_datasets used for the demo run
    finetunable: bool = False            # True if a real short retrain/reconfigure loop exists in pinneapple_arena.external.finetune
    license: str | None = None        # SPDX id where one applies; None: no licence file was found
    license_source: str | None = None # where the licence was read (file, PyPI metadata) and when
    license_note: str | None = None   # restrictions or conflicts worth knowing (weights, PyPI vs repository)
    non_commercial: bool = False         # the code or the published weights forbid commercial use
    maturity: str = "maintained library"  # "maintained library" | "paper code" | "archived"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["domains"] = list(self.domains)
        d["tags"] = list(self.tags)
        return d


@dataclass(frozen=True)
class DomainMeta:
    key: str
    label: str
    glyph: str
    desc: str


@dataclass(frozen=True)
class ArchMeta:
    key: str
    label: str


# --------------------------------------------------------------------------- #
# Datasets
#
# A dataset is not a model: there is nothing to "infer" or "fine-tune", so it
# does not fit ModelSpec or the runner registry. Datasets get their own
# parallel catalog (registry/datasets_data.py) and their own action — a
# "probe" that loads one real sample (pinneapple_arena/external/dataset_probe.py)
# — while reusing the same status vocabulary as model results.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SubDataset:
    name: str                      # exact identifier the upstream loader expects
    label: str
    domains: tuple[str, ...]       # keys into DOMAIN_META
    dims: str                      # "2D" | "3D" | ...
    desc: str


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    name: str
    org: str
    desc: str
    tags: tuple[str, ...]
    repo: str
    ref: str
    size: str                      # human-readable total size
    subsets: tuple[SubDataset, ...]
    splits: tuple[str, ...] = ("train", "valid", "test")
    pip_package: str | None = None
    import_name: str | None = None
    stream_base_path: str | None = None   # remote base path the loader can stream from
    stream_requires: tuple[str, ...] = ()    # extra pip packages remote streaming needs
    download_command: str | None = None   # template with {subset} and {split}
    license: str | None = None
    homepage: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        d["splits"] = list(self.splits)
        d["stream_requires"] = list(self.stream_requires)
        d["subsets"] = [{**asdict(s), "domains": list(s.domains)} for s in self.subsets]
        d["domains"] = sorted({dom for s in self.subsets for dom in s.domains})
        return d
