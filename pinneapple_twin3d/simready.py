"""Opt-in NVIDIA SimReady catalog access without vendoring third-party assets.

This module deliberately downloads no assets during import. A pinned dataset
revision and a per-asset license acknowledgement are required for materialization.
OpenUSD dependencies are preserved in the downloaded package; no physics metadata
is silently converted to GLB.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import json

DATASET = "nvidia/simready-assets"
SOURCE_URL = "https://huggingface.co/datasets/nvidia/simready-assets"
LICENSE = "CC-BY-4.0"
PACKAGE_MANIFEST = "com.nvidia.simready.packaging.json"


@dataclass(frozen=True)
class SimReadyAsset:
    asset_id: str
    package_path: str
    dataset: str
    revision: str
    license: str = LICENSE

    def provenance(self) -> dict:
        return {**asdict(self), "source_url": SOURCE_URL}


def _hub():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ImportError(
            "SimReady catalog access requires 'huggingface_hub'. "
            "Install with: pip install huggingface_hub"
        ) from exc
    return HfApi, snapshot_download


def _check_revision(revision: str) -> str:
    if not isinstance(revision, str) or len(revision) != 40 or any(
        c not in "0123456789abcdefABCDEF" for c in revision
    ):
        raise ValueError("revision must be a pinned 40-character Git commit SHA")
    return revision


def _check_asset_id(asset_id: str) -> str:
    if not isinstance(asset_id, str) or not asset_id or asset_id in (".", ".."):
        raise ValueError("asset_id must be a non-empty directory name")
    if "/" in asset_id or "\\" in asset_id or "\x00" in asset_id:
        raise ValueError("asset_id must be a single directory name")
    return asset_id


def resolve_revision(dataset: str = DATASET) -> str:
    """Resolve the current HEAD once; store the returned SHA for reproducibility."""
    HfApi, _ = _hub()
    info = HfApi().repo_info(repo_id=dataset, repo_type="dataset")
    return _check_revision(info.sha)


def list_assets(
    *,
    revision: str,
    dataset: str = DATASET,
    limit: Optional[int] = None,
) -> list[SimReadyAsset]:
    """List package manifests without downloading large USD payloads."""
    _check_revision(revision)
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    HfApi, _ = _hub()
    files = HfApi().list_repo_files(repo_id=dataset, repo_type="dataset", revision=revision)
    paths = sorted(
        p for p in files if p.endswith("/" + PACKAGE_MANIFEST)
    )
    if limit is not None:
        paths = paths[:limit]
    return [
        SimReadyAsset(
            asset_id=Path(p).parent.name,
            package_path=str(Path(p).parent),
            dataset=dataset,
            revision=revision,
        )
        for p in paths
    ]


def download_asset(
    asset: SimReadyAsset,
    *,
    destination: str | Path,
    acknowledge_license: bool = False,
) -> Path:
    """Download one package, retaining relative USD dependencies and attribution.

    The caller must inspect the package's actual license before redistribution.
    This function intentionally does not claim SimReady profile compliance.
    """
    if not acknowledge_license:
        raise PermissionError("Set acknowledge_license=True after reviewing the asset license")
    _check_revision(asset.revision)
    _check_asset_id(asset.asset_id)
    if not asset.package_path or asset.package_path.startswith("/") or ".." in Path(asset.package_path).parts:
        raise ValueError("Unsafe package path")
    if Path(asset.package_path).name != asset.asset_id:
        raise ValueError("package_path must end with asset_id")
    _, snapshot_download = _hub()
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=asset.dataset,
        repo_type="dataset",
        revision=asset.revision,
        local_dir=str(dest),
        allow_patterns=[asset.package_path + "/**"],
    )
    folder = dest / asset.package_path
    if not (folder / PACKAGE_MANIFEST).is_file():
        raise FileNotFoundError(f"SimReady package manifest missing: {folder / PACKAGE_MANIFEST}")
    provenance = folder / "pinneapple_provenance.json"
    provenance.write_text(json.dumps(asset.provenance(), indent=2) + "\n", encoding="utf-8")
    return folder


__all__ = ["SimReadyAsset", "resolve_revision", "list_assets", "download_asset"]
