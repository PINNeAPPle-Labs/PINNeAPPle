"""Offline tests for the SimReady Hugging Face integration."""
from unittest.mock import MagicMock, patch
import json
import pytest

from pinneapple_twin3d.simready import (
    SimReadyAsset, _check_revision, download_asset, list_assets,
    PACKAGE_MANIFEST,
)

SHA = "a" * 40


def test_revision_requires_immutable_sha():
    assert _check_revision(SHA) == SHA
    for invalid in ("main", "latest", "../foo", "f" * 39):
        with pytest.raises(ValueError):
            _check_revision(invalid)


def test_listing_only_package_manifests():
    api = MagicMock()
    api.list_repo_files.return_value = [
        "machine_a/" + PACKAGE_MANIFEST,
        "machine_a/simready_usd/model.usd",
        "machine_b/" + PACKAGE_MANIFEST,
        "README.md",
    ]
    with patch("pinneapple_twin3d.simready._hub", return_value=(lambda: api, None)):
        assets = list_assets(revision=SHA)
    assert [x.asset_id for x in assets] == ["machine_a", "machine_b"]
    assert all(x.revision == SHA for x in assets)


def test_download_requires_license_acknowledgement(tmp_path):
    asset = SimReadyAsset("machine_a", "machine_a", "nvidia/simready-assets", SHA)
    with pytest.raises(PermissionError):
        download_asset(asset, destination=tmp_path)
    assert not list(tmp_path.iterdir())


def test_download_retains_package_and_writes_provenance(tmp_path):
    asset = SimReadyAsset("machine_a", "machine_a", "nvidia/simready-assets", SHA)
    def fake_snapshot_download(**kwargs):
        assert kwargs["allow_patterns"] == ["machine_a/**"]
        assert kwargs["revision"] == SHA
        folder = tmp_path / "machine_a"
        folder.mkdir()
        (folder / PACKAGE_MANIFEST).write_text("{}")
    with patch("pinneapple_twin3d.simready._hub", return_value=(None, fake_snapshot_download)):
        folder = download_asset(asset, destination=tmp_path, acknowledge_license=True)
    provenance = json.loads((folder / "pinneapple_provenance.json").read_text())
    assert provenance["revision"] == SHA
    assert provenance["license"] == "CC-BY-4.0"


def test_download_rejects_path_traversal(tmp_path):
    asset = SimReadyAsset("..", "../secret", "nvidia/simready-assets", SHA)
    with pytest.raises(ValueError):
        download_asset(asset, destination=tmp_path, acknowledge_license=True)
