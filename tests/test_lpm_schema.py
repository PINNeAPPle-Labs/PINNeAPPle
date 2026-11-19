"""Fast, solver-free checks for the native LPM configuration interface."""
import json
import pytest

from pinneapple.lpm import Campaign, CampaignValidationError


def valid_campaign():
    return Campaign(
        name="cavity-baseline", domain="cfd", solver="openfoam",
        family="cavity2d", samples=24, numerical={"nx": 16, "ny": 16},
        parameters={"viscosity": [0.001, 0.01]},
    )


def test_valid_campaign():
    campaign = valid_campaign()
    campaign.validate()
    assert campaign.to_dict()["solver"] == "openfoam"


def test_rejects_wrong_family():
    campaign = valid_campaign()
    campaign.family = "cantilever3d"
    with pytest.raises(CampaignValidationError):
        campaign.validate()


def test_rejects_invalid_splits():
    campaign = valid_campaign()
    campaign.split = (0.9, 0.1, 0.1)
    with pytest.raises(CampaignValidationError):
        campaign.validate()


def test_rejects_unknown_fields(tmp_path):
    config = valid_campaign().to_dict()
    config["unexpected"] = True
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(config))
    with pytest.raises(CampaignValidationError, match="Unknown"):
        Campaign.load(path)


def test_load_json(tmp_path):
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(valid_campaign().to_dict()))
    assert Campaign.load(path).samples == 24
