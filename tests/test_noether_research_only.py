"""Noether integrations are research only: warned in normal use, refused in commercial mode."""
import warnings

import pytest

from pinneapple_neural._licencas import (
    ResearchOnlyError, ResearchOnlyWarning, commercial_mode, require_research_use,
)


def test_normal_mode_warns(monkeypatch):
    monkeypatch.delenv("PINNEAPPLE_COMMERCIAL_MODE", raising=False)
    with pytest.warns(ResearchOnlyWarning, match="RESEARCH ONLY"):
        require_research_use("x")


def test_commercial_mode_refuses(monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_COMMERCIAL_MODE", "1")
    assert commercial_mode()
    with pytest.raises(ResearchOnlyError):
        require_research_use("x")


def test_model_refused_before_importing_noether(monkeypatch):
    """The guard runs first, so commercial mode refuses even without emmiai-noether installed."""
    monkeypatch.setenv("PINNEAPPLE_COMMERCIAL_MODE", "1")
    from pinneapple_neural.architectures.neural_operators.noether_bridge import NoetherABUPT
    with pytest.raises(ResearchOnlyError):
        NoetherABUPT()


def test_dataset_and_trainer_refused(monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_COMMERCIAL_MODE", "1")
    from pinneapple_data.noether_dataset import NoetherDatasetBridge
    from pinneapple_neural.trainer.noether_trainer import NoetherSurrogateTrainer
    with pytest.raises(ResearchOnlyError):
        NoetherDatasetBridge(noether_dataset=[])
    with pytest.raises(ResearchOnlyError):
        NoetherSurrogateTrainer(model=None, train_loader=None)


def test_model_warns_in_normal_mode(monkeypatch):
    monkeypatch.delenv("PINNEAPPLE_COMMERCIAL_MODE", raising=False)
    from pinneapple_neural.architectures.neural_operators.noether_bridge import NoetherABUPT
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            NoetherABUPT()
        except Exception:
            pass  # emmiai-noether is optional and may be absent: only the warning matters here
    assert any(issubclass(x.category, ResearchOnlyWarning) for x in w)
