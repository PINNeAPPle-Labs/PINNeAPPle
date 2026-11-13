"""CPU-only contract tests: SciREX is optional."""
import numpy as np
import pytest
from pinneapple_integrations.scirex_backend import (
    FieldSpec, ModelSpec, ScirexAdapter, available_models, validate_field,
)

def test_registry():
    assert set(available_models()) == {"fno", "gino", "transolver"}
    with pytest.raises(ValueError):
        ModelSpec("unknown", {})

def test_layout_and_units():
    field = FieldSpec("channels_first", 2, ("Pa", "m/s"), spatial_dim=2)
    assert validate_field(np.ones((3, 2, 8, 8)), field).shape == (3, 2, 8, 8)
    with pytest.raises(ValueError):
        validate_field(np.ones((3, 8, 8, 2)), field)
    with pytest.raises(ValueError):
        FieldSpec("channels_first", 2, ("Pa",))

def test_geometry_validation():
    field = FieldSpec("points_last", 1, ("Pa",), spatial_dim=3)
    with pytest.raises(ValueError):
        validate_field(np.ones((2, 7, 1)), field, coordinates=np.zeros((7, 2)))
    with pytest.raises(ValueError):
        validate_field(np.ones((2, 7, 1)), field, coordinates=np.zeros((6, 3)))

def test_normalizer_train_only():
    field = FieldSpec("channels_first", 1, ("Pa",), spatial_dim=1)
    adapter = ScirexAdapter(ModelSpec("fno", {}), field)
    train = np.arange(8, dtype=np.float32).reshape(2, 1, 4)
    adapter.fit_normalizer(train)
    assert abs(float(adapter.normalize(train).mean())) < 1e-6

def test_backend_layout_guard():
    with pytest.raises(ValueError):
        ScirexAdapter(ModelSpec("gino", {}), FieldSpec("channels_first", 1, ("Pa",)))
