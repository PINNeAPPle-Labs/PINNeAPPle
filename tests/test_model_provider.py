"""One model provider interface (#1477).

The acceptance lines under test: a saved artifact loads through the same call whichever framework produced it; a model
that cannot be exported to ONNX is refused with the operation named rather than written into something that computes
elsewhere; and the same model loaded from a hub layout and reloaded from a local export predicts identically, with the
provenance differing only in what names the provider.

JAX and ``transformers`` are not installed in this environment, so the JAX provider is checked for its *refusal*
(a missing package is named, not raised as a ModuleNotFoundError) and its loading path is not exercised here. That is
stated in the issue, not worked around.
"""
from __future__ import annotations

import json
import sys
import textwrap

import numpy as np
import pytest
import torch

from pinneapple_neural.model_provider import (
    ARTIFACT_SCHEMA,
    ExportNotSupported,
    ModelNotLoadable,
    ProviderUnavailable,
    available_providers,
    detect,
    get_provider,
    load,
    providers,
    register_provider,
)


class Branchy(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:                                  # the trace follows one branch and the file lies about it
            return x * 2
        return x * 3


class Small(torch.nn.Module):
    def __init__(self, n_in: int = 3, n_out: int = 2):
        super().__init__()
        self.linear = torch.nn.Linear(n_in, n_out)

    def forward(self, x):
        return torch.tanh(self.linear(x))


@pytest.fixture
def checkpoint(tmp_path):
    """A checkpoint in the format ``save_checkpoint`` writes (#941): class name plus constructor arguments."""
    torch.manual_seed(0)
    model = Small()
    torch.save({"state_dict": model.state_dict(), "class_name": f"{__name__}.Small", "metadata": {},
                "registry_name": None, "model_kwargs": {"n_in": 3, "n_out": 2}}, tmp_path / "small.pt")
    return tmp_path / "small.pt", model


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(4, 3).numpy()


# ------------------------------------------------------------------------------------------------ the registry
def test_the_six_providers_are_registered_and_say_whether_they_can_run():
    assert {"pytorch", "torchscript", "onnx", "jax", "physicsnemo", "huggingface"} <= set(providers())
    described = available_providers()
    assert described["pytorch"]["available"] and "onnx" in described["pytorch"]["targets"]
    if not described["jax"]["available"]:
        assert "jax" in described["jax"]["detail"] and "pip install" in described["jax"]["detail"]
    assert described["onnx"]["targets"] == [], "ONNX is the end of the road: it is not exported again from here"


def test_an_unknown_provider_is_refused_by_name():
    with pytest.raises(ModelNotLoadable, match="unknown provider"):
        get_provider("tensorflow")


def test_detect_names_what_it_tried_on_an_unknown_file(tmp_path):
    p = tmp_path / "mystery.weights"
    p.write_text("not a model")
    with pytest.raises(ModelNotLoadable) as err:
        detect(p)
    assert "pytorch" in str(err.value) and ".onnx" in str(err.value)


# ------------------------------------------------------------------------------------------------ loading
def test_a_torch_checkpoint_loads_and_predicts_what_the_model_predicts(checkpoint, x):
    path, model = checkpoint
    loaded = load(path)
    assert loaded.provenance.provider == "pytorch"
    assert loaded.provenance.sha256 and loaded.provenance.provider_version
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    assert np.allclose(loaded.predict(x), expected, atol=1e-6)


def test_the_same_model_reaches_the_library_from_a_different_door(checkpoint, x, tmp_path):
    """A checkpoint and its TorchScript export load through the same call and predict the same numbers."""
    path, _ = checkpoint
    torch_loaded = load(path)
    ts = torch_loaded.export("torchscript", tmp_path / "small_ts.pt", example=torch.from_numpy(x))
    again = load(ts)
    assert again.provenance.provider == "torchscript"
    assert np.allclose(again.predict(x), torch_loaded.predict(x), atol=1e-6)


def test_a_hub_layout_and_its_local_export_predict_identically(checkpoint, x, tmp_path):
    """The acceptance: same model, two doors; the provenance differs only in what names the provider."""
    from pinneapple_neural.architectures.registry import ModelRegistry

    path, model = checkpoint
    ModelRegistry.register(name="provider_small", family="test", description="fixture")(Small)
    mirror = tmp_path / "hub"
    mirror.mkdir()
    (mirror / "pinneapple_config.json").write_text(json.dumps(
        {"architecture": "provider_small", "architecture_config": {"n_in": 3, "n_out": 2}}), encoding="utf-8")
    torch.save(model.state_dict(), mirror / "pytorch_model.bin")
    try:
        from_hub = load(mirror)
        assert from_hub.provenance.provider == "huggingface"
        artifact = from_hub.export("artifact", tmp_path / "exported", example=torch.from_numpy(x))
        again = load(artifact)
        assert again.provenance.provider == "torchscript"
        assert np.allclose(from_hub.predict(x), again.predict(x), atol=1e-6)
        a, b = from_hub.provenance.to_dict(), again.provenance.to_dict()
        differing = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert differing <= {"provider", "provider_version", "ref", "sha256", "origin", "architecture", "artifact",
                             "exported_from", "framework"}, f"only provenance, never the model: {differing}"
        assert a["provider"] != b["provider"]
        manifest = json.loads((artifact / "pinneapple_model.json").read_text())
        assert manifest["schema"] == ARTIFACT_SCHEMA and manifest["provider"] == "torchscript"
        assert manifest["source_provenance"]["provider"] == "huggingface", "the artifact remembers what it came from"
    finally:
        ModelRegistry._REGISTRY.pop("provider_small", None)


def test_a_missing_backend_is_a_named_refusal_and_not_a_module_not_found(tmp_path):
    npz = tmp_path / "params.npz"
    np.savez(npz, w=np.zeros((2, 2)))
    provider = get_provider("jax")
    if provider.available()[0]:                          # pragma: no cover - jax is not installed here
        pytest.skip("jax is installed: this test checks the refusal")
    with pytest.raises(ProviderUnavailable) as err:
        load(npz)
    assert "jax is not installed" in str(err.value)


# ------------------------------------------------------------------------------------------------ export
def test_onnx_export_of_a_representable_model_round_trips(checkpoint, x, tmp_path):
    path, _ = checkpoint
    loaded = load(path)
    out = loaded.export("onnx", tmp_path / "small.onnx", example=torch.from_numpy(x))
    again = load(out)
    assert again.provenance.provider == "onnx"
    assert np.allclose(again.predict(x), loaded.predict(x), atol=1e-4)


def test_onnx_export_of_a_model_with_data_dependent_control_flow_is_refused_naming_the_conversion(x, tmp_path):
    torch.save({"state_dict": Branchy().state_dict(), "class_name": f"{__name__}.Branchy", "metadata": {},
                "registry_name": None, "model_kwargs": {}}, tmp_path / "branchy.pt")
    loaded = load(tmp_path / "branchy.pt")
    with pytest.raises(ExportNotSupported) as err:
        loaded.export("onnx", tmp_path / "branchy.onnx", example=torch.ones(2, 3))
    assert "Converting a tensor to a Python boolean" in str(err.value)
    assert err.value.operation and "boolean" in err.value.operation
    assert not (tmp_path / "branchy.onnx").exists(), "a refused export leaves no file that loads and computes"


def test_onnx_export_without_an_example_is_refused_rather_than_guessed(tmp_path):
    torch.save({"state_dict": Small().state_dict(), "class_name": f"{__name__}.Small", "metadata": {},
                "registry_name": None, "model_kwargs": {"n_in": 3, "n_out": 2}}, tmp_path / "small.pt")
    loaded = load(tmp_path / "small.pt")
    with pytest.raises(ExportNotSupported, match="needs an example input"):
        loaded.export("onnx", tmp_path / "small.onnx")


def test_an_onnx_graph_refuses_to_be_exported_further(checkpoint, x, tmp_path):
    path, _ = checkpoint
    onnx_path = load(path).export("onnx", tmp_path / "small.onnx", example=torch.from_numpy(x))
    loaded = load(onnx_path)
    with pytest.raises(ExportNotSupported) as err:
        loaded.export("torchscript", tmp_path / "small.ts")
    assert err.value.target == "torchscript" and "ONNX Runtime runs the graph" in str(err.value)


def test_artifact_export_of_a_torchscript_model_copies_the_archive_rather_than_re_exporting(checkpoint, x, tmp_path):
    path, _ = checkpoint
    ts = load(path).export("torchscript", tmp_path / "s.pt", example=torch.from_numpy(x))
    loaded = load(ts)
    artifact = loaded.export("artifact", tmp_path / "art")
    manifest = json.loads((artifact / "pinneapple_model.json").read_text())
    assert manifest["source_provenance"]["provider"] == "torchscript"
    assert np.allclose(load(artifact).predict(x), loaded.predict(x), atol=1e-6)


# ------------------------------------------------------------------------------------------------ evaluate
def test_evaluate_reuses_the_validation_module(checkpoint, x):
    path, model = checkpoint
    loaded = load(path)
    with torch.no_grad():
        y = model(torch.from_numpy(x)).numpy()
    report = loaded.evaluate({"x": x, "y": y}, inputs=("x",), outputs=("y",))
    assert type(report).__name__ == "ValidationReport", "the evaluation path is pinneapple_analysis.validation"
    checks = {c.id: c for c in report.checks}
    assert "reference.rel_l2" in checks
    assert checks["reference.rel_l2"].value == pytest.approx(0.0, abs=1e-6), "the same model against itself"


def test_a_third_party_provider_can_be_registered(tmp_path):
    from pinneapple_neural.model_provider import ModelProvider, Provenance

    class Constant(ModelProvider):
        name = "constant-test"
        package = ""

        def can_load(self, ref):
            return str(ref).endswith(".const")

        def load(self, ref, **options):
            return {"v": 42.0}, Provenance(provider=self.name, ref=str(ref))

        def predict(self, model, x):
            return np.full((len(np.asarray(x)), 1), model["v"], dtype=np.float32)

    register_provider(Constant())
    try:
        p = tmp_path / "c.const"
        p.write_text("42")
        loaded = load(p, provider="constant-test")
        assert loaded.predict(np.zeros((3, 2))).shape == (3, 1)
        assert "constant-test" in available_providers()
    finally:
        from pinneapple_neural.model_provider import PROVIDERS

        PROVIDERS.pop("constant-test", None)
        sys.modules.pop("constant", None)


_ = textwrap  # imported for the module-writing helper the fixtures above deliberately do not need
