"""Offline regression tests for source-code dataset extraction and integrity."""
import json

import pytest

from pinneapple_llm.code_training import (
    CodeDatasetConfig, finetune_code_dataset, generate_code_dataset
)


def _fixture(root):
    package = root / "pinneapple_example"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "api.py").write_text('''def solve(x: int, scale=2):
    """Return a scaled input value as an integer, for simple examples."""
    return x * scale


def _private():
    """A private helper that should not appear in the output at all."""
    return 0
''')


def test_code_dataset_deterministic_and_excludes_private(tmp_path):
    _fixture(tmp_path)
    cfg = CodeDatasetConfig(
        root=str(tmp_path), output_dir=str(tmp_path / "out"),
        packages=("pinneapple_example",), validation_fraction=0.0
    )
    first = generate_code_dataset(cfg)
    records = [json.loads(line) for line in (tmp_path / "out/train.jsonl").read_text().splitlines()]
    assert first["counts"] == {"train": 1, "validation": 0}
    assert records[0]["symbol"] == "pinneapple_example.api.solve"
    assert "solve(x, scale=2)" in records[0]["response"]
    assert "return x * scale" in records[0]["response"]
    assert first["sha256"] == generate_code_dataset(cfg)["sha256"]


def test_code_dataset_rejects_secrets(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "pinneapple_example/api.py").write_text('''SECRET = "sk-abcdefghijklmnopqrstuvwxyz1234"
def solve():
    """Documentation long enough to be extracted safely."""
    return 1
''')
    with pytest.raises(ValueError, match="No documented"):
        generate_code_dataset(CodeDatasetConfig(
            root=str(tmp_path), output_dir=str(tmp_path / "out"),
            packages=("pinneapple_example",)
        ))


def test_code_dataset_rejects_unsafe_package(tmp_path):
    with pytest.raises(ValueError, match="Unsafe package"):
        generate_code_dataset(CodeDatasetConfig(
            root=str(tmp_path), output_dir=str(tmp_path / "out"),
            packages=("../private",)
        ))


def test_finetune_verifies_checksums_before_training(tmp_path, monkeypatch):
    _fixture(tmp_path)
    cfg = CodeDatasetConfig(
        root=str(tmp_path), output_dir=str(tmp_path / "out"),
        packages=("pinneapple_example",), validation_fraction=0.0
    )
    generate_code_dataset(cfg)
    import pinneapple_llm.finetune as backend

    calls = []
    def fake_train(path, config):
        calls.append((path, config.base_model, config.output_dir))
        return config.output_dir

    monkeypatch.setattr(backend, "finetune_lora", fake_train)
    out = finetune_code_dataset(str(tmp_path / "out"), base_model="test/local", output_dir="adapter")
    assert out == "adapter"
    assert len(calls) == 1
    with (tmp_path / "out/train.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        finetune_code_dataset(str(tmp_path / "out"), base_model="test/local", output_dir="adapter")
