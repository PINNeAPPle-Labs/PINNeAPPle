"""End-to-end tests for :class:`pinneapple_worldmodel.pipeline.PhysicsAIPipeline`.

The pipeline is the package's top-level entry point (``pp.run_pipeline`` /
``python -m pinneapple_worldmodel.train --mode pipeline`` both end up here) and
until now nothing ran it.  These tests run the real thing — no stage is stubbed
— on a configuration small enough to stay in seconds:

* two *custom* :class:`~pinneapple_worldmodel.scenario.PhysicsScenario` objects
  on a 16x16 grid with 3 time steps instead of the built-in 64x64/256 grids.
  The built-in scenarios are the documented default, but their grid is baked
  into the scenario and the specialist model width/depth is baked into
  ``PhysicsAIConfig`` (it exposes no ``WorldModelConfig``), so building the
  default heat_2d case costs ~25 s of CPU for a single epoch — the tiny custom
  scenarios exercise exactly the same stages for ~1 s each.
* two sample trajectories per scenario, one specialist epoch, one foundation
  epoch, ``device="cpu"``.
* ``skip_meta=True``.  The meta stage is deliberately not covered: the built-in
  Reptile loop (``MetaLearner._reptile_loop``) needs catalog entries with at
  least ``n_support + n_query`` = 64 samples each to even build its
  ``TaskDistribution``, and then deep-copies the (fixed, 4.2 M-parameter)
  specialist model 4x per meta-epoch for 5 inner steps each.  Neither the
  sample requirement nor ``n_inner_steps``/``n_tasks_per_batch`` is reachable
  from ``PhysicsAIConfig``, so covering it honestly would cost tens of seconds
  to assert what a direct ``MetaLearner`` test should assert instead.
* ``run_benchmark=False`` for the full run: ``PhysicsBenchmark`` only knows the
  six full-size ``BUILTIN_TASKS`` (64x64 / 256 grids, 10-20 reference
  trajectories each) and ``PhysicsAIConfig`` cannot narrow them.  The stage's
  own guard (no model → ``None``) is covered instead, and the benchmark suite
  itself is a separate unit.
* the legacy curriculum path (``use_curriculum=True``) *is* covered, with a
  one-stage, one-epoch curriculum.

Assertions are about the contract the docstrings promise: which attributes
``PhysicsAIPipelineResult`` carries, what each stage produces, and that the
run's artefacts can be read back.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

import pytest
import torch

from pinneapple_worldmodel import (
    CurriculumConfig,
    CurriculumStage,
    DatasetCatalog,
    FoundationConfig,
    ModelZoo,
    PhysicsAIConfig,
    PhysicsAIPipeline,
    PhysicsAIPipelineResult,
    PhysicsFoundationModel,
    PhysicsScenario,
    PhysicsWorldModel,
)

torch.set_num_threads(1)


class _Run(NamedTuple):
    pipeline: PhysicsAIPipeline
    result: PhysicsAIPipelineResult
    save_dir: object  # pathlib.Path
    scenarios: list


def _tiny_scenarios() -> list:
    """Two 16x16 scenarios so "one specialist per scenario" is a real check."""
    return [
        PhysicsScenario(
            name="tiny_heat",
            pde_kind="heat",
            grid_shape=(16, 16),
            n_steps=3,
            solver="builtin",
        ),
        PhysicsScenario(
            name="tiny_advection",
            pde_kind="advection",
            grid_shape=(16, 16),
            n_steps=3,
            solver="builtin",
        ),
    ]


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> _Run:
    """Run the full 5-stage pipeline once and share the result across tests."""
    save_dir = tmp_path_factory.mktemp("physics_ai_output")
    scenarios = _tiny_scenarios()
    config = PhysicsAIConfig(
        scenarios=scenarios,
        sources=["solver"],
        n_samples=2,
        device="cpu",
        save_dir=str(save_dir),
        specialist_epochs=1,
        specialist_batch_size=8,
        skip_meta=True,
        mega_epochs=1,
        run_benchmark=False,
        verbose=False,
        foundation=FoundationConfig(
            n_modes=4,
            width=8,
            depth=1,
            n_heads=2,
            n_context_layers=1,
            lora_rank=2,
            dropout=0.0,
            device="cpu",
        ),
    )
    pipeline = PhysicsAIPipeline(config)
    return _Run(pipeline, pipeline.run(), save_dir, scenarios)


# ---------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------

def test_result_is_the_documented_dataclass(run):
    """``PhysicsAIPipelineResult`` documents exactly these six attributes."""
    result = run.result

    assert isinstance(result, PhysicsAIPipelineResult)
    assert [f.name for f in dataclasses.fields(PhysicsAIPipelineResult)] == [
        "catalog", "zoo", "meta_model", "mega_model", "benchmark", "elapsed_s",
    ]
    assert isinstance(result.elapsed_s, float)
    assert result.elapsed_s > 0.0


def test_every_stage_populated_its_documented_output(run):
    """Stage 1 → catalog, stage 2 → zoo, stage 3 → meta model, stage 4 → mega
    model, stage 5 → benchmark results.  With ``skip_meta`` and no benchmark
    run, those two stay ``None`` — that is the documented skip behaviour, not
    a silently empty stage."""
    result = run.result

    assert isinstance(result.catalog, DatasetCatalog)
    assert isinstance(result.zoo, ModelZoo)
    assert isinstance(result.mega_model, PhysicsFoundationModel)
    assert result.meta_model is None      # skip_meta=True
    assert result.benchmark is None       # run_benchmark=False

    # run() also leaves the same objects reachable through the pipeline's own
    # (documented) properties.
    assert run.pipeline.catalog is result.catalog
    assert run.pipeline.zoo is result.zoo
    assert run.pipeline.mega_model is result.mega_model


# ---------------------------------------------------------------------------
# Stage 1: multi-source catalog
# ---------------------------------------------------------------------------

def test_catalog_has_one_entry_per_scenario_and_source(run):
    catalog = run.result.catalog

    names = [entry.name for entry in catalog.entries]
    assert names == [f"{sc.name}::solver" for sc in run.scenarios]
    assert {entry.scenario for entry in catalog.entries} == {sc.name for sc in run.scenarios}
    assert {entry.source for entry in catalog.entries} == {"solver"}

    for entry in catalog.entries:
        assert entry.n_samples == len(entry.dataset) > 0
        assert entry.generation_time_s >= 0.0
        assert entry.physics_tags                       # e.g. diffusion/transport/2d
        # The dataset the specialists train on is a real one-step dataset.
        sample = entry.dataset[0]
        assert sample["state_t"].shape == sample["state_tp1"].shape
        assert sample["state_t"].ndim == 3              # (C, H, W)

    # len(catalog) is the total sample count, and merged() concatenates them.
    assert len(catalog) == sum(len(entry.dataset) for entry in catalog.entries)
    assert len(catalog.merged()) == len(catalog)


# ---------------------------------------------------------------------------
# Stage 2: specialist zoo
# ---------------------------------------------------------------------------

def test_zoo_holds_one_trained_specialist_per_scenario(run):
    zoo = run.result.zoo

    assert len(zoo) == len(run.scenarios)
    assert sorted(zoo.list_names()) == sorted(sc.name for sc in run.scenarios)

    for entry in zoo:
        assert entry.scenario == entry.name
        assert isinstance(entry.model, PhysicsWorldModel)
        assert entry.n_params == entry.model.parameter_count() > 0
        # Auto-sized from the dataset: 1 channel on a 16x16 grid.
        assert entry.model.n_fields == 1
        assert entry.model.grid_shape == (16, 16)
        # Trained for one epoch, so the history-derived metrics must be there.
        assert "val_loss" in entry.metrics
        assert entry.metadata["n_samples"] > 0


# ---------------------------------------------------------------------------
# Stage 4: the mega model
# ---------------------------------------------------------------------------

def test_mega_model_matches_the_reference_dataset_and_predicts(run):
    mega = run.result.mega_model
    ref = run.result.catalog.entries[0].dataset

    assert mega.n_fields == ref.n_fields
    assert mega.grid_shape == ref.grid_shape
    assert mega.spatial_dim == 2
    assert mega.parameter_count() > 0

    # It predicts the next state on real data from the catalog it was built on.
    merged = run.result.catalog.merged()
    batch = torch.stack([merged[0]["state_t"], merged[1]["state_t"]])
    context = torch.stack([merged[0]["context"], merged[1]["context"]])
    mega.eval()
    prediction = mega(batch, context)
    assert prediction.shape == batch.shape
    assert torch.isfinite(prediction).all()


def test_summary_reports_every_populated_stage(run, capsys):
    run.result.summary()
    out = capsys.readouterr().out

    assert "PHYSICS AI PIPELINE COMPLETE" in out
    assert f"{len(run.result.catalog):,} total samples" in out
    assert f"{len(run.result.catalog.entries)} entries" in out
    assert f"{len(run.result.zoo)} specialists" in out
    assert f"{run.result.mega_model.parameter_count():,} params" in out
    # Wall time is printed for every run; the benchmark line only when there is one.
    assert "Wall time" in out
    assert "Benchmark" not in out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_run_artifacts_are_written_and_the_mega_model_reloads(run):
    """Stage 1 writes a dataset per (source, scenario), stage 2 writes the zoo,
    stage 4 writes ``mega_model.pt``.  The documented reader for the last one
    must get the same model back."""
    save_dir = run.save_dir

    for scenario in run.scenarios:
        assert (save_dir / "datasets" / "solver" / scenario.name / "dataset_info.json").is_file()
    assert (save_dir / "specialists" / "index.json").is_file()

    checkpoint = save_dir / "mega_model.pt"
    assert checkpoint.is_file()
    reloaded = PhysicsFoundationModel.load(str(checkpoint))
    assert reloaded.grid_shape == run.result.mega_model.grid_shape
    assert reloaded.n_fields == run.result.mega_model.n_fields
    for key, value in run.result.mega_model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], value)


# ---------------------------------------------------------------------------
# Stage 5: benchmark guard
# ---------------------------------------------------------------------------

def test_benchmark_stage_returns_none_when_no_model_is_available():
    """The benchmark stage falls back from the mega model to a zoo specialist,
    and warns + returns ``None`` when neither exists.  Running the real
    ``PhysicsBenchmark`` suite here is out of scope (see the module docstring)."""
    pipeline = PhysicsAIPipeline(PhysicsAIConfig(verbose=False, run_benchmark=False))

    assert pipeline._result.mega_model is None
    assert pipeline._result.zoo is None
    assert pipeline._run_benchmark() is None


# ---------------------------------------------------------------------------
# Legacy curriculum path
# ---------------------------------------------------------------------------

def test_curriculum_mode_surfaces_the_trained_world_model():
    """``use_curriculum=True`` runs the legacy 5-stage curriculum instead of the
    5 stages above, so catalog/zoo/mega_model stay ``None`` — but the model the
    curriculum trains is a ``PhysicsWorldModel`` and the run must not throw it
    away (the result's ``meta_model`` slot is the only ``PhysicsWorldModel``
    field, and it is what ``summary()`` reports as the trained model)."""
    scenario = PhysicsScenario(
        name="tiny_curriculum_heat",
        pde_kind="heat",
        grid_shape=(16, 16),
        n_steps=2,
        solver="builtin",
    )
    curriculum = CurriculumConfig(
        stages=[CurriculumStage(scenarios=[scenario], n_samples=2, epochs=1)],
        device="cpu",
        batch_size=8,
        verbose=False,
    )
    config = PhysicsAIConfig(
        use_curriculum=True,
        curriculum=curriculum,
        device="cpu",
        run_benchmark=False,
        verbose=False,
    )

    result = PhysicsAIPipeline(config).run()

    assert isinstance(result, PhysicsAIPipelineResult)
    assert isinstance(result.meta_model, PhysicsWorldModel)
    assert result.meta_model.grid_shape == (16, 16)
    assert result.meta_model.n_fields == 1
    assert result.catalog is None
    assert result.zoo is None
    assert result.mega_model is None
    assert result.benchmark is None
    assert result.elapsed_s > 0.0
