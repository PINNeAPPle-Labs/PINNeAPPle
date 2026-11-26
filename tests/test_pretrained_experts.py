"""Pre-trained models wrapped as ensemble experts (#348): the registry and its licence records, the adapters, the cost
estimate, and the honest failure of an adapter whose weights are not there."""
import json
from pathlib import Path

import numpy as np
import pytest

from pinneapple_physics.ensemble import PhysicsEnsemble, from_callable
from pinneapple_physics.pretrained import (
    REGISTRY,
    Adapter,
    ArrayArchiveAdapter,
    HuggingFaceAdapter,
    Licence,
    LicenceError,
    PretrainedModel,
    PretrainedUnavailable,
    TorchCheckpointAdapter,
    UnknownPretrainedModel,
    WeatherBench2Adapter,
    from_pretrained_expert,
    known_models,
    licence_report,
    provenance_of,
    register,
    spec,
)
from pinneapple_physics.weather.data import Era5Store, channel_names

CHANNELS = channel_names()
LAT = np.linspace(87.1875, -87.1875, 32)
LON = np.arange(64) * 5.625


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Era5Store:
    """A tiny Era5Store (6 times, all channels) so the adapters have real arrays to read."""
    path = tmp_path_factory.mktemp("era5")
    rng = np.random.default_rng(0)
    state = rng.normal(size=(6, len(CHANNELS), 32, 64)).astype(np.float16)
    np.save(path / "state.npy", state)
    np.save(path / "constants.npy", np.zeros((2, 32, 64), np.float32))
    times = np.array(["2020-03-01T00", "2020-03-01T06", "2020-03-01T12", "2020-03-01T18",
                      "2020-03-02T00", "2020-03-02T06"], dtype="datetime64[ns]")
    np.save(path / "times.npy", times)
    (path / "meta.json").write_text(json.dumps({
        "channels": CHANNELS, "mean": [0.0] * len(CHANNELS), "std": [1.0] * len(CHANNELS),
        "levels": [250, 500, 700, 850], "lat": LAT.tolist(), "lon": LON.tolist(),
        "source": "synthetic", "step_hours": 6}))
    return Era5Store(path)


def archive(tmp_path, name="a.npz", times=None, channels=None) -> str:
    ch = channels or ["t850", "z500"]
    times = np.array(["2020-03-01T00", "2020-03-01T06"], dtype="datetime64[ns]") if times is None else times
    rng = np.random.default_rng(1)
    fields = rng.normal(size=(len(times), 5, len(ch), 32, 64)) + 280.0
    return ArrayArchiveAdapter.save(tmp_path / name, fields, times, ch)


def test_the_registry_records_a_licence_and_where_it_was_read():
    assert known_models("weather") and known_models("vision")
    for name, model in REGISTRY.items():
        assert model.licence.spdx, f"{name} has no licence"
        if model.licence.known:
            assert model.licence.verified, f"{name} claims {model.licence.spdx} without saying where it was read"
    assert REGISTRY["depth-anything-v2-large"].licence.commercial_use is False
    assert REGISTRY["prithvi-100m"].licence.spdx == "Apache-2.0"
    assert spec("pangu-weather").envelope["grid"] == "64x32 equiangular"


def test_spec_names_the_known_models_when_the_name_is_wrong():
    with pytest.raises(UnknownPretrainedModel) as err:
        spec("pangu-weather-2")
    assert "pangu-weather" in str(err.value)


def test_register_needs_a_licence():
    with pytest.raises(LicenceError):
        register(PretrainedModel("mystery", "weather", "some/weights", Licence("")))


def test_from_pretrained_expert_puts_the_licence_in_the_provenance(store, tmp_path):
    e = from_pretrained_expert("pinneapple-weather", ArrayArchiveAdapter(archive(tmp_path), store=store))
    assert e.name == "pinneapple-weather" and e.kind == "weather"
    prov = provenance_of(e)
    assert prov["licence"]["spdx"] == "Apache-2.0" and prov["licence"]["verified"]
    assert prov["pretrained"]["name"] == "pinneapple-weather"
    assert "estimate" in prov["cost"]["method"] or "declared" in prov["cost"]["method"]
    ens = PhysicsEnsemble([e, from_callable("persistence", lambda q: np.zeros((32, 64)))])
    assert ens.licences() == {"pinneapple-weather": "Apache-2.0"}


def test_an_unknown_model_needs_an_explicit_licence(store, tmp_path):
    ad = ArrayArchiveAdapter(archive(tmp_path), store=store)
    with pytest.raises(UnknownPretrainedModel):
        from_pretrained_expert("my-own-model", ad)
    e = from_pretrained_expert("my-own-model", ad, licence=Licence("MIT", "https://example.org/LICENSE", True, True,
                                                                  "2026-11-26: the repository's LICENSE"))
    assert provenance_of(e)["licence"]["spdx"] == "MIT"


def test_a_non_commercial_licence_is_refused_for_a_commercial_use(store, tmp_path):
    ad = HuggingFaceAdapter("depth-anything/Depth-Anything-V2-Large", name="depth", to_input=lambda q: q,
                            loader=lambda a: None)
    with pytest.raises(LicenceError):
        from_pretrained_expert("depth-anything-v2-large", ad, commercial=True)
    from_pretrained_expert("depth-anything-v2-large", ad)               # research use is allowed
    with pytest.raises(LicenceError):                                   # nobody read it at the source
        from_pretrained_expert("pangu-weather", ArrayArchiveAdapter(archive(tmp_path), store=store),
                              require_verified=True)


def test_huggingface_adapter_fails_loudly_instead_of_pretending():
    ad = HuggingFaceAdapter("nonexistent-org/nonexistent-model-000", name="ghost")
    with pytest.raises(PretrainedUnavailable) as err:
        ad.open()
    msg = str(err.value)
    assert "local_path" in msg and ("transformers" in msg or "could not load" in msg)

    def broken(self):
        raise PretrainedUnavailable("ghost: no weights at /nowhere.pt; download them or pass local_path")

    with pytest.raises(PretrainedUnavailable, match="local_path"):
        HuggingFaceAdapter("x/y", name="ghost2", loader=broken).open()


def test_an_expert_that_cannot_load_is_a_failure_and_not_a_prediction(store, tmp_path):
    ok = from_callable("persistence", lambda q: np.full((32, 64), 275.0))
    ghost = from_pretrained_expert("sam2.1", HuggingFaceAdapter("facebook/sam2.1-hiera-large", name="ghost",
                                                               loader=lambda a: _raise()), lazy=True)
    ens = PhysicsEnsemble([ok, ghost])
    q = {"i0": 1, "steps": 2, "variable": "t850"}
    out = ens.predict(q)
    assert ens.failures["ghost"] == 1 and "ghost" not in out["expert_predictions"]
    assert np.all(out["prediction"] == 275.0)                          # the other expert carried the case


def _raise():
    raise PretrainedUnavailable("ghost: the weights are not on this machine")


def test_the_array_archive_adapter_reads_by_initialisation_time(store, tmp_path):
    path = archive(tmp_path)
    ad = ArrayArchiveAdapter(path, store=store, name="archive").open()
    y = ad.predict({"i0": 0, "steps": 2, "variable": "t850"})
    assert y.shape == (3, 32, 64) and np.all(np.isfinite(y))
    with pytest.raises(PretrainedUnavailable, match="no channel"):
        ad.predict({"i0": 0, "steps": 2, "variable": "u10"})
    with pytest.raises(ValueError, match="variable"):
        ad.predict({"i0": 0, "steps": 2})
    with pytest.raises(PretrainedUnavailable, match="leads"):
        ad.predict({"i0": 0, "steps": 9, "variable": "t850"})
    with pytest.raises(PretrainedUnavailable, match="no initialisation"):
        ArrayArchiveAdapter(path, store=store).open().predict({"i0": 5, "steps": 1, "variable": "t850"})


def test_the_weatherbench2_adapter_uses_the_loader_of_the_skill_curves(store, monkeypatch):
    """Injected source: the adapter maps a query to the published array exactly as ``PublishedForecasts`` returns it.

    The published archives carry a handful of channels, not the whole state: what they do not carry arrives as NaN,
    which is what the adapter refuses by name instead of returning a field of missing values.
    """
    fields = np.full((3, len(CHANNELS), 32, 64), np.nan, np.float32)
    fields[:, CHANNELS.index("t850")] = 280.0
    ad = WeatherBench2Adapter(store, "IFS HRES", forecast=lambda i0, steps: fields[:steps + 1])
    y = ad.predict({"i0": 0, "steps": 2, "variable": "t850"})
    assert y.shape == (3, 32, 64) and np.all(y == 280.0)
    with pytest.raises(PretrainedUnavailable, match="no values"):
        ad.predict({"i0": 0, "steps": 2, "variable": "u10"})

    class Absent:
        ds: dict = {}
        errors = {"Pangu-Weather": "ModuleNotFoundError: No module named 'gcsfs'"}

    monkeypatch.setattr("pinneapple_physics.weather.evaluate.PublishedForecasts", lambda *a, **k: Absent())
    with pytest.raises(PretrainedUnavailable) as err:
        WeatherBench2Adapter(store, "Pangu-Weather").open()
    assert "gcsfs" in str(err.value) and "local copy" in str(err.value)


def test_the_cost_estimate_follows_the_query_and_the_ensemble_uses_it(store):
    ad = WeatherBench2Adapter(store, "IFS HRES", forecast=None, latency_s=1.5, throughput_mb_s=0.8)
    ad._forecast, ad._opened = (lambda i0, steps: np.full((steps + 1, len(CHANNELS), 32, 64), np.nan)), True
    short, long = ad.cost({"steps": 4}), ad.cost({"steps": 40})
    assert long > short >= 1.5 and "measured" in ad.cost_method
    e = from_pretrained_expert("ifs-hres", ad)
    assert callable(e.cost) and e.cost_of({"steps": 40}) == pytest.approx(long)

    ens = PhysicsEnsemble([e, from_callable("cheap", lambda q: np.full((5, 32, 64), 0.0), cost=0.01)])
    run = ens.run([{"i0": 0, "steps": 4, "variable": "z500"}], [np.zeros((5, 32, 64))])
    assert run.costs[0, 0] == pytest.approx(short) and run.costs[0, 1] == pytest.approx(0.01)
    # the summary is the cost of the mixture: the weights sum to one, so a case that runs two experts costs the
    # weighted mean of the two, not their sum
    assert run.weights[0].sum() == pytest.approx(1.0)
    assert run.summary()["mean_cost_per_case"] == pytest.approx(float((run.weights * run.costs).sum(axis=1).mean()))


def test_the_torch_adapter_rolls_out_its_checkpoint_and_measures_its_step(store, tmp_path):
    torch = pytest.importorskip("torch")
    from pinneapple_physics.weather.model import SphereUNet
    from pinneapple_physics.weather.train import Forecaster

    torch.manual_seed(0)
    fc = Forecaster(store, SphereUNet(len(CHANNELS), 7, (8, 16), 1, 1))       # zero head: persistence
    ck = tmp_path / "ck.pt"
    torch.save({"state_dict": fc.model.state_dict(), "config": {"widths": [8, 16], "blocks": 1, "patch": 1,
                                                                "channels": CHANNELS, "mean": [0.0] * len(CHANNELS),
                                                                "std": [1.0] * len(CHANNELS)}}, ck)
    ad = TorchCheckpointAdapter(store, ck, name="pinneapple-weather").open()
    y = ad.predict({"i0": 1, "steps": 2, "variable": "t850"})
    x = store.denorm(np.asarray(store.state[1], np.float32))[CHANNELS.index("t850")]
    assert y.shape == (3, 32, 64) and np.allclose(y[0], x)                # the first lead is the initial state
    assert "measured" in ad.cost_method and ad.cost({"steps": 2}) > 0
    with pytest.raises(PretrainedUnavailable, match="checkpoint"):
        TorchCheckpointAdapter(store, tmp_path / "missing.pt").open()
    with pytest.raises(PretrainedUnavailable, match="i0"):
        ad.predict({"i0": 0, "steps": 1, "variable": "t850"})


def test_licence_report_names_what_a_deployment_must_read(store, tmp_path):
    a = from_pretrained_expert("pangu-weather", ArrayArchiveAdapter(archive(tmp_path), store=store))
    b = from_pretrained_expert("depth-anything-v2-large", HuggingFaceAdapter("depth-anything/Depth-Anything-V2-Large",
                                                                            name="depth", loader=lambda x: None))
    rep = licence_report([a, b])
    assert rep["unverified"] == ["pangu-weather"] and rep["non_commercial"] == ["depth"]
    assert set(rep["licences"]) == {"pangu-weather", "depth"}
    assert licence_report(from_callable("plain", lambda q: q))["licences"] == {}


def test_an_adapter_without_an_open_keeps_the_default_cost():
    class Null(Adapter):
        def predict(self, q):
            return np.zeros((2, 2))
    e = from_pretrained_expert("custom", Null(name="custom"), licence=Licence("MIT", verified="2026-11-26: test"))
    assert e.cost_of({"steps": 1}) == 1.0 and e.provenance["licence"]["spdx"] == "MIT"
