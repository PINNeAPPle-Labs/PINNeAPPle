"""Pre-trained models wrapped as ensemble experts (#348): the registry and its licence records, the adapters, the cost
estimate, and the honest failure of an adapter whose weights are not there."""
import json
from pathlib import Path

import numpy as np
import pytest

from pinneapple_physics.ensemble import Expert, PhysicsEnsemble, from_callable
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


def test_the_recipe_of_the_guide_holds_for_every_model_in_the_registry(store, tmp_path):
    """What the guide tells a reader to do with a new model ("Adding a new pre-trained model": a registry entry with the
    licence, an adapter, a domain check, a measurement), executed over every entry that is already there: each one is
    wrapped by an adapter this repository ships, and the provenance carries the licence, the source and the cost
    method that the guide promises. A new registry entry that no adapter can wrap breaks this."""
    ad = ArrayArchiveAdapter(archive(tmp_path), store=store)
    assert len(known_models()) >= 8
    for name in known_models():
        e = from_pretrained_expert(name, ad)                      # step 2 of the guide: the adapter
        prov = provenance_of(e)                                   # step 1: the licence travels with the prediction
        assert prov["licence"]["spdx"] == REGISTRY[name].licence.spdx
        assert prov["pretrained"]["source"] == REGISTRY[name].source
        assert prov["pretrained"]["envelope"] == dict(REGISTRY[name].envelope)
        assert prov["cost"]["method"]
        assert e.kind == REGISTRY[name].family
    own = PretrainedModel("my-own-model", "weather", "gs://my/archive",
                          Licence("MIT", "https://example.org/LICENSE", True, True, "2026-11-27: example.org"))
    register(own)                                                 # step 1 for a model nobody listed yet
    try:
        e = from_pretrained_expert("my-own-model", ad)
        assert provenance_of(e)["licence"]["spdx"] == "MIT"
    finally:
        REGISTRY.pop("my-own-model", None)


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


def test_the_published_archive_has_no_lead_zero_and_the_adapter_fills_it_from_the_analysis(store):
    """``PublishedForecasts`` writes NaN at lead 0 (the analysis is not a forecast), so before the fix every prediction
    of the weather adapter was refused as "no values" and no published model could be an expert at all. The analysis the
    store holds is exactly what the archive's model was initialised with, so that is what lead 0 gets -- and only when
    the source left it empty."""
    nan_lead0 = np.full((3, len(CHANNELS), 32, 64), np.nan, np.float32)
    nan_lead0[1:, CHANNELS.index("z500")] = 5000.0                      # the archive: leads 6 h and 12 h only
    ad = WeatherBench2Adapter(store, "IFS HRES", forecast=lambda i0, steps: nan_lead0[:steps + 1])
    y = ad.predict({"i0": 2, "steps": 2, "variable": "z500"})
    analysis = store.denorm(np.asarray(store.state[2, CHANNELS.index("z500")], np.float32), "z500")
    assert y.shape == (3, 32, 64) and np.allclose(y[0], analysis) and np.all(y[1:] == 5000.0)

    given = np.full((3, len(CHANNELS), 32, 64), np.nan, np.float32)
    given[:, CHANNELS.index("t850")] = 280.0                            # a source that does provide lead 0
    y = WeatherBench2Adapter(store, "IFS HRES", forecast=lambda i0, steps: given[:steps + 1]).predict(
        {"i0": 2, "steps": 2, "variable": "t850"})
    assert np.all(y == 280.0)


def test_one_read_serves_several_variables_and_the_map_says_which(store):
    """A physics check asks one model for the tracer and the wind that should carry it: ``variable`` as a list comes
    back stacked, and ``variables=`` narrows what the archive is read for (and what its cost estimate prices)."""
    from pinneapple_physics.weather.evaluate import WB2_CHECKS

    f = np.full((3, len(CHANNELS), 32, 64), np.nan, np.float32)
    f[:, CHANNELS.index("z500")], f[:, CHANNELS.index("u500")] = 5000.0, 7.0
    ad = WeatherBench2Adapter(store, "IFS HRES", forecast=lambda i0, steps: f[:steps + 1],
                              variables={"z500": ("geopotential", 500), "u500": ("u_component_of_wind", 500)})
    y = ad.predict({"i0": 1, "steps": 2, "variable": ["z500", "u500"]})
    assert y.shape == (2, 3, 32, 64) and np.allclose(y[0], 5000.0) and np.allclose(y[1], 7.0)
    assert ad.predict({"i0": 1, "steps": 2, "variable": "u500"}).shape == (3, 32, 64)   # one name: not stacked
    with pytest.raises(PretrainedUnavailable, match="no channel"):
        ad.predict({"i0": 1, "steps": 2, "variable": ["z500", "t850"]})
    # the estimate prices every (variable, level) of the map: the default map is the six scored channels plus the four
    # the checks add, so the narrow one is the cheaper read
    narrow = WeatherBench2Adapter(store, "IFS HRES", forecast=None,
                                  variables={"z500": ("geopotential", 500)}).cost({"steps": 10})
    wide = WeatherBench2Adapter(store, "IFS HRES", forecast=None).cost({"steps": 10})
    assert wide > narrow and len(WB2_CHECKS) == 10 and "t700" in WB2_CHECKS and "z700" in WB2_CHECKS


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


# ------------------------------------------------------------------------------- the demo (#348, "done when")

DEMO = Path(__file__).resolve().parents[1] / "examples" / "weather_forecasting" / "pretrained_ensemble.py"


@pytest.fixture(scope="module")
def demo():
    """The demo script as a module: ``__name__`` is not ``__main__``, so importing it does not start a run."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("pretrained_ensemble_demo", DEMO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _demo_fields(rng, steps, nlat, noise, energy_step=0.0):
    """One model's window: the analysis, moved by ``noise`` and (optionally) drifting in dry static energy."""
    lat, lon = np.linspace(60.0, -60.0, nlat), np.arange(64) * 5.625
    lead = np.arange(steps + 1)[:, None, None]
    z = 5000.0 + 400.0 * np.cos(np.deg2rad(lat))[None, :, None] + noise * rng.normal(size=(steps + 1, nlat, 64))
    return {"z500": z, "u500": np.full((steps + 1, nlat, 64), 10.0), "v500": np.zeros((steps + 1, nlat, 64)),
            "t700": np.full((steps + 1, nlat, 64), 270.0) + energy_step * lead, "z700": np.full((steps + 1, nlat, 64),
                                                                                              3000.0)}, lat, lon


def test_the_demo_reports_the_ensemble_next_to_every_model_alone(demo, monkeypatch):
    """The issue's "done when": the demo reports the ensemble's skill against each model alone, per (region, season)
    cell. One model is deliberately out of its domain (its dry static energy drifts): it is taken out of the
    combination and still appears in the table with its own error, its residuals and the sentence that took it out --
    a model that only shows up in the ensemble's error is not a comparison."""
    rng = np.random.default_rng(0)
    monkeypatch.setattr(demo, "STEPS", 20)
    nlat, steps = 8, 20
    names = ["IFS HRES", "Pangu-Weather", "pinneapple-weather"]
    fields, lat, lon = {}, None, None
    for n, noise, drift in (("IFS HRES", 0.01, 0.0), ("Pangu-Weather", 0.05, 0.0), ("pinneapple-weather", 0.5, 3.0)):
        f, lat, lon = _demo_fields(rng, steps, nlat, noise, drift)
        fields[n] = {i: f for i in (1, 2)}                       # two cases, the same window each
    truth = {i: _demo_fields(np.random.default_rng(9), steps, nlat, 0.0)[0] for i in (1, 2)}
    experts = {n: Expert(n, lambda q, n=n: fields[n][int(q["i0"])]["z500"], 1.0, "weather") for n in names}
    # the tolerances the demo calibrates from the analysis: wide enough that only the drifting model is out
    cal = {"advection_tolerance": 4.0, "energy_tolerance": 0.05}

    cell = demo.run_cell(experts, names, [1, 2], truth, fields, lat, lon, cal)
    got = demo.summarise(names, cell)
    got["inits"] = ["2020-01-31T00", "2020-07-29T00"]

    assert set(got["error"]) == set(names) and set(got["rmse_day5"]) == set(names)
    assert got["error"]["IFS HRES"] < got["error"]["Pangu-Weather"] < got["error"]["pinneapple-weather"]
    assert got["best_single"]["name"] == "IFS HRES"
    assert got["ensemble"] <= got["error"]["pinneapple-weather"]          # the combination is never the worst model
    # the third model leaves its domain: out of the combination, and still measured
    assert got["out_of_domain"]["pinneapple-weather"] == 1.0 and got["out_of_domain"]["IFS HRES"] == 0.0
    assert got["share"]["pinneapple-weather"] == 0.0 and got["share"]["IFS HRES"] > 0.0
    assert np.isfinite(got["rmse_day5"]["pinneapple-weather"]) and got["rmse_day5"]["pinneapple-weather"] > 0
    assert "spreads per day" in got["failure_reason"]["pinneapple-weather"]
    assert not got["failure_reason"]["IFS HRES"]
    for key in ("advection_residual", "energy_drift", "weights", "share"):
        assert set(got[key]) == set(names), key
    assert got["energy_drift"]["pinneapple-weather"] > got["energy_drift"]["IFS HRES"]


def test_the_demo_names_the_experts_after_the_models_and_keeps_their_licences(demo, store, monkeypatch):
    """The table, the weights and the licence report are keyed by what the ensemble calls each expert. An archive
    adapter carries the published model's display name (``"IFS HRES"``, the key ``BASELINES_64x32`` uses), so keying
    the demo's own dictionary by the registry name made every lookup of a weight raise ``KeyError``."""

    class Unloaded(TorchCheckpointAdapter):
        """The checkpoint of the demo is 40 MB and needs torch: this test is about the names, not the weights."""
        def open(self):
            return self

    monkeypatch.setattr(demo, "TorchCheckpointAdapter", Unloaded)
    experts = demo.build_experts(store, "ck.pt")
    assert list(experts) == ["IFS HRES", "Pangu-Weather", "pinneapple-weather"]
    assert all(e.name == k for k, e in experts.items())
    lic = licence_report(list(experts.values()))
    assert {n: r["spdx"] for n, r in lic["licences"].items()} == {
        "IFS HRES": "NOASSERTION", "Pangu-Weather": "NOASSERTION", "pinneapple-weather": "Apache-2.0"}
    assert lic["unverified"] == ["IFS HRES", "Pangu-Weather"]      # nobody read them at the source, and it says so


def test_the_field_cache_of_the_demo_is_keyed_by_the_number_of_leads(demo, tmp_path, monkeypatch):
    """A smoke run with four steps cached its fields under the model's name; a later run that asks for 40 found the
    short file and scored a 240-hour window that was four steps long. The key carries the lead count, and a file that
    does not hold them is read again."""
    monkeypatch.setattr(demo, "STEPS", 4)
    calls = []

    def predict(q):
        calls.append(q["steps"])
        return np.stack([np.full((q["steps"] + 1, 4, 4), float(i)) for i in range(len(demo.NEEDED))])

    e = from_callable("m", predict)
    first = demo.fetch(e, 7, tmp_path)
    assert calls == [4] and first["z500"].shape == (5, 4, 4)
    demo.fetch(e, 7, tmp_path)                                   # the same query reads the cache
    assert calls == [4]
    monkeypatch.setattr(demo, "STEPS", 8)
    longer = demo.fetch(e, 7, tmp_path)                           # a longer window is a different key
    assert calls == [4, 8] and longer["z500"].shape == (9, 4, 4)
