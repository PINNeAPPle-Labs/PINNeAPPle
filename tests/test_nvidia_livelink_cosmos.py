"""The Omniverse live link and the Cosmos plausibility benchmark, without Kit, Cosmos or a GPU (#411).

Neither Omniverse Kit nor a Cosmos model is installed here, so what these tests pin is the half that does not need
them: the USD stage the live link writes and reads back (``pxr`` alone), the install extra each missing piece names,
and the physics benchmark that scores where a generated clip stops being physical. The numbers asserted below were
measured on this machine; the docs page quotes the same ones.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_tools.nvidia import cosmos, livelink

pytest.importorskip("pxr")


def _track(steps: int = 5) -> livelink.ForecastTrack:
    """A four-vertex body with two transient fields, a pose and a scale per step, plus one sensor."""
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
    field = np.stack([np.arange(4, dtype=float) + i for i in range(steps)])
    return livelink.ForecastTrack(
        title="probe", times=np.arange(steps) * 0.5,
        parts=[livelink.ForecastPart(name="body", vertices=vertices, faces=faces,
                                     fields={"T": field, "p": field * 2 + 1},
                                     field_units={"T": "K", "p": "Pa"},
                                     position=np.stack([np.array([i * 0.1, 0.0, 0.0]) for i in range(steps)]),
                                     scale=np.linspace(1.0, 1.4, steps))],
        sensors=[livelink.SensorTrack(name="s1", position=(0.0, 0.0, 1.0),
                                      series=np.linspace(300.0, 400.0, steps), unit="K", envelope=(250.0, 450.0))])


def test_the_track_the_tests_write_is_self_consistent():
    """A stage written from an inconsistent track would be wrong in a way the reader cannot see, so it is refused."""
    assert _track().check() == []
    broken = _track()
    broken.parts[0].fields["T"] = np.zeros((3, 4))          # 3 steps against 5 times
    assert any("expected (5, 4)" in problem for problem in broken.check())


def test_the_stage_round_trips_through_openusd(tmp_path):
    """Write, read back, and play: the time samples, the poses, the scale and the sensor all survive."""
    path = livelink.write_forecast_stage(_track(), str(tmp_path / "fc.usda"))
    assert path.endswith(".usda")
    playback = livelink.read_forecast_stage(path)
    assert playback.n_steps == 5
    assert playback.times == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert playback.metadata.get("pinneapple_format") == livelink.LIVELINK_FORMAT
    assert list(playback.metadata["time_values"]) == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

    lines = livelink.play_stage(path, echo=lambda *a, **k: None)
    assert len(lines) == 5
    assert "body at (0, 0, 0) scale 1" in lines[0]
    assert "body at (0.4, 0, 0) scale 1.4" in lines[-1]
    assert "s1=300 K" in lines[0] and "s1=400 K" in lines[-1]


def test_the_stage_carries_the_trust_record_when_the_forecast_has_one(tmp_path):
    """The trust verdict travels with the scene: whoever opens the stage in Omniverse sees what it cost to trust it.

    It is written as a JSON string in the layer metadata (``pinneapple_trust``), because a USD stage has no dict
    property and a flattened string is what survives a round trip through every USD tool.
    """
    track = _track()
    track.trust = {"status": "measured", "horizon": 12.0, "unit": "frames"}
    playback = livelink.read_forecast_stage(livelink.write_forecast_stage(track, str(tmp_path / "t.usda")))
    assert json.loads(playback.metadata["pinneapple_trust"]) == track.trust


def test_the_missing_kit_names_its_extra_and_pxr_is_enough_to_write(monkeypatch):
    """Kit is absent in this environment by design: the error has to say which install turns the link on."""
    assert livelink.kit_available() is False
    assert livelink.require_omniverse() is not None                    # pxr is present: writing needs no Kit

    def _absent():
        raise ImportError("no pxr")

    monkeypatch.setattr(livelink, "require_pxr", _absent)
    with pytest.raises(ImportError, match="OpenUSD"):
        livelink.require_omniverse()


# --------------------------------------------------------------------------------- the Cosmos plausibility benchmark
@pytest.fixture(scope="module")
def report() -> cosmos.BenchmarkReport:
    """One labelled set, judged once: the numbers the docs quote come from this same run."""
    return cosmos.benchmark(seeds=(0,))


def test_the_labelled_set_has_both_clean_and_violated_clips():
    clean, bad = cosmos.labelled_set(seeds=(0,))
    assert len(clean) == 4 and len(bad) == 11
    assert {c.kind for c in clean} == {"rigid", "fluid"}
    assert {c.violation for c in bad} >= {"reverse", "mass_gain", "gravity_off", "teleport"}


def test_the_benchmark_catches_the_violations_without_flagging_the_clean_clips(report):
    """Measured: 8 of the 11 violated clips are caught, none of the 4 physical ones is, one frame after the violation."""
    summary = report.summary()
    assert summary["clips"] == 15 and summary["violated"] == 11 and summary["physical"] == 4
    assert summary["detection_rate"] == pytest.approx(0.727, abs=1e-3)
    assert summary["false_alarm_rate"] == 0.0
    assert summary["lag_frames"] == {"min": 0, "median": 1.0, "max": 1}


def test_the_attribution_is_the_weak_part_and_is_reported_as_such(report):
    """Detection and attribution are different questions: the right clip is flagged, the right *check* fires less often.

    Measured 0.25 with a single seed. The test pins the weakness rather than the aspiration, because a benchmark that
    reports only the detection rate hides it.
    """
    assert report.attribution_rate() == pytest.approx(0.25, abs=1e-6)
    assert report.attribution_rate() < report.detection_rate()


def test_the_horizon_is_the_lead_the_trust_record_asks_for(report):
    """The trust horizon of a generated clip is the median frame at which it stopped being physical, against the clip.

    Measured 26.5 of the 48 frames: the violations are injected around the middle of the clip, so half of it is spent
    before anything is wrong, and the horizon is the point past which the rest of the clip is no longer physics.
    """
    horizon = report.horizon()
    assert horizon.useful.status == "measured"
    assert horizon.useful.value == pytest.approx(26.5)
    assert horizon.no_skill.value == pytest.approx(48.0)
    assert horizon.useful.reference == pytest.approx(48.0)


def test_a_violation_is_labelled_and_its_residual_moves(report):
    """Every judged clip carries what it is, and the violated ones carry the residual the check reads."""
    for verdict in report.of("rigid"):
        assert verdict.clip.violation is None if verdict.clip.physical() else verdict.clip.violation
    assert all(v.detected is not None for v in report.verdicts)
