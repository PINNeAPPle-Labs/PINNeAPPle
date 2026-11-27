"""The consumers read the record, and draw what it decided (#345).

The last scope item of the issue is "twin3d and the weather ``earth2_gif`` read it (lead bar, envelope colours)". A
consumer that keeps its own opinion about the same forecast is what the record exists to remove, so what is held here
is that the figure and the export follow the ``Trust`` and not a second set of numbers:

* the weather ``earth2_gif``, given ``trust=``, ends its lead bar at the leads the record measured, paints no colour
  at all when the record measured no lead, and writes the record's badge under the lead — even when the ``horizon=``
  it was also given says something else, which is how the test would fail if the record were ignored;
* the twin3d ``Scene`` writes the record into ``scene.json``, the document ``Trust.to_json`` gives and nothing else.

The figure is checked without rendering an animation: ``FuncAnimation`` is replaced by a recorder, and the frame
functions it captured are called so the texts and the bar patches can be read back.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pinneapple_analysis.trust.adapters import weather_horizon
from pinneapple_analysis.trust.record import Horizon, Trust, TrustValue
from pinneapple_physics.weather.viz import _BADGE_COLOUR as _BADGE

SOURCE = "pinneapple_physics.weather.evaluate.horizons"


def _record(useful: float | None, no_skill: float | None, *, in_domain: float | None = 1.0) -> Trust:
    """A weather record whose horizon is the two leads given (``None``: the record did not measure it).

    ``None`` goes through ``evaluate.horizons``' own convention — a lead that was never crossed inside the scored
    window is ``inf`` there — so the record ends up with ``not_measured`` and the reason that says why.
    """
    horizons = {k: (float("inf") if v is None else v)
                for k, v in (("useful_hours", useful), ("no_skill_hours", no_skill))}
    horizon = weather_horizon(horizons, requested_lead=48.0)
    in_domain_value = (TrustValue.not_measured("no gate was fitted", source=SOURCE) if in_domain is None
                       else TrustValue.measured(in_domain, unit="[-]", source="a TrustGate on the training sample"))
    return Trust(kind="weather", in_domain=in_domain_value,
                 domain_score=TrustValue.measured(0.5, unit="[-]", source="a TrustGate on the training sample"),
                 horizon=horizon)


class _Recorder:
    """Stands in for ``matplotlib.animation.FuncAnimation``: keeps the figure and the frame function, writes a file."""

    last: _Recorder | None = None

    def __init__(self, fig, frame, frames=0, **kw):
        self.figure, self.frame = fig, frame
        self.n = frames if isinstance(frames, int) else len(frames)
        _Recorder.last = self

    def save(self, path, **kw):
        from pathlib import Path
        Path(path).write_bytes(b"GIF89a")


@pytest.fixture
def recorder(monkeypatch):
    import matplotlib.animation as animation
    monkeypatch.setattr(animation, "FuncAnimation", _Recorder)
    return _Recorder


def _draw(tmp_path, *, trust=None, horizon=None, leads=72.0):
    """An ``earth2_gif`` of six synthetic states up to ``leads`` hours, returning the recorder."""
    from pinneapple_physics.weather.viz import earth2_gif

    lat = np.linspace(84.375, -84.375, 32)
    lon = np.arange(0.0, 360.0, 5.625)
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(6, lat.size, lon.size))
    forecast = truth + 0.1 * rng.normal(size=truth.shape)
    times = np.datetime64("2020-01-01T00") + np.arange(6) * np.timedelta64(6, "h")
    lead_hours = np.linspace(0.0, leads, 6)
    return earth2_gif(tmp_path / "forecast.gif", "t2m", truth, forecast, lat, lon, times,
                      lead_hours=lead_hours, horizon=horizon or {"useful_hours": 1e9, "no_skill_hours": 1e9},
                      title="testbed forecast", substeps=1, size=48, dpi=20, fps=1, trust=trust)


def _texts(recorder) -> dict[str, tuple[str, str]]:
    """Every text of the figure once the last frame has been drawn: text -> (text, colour)."""
    recorder.last.frame(recorder.last.n - 1)
    return {artist.get_text(): (artist.get_text(), artist.get_color()) for artist in recorder.last.figure.texts}


def _spans(recorder, end: float) -> list[tuple[float, float]]:
    """The coloured segments of the lead bar, as (start, end) in lead hours; the white progress marker is not one."""
    bar = [ax for ax in recorder.last.figure.axes if tuple(ax.get_xlim()) == (0.0, end)][0]
    return sorted((float(p.get_x()), float(p.get_x() + p.get_width())) for p in bar.patches
                  if p.get_width() > 0 and float(p.get_height()) == 1.0)


# ----------------------------------------------------------------------------------- the weather gif
def test_the_gif_draws_the_lead_bar_at_the_leads_the_record_measured(tmp_path, recorder):
    """The bar ends where the record says it does — not where the ``horizon=`` also passed says."""
    _draw(tmp_path, trust=_record(24.0, 48.0), horizon={"useful_hours": 1e9, "no_skill_hours": 1e9}, leads=72.0)
    assert _spans(recorder, 72.0) == [(0.0, 24.0), (24.0, 48.0), (48.0, 72.0)]


def test_the_gif_writes_the_badge_of_the_record_under_the_lead(tmp_path, recorder):
    """The caption under the lead is the record's badge, and it turns red when the record says no skill."""
    facing = _record(24.0, 48.0)
    _draw(tmp_path, trust=facing, leads=72.0)
    texts = _texts(recorder)
    assert f"trust · {facing.badge_text()}" in texts, sorted(texts)
    assert texts[f"trust · {facing.badge_text()}"][1] == _BADGE["fading"]

    outside = _record(24.0, 48.0, in_domain=0.0)          # the inputs are outside the training distribution
    assert outside.badge() == "no_skill"
    _draw(tmp_path, trust=outside, leads=72.0)
    texts = _texts(recorder)
    assert "trust · no skill" in texts
    assert texts["trust · no skill"][1] == _BADGE["no_skill"]


def test_the_gif_draws_no_colour_for_a_horizon_the_record_did_not_measure(tmp_path, recorder):
    """A record that measured no lead gets no green bar and a grey caption that says why — never a silent ``good``."""
    record = _record(None, None)
    assert record.horizon.useful.status == "not_measured"
    _draw(tmp_path, trust=record, leads=72.0)
    assert _spans(recorder, 72.0) == []                        # no segment at all
    texts = _texts(recorder)
    assert "horizon not measured" in texts, sorted(texts)
    assert "beyond them" in next(t for t in texts if t.startswith("lead bar:"))


def test_the_gif_the_record_replaces_the_horizon_it_was_also_given(tmp_path, recorder):
    """Given both, the figure follows the record: the ``horizon=`` argument is not a second opinion."""
    record = _record(None, None)
    _draw(tmp_path, trust=record, horizon={"useful_hours": 6.0, "no_skill_hours": 12.0}, leads=72.0)
    texts = _texts(recorder)
    assert "horizon not measured" in texts                     # not "good", which the horizon= would have said
    assert _spans(recorder, 72.0) == []                        # and not the 6 h / 12 h segments either


# ----------------------------------------------------------------------------------- twin3d
def test_the_twin_from_export_carries_the_record_the_scene_was_given(tmp_path):
    """``Scene.attach_trust`` writes the record into ``scene.json``, as the document its own JSON gives."""
    from pinneapple_twin3d import Scene

    vertex, faces = _triangle()
    scene = Scene("surrogate field", times=[0.0, 1.0])
    scene.add_part("blade", vertex, faces)
    scene.add_field("blade", "pressure", np.array([1.0, 2.0, 3.0]), unit="Pa")
    record = _record(24.0, 48.0)
    scene.attach_trust(record)

    manifest = json.loads(open(scene.export(str(tmp_path / "twin"), with_viewer=False)).read())
    assert manifest["trust"] == json.loads(record.to_json())
    assert manifest["trust"]["schema"] == "pinneapple.trust/1"
    assert manifest["trust"]["badge"] == record.badge()
    assert Trust.validate_document(manifest["trust"]) == []


def test_the_twin_without_a_record_says_so_instead_of_inventing_one(tmp_path):
    """No ``attach_trust``: the key is there and null, and a non-record is refused by name."""
    from pinneapple_twin3d import Scene

    vertex, faces = _triangle()
    scene = Scene("no record")
    scene.add_part("blade", vertex, faces)
    manifest = json.loads(open(scene.export(str(tmp_path / "twin"), with_viewer=False)).read())
    assert manifest["trust"] is None
    with pytest.raises(TypeError, match="takes a pinneapple_analysis.trust.Trust"):
        scene.attach_trust({"badge": "good"})


def _triangle():
    return (np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.uint32))


def test_the_horizon_of_a_record_is_read_in_hours_whatever_unit_it_was_measured_in():
    """``horizon_of_trust`` converts the record's own unit and says which leads it could not convert."""
    from pinneapple_analysis.trust.adapters import horizon_of_trust

    days = Horizon(useful=TrustValue.measured(2.0, unit="d", source=SOURCE),
                   no_skill=TrustValue.measured(4.0, unit="d", source=SOURCE),
                   requested=TrustValue.measured(48.0, unit="h", source="the case"),
                   unit="d", source=SOURCE)
    read = horizon_of_trust(Trust(kind="weather", in_domain=TrustValue.not_measured("x"),
                                  domain_score=TrustValue.not_measured("x"), horizon=days))
    assert (read["useful_hours"], read["no_skill_hours"]) == (48.0, 96.0)
    assert read["lead_axis"] == "measured"

    fortnights = Horizon(useful=TrustValue.measured(1.0, unit="fortnight", source=SOURCE),
                         no_skill=TrustValue.not_measured("x", source=SOURCE),
                         requested=TrustValue.not_measured("x"), unit="fortnight", source=SOURCE)
    read = horizon_of_trust(Trust(kind="weather", in_domain=TrustValue.not_measured("x"),
                                  domain_score=TrustValue.not_measured("x"), horizon=fortnights))
    assert read["lead_axis"] == "not_measured"
    assert "does not convert to 'h'" in read["note"]
