"""The turntable (#64): an orbiting camera and a field advancing in time, one call, GIF or MP4.

What the issue promised is that one command turns a case into a clip that still says what it shows after it
leaves the report: 1080p, the camera orbiting, the time advancing, a discreet mark and a unit caption. These
tests hold each of those to its word, and they check the failure modes that would otherwise surface on a user's
machine -- a 2-D field, one instant, a NaN, a suffix nobody can encode.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from pinneapple_tools.visualization.orbit import DEFAULT_SIZE, turntable


def _walk(n_t: int = 6, n_s: int = 24, speed: float = 1.0) -> np.ndarray:
    """A (nt, nx, ny) field that travels in x and bulges in y: both axes change with the instant."""
    t = np.linspace(0.0, 1.0, n_t)
    x = y = np.linspace(0.0, 1.0, n_s)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return (np.sin(2 * np.pi * (X[None] - speed * t[:, None, None]))
            * np.cos(2 * np.pi * Y[None]))


SMALL = (240, 135)                     # a frame small enough to keep the suite in seconds


def _frame(path):
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def test_the_default_frame_is_1080p():
    """The issue asks for 1080p: that is the default, not an option a caller has to remember."""
    frames = turntable(_walk(n_t=2), frames=2)
    assert DEFAULT_SIZE == (1920, 1080)
    assert [f.shape for f in frames] == [(1080, 1920, 3), (1080, 1920, 3)]
    assert frames[0].dtype == np.uint8


def test_the_camera_orbits_and_the_field_advances():
    """Both halves of the animation move: the frames differ, and they differ for two independent reasons."""
    moving = turntable(_walk(n_t=3), size=SMALL)
    assert not np.array_equal(moving[0], moving[1])
    # the same instant at every frame: the only thing that can change is the camera
    still = np.repeat(_walk(n_t=1)[:1], 3, axis=0)
    orbiting = turntable(still, size=SMALL)
    assert np.abs(orbiting[0].astype(int) - orbiting[-1].astype(int)).max() > 40
    # and with the camera held still, the same field gives the same frame
    frozen = turntable(still, size=SMALL, orbit=0.0)
    assert np.array_equal(frozen[0], frozen[-1])


def test_a_full_turn_does_not_end_on_the_frame_it_started_from():
    """A looping GIF that repeats its first frame stutters once per loop; the sweep stops one step short."""
    still = np.repeat(_walk(n_t=1)[:1], 12, axis=0)
    frames = turntable(still, size=SMALL, orbit=360.0)
    for previous, following in zip(frames, frames[1:], strict=False):
        assert not np.array_equal(previous, following)
    assert not np.array_equal(frames[0], frames[-1])



def test_frames_resamples_and_repeats_the_instants_it_has():
    """A smooth orbit out of a few instants: asking for more frames than instants holds the nearest instant."""
    assert len(turntable(_walk(n_t=5), frames=5, size=SMALL)) == 5
    assert len(turntable(_walk(n_t=5), frames=3, size=SMALL)) == 3
    twelve = turntable(_walk(n_t=4), frames=12, size=SMALL)
    assert len(twelve) == 12
    assert not np.array_equal(twelve[0], twelve[1])      # same instant, camera moved on


def test_the_mark_is_discreet_and_in_the_corner():
    marked = turntable(_walk(n_t=2), size=SMALL, frames=2, caption=False)
    unmarked = turntable(_walk(n_t=2), size=SMALL, frames=2, caption=False, mark="")
    corner = (slice(-24, None), slice(-90, None))        # bottom-right
    assert np.abs(marked[0][corner].astype(int) - unmarked[0][corner].astype(int)).max() > 30
    elsewhere = (slice(0, 40), slice(0, 90))             # top-left: the mark is not there
    assert np.abs(marked[0][elsewhere].astype(int) - unmarked[0][elsewhere].astype(int)).max() == 0
    # discreet: it is a corner annotation, not the subject -- a few hundred pixels at most
    changed = (np.abs(marked[0].astype(int) - unmarked[0].astype(int)).max(axis=2) > 8)
    assert 0 < changed.sum() < 0.02 * changed.size


def test_the_caption_says_what_the_clip_shows():
    """The caption carries the unit and the instant: that is what makes the clip readable out of context."""
    with_caption = turntable(_walk(n_t=2), size=SMALL, frames=2, name="u", unit="m/s", t=[0.0, 1.5], mark="")
    without = turntable(_walk(n_t=2), size=SMALL, frames=2, name="u", unit="m/s", t=[0.0, 1.5], mark="",
                        caption=False)
    bottom_left = (slice(-30, None), slice(0, 200))
    assert np.abs(with_caption[0][bottom_left].astype(int) - without[0][bottom_left].astype(int)).max() > 30
    # the second frame's caption differs from the first's, because the time in it does
    first, last = with_caption[0][bottom_left], with_caption[-1][bottom_left]
    assert not np.array_equal(first, last)


def test_a_gif_is_written_with_the_frames_and_the_size_asked_for(tmp_path):
    path = tmp_path / "case.gif"
    written = turntable(_walk(n_t=4), path, size=SMALL, frames=6, fps=12)
    assert written == path and path.is_file() and path.stat().st_size > 0
    from PIL import Image

    with Image.open(path) as image:
        assert image.format == "GIF" and image.n_frames == 6 and image.size == SMALL


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="MP4 export needs the ffmpeg binary on PATH")
def test_an_mp4_is_written_and_is_h264(tmp_path):
    path = tmp_path / "case.mp4"
    turntable(_walk(n_t=4), path, size=(320, 180), frames=4, fps=10)
    data = path.read_bytes()
    assert len(data) > 1000
    assert b"ftyp" in data[:64]                          # an ISO base media file, which is what an MP4 is


def test_without_ffmpeg_an_mp4_says_so_instead_of_failing_quietly(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name, *a, **kw: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        turntable(_walk(n_t=2), tmp_path / "case.mp4", size=SMALL, frames=2)


def test_the_frames_come_back_when_no_path_is_given():
    frames = turntable(_walk(n_t=3), size=SMALL)
    assert len(frames) == 3 and all(f.shape == (SMALL[1], SMALL[0], 3) for f in frames)


def test_pp_viz_turntable_is_the_same_call():
    """`pp.viz.turntable` is the documented entry point; it must be the function that is tested here."""
    import pinneapple as pp

    frames = pp.viz.turntable(_walk(n_t=3), size=SMALL)
    assert len(frames) == 3


@pytest.mark.parametrize("kwargs, message", [
    ({"field": np.zeros((8, 8))}, "nt, nx, ny"),
    ({"field": np.zeros((1, 8, 8))}, "one instant"),
    ({"field": np.full((2, 8, 8), np.nan)}, "non-finite"),
    ({"field": np.zeros((2, 8, 8)), "size": (32, 32)}, "64x64"),
    ({"field": np.zeros((2, 8, 8)), "frames": 1}, "at least 2"),
    ({"field": np.zeros((2, 8, 8)), "clim": (1.0, 0.0)}, "clim"),
    ({"field": np.zeros((2, 8, 8)), "x": np.zeros(3)}, "long"),
    ({"field": np.zeros((2, 8, 8)), "t": np.zeros(3)}, "one value per instant"),
])
def test_a_field_or_option_the_turntable_cannot_draw_is_refused(kwargs, message):
    field = kwargs.pop("field")
    with pytest.raises(ValueError, match=message):
        turntable(field, **kwargs)


def test_an_unknown_suffix_names_the_two_that_work(tmp_path):
    with pytest.raises(ValueError, match=r"\.gif or a \.mp4"):
        turntable(_walk(n_t=2), tmp_path / "case.png", size=SMALL, frames=2)


def test_an_unknown_colour_map_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError):
        turntable(_walk(n_t=2), size=SMALL, frames=2, cmap="not-a-colour-map")
