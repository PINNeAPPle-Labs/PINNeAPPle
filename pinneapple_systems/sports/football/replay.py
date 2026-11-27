"""Broadcast-style 3-D replays and the club-grade match report (#917).

A research figure answers "does the physics hold?"; a broadcast replay has to answer "what happened?" in the
vocabulary an analyst already reads: a 3-D pitch with real markings, discs with numbers and velocity trails, the
ball drawn as a tube whose width is the model's own uncertainty, the fitted trajectory beside the feed that
produced it, a pitch-control layer that can be switched off, a time code and a data credit.

:func:`sequence` slices a tracking record into the flat arrays every renderer reads, applying the
:class:`~.ball.HeightMap` of #911 to the ball's ``z``; :func:`replay` writes the MP4 (1080p, 25 fps, CPU only) and
the interactive page (self-contained three.js). Because both read the same :class:`Sequence`, the video, the still
and the page cannot disagree about what they show.

:func:`report` writes the one-page match report. It does not invent a second report engine: the body is the
Markdown of :func:`pinneapple_autolab.report.write_report`, the same generator as the engineering report of the
visualization epic, wrapped in the house page.

Identifiability is declared, never hidden (#911): the spin is printed with the flag the Cramér-Rao bound gives it,
and a coefficient the flights do not identify is reported as not identified rather than as a number.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Sequence", "sequence", "replay", "report", "CREDIT", "PLAYER_RADIUS", "BALL_RADIUS"]

#: the credit every asset carries -- SkillCorner's MIT licence, and no club or player claim anywhere in the assets
CREDIT = "Data: SkillCorner open data (MIT license), 10 Hz broadcast tracking. Physics: PINNeAPPle football."

PLAYER_RADIUS = 0.42          # m: a disc wide enough to read at 1080p over half a pitch
BALL_RADIUS = 0.11

#: house palette, the same order as :mod:`...football.plots`, so a team keeps its colour across every asset
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TURF = "#0b1a10"
INK = "#f2f0ea"
MUTED = "#9a988f"
#: how many frames a velocity trail lasts (0.4 s at 25 fps): long enough to read a run, short enough not to clutter
TRAIL_FRAMES = 10


@dataclass
class Sequence:
    """One stretch of the broadcast feed, ready to draw.

    ``t`` is seconds from the start of the clip. ``ball`` is the **corrected** height (the map of #911 applied) and
    ``ball_feed`` what the provider reported, kept so the two can be drawn together. ``band`` is ``(lo, hi)``, the
    uncertainty of the corrected height from the map's bootstrap band -- it is what the tube's width shows.
    ``players[team]`` holds ``pos`` ``(n, p, 2)`` with NaN where the detector lost the player, ``vel`` ``(n, p, 2)``
    in m/s and ``numbers`` (the provider's anonymous track ids: these discs claim no identity). ``control`` is the
    pitch-control surface per frame, ``(n, ny, nx)``, with the grid it lives on, or ``None``.
    """

    t: np.ndarray
    ball: np.ndarray
    ball_feed: np.ndarray
    band: tuple[np.ndarray, np.ndarray]
    players: dict[str, dict[str, np.ndarray]]
    teams: list[str]
    fps: float
    label: str = ""
    fit: np.ndarray | None = None
    control: np.ndarray | None = None
    control_grid: tuple[np.ndarray, np.ndarray] | None = None
    identifiability: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t)

    def at(self, i: int) -> dict[str, Any]:
        """Everything a renderer needs for frame ``i`` (no computation, only indexing)."""
        return {"t": float(self.t[i]), "ball": self.ball[i], "ball_feed": self.ball_feed[i],
                "band": (float(self.band[0][i]), float(self.band[1][i])),
                "fit": None if self.fit is None else self.fit[i],
                "control": None if self.control is None else self.control[i],
                "players": {k: {"pos": v["pos"][i], "vel": v["vel"][i], "numbers": v["numbers"]}
                            for k, v in self.players.items()}}

    def time_code(self, i: int) -> str:
        """``MM:SS.mmm`` of frame ``i``: what a broadcast puts in the corner."""
        m, s = divmod(float(self.t[i]), 60.0)
        return f"{int(m):02d}:{s:06.3f}"


def velocity(pos: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference velocity of ``(n, p, 2)`` positions in m/s, NaN where either neighbour is missing."""
    pos = np.asarray(pos, float)
    v = np.full_like(pos, np.nan)
    if len(pos) < 2:
        return v
    v[1:-1] = (pos[2:] - pos[:-2]) * (fps / 2.0)
    v[0] = (pos[1] - pos[0]) * fps
    v[-1] = (pos[-1] - pos[-2]) * fps
    return v


def _resample(a: np.ndarray, n_out: int) -> np.ndarray:
    """Linear resample of ``(n, ...)`` to ``n_out`` frames (the feed's 10 Hz to the video's 25 fps)."""
    a = np.asarray(a, float)
    n = len(a)
    if n == n_out or n < 2:
        return a
    src, dst = np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n_out)
    flat = a.reshape(n, -1)
    out = np.empty((n_out, flat.shape[1]), float)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(dst, src, flat[:, j])
    return out.reshape((n_out,) + a.shape[1:])


def _split_sides(pos: np.ndarray) -> list[list[int]]:
    """Group the players of one frame into the two sides of the most populated gap along ``x``.

    The feed carries no team for a track, so the sides are read off the pitch itself: at any frame the ten outfield
    players of one team sit on one side of the other ten more often than not, and the largest gap in the sorted
    ``x`` is where the two blocks meet. It is a split of *sides*, not of clubs -- which is exactly what an asset
    without a club claim should show.
    """
    x = np.sort(np.asarray(pos[:, 0], float))
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return [list(range(len(pos))), []]
    gaps = np.diff(x)
    k = int(np.argmax(gaps))
    cut = 0.5 * (x[k] + x[k + 1])
    left = [i for i in range(len(pos)) if not np.isfinite(pos[i, 0]) or pos[i, 0] < cut]
    right = [i for i in range(len(pos)) if i not in set(left)]
    return [left, right]


def _control_stack(players: dict[str, dict[str, np.ndarray]], ball: np.ndarray, control, n_out: int):
    """The pitch-control surface of every drawn frame, for the toggle layer."""
    if len(players) < 2:
        return None, None
    keys = list(players)
    q = None
    grid = None
    out = None
    for i in range(n_out):
        a, b = players[keys[0]], players[keys[1]]
        ok_a = np.all(np.isfinite(a["pos"][i]), axis=1)
        ok_b = np.all(np.isfinite(b["pos"][i]), axis=1)
        if ok_a.sum() < 3 or ok_b.sum() < 3 or not np.all(np.isfinite(ball[i, :2])):
            continue
        q, grid = control(a["pos"][i][ok_a], a["vel"][i][ok_a], b["pos"][i][ok_b], b["vel"][i][ok_b],
                          ball[i, :2])
        if out is None:
            out = np.zeros((n_out,) + np.shape(q))
        out[i] = q
    return out, grid


def sequence(tracks: dict[str, Any], t0: float, t1: float, *, height=None, fit=None, fps: float = 25.0,
             control=None, label: str = "", identifiability: dict[str, Any] | None = None) -> Sequence:
    """Cut ``[t0, t1]`` out of a :func:`...data.skillcorner_tracking` record, at ``fps``.

    ``height`` is the :class:`...ball.HeightMap` of #911 (``None`` keeps the feed's own ``z``); ``fit`` a fitted
    :class:`...ball.FlightFit` whose trajectory is overlaid; ``control`` an optional callable
    ``(att_pos, att_vel, def_pos, def_vel, ball_xy) -> (q, (xs, ys))`` returning the pitch-control surface of one
    frame, evaluated only for the frames that are drawn.
    """
    ball = tracks["ball"]
    tb = np.asarray(ball["t"], float)
    ok = np.isfinite(tb) & (tb >= t0) & (tb <= t1)
    if ok.sum() < 10:
        raise ValueError("fewer than 10 ball samples in the requested window")
    tt = tb[ok]
    n_out = max(10, int(round((t1 - t0) * fps)))
    feed = _resample(np.column_stack([ball["x"][ok], ball["y"][ok], ball["z"][ok]]), n_out)
    t_out = np.linspace(tt[0], tt[-1], n_out) - tt[0]

    has_z = np.isfinite(feed[:, 2])
    corrected = feed.copy()
    lo = np.full(n_out, np.nan)
    hi = np.full(n_out, np.nan)
    if height is not None:
        corrected[has_z, 2] = height(feed[has_z, 2], feed[has_z, 0], feed[has_z, 1])
        lo[has_z], hi[has_z] = height.band_at(feed[has_z, 2])
    else:
        lo[has_z] = hi[has_z] = feed[has_z, 2]

    player_frames = np.asarray(tracks["players"][:, 0], float) if len(tracks["players"]) else np.zeros(0)
    frames = np.asarray(ball["frame"], float)
    f0 = float(np.interp(t0, tb[ok], frames[ok]))
    f1 = float(np.interp(t1, tb[ok], frames[ok]))
    sel = (player_frames >= f0) & (player_frames <= f1)
    pdat = tracks["players"][sel] if len(player_frames) else np.zeros((0, 6))
    tf_all = np.interp(player_frames[sel], frames[ok], tt[ok]) if len(pdat) else np.zeros(0)

    teams, players = [], {}
    ids = np.asarray([i for i in np.unique(pdat[:, 1]) if i >= 0], float) if len(pdat) else np.zeros(0)
    if len(ids) >= 4:
        pos = np.full((n_out, len(ids), 2), np.nan)
        for j, pid in enumerate(ids):
            row = pdat[pdat[:, 1] == pid]
            if len(row) < 3:
                continue
            order = np.argsort(row[:, 0])
            xy = row[order, 3:5]
            present = row[order, 2] > 0.5                      # is_detected
            tf = tf_all[order]
            good = present & np.all(np.isfinite(xy), axis=1) & np.isfinite(tf)
            if good.sum() < 3:
                continue
            pos[:, j, 0] = np.interp(t_out + tt[0], tf[good], xy[good, 0], left=np.nan, right=np.nan)
            pos[:, j, 1] = np.interp(t_out + tt[0], tf[good], xy[good, 1], left=np.nan, right=np.nan)
        # which side each track plays on, decided across the whole clip (a side, not a club)
        mid = np.nanmedian(pos[:, :, 0], axis=0)
        cut = float(np.nanmedian(mid))
        for name, take in (("A", mid <= cut), ("B", mid > cut)):
            if take.sum() < 2:
                continue
            sub = pos[:, take]
            teams.append(name)
            players[name] = {"pos": sub, "vel": velocity(sub, fps),
                             "numbers": [str(int(i)) for i in ids[take]]}
    ctrl, grid = _control_stack(players, corrected, control, n_out) if control is not None else (None, None)

    fl = None
    if fit is not None:
        tr = fit.predict(t_out + tt[0])
        fl = np.full((n_out, 3), np.nan)
        m = np.all(np.isfinite(tr), axis=1)
        fl[m, :2] = tr[m, :2]
        fl[m, 2] = height(tr[m, 2]) if height is not None else tr[m, 2]

    return Sequence(t=t_out, ball=corrected, ball_feed=feed, band=(lo, hi), players=players, teams=teams,
                    fps=fps, label=label, fit=fl, control=ctrl, control_grid=grid,
                    identifiability=dict(identifiability or {}))
