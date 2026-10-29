"""Public football data, downloaded at run time, cached outside git with the SHA-256 of every file.

Sources (none needs credentials; every one is redistributed by its owner on GitHub):

* **Metrica Sports sample data** -- two anonymised matches, 25 Hz player and ball tracking (2-D, ball height not
  given) synchronised with events (passes with sender, receiver, start and end frames).
  github.com/metrica-sports/sample-data; terms: "if you use it for anything public, please acknowledge the source".
* **SkillCorner open data** -- ten A-League 2024/25 matches of broadcast tracking at 10 Hz, ball with an estimated
  height ``z``; MIT licence. github.com/SkillCorner/opendata (files under Git LFS, served by media.githubusercontent).
* **StatsBomb open data** -- event data with 360 freeze frames (every visible player at the moment of a shot).
  github.com/statsbomb/open-data; StatsBomb's user agreement (non-commercial, attribute "Data: StatsBomb").

Only small derived tables are ever committed; raw files stay in the cache (``$PINNEAPPLE_DATA_CACHE`` or
``~/.cache/pinneapple/datasets``, sub-folder ``football``). :func:`manifest` lists what was downloaded with its hash.

Coordinates returned by the parsers are metres with the origin at the centre spot, ``x`` along the long side.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["SOURCES", "cache_dir", "fetch", "manifest", "MetricaMatch", "metrica_match", "skillcorner_ball",
           "skillcorner_matches", "statsbomb_shots", "STATSBOMB_360"]

METRICA = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
SKILLCORNER_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
SKILLCORNER_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"
STATSBOMB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

SOURCES: dict[str, dict[str, str]] = {
    "metrica": {"name": "Metrica Sports sample data (2 matches, 25 Hz tracking + events)",
                "url": "https://github.com/metrica-sports/sample-data",
                "license": "free sample data; acknowledge Metrica Sports when used publicly"},
    "skillcorner": {"name": "SkillCorner open data (10 A-League 2024/25 matches, 10 Hz broadcast tracking)",
                    "url": "https://github.com/SkillCorner/opendata", "license": "MIT (credit SkillCorner)"},
    "statsbomb": {"name": "StatsBomb open data (events + 360 freeze frames)",
                  "url": "https://github.com/statsbomb/open-data",
                  "license": "StatsBomb public data user agreement (non-commercial; credit 'Data: StatsBomb')"},
}

#: competition_id, season_id of the StatsBomb seasons with 360 data used by the xG study (men's tournaments)
STATSBOMB_360 = {"world_cup_2022": (43, 106), "euro_2020": (55, 43), "euro_2024": (55, 282)}

PITCH = (105.0, 68.0)


def cache_dir(root: str | Path | None = None) -> Path:
    """The football cache folder (created)."""
    import os
    base = Path(root or os.environ.get("PINNEAPPLE_DATA_CACHE", Path.home() / ".cache" / "pinneapple" / "datasets"))
    d = base / "football"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(root) -> Path:
    return cache_dir(root) / "manifest.json"


def manifest(root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """URL -> {"path", "sha256", "bytes"} of every file fetched into the cache."""
    p = _manifest_path(root)
    return json.loads(p.read_text()) if p.exists() else {}


def fetch(url: str, rel: str, root: str | Path | None = None, *, timeout: float = 120.0) -> Path:
    """Download ``url`` once to ``<cache>/rel`` and record its SHA-256; return the local path.

    Raises ``OSError`` when offline and the file is not cached (callers and tests turn that into a skip).
    """
    path = cache_dir(root) / rel
    if path.exists() and path.stat().st_size > 0:
        m = manifest(root)
        if url not in m:                       # placed by hand or by an older run: hash it now
            data = path.read_bytes()
            m[url] = {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            _manifest_path(root).write_text(json.dumps(m, indent=1, sort_keys=True))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pinneapple-football"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if data.startswith(b"version https://git-lfs"):
        raise OSError(f"{url} returned a Git LFS pointer, not the file")
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    m = manifest(root)
    m[url] = {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    _manifest_path(root).write_text(json.dumps(m, indent=1, sort_keys=True))
    return path


# --------------------------------------------------------------------------------------------------------- Metrica
@dataclass
class MetricaMatch:
    """One Metrica sample match in metres (origin at the centre spot, x along the long side).

    ``players[team]`` is ``(T, n, 2)`` with NaN when a player is off the pitch; ``ball`` is ``(T, 2)``;
    ``events`` is the event table (pandas) with positions converted to metres.
    """

    game: int
    t: np.ndarray
    period: np.ndarray
    players: dict[str, np.ndarray]
    names: dict[str, list[str]]
    ball: np.ndarray
    events: Any
    fps: float = 25.0
    gk: dict[str, int] = field(default_factory=dict)

    def attack_sign(self, team: str, period: int) -> float:
        """+1 when ``team`` attacks towards +x in ``period`` (from the goalkeepers' mean x in the first frames)."""
        sel = np.where(self.period == period)[0][:250]
        gx = np.nanmean(self.players[team][sel, self.gk[team], 0])
        return 1.0 if gx < 0 else -1.0


def _metrica_tracking(path: Path):
    import pandas as pd
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    names, xy = [], []
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            names.append(c)
            xy.append(raw.iloc[:, i:i + 2].to_numpy(float))
    bi = cols.index("Ball")
    ball = raw.iloc[:, bi:bi + 2].to_numpy(float)
    P = np.stack(xy, axis=1)
    return raw["Period"].to_numpy(int), raw["Time [s]"].to_numpy(float), names, P, ball


def _to_metres(a: np.ndarray) -> np.ndarray:
    out = np.empty_like(a, dtype=float)
    out[..., 0] = (a[..., 0] - 0.5) * PITCH[0]
    out[..., 1] = -(a[..., 1] - 0.5) * PITCH[1]
    return out


def metrica_match(game: int, root: str | Path | None = None) -> MetricaMatch:
    """Metrica sample game 1 or 2: tracking of both teams, ball and events in metres (downloaded once, ~65 MB)."""
    import pandas as pd
    if game not in (1, 2):
        raise ValueError("Metrica CSV sample games are 1 and 2")
    base = f"{METRICA}/Sample_Game_{game}/Sample_Game_{game}_"
    paths = {k: fetch(base + f"{k}.csv", f"metrica/Sample_Game_{game}_{k}.csv", root)
             for k in ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData")}
    npz = cache_dir(root) / f"metrica/game{game}_tracking.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=False)
        period, t, ball = z["period"], z["t"], z["ball"]
        players = {"Home": z["home"], "Away": z["away"]}
        names = {"Home": list(z["home_names"]), "Away": list(z["away_names"])}
    else:
        period, t, hn, H, ball = _metrica_tracking(paths["RawTrackingData_Home_Team"])
        _, _, an, A, _ = _metrica_tracking(paths["RawTrackingData_Away_Team"])
        players = {"Home": _to_metres(H), "Away": _to_metres(A)}
        ball = _to_metres(ball)
        names = {"Home": hn, "Away": an}
        np.savez_compressed(npz, period=period, t=t, ball=ball, home=players["Home"], away=players["Away"],
                            home_names=np.array(hn), away_names=np.array(an))
    ev = pd.read_csv(paths["RawEventsData"])
    for a, b in (("Start X", "Start Y"), ("End X", "End Y")):
        m = _to_metres(ev[[a, b]].to_numpy(float))
        ev[a], ev[b] = m[:, 0], m[:, 1]
    # goalkeeper: the player nearest his own goal line in the first frame with everyone on the pitch
    gk = {}
    for team, P in players.items():
        f0 = int(np.argmax(np.sum(np.isfinite(P[:, :, 0]), axis=1) >= 11))
        gk[team] = int(np.nanargmax(np.abs(P[f0, :, 0])))
    return MetricaMatch(game=game, t=t, period=period, players=players, names=names, ball=ball, events=ev, gk=gk)


# ----------------------------------------------------------------------------------------------------- SkillCorner
def skillcorner_matches(root: str | Path | None = None) -> list[dict]:
    """The ten SkillCorner open-data matches (``id``, teams, date)."""
    return json.loads(fetch(f"{SKILLCORNER_RAW}/matches.json", "skillcorner/matches.json", root).read_text())


def skillcorner_ball(match_id: int, root: str | Path | None = None) -> dict[str, np.ndarray]:
    """Ball track of a SkillCorner match at 10 Hz: ``frame, t, period, x, y, z, detected`` (NaN when missing).

    The ~90 MB JSON-lines tracking file is downloaded once and reduced to a small ``.npz``; the raw file is kept
    only if ``keep_raw`` is set in the environment (``PINNEAPPLE_KEEP_RAW=1``).
    """
    import os
    npz = cache_dir(root) / f"skillcorner/ball_{match_id}.npz"
    if not npz.exists():
        rel = f"skillcorner/{match_id}_tracking_extrapolated.jsonl"
        p = fetch(f"{SKILLCORNER_LFS}/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl", rel, root)
        rows = []
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                b = d["ball_data"]
                ts = d.get("timestamp")
                sec = np.nan
                if ts:
                    h, m_, s = ts.split(":")
                    sec = 3600 * float(h) + 60 * float(m_) + float(s)
                rows.append([d["frame"], sec, d.get("period") or 0,
                             *(np.nan if b.get(k) is None else b[k] for k in ("x", "y", "z")),
                             1.0 if b.get("is_detected") else 0.0])
        a = np.asarray(rows, float)
        np.savez_compressed(npz, a=a)
        if not os.environ.get("PINNEAPPLE_KEEP_RAW"):
            p.unlink()
    a = np.load(npz)["a"]
    keys = ("frame", "t", "period", "x", "y", "z", "detected")
    return {k: a[:, i] for i, k in enumerate(keys)}


# ------------------------------------------------------------------------------------------------------ StatsBomb
def _sb_json(rel: str, root) -> Any:
    url = f"{STATSBOMB}/{rel}"
    p = fetch(url, f"statsbomb/{rel}", root)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:                 # a truncated earlier download: fetch it again once
        p.unlink()
        m = manifest(root)
        m.pop(url, None)
        _manifest_path(root).write_text(json.dumps(m, indent=1, sort_keys=True))
        return json.loads(fetch(url, f"statsbomb/{rel}", root).read_text())


def statsbomb_shots(seasons: dict[str, tuple[int, int]] | None = None, root: str | Path | None = None,
                    *, max_matches: int | None = None) -> list[dict]:
    """Every open-play and set-piece shot with a 360 freeze frame in the given StatsBomb seasons.

    Each shot: ``match_id, season, x, y`` (StatsBomb yards, 120 x 80, attacking towards x = 120), ``goal`` (0/1),
    ``body_part, shot_type, technique, first_time, under_pressure, statsbomb_xg`` and ``freeze`` (list of
    ``(x, y, teammate, keeper)`` of the players visible in the 360 frame). The reduced table is cached as
    ``statsbomb/shots_<season>.json.gz`` so the ~3 GB of raw match files are read only once.
    """
    seasons = seasons or STATSBOMB_360
    out: list[dict] = []
    for name, (cid, sid) in seasons.items():
        cached = cache_dir(root) / f"statsbomb/shots_{name}.json.gz"
        if cached.exists() and max_matches is None:
            out += json.loads(gzip.decompress(cached.read_bytes()))
            continue
        shots = []
        matches = _sb_json(f"matches/{cid}/{sid}.json", root)[:max_matches]
        for m in matches:
            mid = m["match_id"]
            try:
                ff = {f["event_uuid"]: f for f in _sb_json(f"three-sixty/{mid}.json", root)}
            except OSError:
                continue
            for e in _sb_json(f"events/{mid}.json", root):
                if e["type"]["name"] != "Shot" or e["id"] not in ff:
                    continue
                s = e["shot"]
                if s["type"]["name"] == "Penalty":
                    continue
                shots.append({
                    "match_id": mid, "season": name, "x": e["location"][0], "y": e["location"][1],
                    "goal": int(s["outcome"]["name"] == "Goal"), "body_part": s["body_part"]["name"],
                    "shot_type": s["type"]["name"], "technique": s["technique"]["name"],
                    "first_time": bool(s.get("first_time", False)),
                    "under_pressure": bool(e.get("under_pressure", False)),
                    "play_pattern": e["play_pattern"]["name"], "statsbomb_xg": s["statsbomb_xg"],
                    "freeze": [(p["location"][0], p["location"][1], bool(p["teammate"]), bool(p["keeper"]))
                               for p in ff[e["id"]]["freeze_frame"] if not p.get("actor")],
                })
        if max_matches is None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(gzip.compress(json.dumps(shots).encode()))
        out += shots
    return out
