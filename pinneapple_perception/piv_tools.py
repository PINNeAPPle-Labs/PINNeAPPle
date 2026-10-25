"""Around :mod:`.video_piv`: reading frames from a video or image files, the vector-grid size a setting gives, a
summary of a PIV result, and window / search-margin advice from the expected displacement.

* :func:`decode_video` -- a video file to a (T, H, W) float grayscale array through ``ffprobe`` / ``ffmpeg``
  subprocesses (``FFmpegMissing`` when they are not on PATH).
* :func:`decode_image_frames` -- image files (PNG, JPEG, ...; Pillow) to (T, H, W), in file-name order.
* :func:`piv_grid_count` -- vectors per frame pair, with the same loop bounds as ``piv_velocity_field``: a window
  is used only when its whole search region fits in the image (half window + search margin from every border).
* :func:`summarize_piv` -- vector count, the windows withheld for border safety (against a grid that only needs
  room for the window), mean / max speed. ``piv_velocity_field`` has no correlation-quality (signal-to-noise)
  rejection; the only vectors it drops are those border windows.
* :func:`piv_parameter_guide` -- the one-quarter rule (Raffel, Willert, Wereley & Kompenhans, *Particle Image
  Velocimetry: A Practical Guide*, 3rd ed., Springer 2018): keep the particle displacement below about a quarter
  of the interrogation window, a search margin of 1.5 x the displacement, 50 % overlap.

Moved here from the PINNeAPPle-apps ``piv_analysis_api`` product (FlowTrace).
"""
from __future__ import annotations

import io
import math
import shutil
import subprocess
from collections.abc import Iterable

import numpy as np

__all__ = ["FFmpegMissing", "decode_video", "decode_image_frames", "piv_grid_count", "summarize_piv",
           "piv_parameter_guide"]


class FFmpegMissing(RuntimeError):
    """``ffmpeg`` or ``ffprobe`` is not on PATH."""


def decode_video(path: str) -> np.ndarray:
    """(T, H, W) float64 grayscale frames of the video at ``path`` (ffprobe for the size, ffmpeg for the pixels)."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise FFmpegMissing(f"{tool} is not installed: install ffmpeg or pass the frames as image files")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=s=x:p=0", path], capture_output=True, text=True)
    if probe.returncode != 0 or "x" not in probe.stdout:
        raise ValueError(f"not a readable video: {probe.stderr.strip()[:300] or 'no video stream'}")
    w, h = (int(v) for v in probe.stdout.strip().splitlines()[0].split("x"))
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                          capture_output=True)
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg could not decode the video: {proc.stderr.decode(errors='replace')[:300]}")
    n = len(proc.stdout) // (w * h)
    if n < 2:
        raise ValueError(f"the video has {n} frame(s); a velocity field needs at least 2")
    return np.frombuffer(proc.stdout[: n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.float64)


def decode_image_frames(files: Iterable[tuple[str, bytes]]) -> np.ndarray:
    """(T, H, W) grayscale frames from ``(file name, bytes)`` pairs, sorted by file name (use zero-padded names)."""
    from PIL import Image

    frames, shape = [], None
    for name, data in sorted(files, key=lambda f: f[0]):
        try:
            arr = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float64)
        except Exception as e:  # noqa: BLE001 -- any decoder error, named per file
            raise ValueError(f"{name!r} is not a readable image: {e}") from e
        if shape is not None and arr.shape != shape:
            raise ValueError(f"{name!r} is {arr.shape}, the earlier frames are {shape}: all frames need one size")
        shape = arr.shape
        frames.append(arr)
    if len(frames) < 2:
        raise ValueError(f"{len(frames)} frame(s); a velocity field needs at least 2")
    return np.stack(frames)


def piv_grid_count(height: int, width: int, window_size: int, search_margin: int, step: int | None = None,
                   *, margin: int | None = None) -> tuple[int, int]:
    """(rows, cols) of interrogation windows ``piv_velocity_field`` uses on an H x W frame."""
    step = step or window_size
    m = (window_size // 2 + search_margin) if margin is None else margin
    rows = len(range(m, height - m, step)) if height - m > m else 0
    cols = len(range(m, width - m, step)) if width - m > m else 0
    return rows, cols


def summarize_piv(seq: dict, frame_shape: tuple, window_size: int, search_margin: int,
                  step: int | None = None) -> dict:
    """Counts and speeds of a ``piv_velocity_sequence`` result on frames of shape (T, H, W)."""
    T, H, W = frame_shape
    pairs = max(T - 1, 0)
    u, v = np.asarray(seq["u"]), np.asarray(seq["v"])
    speed = np.hypot(u, v)
    r, c = piv_grid_count(H, W, window_size, search_margin, step)
    rn, cn = piv_grid_count(H, W, window_size, search_margin, step, margin=window_size // 2)
    return {"frame_pairs": pairs, "vectors": int(u.size), "vectors_per_pair": r * c,
            "withheld_near_border": max((rn * cn - r * c) * pairs, 0),
            "rejection": "border windows only (whole search region must fit); no correlation-quality rejection",
            "mean_speed": float(speed.mean()) if speed.size else 0.0,
            "max_speed": float(speed.max()) if speed.size else 0.0,
            "mean_u": float(u.mean()) if u.size else 0.0, "mean_v": float(v.mean()) if v.size else 0.0}


def piv_parameter_guide(displacement_px: float, width_px: int, height_px: int) -> dict:
    """Window size, search margin and step for an expected displacement (px per frame), with warnings."""
    if displacement_px <= 0 or width_px <= 0 or height_px <= 0:
        raise ValueError("displacement and image size must be positive")
    sizes = (8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512)
    side = min(width_px, height_px)
    min_window = 4.0 * displacement_px
    window = next((s for s in sizes if s >= min_window), sizes[-1])
    if window > side // 2:
        window = max(8, side // 2)
    margin = max(4, int(math.ceil(1.5 * displacement_px)))
    step50 = max(window // 2, 4)
    r50, c50 = piv_grid_count(height_px, width_px, window, margin, step50)
    r0, c0 = piv_grid_count(height_px, width_px, window, margin, window)
    warnings = []
    if min_window > side // 2:
        warnings.append(f"the quarter rule asks for a {int(min_window)} px window, more than half the image "
                        f"({side} px): the displacement may be too large to resolve; a higher frame rate helps")
    if r50 * c50 < 4:
        warnings.append(f"only {r50 * c50} vectors per frame pair at 50 % overlap ({r0 * c0} without): "
                        "try a smaller window if the seeding allows")
    return {"window_size": window, "search_margin": margin, "step_50pct_overlap": step50, "step_no_overlap": window,
            "vectors_50pct_overlap": {"rows": r50, "cols": c50, "total": r50 * c50},
            "vectors_no_overlap": {"rows": r0, "cols": c0, "total": r0 * c0},
            "rule": "displacement < window / 4, search margin 1.5 x displacement, 50 % overlap (Raffel et al. 2018)",
            "warnings": warnings}
