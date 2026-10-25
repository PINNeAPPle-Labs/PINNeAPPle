# FlowTrace

**Upload a video or an image sequence of a seeded flow; get a cross-correlation PIV velocity field back. Any camera.**

Imported from PINNeAPPle-apps (`piv_analysis_api`). PIV is `pinneapple_perception.video_piv` (FFT-normalised
cross-correlation per interrogation window, three-point parabolic sub-pixel peak; Raffel et al., *Particle Image
Velocimetry: A Practical Guide*); decoding (ffmpeg / Pillow), the vector-grid size, the summary and the
window / search-margin advice (the one-quarter rule) are `pinneapple_perception.piv_tools`. No other app does PIV.

## Run

```bash
pip install -e . -r apps/flowtrace/requirements.txt          # video uploads also need ffmpeg on PATH
cd apps/flowtrace && uvicorn flowtrace.api:app --port 8145
curl -F video=@flow.mp4 -F window_size=32 -F search_margin=16 localhost:8145/v1/analyze_sequence -o field.npz
curl "localhost:8145/v1/parameters_guide?approx_displacement_px=5&image_width_px=1920&image_height_px=1080"
```

`frames=@frame_000.png ...` instead of `video` (sorted by file name); `output=json` returns the summary and the NPZ
(x, y, t, u, v) as base64. Logins `FLOWTRACE_USER` / `FLOWTRACE_PASSWORD_HASH`; `FLOWTRACE_UPLOAD_MAX_MB` (200),
`FLOWTRACE_MAX_FRAMES` (200), `FLOWTRACE_FREE_ANALYSES` (100). Deployed as `flowtrace` (image with ffmpeg).

## Tests and what they show

`pytest tests/test_piv_tools.py tests/test_app_flowtrace.py` (6 + 3 pass, ffmpeg installed here). The source TODO
said FlowTrace had no test; its later `examples/vortex_case.py` (a solid-body vortex, particles moved by the exact
rotation) is now a test: 81 vectors, mean error **0.13 px/frame**, mean relative error **3.9 %** (max 16 %, near
the centre where the speed is small). A uniform 3 / -2 px shift uploaded as PNG frames comes back within 0.3 px.

## Known limitations

- The only vectors dropped are windows whose search region would cross the image border: no signal-to-noise
  (correlation-quality) rejection, so bad vectors are not flagged.
- Motion blur degrades the result. The library's `tests/test_perception.py` (real libx264 fixture
  `tests/fixtures/perception/known_shift_real.mp4`, lens distortion, frame blending) passes here (6 PIV tests);
  the per-case error figures quoted in the source README (0.09-0.14 px, 0.54 px with frame blending) are its own
  and were not re-printed here.
- One request is analysed synchronously; long videos are cut at `FLOWTRACE_MAX_FRAMES`.
