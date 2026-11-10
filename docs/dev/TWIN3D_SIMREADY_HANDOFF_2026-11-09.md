# Twin3D browser optimization + NVIDIA SimReady — handoff (2026-11-09)

**Status:** work preserved on GitHub branch `feat/twin3d-web-asset-optimization`, draft PR #1591. **Not merged. No successful CI, browser or real-asset run verified.**

## Implemented in this branch

- `pinneapple_twin3d/lod.py`: visual-only GLB LOD generation; preserves authoritative geometry and fields.
- `pinneapple_twin3d/viewer/viewer.js`: explicit `?lod=N` and experimental `?lod=auto` camera-distance selection, hysteresis, cached geometries, field-safety guard.
- `pinneapple_twin3d/simready.py`: optional Hugging Face SimReady dataset catalog with pinned SHA, selective download, explicit license acknowledgement and provenance JSON.
- `pinneapple_twin3d/simready_usd.py`: optional `usd-core` stage inspection, static polygon mesh triangulation and transformed visual import into Twin3D; does **not** certify or fully map physics.
- `pinneapple_twin3d/compress.py`: optional gltfpack Meshopt-compressed visual GLBs, separate from authoritative originals; browser decoder wiring is pending.
- Tests written but **not executed**: `tests/test_twin3d_lod.py`, `tests/test_twin3d_simready.py`, `tests/test_twin3d_simready_pipeline.py`.
- Guides: `docs/guides/twin3d_browser_assets.md`, `docs/guides/twin3d_simready.md`.

## Issues / remaining work

- #1593 adaptive LOD: initial implementation committed; still needs runtime validation and completion criteria.
- #1594 Draco/Meshopt/KTX2: offline Meshopt export begun; viewer support pending.
- #1595 instancing and deduplication.
- #1596 provenance, catalog and reproducible benchmarks.
- #1597 physics-safe simplification.
- #1605 full SimReady OpenUSD integration and validation.
- #1614 real SimReady asset end-to-end and CI verification.
- #1615 Meshopt/KTX2 decoder integration and fallbacks.
- #1616 full UsdPhysics metadata mapping and profile validation.
- #1617 instancing and performance measurements.

## Next steps before merging

1. Inspect and run Python unit tests, lint and JavaScript syntax/runtime checks; correct any implementation defects.
2. Download one **pinned, licensed** small NVIDIA SimReady package, verify its real package layout, import its USD stage, render in browser and retain evidence.
3. Wire Meshopt/KTX2 decoders; ensure compressed GLBs load and uncompressed fallback works.
4. Map and validate physics schemas while keeping original OpenUSD authoritative.
5. Measure triangle counts, geometric error, bytes, latency, FPS and VRAM on fixed hardware.
6. Review CI and only then mark PR ready for review and merge.

**Important limitations:** No benchmark numbers or passing tests are asserted. Fan triangulation of concave USD polygons is not guaranteed correct. Downloaded assets are not committed to the repository. The license and provenance must accompany any redistributed asset. The branch contains the implementation and the issue tracker contains follow-up work; no requirement to keep the ChatGPT session open.
