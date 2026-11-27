- **The four `pinneapple_data.cae` modules state their reason in the Flow coverage**: `api/core.json` now carries the
  structural CAE surface (#2518), so `pinneapple_flow.coverage` asks each of its modules for a node or a reason:
  `structural` (`StructuralRun` and its four constructors -- a run is an object a form cannot hold), `training_set`
  (deduplication, decimation, units and the split by geometry family over a campaign of runs), `lsdyna` (the open
  ASCII half of an LS-DYNA run: keyword deck, `nodout` history, the deck's mesh) and `vtk_series` (the legacy-VTK
  time series round trip). The gate goes back to 0 unreached.
