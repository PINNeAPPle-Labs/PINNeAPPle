`pinneapple_simulation.pipelines.drivaernetplusplus.surface`: the AirfRANS campaign's 3-D counterpart, restricted to the
one thing that dataset scores — the surface pressure at every native point. `read_legacy_polydata` reads the release's
binary VTK legacy surfaces (points, POINT_DATA scalars, polygons) and refuses what it cannot read instead of guessing;
`load_support` reads the released per-case support (truth `p`, `dual_area_m2`, profile indices);
`surface_pressure_metrics` computes both published reductions exactly — area-weighted and uniform relative L2, in
percent — and `aggregate_metrics` the declared macro average over geometries. The scoring side is checked against the
released support of `E_S_WWC_WM_008` (511 063 points, 32.3518 m² of dual area, `p` from −1 853.76 to 449.888 m²/s²),
fetched from the archive whose SHA-256 matches the manifest. The geometry side is deliberately absent: the release is
behind a Harvard Dataverse guestbook and the support carries no coordinates.
