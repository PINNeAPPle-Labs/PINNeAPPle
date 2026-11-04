Library helpers moved out of the October apps (#1226): `pinneapple_core.series.downsample` (min/max preserving, plus
the even and strided picks the library used), `pinneapple_tools.visualization.studio.cached_view` / `save_view` /
`prune_views`, `pinneapple_simulation.external_solvers.openfoam.find_case_root` / `extract_case` / `is_case`,
`pinneapple_data.doe.design_size` and `CampaignPlan.view()`, `write_campaign(..., archive=)`, and in
`pinneapple_simulation.meshless`: `Domain2D.perimeter()` / `estimate_nodes()`, `default_spacings`,
`point_cloud_from_table`, `domain_from_spec`, `MeshlessStudy.solutions_csv()`. The default stored-results path of
`inverse_problems.heat_transfer` now points at the checkout's `examples/` (it pointed one folder too high; the app
shim hid it), and `thermal.heatsink.surrogate.load_surrogates(directory)` no longer reads an app folder by default.
