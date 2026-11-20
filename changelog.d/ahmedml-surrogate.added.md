`pinneapple_simulation.pipelines.ahmedml.surface_model`: the surface surrogate the AhmedML contract asks for on the two
fields this machine can reach — the pressure and the three-component wall shear, as two heads on one point trunk over
normalised centroids, normals and the release's eight geometric parameters. The two quantities are 500× apart
(1.5·10⁻³ m²/s² of typical shear against a pressure of order 1), so each head is normalised by its own scale: the
AirfRANS wall readout cost three runs to learn that a small field sharing a head with a large one does not train.
`fit_surface_model` weights every polygon by its published area, because that is the weight the metric uses;
`normalise_case` reads geometry only, so the frame at fit time and at prediction time cannot disagree, and
`predict_case` needs no truth at all. Eight tests cover the reader against a written-back PolyData, both reductions, the
vector rule, the macro average, and a two-case fit that has to beat the training mean.
