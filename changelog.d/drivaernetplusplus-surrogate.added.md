`pinneapple_simulation.pipelines.drivaernetplusplus.surface_model`: the surrogate side of the DrivAerNet++ contract,
ready for the geometry. `CaseGeometry` carries a case the way the contract hands it over (points, truth, dual areas,
design vector, dynamic head); `normalise_case` centres and scales the geometry by its longest side and puts the pressure
in dynamic heads, which is the only way a point-wise network sees comparable objects across variants and does **not** read
the truth's scale — `predict_case` refuses a case without a declared dynamic head for exactly that reason;
`SurfacePressureNet` is a point-wise MLP over the normalised coordinates with Fourier bands and the case's design vector
appended; `fit_surface_model` weights every point by its dual area, because the metric does.
