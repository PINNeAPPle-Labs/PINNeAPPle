`pinneapple_simulation.pipelines.airfrans_benchmark.clip_wall_gradients`: the gradient clip now acts on the residual
trunk and the wall readout against separate norms. A single `clip_grad_norm_` over every parameter measured ~140 at
initialisation — the force term dominates the trunk — so its 0.007 scale factor was applied to the slope head as well
and trained it at 1/140 of the learning rate. The head could not fit its own training cases (356 % relative error on
the training set, correlation 0.24 with the truth), which is what kept the AirfRANS wall-shear component above its cap
and the drag R² negative. With the two groups clipped apart, the same readout that reaches 30 % when trained alone now
trains inside the model.
