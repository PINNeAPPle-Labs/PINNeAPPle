`WallAwareNet(slope_prior=True)` and `flat_plate_slope`: the wall readout predicts a log-correction to the turbulent
flat-plate skin friction `Cf = 0.0576 Re_x^{-1/5}` at the predicted edge speed, instead of the slope itself, so the
magnitude comes from a law and only its shape has to be learned. The untrained law sits at 168 % relative error with a
magnitude ratio of 1.08 on the training wall points; on the 20-case `full` holdout the trained model brings the wall
shear to 38.5 % against 43.5 % without the prior, and `Cd R²` from −5.03 to −1.88. The release's viscosity is the same
for every case (1.54981e-05) and every airfoil is normalised to a unit chord, both read from the wall store, so the law
needs no new input channel.
