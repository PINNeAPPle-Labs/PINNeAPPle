`WallSpec(readout_lr=...)`: the wall readout can follow the field at its own, slower learning rate instead of either
co-adapting with it or being frozen. Measured on the 20-case holdout, co-adapting costs the near-wall consistency that
the evaluator's wall shear depends on (43.5 % against 30.9 % when the readout is frozen) while freezing it costs field
accuracy (velocity 4.31 % against 5.80 %); this is the knob between them.
