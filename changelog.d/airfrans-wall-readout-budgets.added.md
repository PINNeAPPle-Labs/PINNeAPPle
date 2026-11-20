`WallSpec(init_from=..., freeze_trunk=..., freeze_readout=...)` and the matching `train_v2.py` flags: the field and the
wall readout can be trained on separate budgets. Freezing the readout while the field trains keeps the near-wall layer
consistent with the flat-plate law by construction and gives the campaign's best wall shear on the 20-case holdout
(43.5 % → 30.9 %) and best `Cd R²` (−5.03 → −0.97), at 4.31 % → 5.80 % velocity; refitting the readout on a frozen trunk
afterwards barely moves it, and refitting it on a trunk trained with a co-adapting readout makes it much worse (38.5 % →
75.4 %). The wall shear is a property of the pair, not of the readout.
