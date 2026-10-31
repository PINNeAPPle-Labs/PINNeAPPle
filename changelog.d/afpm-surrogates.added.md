- Surrogates of the axial-flux machine in 3-D with calibrated uncertainty and VeriPhysics checks (issue #929, sixth
  step): `machines3d.afpm_surrogate` (`run`, `simulator`, `space`, `physics_checks`, `certify`) and
  `examples/electrodynamics/25_afpm_surrogates.py`. 48 designs over magnet arc, slot opening, gap and magnet
  thickness, each checked in 3-D on the Axial-Flux Motor Designer's grid (54 s each); Gaussian process, MLP ensemble
  and quadratic surface compared with cross-conformal 90 % intervals and physics checks; each selected surrogate
  certified into a trust decision and an evidence ledger. Tooth flux linkage and its quasi-3D / 3-D ratio: APPROVED
  (test and extrapolation `R^2` 0.999 or better); cogging peak-to-peak: REVIEW (extrapolation `R^2` 0.855); the
  cogging ratio: REJECT (test `R^2` 0.756, a ratio of small numbers near a cogging minimum).
