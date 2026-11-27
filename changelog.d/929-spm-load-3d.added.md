- `machines3d` at load (#929): `SPM3D.current_density(currents)` injects the stranded armature current of the
  slots as a second source beside the magnets, `SPM3D.solve(..., currents=)` and
  `SPM3D.torque_series(thetas, skew, currents_of=)` give the torque at a synchronous drive, `SPM3D.multislice`
  carries the same currents per slice, and `machines3d.load.load_study` measures the mean torque and the ripple of
  a skewed machine in 3-D against multi-slice 2-D on the same mesh. One-layer 3-D against the 2-D code: 0.03 % of
  the torque at 72 angular steps. `examples/electrodynamics/27_spm_load_3d.py`.
