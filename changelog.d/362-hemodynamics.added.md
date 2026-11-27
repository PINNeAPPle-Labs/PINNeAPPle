- **Haemodynamics from 4D-flow MRI, measured on a phantom (#362)**: `pinneapple_physics.hemodynamics` carries
  Womersley's closed form for pulsatile flow in a rigid straight artery (velocity, flow rate, wall shear stress and
  pressure gradient from Bessel functions of the Womersley number), synthesises the 4D-flow acquisition the modality
  would make of it -- voxel centres across the radius, cardiac phases, independent noise per plane, no sample at the
  wall -- and recovers the pressure gradient and the wall shear stress from that with an ensemble of axisymmetric
  Navier-Stokes PINNs whose no-slip, axis symmetry and periodicity are built into the ansatz. Reported against the
  closed form over four synthetic vessels and one five-harmonic waveform, together with the two-point difference a
  4D-flow pipeline starts from and the same errors as a function of the voxel size and of the noise.
  `examples/hemodynamics_4dflow` writes the numbers and four figures; `docs/guides/hemodynamics_4dflow.md` reads them,
  including where the wall shear stress stops being recoverable.
