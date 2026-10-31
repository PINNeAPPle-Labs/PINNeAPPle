- Axial-Flux Motor Designer app, `apps/afpm_designer` (issue #929, fifth step): a design form, a quasi-3D estimate
  (cogging, tooth flux linkage, back-EMF constant per tooth turn) and "Verify in 3-D", which solves the same design
  in full 3-D on the same edge-element grid and gives the estimate / 3-D ratios, with a 3-D view of `|B|`. On the
  default design (coarse grid) the estimate overstates the cogging by 13.1 % and the flux linkage by 2.1 %; the
  check takes 21 s. Library: `machines3d.afpm_design` (`estimate`, `verify_3d`, `grid_for`, `field_view_of`,
  `machine_from`), `machines3d.afpm.quasi3d_slices`, and `AFPM3D(edges="fraction")`: magnet and tooth edges as
  shares of the cells they cut, so the magnet arc and the slot opening are the ones asked for on any grid (the
  default, `"cell"`, moves them in whole grid steps).
