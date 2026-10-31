- Axial-flux machine and magnet eddy loss in 3-D (issue #929, fourth step).
  - `machines3d.magnet_loss`: eddy loss of a segmented magnet in a uniform normal field, 3-D against the
    closed-form resistance-limited loss (double sine series) and the 2-D strip value. Example
    `examples/electrodynamics/23_magnet_eddy_segmentation.py`: 3-D matches the closed form within -0.6 to +4.8 %
    where the loss is resistance-limited; 2-D overestimates short pieces (eight axial pieces keep 1.3 % of the 2-D
    loss); at 50 kHz the whole magnet is inductance-limited and two axial pieces lose more than one.
  - `machines3d.afpm`: `AFPM3D`, an axial-flux PM machine (single-sided or double-rotor) on a structured grid
    closed round the axis, no helical bias (zero torque on the aligned rotor), with quasi-3D radial slices on the
    same grid. Example `examples/electrodynamics/24_afpm_quasi3d.py`: quasi-3D overestimates the cogging by 11 to
    12 % and the tooth flux linkage by 2 to 3 %, from the fringing at the inner and outer radius (one and five
    slices agree within 0.3 %).
  - Flow nodes `em.magnet_eddy_segmentation` and `em.afpm_quasi3d`.
