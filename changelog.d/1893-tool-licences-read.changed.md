The last ten tool licences in the stage map are read (#1893): Gmsh, SALOME, FluidX3D, Genesis, Newton Physics,
OpenRadioss, REBOUND, CGNS, Blender and Matplotlib carried `license_verified=False`, so no delivery decision could
be made on them. Each was read at the place its `license_source` now names, with the date -- the project's own
LICENSE file, its licence page, or, for OpenRadioss, the archived mirror of the distribution since
openradioss.org and its repository were withdrawn in 2026-10 -- and the row records what the text implies for
commercial use and for distribution. Two rows changed rather than being confirmed: REBOUND's header says "version 3
or any later version", so it is `GPL-3.0-or-later`, and Matplotlib's agreement is the Matplotlib Development
Team's own, not the BSD licence the row called it. FluidX3D forbids commercial use outright, which is why
`Tool.is_open` now answers `False` for a custom non-commercial licence. `stages.coverage()['totals']
['license_unverified']` is 0. All ten are also covered by a test that keeps the `is_open` answer honest.
