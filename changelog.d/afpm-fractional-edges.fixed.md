- Axial-flux machine in 3-D (issue #929): the Axial-Flux Motor Designer took the magnet arc and the slot opening to
  whole grid cells and turned the rotor by even grid steps only. On its coarse grid the magnet arc moved in steps of
  1/12 of the pole pitch (the default design's cogging jumped from 0.52 to 11.67 N m between an arc of 0.79 and
  0.80) and the cogging was sampled at 4 angles per period, which put its peak-to-peak 29 % low.
  `AFPM3D(edges="fraction")` gives the cells on a magnet or tooth edge their share of remanence and of iron and
  turns the rotor by any angle; the designer uses it with 16 angles per period. Default design, coarse grid: cogging
  8.60 N m estimated, 7.63 N m in 3-D (was 11.67 and 10.37 N m), ratio 1.128 (was 1.125); flux linkage 1.063 and
  1.041 mWb (was 1.083 and 1.060). The guide's comparison of the two grids of the axial-flux study is corrected:
  they carried different magnet arcs (83.3 % and 79.2 %) and 4 against 8 rotor angles per period.
