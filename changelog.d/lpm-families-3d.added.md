- The five design spaces, three of them in three dimensions (`pinneapple_neural.lpm_families_3d`): the same pipeline,
  the same card, the same refusal to claim more than was measured. `pp_wing3d` is a swept, tapered, twisted wing at a
  compressible free stream — a 3-D vortex lattice on the camber surface, strip-coupled to the 2-D Hess-Smith solution
  of the wing's own section, carried to Mach by Prandtl-Glauert, with Cp over 960 surface points. `pp_fin3d` is an
  extruded-fin heat sink under a chip of known power: 3-D conduction with a Robin boundary by the immersed finite-cell
  method, temperature over 950 surface points. `pp_bracket3d` is an L-bracket under a tip load: displacement and von
  Mises by 3-D linear elasticity on conforming quadratic tetrahedra, the path verified against CalculiX.
- `Wing3D.surface_measure` returns the panel length each predicted point represents, so a prediction can be
  integrated into the load it stands for: a field is only a force once it is summed with the measure of the surface.
- Three checks that do not come from the families: the wing's composed pressure integrates section by section to the
  load the lattice asked for, its lift obeys Prandtl's lifting-line law as the aspect ratio grows, and Cp at Mach is Cp
  at zero Mach over beta; the heat sink balances its energy; the bracket is linear in its load and converges with the
  mesh. The immersed cut-cell grid was tried for the bracket and rejected on measurement: order one on a thin bending
  plate moved 55 % when the grid halved, against 0.9 % for conforming P2 tetrahedra.
- `pp_nozzle3d` and `pp_exchanger3d`, the last two design spaces in three dimensions. `pp_nozzle3d` is a scarfed
  (vectoring) nozzle: the exit plane is cut at an angle, so the expansion ratio, the exit Mach and the wall pressure
  all vary with the azimuth, and the field is Mach and p/p_chamber over 128 points of a surface of revolution. Its
  reference is sector-wise quasi-1-D isentropic expansion — each meridional sector is its own streamtube and the
  sectors do not communicate, which is the assumption to read twice, because the non-uniformity that communication
  leaves behind is exactly the loss a scarf produces. `pp_exchanger3d` is the tube sheet of a shell-and-tube
  exchanger: nine bores, the hot tube-side film on the bore walls, the cool shell-side film on the faces and the rim,
  and the temperature through the metal that a thermal-stress analysis would take as its input; 3-D conduction by the
  immersed finite-cell method, 194 surface points.
- Checks that come from outside the two new families: with the scarf at zero every sector must be the same sector and
  must carry the closed-form area-Mach answer; the A/A*(M) round trip from the Mach they report must return the area
  ratio of the station; the shortest sector must be the slowest and the longest the fastest (and, since Mach is convex
  in the area ratio, the mean is *not* below the unscarfed value — the test says so instead of asserting otherwise);
  the tube sheet must balance its energy, run hotter on the bores than on the faces, cool when the shell side gets
  faster, and refuse bores that touch.
