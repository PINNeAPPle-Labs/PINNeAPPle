- The dependency licence record is complete: `docs/dev/dependency_licenses.json` carries a licence for **96 of the 101**
  declared dependencies (it had 32) and the table in `docs/licensing.md` is regenerated from it. The record was
  refreshed in an environment built for the job — Python 3.11 with the project installed extra by extra, so a failure
  names one extra instead of the whole list. Five are still not in the table, and the record says why: `emmiai-noether`
  (every published release needs Python >=3.12), `fenics-dolfinx` (no distribution on PyPI), `petsc4py` (sdist only,
  built against PETSc), and `fmpy` + `genesis-world`, which install but declare no licence in their metadata — those
  two are now listed apart, in `unreadable`, instead of being reported as "not installed here", which they are not.
  `asyncua` (LGPL-3.0-or-later, the `twin` and `digital_twin` extras) is the second copyleft dependency after `gmsh`
  and is written up as a decision in `docs/licensing.md`. The gate also stops calling a true record wrong when a
  different release of the same dependency spells the same licence differently — pandas 3.0.5 says `BSD License`,
  3.1.1 says `BSD-3-Clause` — while a copyleft or unreadable licence still fails on any release (issue #96).
