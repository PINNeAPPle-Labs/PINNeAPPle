"""End-to-end pipelines that chain external tools (CAD, mesher, solver) into one checked, reproducible run.

* :mod:`.freecad_openfoam`: a parametric FreeCAD model to an OpenFOAM result, sweeps and surrogates.
* :mod:`.usd_structural_surrogate`: an OpenUSD asset measured into a design family, solved by CalculiX and
  learned by a 3-D FNO, with the checks and the dashboard that go with it.
"""
