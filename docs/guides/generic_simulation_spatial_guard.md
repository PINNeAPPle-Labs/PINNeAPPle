# Generic simulation and sampling guards (issues #1673, #1674)

This is an **additive MVP**, not full completion of either issue. It does not
convert arbitrary OpenFOAM/CalculiX/FEniCSx field arrays to a common mesh,
support MPI or claim the CFD studies have been rerun.

## Native-case execution

```python
from pathlib import Path
from pinneapple_simulation.external_solvers.generic_contract import SimulationSpec, simulate

# Existing OpenFOAM case with system/controlDict and a solver on PATH:
result = simulate(SimulationSpec("openfoam", Path("/work/foam_case"),
                                 options={"solver": "interFoam", "timeout": 7200}))
# Existing CalculiX job beam.inp:
result = simulate(SimulationSpec("calculix", Path("/work/beam"), case="beam",
                                 options={"threads": 4, "timeout": 3600}))
# Trusted in-process FEniCSx variational solve (caller owns DOF/mesh export):
result = simulate(SimulationSpec("fenicsx", Path("/work/fenics")),
                  fenicsx_solve=my_dolfinx_solve)
```

Missing binaries, missing input decks, unsupported execution modes, nonzero
exit codes and missing CalculiX artifacts raise explicit exceptions. A
completed result does **not** mean scientifically verified.

## Prevent incompatible scientific comparisons

```python
from pinneapple_analysis.spatial_sampling import (
    SpatialReference, SamplingProtocol, require_compatible,
    absolute_distance, liquid_peak,
)
origin = SpatialReference("inlet", (-7.429,), ("y",), location="cell_center")
protocol = SamplingProtocol(origin, "k", "m2/s2", "peak", (20, 30),
                            phase="liquid", phase_alpha_min=0.9)
require_compatible(protocol, other_protocol)  # raises on mismatch
distance_from_inlet = absolute_distance(-7.429 + 1.364, -7.429)
idx, peak_k = liquid_peak(k_values, alpha_values)
```

The canal investigation established offsets of 1.364 m (slag), 1.819 m
(pig iron), and up to 3.637 m (pig iron near wall). The gate also rejects
peak-versus-mean, 0–30 versus 20–30 s, mismatched phase filters, node-versus-
cell-center and different spatial frames. Alignment must be explicit and
auditable. This is a **methodology compatibility check**, not evidence that
the magnitudes of turbulent kinetic energy agree.

## Next requirements before resolving the issues

- Canonical topology/field/time/units/provenance conversion, including
  nodal vs cell and integration-point results.
- Full capability negotiation, safe resource-isolated workers, MPI and
  arbitrary native-case support with version-aware preflight.
- Hook the sampling gate into existing VeriPhysics decisions and Engineering
  API dataset/benchmark comparison paths, not only the new standalone API.
- Reproduce the blast-furnace channel and valve experiments with actual
  OpenFOAM logs, solver versions, mesh checks and quantitative acceptance.
- Run `pytest -q tests/test_generic_simulation_spatial.py` and real-solver
  integration CI; these have not been executed in the GitHub connector session.

The source campaigns and unresolved k discrepancies remain documented in
PINNeAPPle-CFD issues #156–#158. The globe-valve implementation is in
`pinneapple_systems/process_components/valve_cfd.py`.
