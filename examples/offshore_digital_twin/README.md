# Offshore Digital Twin — experimental seed

This is a **0D/1D lumped, quasi-steady hydraulic experiment**, not an operational digital twin.
It is a deliberately small reference implementation to begin the progressive-fidelity pipeline.
No sensor connection, BOP automation, multiphase transient solver, reservoir coupling, 3D scene,
trained surrogate, VeriPhysics integration, or field validation is claimed.

## Run

From this directory:

```bash
python -m unittest discover -s . -p 'test_*.py'
python -c "from offshore_twin import WellConfig,evaluate; print(evaluate(WellConfig()))"
python -c "from scenarios import generate_scenarios,manifest; print(manifest(generate_scenarios()))"
```

Dependencies: Python standard library only. All physical inputs are SI.
The `water_depth_m` field is metadata in this first vertical-well model, not a separate
hydrostatic water-column calculation.

## Fidelity ladder

1. 0D lumped hydraulic pressure model (implemented).
2. 1D spatial transient single-phase pressure/flow and energy conservation.
3. 1D coupled drillstring and riser dynamics, formation inflow and pressure boundaries.
4. 2D axisymmetric FEM/FVM and 3D CFD/FEA with geometry convergence.
5. Validated surrogates, state estimation, uncertainty, verified real-time visualization.

Every higher fidelity must match lower fidelity in its asymptotic limit, pass conservation
checks and include benchmark datasets with documented sources, units, boundary conditions,
mesh/time convergence and train/test leakage controls.

## Safety

For scientific research and educational experimentation only. Do not use for drilling,
pressure-control decisions, safety-critical automation, or field operations.
