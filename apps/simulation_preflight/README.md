# Simulation Preflight

**Before spending hours of solver time, find out whether your simulation is set up correctly.**

Day 8 of the PINNeAPPle 30-app program. A simulation that fails after hours of processing is money and time thrown
away; worse is one that finishes and is wrong. Upload the case before running it:

| You give it | You get |
|---|---|
| An OpenFOAM case (folder or zip: `0/`, `constant/`, `system/`), or a CalculiX / Abaqus `.inp` deck with its `*INCLUDE` files | **PASS / WARNING / FAIL**, a pre-flight summary per section (mesh, materials, boundary conditions, initial conditions, solver settings), and every problem with its severity, file and line (with the lines shown), an explanation, a fix, and what the real solver does with it; checklist export (Markdown, CSV, JSON) |

```
SIMULATION PRE-FLIGHT
──────────────────────────────
✓ Mesh found
✓ Boundary conditions
✓ Solver settings

✗ Missing thermal conductivity in material STEEL

Overall: FAIL
```

## Rules

| Section | OpenFOAM | CalculiX / Abaqus |
|---|---|---|
| Mesh | polyMesh present and readable; checkMesh-equivalent quality (non-orthogonality, skewness, volumes, regions) | nodes/elements; inverted elements; element quality |
| Materials | `nu` present and plausible (unit errors), non-Newtonian coefficients, thermophysical entries, `g` for buoyant solvers | every element has a section; sections point to defined sets and materials; the properties each procedure needs (elastic, density for modes/gravity, conductivity and specific heat for heat transfer, expansion for thermal loads); Poisson's ratio; **E vs density unit consistency** |
| Boundary conditions | a condition for every patch in every field (exact, group or regex entries), constraint types (empty, wedge, symmetry, cyclic), required entries (`value`, `inletValue`, ...), pressure reference, walls (no-slip, wall functions) | sets referenced by `*BOUNDARY` and loads exist; **every connected part supported against rigid-body motion** (all three directions, not a single point); a temperature sink for heat transfer; loads present |
| Initial conditions | every required field present (from the solver and turbulence model), turbulence fields not zero, field sizes match the mesh, `0.orig` vs `0` | initial temperatures for transient heat and thermal stress |
| Solver settings | `controlDict` (time step, end time, number of steps), relaxation for SIMPLE, convection schemes for every transported field, linear solvers for every solved field (`Final` for PISO/PIMPLE), non-orthogonal correctors vs mesh, **Courant number estimated before the run** | a procedure and `*END STEP` per step, increment control, number of modes, output requests, contact interactions |

## Validation (`tests/test_preflight.py`)

Each example is a real case or a copy broken in one way, run through the actual solver (OpenFOAM v1912, CalculiX 2.21)
to record what happens (logs in [`examples/solver_logs`](examples/solver_logs)):

| Case | Solver | Preflight |
|---|---|---|
| pitzDaily (simpleFoam, k-ε), as shipped | converged in 282 iterations | PASS |
| outlet entry deleted from `0/k` | stops: "Cannot find patchField entry for outlet" | FAIL `0/k`, patch outlet |
| outlet pressure zeroGradient | stops: "Unable to set reference cell for field p" | FAIL, `system/fvSolution` |
| epsilon initialised to 0 | floating-point exception, first iteration | FAIL `0/epsilon` |
| plain SIMPLE, no relaxation | floating-point exception, first iteration | FAIL `system/fvSolution` |
| cavity (icoFoam), time step × 4 | runs at Courant 3.4, transient not time-accurate | WARNING, Courant ≲ 4 (upper bound) |
| cavity, `nu` deleted | stops: "Entry 'nu' not found" | FAIL `constant/transportProperties` |
| cantilever (ccx, NLGEOM) | converged, tip 7.57 mm | PASS |
| heat transfer, no `*CONDUCTIVITY` | warns, aborts while factoring, no results | FAIL, material STEEL, line 1681 |
| loaded but not supported | "Job finished", **tip displacement 1.8e11 mm** | FAIL, no supports |
| modal, E in MPa and density in kg/m³ | **first frequency 0.16 Hz instead of 209 Hz**, no warning | FAIL, inconsistent units |
| `*BOUNDARY` on an undefined set | stops: "node set CLAMP" | FAIL, line 1685 |
| one element with reversed nodes | stops: "nonpositive jacobian determinant in element 1" | FAIL, element 1 |
| 40 elements outside the section | stops: "no material was assigned to element 638" | FAIL, elements 601-640 |

Two of these the solver never reports. The model without supports and the inconsistent units both run to the end.

## Run

```bash
pip install -e . -r apps/simulation_preflight/requirements.txt          # from the repository root
cd apps/simulation_preflight && uvicorn simulation_preflight.api:app --port 8087
```

Environment: `PFL_USER` / `PFL_PASSWORD` (HTTP Basic login), `PFL_MAX_MB` (default 200), `PFL_MAX_HEAVY` (default 2).
Docker: `docker build -f apps/simulation_preflight/Dockerfile -t simulation-preflight .`; deployment with the other
apps in [`apps/deploy`](../deploy) (service `preflight`).

## API and library

`POST /api/preflight` (multipart `files`) returns `status`, `sections`, `findings` (section, status, title, detail,
fix, file, line, entity, evidence, snippet) and `info`. Library: `from pinneapple_data.preflight import preflight`.
