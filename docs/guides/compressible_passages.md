# Compressible flow in an internal passage, solved and checked

A valve, an orifice, a nozzle, a venturi and a pipe reducer are the same CFD problem: a bore that
runs along one axis and changes radius, a gas, and a pressure drop. This guide is how PINNeAPPle
solves that class of problem — not one valve — and how it decides whether the answer may be used.

```python
from pinneapple_simulation.pipelines.compressible_internal import (
    Conditions, StudySpec, globe_valve, orifice_plate, converging_nozzle, run_study)

passage = globe_valve(lift=4.0)                       # or orifice_plate(hole_ratio=0.5),
                                                      #    converging_nozzle(area_ratio=4)
spec = StudySpec(passage, gas="methane",
                 conditions=Conditions(p_out_bar=60.0, T_in_K=300.0, p_in_target_bar=70.0))
out = run_study(spec, "runs/valve4mm", sweep_values=[0.5, 1.0, 2.0, 4.0])
print(out["summary"])          # one line with the flow, Kv, dT and how many checks passed
print(out["decision"])         # what failed, what had no reference
```

`run_study` writes the CSV tables, the figures, a Blender render and a video, and returns the
`checks` list the Flow node `cfd.compressible_passage` turns into evidence.

## What the pipeline is made of

| module | what it holds | why it is separate |
|---|---|---|
| `passage` | `Ring`, `Span`, `PassageSpec`, the `polyMesh` writer | the geometry is a span list; the mesher knows no valve |
| `physics` | `Gas` (Peng-Robinson + Sutherland), `Conditions`, `GASES` | the gas reaches the dictionaries, the checks and the tables from one place |
| `case` | the OpenFOAM case, `checkMesh`/`rhoSimpleFoam` runner | one writer for any passage |
| `post` | patch averages, mass flow, Kv, CSV | the definitions live in one file |
| `checks` | the physics checks | each one falsifiable, none passed by omission |
| `visual` | the `Scene`, the Blender render, the video | the same scene serves the page and the report |
| `studies` | `globe_valve`, `orifice_plate`, `converging_nozzle` | adding a passage is a new function here and nothing else |

## The passage

A passage is a list of `Span`s along `x`, each with the radial `Ring`s it is meshed with and its
axial cell size. `Ring` carries the radii at both ends of the span, so a band can taper: a cone is
one span, not a staircase. Two neighbouring spans must share their radii at the join — where they
do not, the extra faces become walls, which is what a shoulder in the bore is. `PassageSpec.problems`
refuses a spec that breaks that before any meshing happens.

```python
from pinneapple_simulation.pipelines.compressible_internal import PassageSpec, Ring, Span

# a sudden contraction: 40 mm bore to 20 mm, 60 mm of each, then 10 diameters of outlet
duct = (Ring(0.5, 20.0, 12),)
wide = (Ring(0.5, 20.0, 12), Ring(20.0, 40.0, 10))
passage = PassageSpec(spans=(
    Span("inlet",  -240.0,  0.0, wide, 3.0),
    Span("throat",   0.0, 60.0, duct, 2.0),
    Span("outlet",  60.0, 260.0, duct, 3.0),
), n_theta=24, sector_deg=90.0)
print(passage.problems())     # [] when it can be meshed
```

The axis is a 0.5 mm core with a `symmetry` patch rather than a degenerate line of points: every
cell stays a hexahedron and `checkMesh` is clean. The run uses a sector (90° by default) closed by
two `symmetryPlane` patches, which is exact for a solid of revolution; integral quantities are
scaled to the whole passage and that scaling is stated wherever it is used.

## The conditions, and why the flow is imposed

| setting | default | note |
|---|---|---|
| `p_out_bar` | 60 | static, `fixedValue` at the outlet |
| `T_in_K` | 300 | static, `fixedValue` at the inlet |
| `mass_flow_kg_s` | `None` | `None` estimates it from `Cd A sqrt(2 rho dP)` and the outer loop refines it |
| `p_in_target_bar` | `None` | when set, `solve_to_pressure_drop` iterates the flow onto this inlet pressure |
| `scheme` | `linearUpwind` | second order; `upwind` is available for a stubborn mesh |
| `turbulence` | `kOmegaSST` | `laminar` is available, and is what the checks tolerate best |

Imposing static pressure at *both* ends of a small domain makes the first pressure correction take
the whole start-up imbalance. On the 73 cm³ valve of CFD-08 the pressure jumped to 3.3·10⁸ Pa in one
iteration and the run died. With the flow imposed and the inlet pressure read back, the same case
converges in about a thousand iterations and the loop lands on the pressure drop that was asked for.

## The checks

| check | compares | passes when |
|---|---|---|
| `mesh` | `checkMesh` verdict | the mesh is one region and `Mesh OK` |
| `converged` | the residual floor | `SIMPLE solution converged` |
| `conservacao.de.massa` | mass in against mass out | relative imbalance < 0.1 % |
| `entalpia.de.estagnacao` | `h0` across an adiabatic passage | variation < 1 % |
| `regime.de.escoamento` | `p_out/p_in` against the critical ratio, and `Ma_max` | subsonic when not choked, `Ma_max > 0.9` when choked |
| `joule.thomson` | the outlet temperature against the library's isenthalpic path | within 1 K of GERG-2008 (or Peng-Robinson, and it says which) |
| `densidade.gas.real` | the inlet density against Peng-Robinson | within 2 % (the ideal gas is printed alongside) |
| `coeficiente.de.fluxo` | the run's `Kv` against IEC 60534-2-1 | `Cd` between 0.4 and 1.0 |

Two of them catch a wrong valve result without any CFD at all, which is why they run against any
solution: a Mach number that the pressure ratio cannot produce, and a temperature drop larger than
the isenthalpic path allows. See `PINNeAPPle-CFD/casos/cfd08-valvula-globo` for both, measured.

## The pictures

`visual.build_scene` returns a `pinneapple_twin3d.Scene` (walls revolved to four copies, one
meridian cut coloured by the cell it bounds). `visual.render` renders it with a real Blender
(`blender --background`, Cycles) and `visual.render_video` orbits it into an MP4 with `ffmpeg`. The
same scene exports to the interactive viewer, to OpenUSD and to the Studio, so the page and the
report show the same object.

## From the app

The Flow node `cfd.compressible_passage` wraps `run_study`; the Capability is registered with the
keywords a person would write in Portuguese or English, so `pinneapple-engineering-api` finds it with
no code of its own. The app's job budget (300 s per job, 900 s per flow) is smaller than a converged
valve, so the node defaults to a coarse mesh and ~900 iterations; the full study is the case folder.
