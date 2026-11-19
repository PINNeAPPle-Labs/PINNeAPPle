# A structural surrogate from an OpenUSD asset, solved by CalculiX, trained with PhysicsNeMo

The UR10 mounting pedestal that ships inside NVIDIA's OpenUSD content, read with OpenUSD, meshed and solved with
CalculiX, learned by a 3-D Fourier neural operator, checked against the solver, and driven live from a dashboard.

```bash
python examples/usd_structural_surrogate/run.py all --out run/usd_pedestal     # ~15 min on a 10-core laptop
python examples/usd_structural_surrogate/run.py dashboard --out run/usd_pedestal
```

## Why this part

NVIDIA's public USD library — the `omniverse-content-production` bucket Isaac Sim streams assets from — is a robotics
and warehouse prop library: bins, mugs, tables, screws, robot arms. **It ships no structural-mechanics benchmark
part**, so "take a ready geometry from OpenUSD" cannot mean "download a bracket". What it does ship, under
`Props/Mounts/`, is the pedestal a UR10 arm is bolted to: a round base plate with a bolt circle, a column up to the
robot flange. That is a real mounting interface, it is a real structural part, and it is what this example analyses.

The numbers the family is built from are **measured from the asset**, not typed in:

| from `ur10_mount.usd`          | measured       | used as                                    |
|--------------------------------|----------------|--------------------------------------------|
| base plate diameter            | 271.9 mm       | fixed: every design has the same footprint |
| base plate thickness           | 6.0 mm         | the thinnest allowed plate thickness       |
| height                         | 515 mm         | the top of the `h` range                   |
| `metersPerUnit` / up axis      | 1.0 / Z        | read as millimetres, recorded in the manifest |

## The four design variables

| variable | meaning              | range      | nominal |
|----------|----------------------|------------|---------|
| `t`      | plate thickness      | 6–16 mm    | 6.0 mm (the asset) |
| `dc`     | column diameter      | 50–110 mm  | 60 mm   |
| `r`      | flare radius         | 6–35 mm    | 12 mm   |
| `h`      | load height          | 300–500 mm | 500 mm  |

The flare is a cone rather than a sharp re-entrant corner on purpose: a sharp corner makes the peak stress a
property of the mesh, and the whole point of the exercise is that the peak stress is a property of the design.

## The pipeline

1. **asset** — fetch, read with `pxr`, measure the mounting interface (`usd_asset.py`).
2. **campaign** — Latin hypercube over the four variables; each design goes through gmsh (`C3D4`) and `ccx`, and the
   nodal fields are sampled onto one fixed 40 × 40 × 56 grid (`campaign.py`, `grid.py`).
3. **train** — PhysicsNeMo's 3-D `FNO` (2 in-channels: occupancy and signed distance; 4 out: `u_x`, `u_y`, `u_z`,
   von Mises) under a `ComputeBudget` (`surrogate.py`).
4. **validate** — relative L2 per field, the **peak** von Mises, the clamped face, and the field trust card
   (`checks.py`).
5. **viz** — Twin3D scene to the browser viewer, to Blender and to `.usda` (`viz.py`).
6. **dashboard** — sliders, `ccx` on demand, and a fine-tuning step on the solved case (`dashboard.py`).

The dashboard in use, 72 s — the four sliders, CalculiX on the same design, and the optimiser on the operator:
[`media/dashboard.mp4`](media/dashboard.mp4).

## What the geometry enters as, and why a grid

Every sample has to live on the same lattice, so the box is fixed for the whole family and the geometry enters as a
field: `occupancy` and `signed_distance`. That is what lets a design the campaign never solved cost the same single
forward pass as one it did — the sliders move, the answer changes, and no mesh is generated.

The fields are stored **unmasked**: cells outside the part take the value of the nearest cell inside it
(`grid.fill_outside`). This is not cosmetic. The plate is 6 mm thick and a cell is ~9 mm, so it is *outside* cells
that touch it; with zeros there, reading the field back at the finite-element nodes is off by **21 %** of the peak
displacement. With the fill it is **0.4 %**, and refining the grid changes neither number.

## The operator stands where `ccx` stands

`SurrogatePredictor` implements the `virtual_ccx` interface: a `ccx`-shaped command line, an `.inp` in, an `.frd`
out, and an envelope. The deck's own mesh builds the occupancy channel, so any pre-processor that exports a design of
the family is answered without being told which design it is — and outside the trained box it refuses rather than
extrapolating:

```bash
PINNEAPPLE_SURROGATE=run/usd_pedestal/surrogate pinneapple-ccx -i job --fallback
```

## What it does not do

- **One load direction.** The fields are solved for a horizontal load at the top of the column and scale with it
  exactly (linear elastic, checked by `checks.linearity`). A load in another direction is outside the envelope.
- **Linear elastic only.** No plasticity, no contact with the floor, no bolt preload: the bottom face is fully
  clamped.
- **`C3D4` at one element size.** The peak von Mises is a discretisation of the true peak; no mesh-refinement study
  of the campaign's element size is reported here.
- **Plates thinner than a cell.** The grid's vertical spacing is ~9 mm and `t` goes down to the asset's 6 mm, so for
  the thinnest designs no cell centre lies inside the plate at all: its field is carried by `fill_outside` from the
  neighbouring cells instead of being resolved. The held-out errors include that, and a finer grid is the fix —
  not a wider tolerance.
