# An OpenUSD asset, CalculiX and a PhysicsNeMo FNO: the structural surrogate pipeline

`pinneapple_simulation.pipelines.usd_structural_surrogate` takes a part that exists in NVIDIA's OpenUSD content,
measures it, solves a family of variants with CalculiX, learns the family with a 3-D Fourier neural operator, checks
the operator against the solver, and serves it behind sliders.

The worked example, with its measured numbers, is [`examples/usd_structural_surrogate`](../../examples/usd_structural_surrogate).

## What the OpenUSD asset actually gives

NVIDIA's public USD library is a robotics and warehouse prop library. It ships **no** structural benchmark part, so
the honest reading of "get a geometry from OpenUSD" is to take the one mechanically meaningful asset in it: the
pedestal a UR10 arm is bolted to (`Props/Mounts/ur10_mount.usd`). `usd_asset.mount_interface` measures its mounting
plate — diameter, thickness, height — and those measured numbers become the anchors of the design family rather than
numbers typed into a script.

```python
from pinneapple_simulation.pipelines.usd_structural_surrogate import DesignSpace
from pinneapple_simulation.pipelines.usd_structural_surrogate.usd_asset import probe

iface = probe()                                   # fetch + read with usd-core + measure
space = DesignSpace.from_interface(iface)         # the family, anchored to the asset
```

Four design variables: plate thickness, column diameter, flare radius, load height. The base plate's diameter is
**not** one of them — it is the asset's, held fixed, so every design carries the same bolted footprint.

## One grid, from the solver to the dashboard

A neural operator needs every sample on the same lattice; a finite-element result lives on a mesh that changes with
every design. The pipeline fixes one box for the whole family and maps both ways:

| step | function | what it does |
|---|---|---|
| mesh → mask | `grid.occupancy` | exact point-in-tetrahedron test, KD-tree pre-filtered and bounding-box clipped |
| mesh → grid | `grid.rasterize_barycentric` | linear interpolation inside the element holding each cell |
| grid → mesh | `grid.sample_field` | trilinear, so a prediction answers a deck's nodes |
| gap filling | `grid.fill_outside` | outside cells take the nearest inside value |

Two of those deserve their reasons:

- **Barycentric, not inverse distance.** IDW is fine for displacement (0.6 % of peak, round trip) and not for a
  stress concentration: the largest node-level error of von Mises through IDW is 27 % of the peak, at the flare — the
  one number the surrogate is judged on.
- **The fill is not cosmetic.** The plate is 6 mm thick and a cell is ~9 mm, so it is *outside* cells that touch it.
  With zeros there, reading the field back at the nodes is off by 21 % of the peak displacement; with the fill, 0.4 %.
  Refining the grid changes neither number.

## The checks

`checks.compare` scores the operator on designs the training split held out: relative L2 per field, the **99.5th
percentile of von Mises** (the number an engineer reads), and the displacement on the bolted face, which has to be
zero whatever the model says. `checks.field_trust` hands the predicted field to
`pinneapple_analysis.trust.field_card.trust_card` against the CalculiX reference, so the deploy decision is the
library's and not this pipeline's. `checks.linearity` solves one design at two loads and measures whether the fields
really scale exactly — the assumption that makes the dashboard's load slider honest.

## Standing where `ccx` stands

`SurrogatePredictor` implements the `virtual_ccx` predictor interface, so the trained operator can replace the solver
executable for a pre-processor that only knows `ccx -i job`:

```bash
PINNEAPPLE_SURROGATE=run/usd_pedestal/surrogate pinneapple-ccx -i job --fallback
```

The deck's own mesh builds the occupancy channel, so the operator does not need to be told which design it is
answering. Outside the trained box it raises `OutOfEnvelope`; with `--fallback` the real `ccx` runs instead.

## The dashboard

```bash
python -m pinneapple_simulation.pipelines.usd_structural_surrogate.dashboard run/usd_pedestal/surrogate --port 8131
```

The sliders revolve the part's half-section (`design.revolve_surface`, microseconds — gmsh takes seconds and would
not be interactive) and sample the predicted fields on the surface, so nothing is cached between designs. Three
buttons, each a check rather than a flourish: **optimise** descends on the surrogate for the lightest design under a
stress limit; **solve for real** runs `ccx` on the geometry the sliders describe and puts its numbers beside the
prediction; **refine** fine-tunes the operator on that solved case.
