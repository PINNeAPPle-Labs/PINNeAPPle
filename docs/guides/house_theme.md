# One house theme for every 2-D figure

Roadmap item #918. Before this, each example chose its own look: 7 of the 88
`examples/` folders imported the style module, the rest used matplotlib defaults
or a palette copied from somewhere. Two figures of the same quantity ended up in
different colours, and the default CFD map was a rainbow, which is not
perceptually uniform: a reader cannot rank two colours by value, and a
colour-blind reader cannot separate some of them at all.

`pinneapple_tools.visualization.style` now carries one theme. Import it, call
`use_theme()` once, and take the colour of a field from `field_cmap`.

```python
from pinneapple_tools.visualization import use_theme, field_cmap, colorbar, label, save_figure

use_theme("report")                    # or "slide" for a projected deck, "paper" for print
...
mesh = ax.pcolormesh(x, y, temperature, cmap=field_cmap("temperature"))
colorbar(ax, mesh, "Temperature", "K")
label(ax, x=("x", "m"), y=("y", "m"), title="Heated plate, steady state")
save_figure(fig, "figures/plate")      # PNG at 200 dpi plus a vector SVG
```

## What `use_theme` sets

- one font stack (DejaVu Sans, then Helvetica, then Arial) and body sizes that
  stay readable at 50 % zoom;
- the Okabe-Ito categorical palette, which stays distinguishable under
  deuteranopia, protanopia and tritanopia;
- perceptually uniform sequential, diverging and cyclic maps, monotone in
  lightness so a grey-scale print keeps the ordering;
- grid and spine style, and figure sizes for 1-, 2- and 3-panel layouts through
  `make_panel_figure(ncols)`;
- DPI 200 for PNG plus a vector copy, through `save_figure`.

`use_theme` rejects an unknown kind instead of silently doing nothing.

## One quantity, one colour

`field_cmap(quantity)` reads a registry, not a guess. `"temperature"`,
`"pressure"`, `"error"` and the other magnitudes resolve to the sequential map;
`"vorticity"`, `"pressure_coefficient"` and the other signed quantities to the
diverging one; `"angle"` and `"phase"` to the cyclic one. An unregistered
quantity raises, because a silent fallback is exactly how two figures of the
same quantity end up with different colours. To add a quantity, add a line to
`FIELD_CMAPS`.

## One theme file, three renderers

Roadmap item #65. The colours above do not live in `style.py` any more:
`pinneapple_tools/visualization/theme.json` is the file, and everything that
draws reads it through `pinneapple_tools.visualization.theme`.

```json
"brand":  {"ink": "#262626", "blue": "#0081be", "ember": "#a23a00"},
"surface": {"light": {"plot": "#e9ecf1", "paper": "#ffffff", ...},
            "dark":  {"plot": "#0b1220", "paper": "#1a1a2e", ...}},
"categorical": ["#0072B2", "#E69F00", ...],
"colormaps": {"cfd_sequential": {"kind": "sequential", "stops": [...]}, ...}
```

| Renderer | Reads it through |
|---|---|
| matplotlib / PyVista figures | `style.py` (`use_theme`, `field_cmap`, `CMAPS`), `pyvista_bridge.BACKGROUND` |
| the studio viewer and the apps (`/studio`) | `studio/web/theme.js`, generated, imported by `studio-core.js` |
| the twin3d viewer (`pinneapple_twin3d.Viewer`) | `pinneapple_twin3d/viewer/theme.js`, generated, imported by `viewer.js` |
| the apps' page chrome and charts | `apps/_shared/static/shared/theme.css`, generated, imported by `app.css` |

A browser page cannot read the JSON, and `pp.viz`'s single-file page inlines its
modules, so the file is copied into those three places in their own language by
`scripts/build_theme.py`:

```bash
python scripts/build_theme.py            # rewrite the three generated copies
python scripts/build_theme.py --check    # the gate: fail when one has drifted
```

The copies are checked in, so a page needs no fetch to draw. `--check` is what
keeps "one theme file" true: a colour changed in `theme.json` and not rebuilt is
a consumer drawing yesterday's palette.

The plot background is the theme's, and there is one of it: `#0b1220` dark,
`#e9ecf1` light, for the PyVista renders, the studio viewer and the twin3d
viewer alike. The twin3d page no longer declares `--viewport` in its own CSS;
`viewer.js` writes it from the theme, so the canvas and the page around it agree.

The brand face, Oxanium, is used where the site uses it — the app title — and
only if the machine has it: no webfont is shipped with the library, so the stack
falls back to the system sans.

`tests/test_house_theme_shared.py` checks both ends of that seam: that the
matplotlib maps are the file's, that the generated copies are in step, that a
stop means the same colour in Python and in the browser to 1/255, and that the
palette stays separable under simulated protanopia, deuteranopia and tritanopia
(the same Viénot/Brettel/Mollon approximation `scripts/render_house_theme.py`
draws with; the worst pair of the palette is 0.127 in linear RGB, the test's
floor is 0.10).

## Helpers

| Helper | Does |
|---|---|
| `colorbar(ax, mappable, quantity, unit)` | colour bar that refuses to exist without its quantity and unit; `""` prints `[-]` |
| `label(ax, x=(name, unit), y=(name, unit))` | axis labels in the same convention |
| `annotate_verdict(ax, text, ok)` | pass/fail stamp that carries a word and a marker as well as a colour |
| `panel_letters(fig)` | (a), (b), (c) in reading order |
| `source_note(fig, "Data: …")` | provenance line; refuses an empty string |
| `save_figure(fig, path)` | PNG at 200 dpi and an SVG |

## Colour-blind check

![Palette and maps under simulated colour vision](assets/house_theme/palette.png)

The figure above is generated by `scripts/render_house_theme.py`, which applies
the three dichromacy matrices of Viénot, Brettel and Mollon (1999) to the
palette and to each map. Run it to regenerate; the tests in
`tests/test_house_theme.py` assert the properties the picture is meant to show,
so a change that breaks them fails before the picture is looked at.

## Migration

`cfd_rainbow` stays registered and `use_cfd_style()` still applies the older
dark theme, for comparison pictures against commercial CAE. It is no longer the
default: `DEFAULT_CMAP` is `cfd_sequential`.

The examples migrate in batches, and both halves are mechanical — 146 scripts
under `examples/` import matplotlib, 37 call `use_theme` and 102 are still on the
debt list — so the edit is one anybody can review line by line, and the gate
reads the files so the number only goes down.

```bash
python scripts/apply_house_theme.py examples/a.py examples/b/c.py   # migrate
python scripts/apply_house_theme.py --check                         # the gate
python scripts/apply_house_theme.py --baseline                      # after a batch
```

The codemod adds two lines where the module imported pyplot — after the
top-level import block, or inside the `try` block that imported matplotlib when
the module has one, so the call runs exactly where matplotlib did:

```python
from pinneapple_tools.visualization.style import use_theme
use_theme("report")
```

It also deletes a `plt.rcParams.update(...)` that only restates what the theme
sets (font size 10, no top or right spine): after the theme that line is a
second place to edit, not a choice. `tests/test_house_theme_gate.py` compares
that table against the theme's own `rc`, so the two cannot drift.

The gate fails on an example that draws and calls neither `use_theme` nor
`use_cfd_style`, unless it is listed in `examples/.house_theme_baseline`. It
fails the other way too: a baseline line for a file that is themed, or that no
longer exists, has to go. The baseline is the debt, and it only shrinks — 102
examples are left after three batches.

What is left is not a special case. Of the 102, two draw with PyVista
(`use_cases/fin_convection_inverse/plots_2d3d.py`,
`use_cases/physics_data_factory/ultrarealistic_digital_twin.py`) and eight write
an animation (`FuncAnimation`, a GIF, or a frame loop); the rest is plain
matplotlib that enters a batch like the ones before it. The two families wait
for a batch of their own because they need a second thing the 2-D batches did
not: the first a PyVista scene on the theme's surface, the second a map whose
stops the frames share with the figures beside them.

### Batches

| Batch | Examples | What changed |
|---|---|---|
| 1 | `first_example`, `pinn_solver/03_symbolic_pde_hard_bc`, `physics_learning_verification/run_study`, `physics_query/run_demo`, `electrodynamics/{20,21,22,23}` | 8 scripts on the theme, 4 redundant `rcParams.update` lines removed |
| 2 | `electrodynamics/01`–`06`, `geometry/07_csg_domain_demo`, `getting_started/01`–`10` | 17 scripts; each one run on both sides for the sheet (`getting_started/04` trains for over an hour and was left running past the check window) |
| 3 | `topology_optimization/01`–`03`, `control/{cartpole_swing_up,submarine_depth}`, `time_series/adaptive_forecasting`, `geophysics/01_sphere_and_assimilation`, `electrodynamics/17_prius_tolerances`, `geometry_library/02_wind_turbine_bem`, `motor_magnetics/01_spm_cogging_and_torque`, `physics_audit/pipe_regimes` | 12 scripts; the run and the sheet are one command (`scripts/run_theme_batch.py`), which is the whole cost of a batch now. The debt went from 114 to 102 |

![Batch 1, before and after](assets/house_theme/batch1_antes_depois.png)

![Batch 2, before and after](assets/house_theme/batch2_antes_depois.png)

![Batch 3, before and after](assets/house_theme/batch3_antes_depois.png)

The two runs and the sheet are one command, `scripts/run_theme_batch.py`: it copies each example into two
scratch trees (the branch's file and the same file out of `origin/main`), runs both with matplotlib on Agg,
picks the figure each side wrote, and calls `scripts/render_theme_migration.py` with the pairs. An example that
fails on either side is reported and left out of the sheet, so a batch never looks complete when it is not.

The sheet itself puts the same figure from two runs — the branch and `origin/main` — side by side at the same
height. What it shows is that the content did not move: same curves,
same axes, one look.

