- One theme file for every renderer: `pinneapple_tools/visualization/theme.json` carries the house brand inks, the
  two surfaces, the colour-blind-safe categorical palette, the colour maps with their stops and the quantity -> map
  registry, and the 2-D figures, the studio viewer, the twin3d viewer and the apps read it. `scripts/build_theme.py`
  generates the three browser copies and `--check` fails when one has drifted. `style.py` keeps no colour of its own;
  PyVista and the turntable draw on the theme's plot background. **Behavior:** the twin3d viewer opens on the house
  sequential map instead of turbo, and its `--viewport` now comes from the theme.
- `caption_from_figure(fig)` reads a caption back out of a drawn figure (colour bar, then axes, then title) and
  refuses one that declares no unit; the reference catalogue saves a caption sidecar beside every figure, and a
  figure without a unit fails that run instead of being catalogued as a picture of colours (#67).
