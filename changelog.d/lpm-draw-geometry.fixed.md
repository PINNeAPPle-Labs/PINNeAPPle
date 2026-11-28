- `pinneapple_neural/lpm_pipeline.py`, `pinneapple_neural/lpm_families_3d.py`: two reasons the campaign stopped at nine
  of eleven design spaces. **`draw` re-drew the wrong thing**: a rejection can be about the geometry — "the bores touch"
  in `pp_exchanger3d` is decided before any operating point exists — and re-drawing only the operating point spent all
  60 attempts on one impossible plate and raised. The geometry is re-drawn too, and the error names the constraint.
  **`pp_fin3d` left NaN on half its surface**: a fin is thinner than the background cell in most of that design space (a
  1 mm fin on the 8 mm grid the cost budget allows, with the gap as wide as the fin), so its faces fall in cells the
  grid classifies as outside the body. The reference now halves the grid until the whole surface reads back finite —
  about one design in fifty needs it — and raises with the cell size if even that is not enough.
