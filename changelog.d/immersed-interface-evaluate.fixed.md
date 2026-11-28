- `pinneapple_simulation/immersed/assembly.py`: `Discretization.evaluate` no longer returns NaN for a point that sits
  exactly on the immersed interface. The grid is shifted off the axes a part is usually aligned with, but a face can
  still land on a grid plane, and then the surface samples on that face are located in the **empty** cell just outside
  the body (`dofs.index == -1`) and read back as NaN. `pp_fin3d` lost 105 of its 950 samples this way — all on the
  outermost fin's x- face — and the family aborted with "non-finite values in the geometry or the field", taking the
  campaign from 11 design spaces to 9. Those points are now read from the nearest active cell in the 3^d neighbourhood,
  with the basis extrapolated across the face by the distance the point actually sits outside; a point with no active
  cell anywhere near it is still NaN, which is what a point outside the body deserves.
