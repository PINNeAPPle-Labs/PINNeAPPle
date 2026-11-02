- `UNONdGrid` and `UniversalUNO` no longer crash with an `einsum` size error when a coarse level of the U has fewer
  Fourier modes than `modes` (for example, the default 12 modes on a 16-point grid with 2 levels): each spectral
  convolution keeps the modes the grid has (#1129).
