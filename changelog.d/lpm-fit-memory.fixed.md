- `pinneapple_neural/lpm.py`: `fit` backpropagates each chunk of designs as it goes instead of summing the loss of
  every chunk and calling `backward` once per epoch. The update is the same — the gradient of a sum is the sum of the
  gradients, and the optimizer still steps once per epoch — but only one chunk's graph is alive at a time. The old way
  kept the graphs of all 150 chunks of `pp_wing3d` (900 training designs of 960 points each, attention quadratic in
  them) simultaneously: measured peak **4.4 GB** and a macOS jetsam kill inside the first epoch, which is why the
  campaign's last three families never started. With one chunk at a time the same fit peaks at **1.6 GB** and runs at
  the same speed (21.9 s per epoch at batch 24). `tests/test_lpm.py::test_chunked_epoch_equals_one_chunk` pins that a
  chunked epoch and a single-chunk epoch produce the same loss history, and the `fit` docstring now says what
  `batch_designs` actually bounds.
