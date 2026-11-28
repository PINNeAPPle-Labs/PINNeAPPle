- `pinneapple_neural/lpm.py`, `pinneapple_neural/lpm_pipeline.py`, `benchmarks/lpm_campaign.py`: a fit now survives
  being killed. `fit(..., checkpoint_dir=...)` rewrites `fit.pt` at most every `checkpoint_every_s` seconds (and after
  the last epoch) with the weights, the optimizer and scheduler state and the loss history, and a later call that finds
  it continues from the next epoch; the file is deleted when the fit reaches `epochs`, and a checkpoint whose recipe
  (designs, epochs, batch, learning rate, seed) does not match the call is ignored, so one run's history is never
  reported as another's. `train(..., resume=True)` — what the campaign now passes — turns it on. This is the answer to
  the process dying with no crash report and no jetsam entry: the wing fit had reached epoch 128 of 600 on a 2.3 h run
  that then had to start over. Verified on the wing's own data on MPS: killed at epoch 2, resumed at epoch 3, history
  `[0.302553, 0.3069, 0.305914, 0.281506]` — the first two entries are the pre-kill ones — and the finished fit left no
  `fit.pt` behind. `tests/test_lpm.py::test_fit_resumes_from_its_checkpoint` pins the resume, the history continuity,
  the cleanup and the mismatched-recipe case.
