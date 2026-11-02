- The DDP gradient-compression hooks in `pinneapple_neural.trainer.hpc` work again with current PyTorch:
  - `register_topk_hook` returned a list from its future instead of a tensor, so the backward pass failed;
  - `register_powersgd_hook` passed `start_powersgd_iter` to `PowerSGDState`, which only accepts `start_powerSGD_iter`, so registering the hook failed.

  Both hooks are now tested in a single-process `gloo` group on CPU (#1129).
