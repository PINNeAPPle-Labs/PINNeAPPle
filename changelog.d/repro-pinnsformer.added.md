- `pinneapple_neural.architectures.pinns.PINNsFormerPaper`: the PINNsFormer of Zhao, Ding & Prakash (ICLR 2024),
  following the official code: a pseudo sequence of one point and its k time-shifted copies, an encoder-decoder
  and WaveAct.
  - It is pointwise, with 453 561 parameters at the paper's settings.
  - Registered as `pinnsformer_paper`.
  - The existing `PINNsFormer` is a different model (attention across collocation points). Its registry capability
    is now `"sequence"`, and its docs point to the new class.
- `examples/reproductions/pinnsformer_zhao2024`: the PINNsFormer reproduction, with frozen targets, a checkpointing
  runner and measured results.
  - Table 1 at seed 0: 6 of 12 targets reproduced, 3 partially reproduced and 3 not reproduced.
  - Wave (Table 2) is not attempted because of its CPU cost.
