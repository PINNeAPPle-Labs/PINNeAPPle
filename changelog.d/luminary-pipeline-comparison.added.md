- `docs/strategy/LUMINARY_PIPELINE_VS_OURS.md`: the Luminary Cloud pipeline for their Large Physics Models, read from
  their five technical posts, next to ours stage by stage and with file-and-line references, and the differences grouped
  by nature — data scale (400–900 geometries against 700–5 000 high-fidelity simulations), where the data comes from
  (their own GPU solver and parametric CAD against our analytic/FEM/VLM references), architecture, transfer learning
  (they fine-tune a pretrained SHIFT model, we train from zero), confidence (a calibrated 0–1 score with a published
  shift table against our distance ratio), and product. It also lands one lesson that is ours as much as theirs:
  `_multiscale` recomputes `torch.cdist` on every forward over designs that never change, which is the same bottleneck
  their neighbour-index cache removed for a 50× speed-up per epoch.
