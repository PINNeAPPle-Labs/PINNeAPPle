# SciREX/JAX integration (experimental)

This adapter is intentionally optional and does not import JAX at package import.
SciREX upstream: https://github.com/zenteiq-research/scirex (LGPL-3.0-only).
PINNeAPPle remains Apache-2.0; retain upstream license and notices when distributing.

## Status
- Implemented: model discovery (FNO/GINO/Transolver), model initialization,
  inference entry point, layout/unit/coordinate validation, training-only normalization,
  FNO supervised training loop, native SciREX checkpoints with a provenance manifest.
- Not implemented: geometry-aware GINO/Transolver training, full optimizer state resume,
  production worker scheduling, automatic VeriPhysics checks, cross-framework model
  parity, checkpoint loading, AirfRANS/Pipe/Darcy benchmarks.
- No physical validity is implied by successful model initialization or inference.

## Install
Use an isolated Python 3.10-3.12 worker with compatible SciREX 1.0 dependencies:
`pip install 'scirex==1.0.0' pytest`

## Minimal example
```python
import numpy as np
from pinneapple_integrations.scirex_backend import FieldSpec, ModelSpec, ScirexAdapter
spec = ModelSpec("fno", dict(n_modes=(8, 8), hidden_channels=16,
                             n_layers=2, in_channels=1, out_channels=1))
field = FieldSpec("channels_first", 1, ("Pa",), spatial_dim=2)
adapter = ScirexAdapter(spec, field)
x = np.ones((2, 1, 16, 16), dtype=np.float32)
adapter.initialize(x[:1])
print(adapter.predict(x).shape)
```

## Verification gates
1. Split data by geometry/parameter regime before fitting any normalization.
2. Check coordinate reference frame, SI units, channel order and mesh connectivity.
3. Test conservation, PDE residuals, boundary conditions, OOD and independent references.
4. Report wall time, JIT compilation time, device memory, error and confidence.
5. Treat any `verification_status=not_run` as unverified, never as approved.
