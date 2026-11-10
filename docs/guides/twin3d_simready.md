# NVIDIA SimReady assets in PINNeAPPle

The adapter in `pinneapple_twin3d.simready` discovers and downloads individual
OpenUSD SimReady packages from Hugging Face. It is opt-in, downloads no assets
at import time, and preserves USD files and dependencies rather than flattening
simulation semantics into browser geometry.

Source: https://huggingface.co/datasets/nvidia/simready-assets
Official SimReady documentation:
https://docs.omniverse.nvidia.com/simready/latest/simready-central.html

## Installation and example

```bash
pip install huggingface_hub
```

```python
from pinneapple_twin3d.simready import resolve_revision, list_assets, download_asset

revision = resolve_revision()  # resolve once, save this immutable commit
assets = list_assets(revision=revision, limit=20)
print([(a.asset_id, a.package_path) for a in assets])

# Review the source dataset and individual asset licensing first.
if assets:
    package = download_asset(
        assets[0], destination="data/simready",
        acknowledge_license=True,
    )
    print(package)
```

This downloads the selected asset folder and records
`pinneapple_provenance.json` with dataset, revision, asset and license.
The dataset is CC BY 4.0; retain attribution to NVIDIA and check individual
package metadata and third-party dependencies before redistribution.
Do not commit large assets to the PINNeAPPle repository.

## Boundaries and safety

- The adapter does **not** claim that an asset passed SimReady validation.
- It does **not** convert OpenUSD physical metadata into PINNeAPPle PDE models.
- It does **not** import or expose NVIDIA trademarks as PINNeAPPle branding.
- It does **not** use unpinned `main` for downloads.
- The existing Twin3D GLB/LOD viewer requires an explicit, separately validated
  USD-to-GLB conversion before an imported SimReady asset can be displayed.
- The browser optimization path must never substitute a decimated visual mesh
  for collision, CFD, FEA, or other physics-authoritative geometry.

Future work: parse SimReady package profiles, validate metadata and dependency
closure, map physical properties into Twin3D, provide a licensed conversion
pipeline, and benchmark with a reproducible asset subset.
