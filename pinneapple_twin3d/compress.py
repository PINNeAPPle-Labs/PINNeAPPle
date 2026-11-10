"""Optional meshoptimizer/glTFpack browser-asset compression.

Only geometry.glb and generated visual LOD files are optimized. Authoritative
USD sources, field buffers and physics metadata are never rewritten.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def optimize_visual_assets(
    folder: str | Path, *, gltfpack: str = "gltfpack",
    texture_mode: str = "none", timeout: int = 600,
) -> dict:
    """Generate meshopt-compressed GLBs using an installed gltfpack executable.

    The caller must configure MeshoptDecoder in the Three.js GLTFLoader before
    loading compressed assets. Existing geometry and physics data are untouched.
    """
    if texture_mode not in ("none", "ktx2"):
        raise ValueError("texture_mode must be 'none' or 'ktx2'")
    executable = shutil.which(gltfpack)
    if not executable:
        raise FileNotFoundError("Install gltfpack (meshoptimizer) to compress visual assets")
    folder = Path(folder)
    manifest_path = folder / "scene.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = [manifest["geometry"]] + [x["file"] for x in manifest.get("visual_lods", [])]
    outputs = []
    for name in names:
        source = (folder / name).resolve()
        if source.parent != folder.resolve() or not source.is_file():
            raise ValueError(f"Invalid visual geometry path: {name}")
        target = source.with_name(source.stem + "_meshopt.glb")
        command = [executable, "-i", str(source), "-o", str(target), "-cc"]
        if texture_mode == "ktx2":
            command += ["-tc"]
        subprocess.run(command, check=True, timeout=timeout, capture_output=True, text=True)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"gltfpack did not produce a valid file: {target}")
        outputs.append({"source": name, "file": target.name,
                        "source_bytes": source.stat().st_size,
                        "output_bytes": target.stat().st_size,
                        "codec": "EXT_meshopt_compression"})
    manifest["visual_compressed"] = outputs
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"assets": outputs, "requires_meshopt_decoder": True,
            "requires_ktx2_loader": texture_mode == "ktx2"}


__all__ = ["optimize_visual_assets"]
