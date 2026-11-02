"""Build the Blender extension zip: ``python -m pinneapple_blender.step_extension.build --out dist [--wheels]``.

The zip holds the extension, a copy of :mod:`pinneapple_design.geometry.brep` under ``vendor/brep`` (so Blender needs
no PINNeAPPle install) and, with ``--wheels``, the wheels of smlibpy, mapbox-earcut and scipy for every platform
Blender ships on, listed in the manifest so Blender installs the right ones (Blender bundles numpy).
Install the zip with Edit > Preferences > Get Extensions > Install from Disk, or
``blender --command extension install-file -r user_default <zip>``.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BREP = HERE.parents[1] / "pinneapple_design" / "geometry" / "brep"
EXTENSION_FILES = ("__init__.py", "addon.py", "core.py", "blender_manifest.toml")
WHEELS = ("smlibpy", "mapbox-earcut", "scipy")
PLATFORMS = {                     # Blender platform tag -> pip platform tags
    "linux-x64": ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"],
    "windows-x64": ["win_amd64"],
    "macos-arm64": ["macosx_11_0_arm64", "macosx_12_0_arm64", "macosx_14_0_arm64"],
    "macos-x64": ["macosx_10_9_x86_64", "macosx_10_13_x86_64", "macosx_11_0_x86_64"],
}


def download_wheels(dest: Path, python: str = "3.11", platforms=PLATFORMS, packages=WHEELS) -> list[Path]:
    """pip-download one wheel per package and platform (no dependencies: Blender bundles numpy)."""
    got = []
    for tags in platforms.values():
        for pkg in packages:
            cmd = [sys.executable, "-m", "pip", "download", pkg, "--no-deps", "--only-binary=:all:",
                   "--python-version", python, "-d", str(dest)]
            for t in tags:
                cmd += ["--platform", t]
            subprocess.run(cmd, check=True, capture_output=True)
    for w in sorted(dest.glob("*.whl")):
        got.append(w)
    return got


def build(out: str | Path, wheels: bool = False, python: str = "3.11") -> Path:
    """Write ``pinneapple_step-<version>.zip`` into ``out`` and return its path."""
    text0 = (HERE / "blender_manifest.toml").read_text()
    manifest = {k: re.search(rf'^{k} = "([^"]+)"', text0, re.M).group(1) for k in ("id", "version")}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pinneapple_step"
        root.mkdir()
        for f in EXTENSION_FILES:
            shutil.copy2(HERE / f, root / f)
        (root / "vendor").mkdir()
        (root / "vendor" / "__init__.py").write_text('"""A copy of pinneapple_design.geometry.brep."""\n')
        shutil.copytree(BREP, root / "vendor" / "brep", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        text = (root / "blender_manifest.toml").read_text()
        if wheels:
            wd = root / "wheels"
            wd.mkdir()
            listed = [f'"./wheels/{w.name}"' for w in download_wheels(wd, python)]
            text = text.replace("# wheels = []", "wheels = [\n  " + ",\n  ".join(listed) + ",\n]")
            text = text.replace('[permissions]', 'platforms = ["' + '", "'.join(PLATFORMS) + '"]\n\n[permissions]')
        (root / "blender_manifest.toml").write_text(text)
        z = out / f"{manifest['id']}-{manifest['version']}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(root).as_posix())
    return z


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="dist")
    ap.add_argument("--wheels", action="store_true", help="bundle smlibpy, mapbox-earcut and scipy per platform")
    ap.add_argument("--python", default="3.11", help="Blender's Python version (4.2 to 5.x: 3.11)")
    a = ap.parse_args(argv)
    print(build(a.out, a.wheels, a.python))


if __name__ == "__main__":
    main()
