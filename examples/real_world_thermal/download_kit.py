#!/usr/bin/env python3
"""Download public KIT measurement files, verify Zenodo-published MD5 checksums.

No synthetic fallback. Uses the record's public version-pinned file endpoints.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

BASE = "https://zenodo.org/records/21157950/files/"
FILES = {
    "Exp1_tot.csv": "b9f2c7f40d5a243ce8a05021ef9308ce",
    "Exp2_tot.csv": "5bd5bdf5e1a1d5b573717ceb7332094b",
    "Metadata.json": "ff2d40c52453e63697876e50033dab25",
}


def download(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, expected in FILES.items():
        destination = directory / name
        temporary = directory / (name + ".part")
        if destination.exists():
            data = destination.read_bytes()
            if hashlib.md5(data).hexdigest() == expected:
                manifest[name] = {"md5": expected, "sha256": hashlib.sha256(data).hexdigest()}
                continue
            raise ValueError(f"Existing file checksum mismatch: {destination}")
        try:
            with urlopen(BASE + name + "?download=1", timeout=120) as response, temporary.open("wb") as out:
                import shutil
                shutil.copyfileobj(response, out)
            data = temporary.read_bytes()
            actual = hashlib.md5(data).hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch for {name}: {actual} != {expected}")
            temporary.replace(destination)
            manifest[name] = {"md5": actual, "sha256": hashlib.sha256(data).hexdigest()}
        finally:
            temporary.unlink(missing_ok=True)
    import json
    (directory / "checksums.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Verified public experimental files at", directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/kit_thermal"))
    args = parser.parse_args()
    download(args.output)
