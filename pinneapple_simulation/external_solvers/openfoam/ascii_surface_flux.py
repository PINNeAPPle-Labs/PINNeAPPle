"""Strict ASCII OpenFOAM surfaceScalarField -> complete face flux CSV.

Supports ASCII nonuniform internal scalar lists and uniform/nonuniform
boundary patch 'value' entries. Unsupported patch syntax fails closed.
Only scalar fields already expressed as integrated fluxes are accepted;
this does not convert velocity or density into flux.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from . import binary_reader


def _blocks(raw: str) -> list[tuple[str, str]]:
    """Read named top-level patch blocks from an OpenFOAM dictionary body."""
    result = []
    pos = 0
    while True:
        match = re.search(r"([A-Za-z_][\w.\-]*)\s*\{", raw[pos:])
        if match is None:
            if raw[pos:].strip(" \t\r\n;"):
                raise ValueError("unsupported trailing dictionary content")
            return result
        start = pos + match.end()
        if raw[pos:pos + match.start()].strip(" \t\r\n;"):
            raise ValueError("unsupported dictionary content")
        depth = 1
        end = start
        while end < len(raw) and depth:
            if raw[end] == "{":
                depth += 1
            elif raw[end] == "}":
                depth -= 1
            end += 1
        if depth:
            raise ValueError("unclosed patch block")
        result.append((match.group(1), raw[start:end - 1]))
        pos = end


def _scalar_list(entry: str, expected: int) -> np.ndarray:
    uniform = re.fullmatch(r"\s*uniform\s+([^;()]+)\s*;\s*", entry, re.S)
    if uniform:
        value = float(uniform.group(1))
        out = np.full(expected, value, dtype=np.float64)
    else:
        match = re.fullmatch(
            r"\s*nonuniform\s+List<scalar>\s+(\d+)\s*\(([^()]*)\)\s*;\s*",
            entry, re.S)
        if match is None or int(match.group(1)) != expected:
            raise ValueError("unsupported scalar list or wrong element count")
        tokens = match.group(2).split()
        if len(tokens) != expected:
            raise ValueError("scalar list element count mismatch")
        out = np.array([float(token) for token in tokens], dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("nonfinite face flux")
    return out


def _entry(body: str, key: str) -> str:
    match = re.search(
        rf"\b{re.escape(key)}\s+((?:uniform\s+[^;]+|nonuniform\s+List<scalar>\s+\d+\s*\([^()]*\)))\s*;",
        body, re.S)
    if match is None:
        raise ValueError(f"missing supported {key} entry")
    return match.group(1) + ";"


def _read_boundary_patches(raw: str) -> list[tuple[str, int, int]]:
    match = re.search(r"\b\d+\s*\(", raw)
    if match is None:
        raise ValueError("invalid polyMesh/boundary list")
    start = match.end()
    end = raw.rfind(")")
    if end <= start:
        raise ValueError("invalid boundary list terminator")
    patches = []
    for name, body in _blocks(raw[start:end]):
        count = re.search(r"\bnFaces\s+(\d+)\s*;", body)
        first = re.search(r"\bstartFace\s+(\d+)\s*;", body)
        if not count or not first:
            raise ValueError("boundary patch missing nFaces/startFace")
        patches.append((name, int(first.group(1)), int(count.group(1))))
    return patches


def extract_ascii_surface_scalar_flux(
    case_dir: str | Path, *, time: str, field: str = "phi",
    output_csv: str | Path,
) -> dict:
    """Extract all face fluxes in polyMesh order from an ASCII surface field.

    Rejects binary surface fields, missing boundary values, ambiguous
    patch ordering and non-contiguous boundary face coverage.
    """
    root = Path(case_dir)
    if (root.is_symlink() or not root.is_dir() or not time or "/" in time
            or "\\" in time or not field.isidentifier()):
        raise ValueError("invalid case, time or field")
    mesh = root / "constant" / "polyMesh"
    files = [mesh / "owner", mesh / "neighbour", mesh / "boundary", root / time / field]
    if any(p.is_symlink() or not p.is_file() or p.stat().st_size > 64 * 1024 * 1024 for p in files):
        raise ValueError("missing, symlinked or oversized OpenFOAM input")
    owner = binary_reader.read_label_list(files[0].read_bytes(), str(files[0]))
    neighbour = binary_reader.read_label_list(files[1].read_bytes(), str(files[1]))
    total = int(owner.size)
    internal = int(neighbour.size)
    if total < 1 or internal > total:
        raise ValueError("invalid polyMesh face counts")
    raw = files[3].read_text(encoding="utf-8")
    raw = re.sub(r"/\*.*?\*/|//[^\n]*", "", raw, flags=re.S)
    header = binary_reader.read_header(raw.encode())
    if header.fmt != "ascii" or header.klass != "surfaceScalarField":
        raise ValueError("only ASCII surfaceScalarField is supported")
    values = np.empty(total, dtype=np.float64)
    values[:internal] = _scalar_list(_entry(raw, "internalField"), internal)
    boundary = files[2].read_text(encoding="utf-8")
    boundary = re.sub(r"/\*.*?\*/|//[^\n]*", "", boundary, flags=re.S)
    patches = _read_boundary_patches(boundary)
    field_boundary = re.search(r"\bboundaryField\s*\{", raw)
    if not field_boundary:
        raise ValueError("missing boundaryField")
    blocks = _blocks(raw[field_boundary.end():raw.rfind("}")])
    patch_values = dict(blocks)
    if len(patch_values) != len(blocks) or set(patch_values) != {name for name, _, _ in patches}:
        raise ValueError("boundary patch names mismatch")
    next_face = internal
    for name, start, count in patches:
        if start != next_face or start + count > total:
            raise ValueError("noncontiguous or out-of-order boundary patches")
        values[start:start + count] = _scalar_list(_entry(patch_values[name], "value"), count)
        next_face += count
    if next_face != total:
        raise ValueError("incomplete boundary face coverage")
    out = Path(output_csv)
    if out.is_symlink() or (out.exists() and not out.is_file()):
        raise ValueError("invalid output path")
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["face_id", "integrated_flux"])
        writer.writerows((i, format(float(v), ".17g")) for i, v in enumerate(values))
    return {"path": str(out), "face_count": total, "internal_face_count": internal,
            "boundary_face_count": total - internal,
            "source": "openfoam.ascii.surfaceScalarField",
            "scientific_status": "face_flux_extracted_source_terms_not_verified"}
