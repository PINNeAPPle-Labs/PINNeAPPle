"""What CFL3D and FUN3D published at the station the flat-plate case measures, from the files next to this module (#265).

:mod:`~pinneapple_simulation.cfd_validation.tmr` runs the TMR 2D zero-pressure-gradient flat plate in OpenFOAM and
reports the skin friction at ``x = 0.97`` on three grids refined by two. This module holds the other side of that
comparison, in the form the TMR publishes it, as data:

* ``cf_convergence_sa.dat`` and ``cf_convergence_sst.dat`` — ``cf`` at ``x = 0.970084071`` on the **five nested grids**
  of the verification case, from CFL3D and FUN3D. This is the comparison the case's own station supports: the same
  quantity, at the same place, on a family that can be extrapolated to zero grid spacing exactly as the case does with
  its own.
* ``cf_incomp_sa.dat`` and ``cf_incomp_sst.dat`` — the same station from codes that ran the case **without the energy
  equation** (FUN3D incompressible, SC/Tetra and, for SST, GHOST). An OpenFOAM ``simpleFoam`` run is one of those, so
  these files measure what the missing compressibility is worth before any model difference is blamed for it.
* ``sa-cf_cfl3d.dat`` and ``sst-cf_cfl3d.dat`` — ``cf`` against ``Re_theta`` along the plate, the validation case's
  own quantity of interest, which :func:`~pinneapple_simulation.cfd_validation.tmr.study` uses through the momentum
  thickness of the solved field.

Provenance
----------
Every file was downloaded unchanged from the TMR mirror on 2026-11-25 and is stored byte for byte; :data:`SOURCES`
carries its URL, size and SHA-256, and :func:`check_sources` verifies the copy in the repository against that hash. A
number below can therefore be traced to a file, and the file to the publication.

Why the files and not a transcription
-------------------------------------
A table typed into a Python literal is a table that can be wrong in a way nobody notices — a decimal moved, a row
dropped. Keeping the published file and parsing it removes that failure mode entirely: the hash is the transcription.
The reader is :func:`~pinneapple_simulation.cfd_validation.tmr.read_reference`, which maps columns by the names the
file declares (these files lead with the value and put the abscissa second) and refuses a multi-zone file unless a
zone is named, because ``cf_as_function_of_x.dat`` in the same directory is three zones of White's theory and
interleaving them produces a curve no theory predicts.

The station
-----------
``x = 0.970084071`` is the TMR grid point, and ``x = 0.97`` is where the case samples. The two differ by one grid
point of a 449-point plate, which moves ``cf`` by about 0.02 % — below every number in the tables here. The plate in
this repository is half as long (``x in [0, 1]``) as the published one (``x in [0, 2]``) at the same Reynolds number
per unit length, so the same ``x`` is the same ``Re_x = 5e6 x`` and the same boundary layer: no scaling is needed,
and none is applied.

What separates an OpenFOAM run from these numbers
-------------------------------------------------
:data:`COMMON_DIFFERENCES` and :data:`MODEL_DIFFERENCES` list them item by item, with the size of each where the TMR's
own files measure it. The first is the one that matters: the reference resolved the wall at ``y+ ~ 0.1`` and the case
here runs wall functions at ``y+ = 30``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from . import tmr

__all__ = ["DATA", "STATION", "RETRIEVED", "SOURCES", "STATION_FILES", "CURVE_FILES", "MODEL_TABLE",
           "COMMON_DIFFERENCES", "MODEL_DIFFERENCES", "differences", "check_sources", "station_table",
           "reference_at_station", "compare"]

#: Where the published files live: next to the code that reads them, so no comparison needs a download.
DATA = Path(__file__).resolve().parent / "data" / "tmr"

#: The station the TMR verification case tabulates, in the published grid's own coordinates.
STATION = 0.970084071

#: The date every file was fetched.
RETRIEVED = "2026-11-25"

#: This repository's model name -> the TMR's name for the same model.
MODEL_TABLE: dict[str, str] = {"SpalartAllmaras": "SA", "kOmegaSST": "SST"}

_MIRROR = "https://tmbwg.github.io/turbmodels"

#: Every file, with the URL it came from and the SHA-256 a copy must have to be the published material.
SOURCES: dict[str, dict[str, Any]] = {
    "cf_convergence_sa.dat": {
        "url": f"{_MIRROR}/FlatPlate/SA/cf_convergence.dat", "bytes": 653,
        "sha256": "966a9f9c8036c3c6d8a1f2c4f1744b6a79f395f5986c8a2b37941803faf67695",
        "what": "cf at x = 0.970084071 on the five nested grids, CFL3D and FUN3D, SA"},
    "cf_convergence_sst.dat": {
        "url": f"{_MIRROR}/FlatPlate/SST/cf_convergence_sstv.dat", "bytes": 666,
        "sha256": "e5cde959873804d9b37a83640535309e18d5f1bc269b7bb6a46c460c40cd2a30",
        "what": "the same for SST-V (the page states SST, SST-V and SSTm differ by well under 1 % here)"},
    "cf_incomp_sa.dat": {
        "url": f"{_MIRROR}/FlatPlate_validation/cf_incomp_results_sa.dat", "bytes": 670,
        "sha256": "23dbf72d3a8396b1524c03fd7fc2d56b93b7f64ef0955429ee4b37b68b3c78db",
        "what": "the same station from two incompressible codes, SA"},
    "cf_incomp_sst.dat": {
        "url": f"{_MIRROR}/FlatPlate_validation/cf_incomp_results_sstv.dat", "bytes": 986,
        "sha256": "65bbdbae974af5297db4387d67f2f9ac8d3c46750815d684e108a52e850647b9",
        "what": "the same from three incompressible codes, SST/SST-V"},
    "sa-cf_cfl3d.dat": {
        "url": f"{_MIRROR}/FlatPlate_validation/sa-cf_cfl3d.dat", "bytes": 6249,
        "sha256": "e089c0ada436ab74308b20b3e7913a0ced691de85ccef83c3a5961f9f7f35c19",
        "what": "cf against Re_theta along the plate, CFL3D on 545x385, SA"},
    "sst-cf_cfl3d.dat": {
        "url": f"{_MIRROR}/FlatPlate_validation/sst-cf_cfl3d.dat", "bytes": 6353,
        "sha256": "d694caf86c8886ad40016f63e76f738e2c8caa982437ba2a1074d5d58efc2109",
        "what": "the same for SST"},
}

#: Which file holds the station's nested grids, per model, split by whether the codes solved the energy equation.
STATION_FILES: dict[str, dict[str, str]] = {
    "SA": {"compressible": "cf_convergence_sa.dat", "incompressible": "cf_incomp_sa.dat"},
    "SST": {"compressible": "cf_convergence_sst.dat", "incompressible": "cf_incomp_sst.dat"},
}

#: Which file holds ``cf`` against ``Re_theta``, per model (the validation case's quantity of interest).
CURVE_FILES: dict[str, str] = {"SA": "sa-cf_cfl3d.dat", "SST": "sst-cf_cfl3d.dat"}

#: What separates any OpenFOAM run of this case from these numbers, with the size of each where a file measures it.
COMMON_DIFFERENCES: tuple[str, ...] = (
    "wall treatment: the TMR finest grid has 5e-7 wall spacing, an average y+ of about 0.1, and resolves the viscous "
    "sublayer (turbmodels/flatplate_grids.html); the case here runs wall functions with the first cell at y+ = 30, so "
    "it compares a modelled wall with a computed one",
    "compressibility: CFL3D and FUN3D ran M = 0.2 with the energy equation; simpleFoam has none. The TMR's own "
    "incompressible files measure what that is worth at this station: SA 0.0027291 (FUN3D incompressible) against "
    "0.0027054 (FUN3D compressible), +0.88 %, and SST 0.0027174 against 0.0026905, +1.00 %",
    "grid: 11 520 cells (120 x 96, 120 of them on the plate) at the finest level against the reference's 208 896 "
    "(545 x 385, 449 on the plate), and the family is regenerated here rather than converted from the published "
    "Plot3D grids - which is why the number quoted for this case is the Richardson-extrapolated one, GCI 0.13 %",
    "definition of cf: 2 tau_w / U^2 from the solver's own wallShearStress function object on one side, "
    "tau_w / (0.5 rho U^2) with the freestream density on the other; the same in the incompressible limit",
)

#: The model-specific half. Named separately because it is what a reader has to change to close a gap.
MODEL_DIFFERENCES: dict[str, tuple[str, ...]] = {
    "SA": (
        "freestream: both sides use nuTilda/nu = 3 (the SA results page states it for CFL3D and FUN3D; the case here "
        "writes ninf = 3 nu), so this one matches",
        "variant: the TMR baseline SA includes the ft2 term, as OpenFOAM's SpalartAllmaras does; SA-noft2, which the "
        "TMR says differs only for low inflow values, is not what either side ran",
        "nut at the wall: the case writes nutUSpaldingWallFunction on the plate for both models, while OpenFOAM's SA "
        "computes nut from nuTilda (its own setup is 'calculated'). That override is the first candidate for the SA "
        "gap and has not been measured yet",
    ),
    "SST": (
        "freestream: the TMR used turbulent intensity 0.039 % and mut/mu = 0.009; the case here writes 1 % and "
        "nut/nu = 10. The TMR states its SST result moves by well under 1 % between its own two freestream levels, but "
        "both of those are far below 1 % intensity, so this is a difference and not a measured equality",
        "variant: the TMR table is SST-Vm (its page: 'SST, SST-V, SSTm yield nearly identical results for this "
        "case'); OpenFOAM's kOmegaSST is the 2003 Menter form",
    ),
}


def _tmr_model(model: str) -> str:
    """``"SpalartAllmaras"`` -> ``"SA"``; a TMR name passes through; anything else lists what exists."""
    if model in MODEL_TABLE:
        return MODEL_TABLE[model]
    if model in STATION_FILES:
        return model
    raise KeyError(f"{model!r}: no TMR reference for that model (have: "
                   f"{', '.join(sorted(MODEL_TABLE))} as models, {', '.join(sorted(STATION_FILES))} as TMR names)")


def differences(model: str) -> tuple[str, ...]:
    """Every implementation difference for one model, the common ones first."""
    return COMMON_DIFFERENCES + MODEL_DIFFERENCES[_tmr_model(model)]


def check_sources() -> dict[str, Any]:
    """Verify every bundled file against :data:`SOURCES`: ``{"ok": bool, "files": [...], "problems": [...]}``.

    A file that has been edited, truncated or replaced is not the published material any more, and a comparison
    against it would be a comparison with something else. This is the gate that makes "these are the published
    numbers" a checkable statement rather than a claim.
    """
    files, problems = [], []
    for name, info in SOURCES.items():
        path = DATA / name
        if not path.is_file():
            problems.append(f"{name}: not in {DATA}")
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != info["sha256"] or len(raw) != info["bytes"]:
            problems.append(f"{name}: {len(raw)} bytes, sha256 {digest}; published is "
                            f"{info['bytes']} bytes, sha256 {info['sha256']}")
        files.append({"file": name, "ok": digest == info["sha256"] and len(raw) == info["bytes"],
                      "bytes": len(raw), "sha256": digest, "url": info["url"]})
    return {"ok": not problems, "files": files, "problems": problems}


def station_table(model: str, family: str = "compressible") -> dict[str, dict[str, Any]]:
    """The nested-grid tables at :data:`STATION` for one model: ``{code: {"n", "cf"}}``, finest grid first.

    ``family`` is ``"compressible"`` (CFL3D and FUN3D, the codes the TMR calls its reference) or ``"incompressible"``
    (the codes that ran the case without the energy equation, which is what an OpenFOAM ``simpleFoam`` run is).
    """
    tmr_model = _tmr_model(model)
    if family not in STATION_FILES[tmr_model]:
        raise KeyError(f"{family!r}: no such family (have: {', '.join(STATION_FILES[tmr_model])})")
    path = str(DATA / STATION_FILES[tmr_model][family])
    first = tmr.read_reference(path)
    out: dict[str, dict[str, Any]] = {}
    for zone in first["zones"]:
        ref = first if zone == first["zone"] else tmr.read_reference(path, zone=zone)
        cf = np.asarray(ref["cf"], dtype=float)
        n = np.asarray(ref["n"], dtype=float) if "n" in ref else np.arange(cf.size)[::-1].astype(float)
        out[zone] = {"n": n, "cf": cf}
    return out


def reference_at_station(model: str, code: str = "CFL3D") -> dict[str, Any]:
    """One code's reference at :data:`STATION`: the finest grid's value and the family extrapolated to zero spacing.

    The extrapolation is :func:`~pinneapple_analysis.verification.convergence.richardson_extrapolate` — the same
    function the case uses on its own three grids — applied to the three finest of the published five. It reproduces
    the TMR's published apparent order and relative extrapolated error digit for digit (``observed_order`` 1.98 and a
    relative extrapolated error of 0.014 % for SA on CFL3D; 1.21 and 0.221 % for SST), which is what makes this the
    published table rather than one that merely looks plausible.
    """
    from pinneapple_analysis.verification.convergence import richardson_extrapolate

    tmr_model = _tmr_model(model)
    for family in STATION_FILES[tmr_model]:
        table = station_table(tmr_model, family)
        if code not in table:
            continue
        cf = table[code]["cf"]
        if cf.size < 3:
            raise ValueError(f"{tmr_model} {code}: {cf.size} grids at the station, three are needed to extrapolate")
        fine, medium, coarse = float(cf[0]), float(cf[1]), float(cf[2])
        conv = richardson_extrapolate(coarse, medium, fine, 2.0)
        return {"model": tmr_model, "code": code, "family": family, "station": STATION,
                "grids": [int(v) for v in table[code]["n"]], "cf_grids": [float(v) for v in cf],
                "cells_fine": int(table[code]["n"][0]), "cf_fine": fine,
                "cf_converged": float(conv.extrapolated_value),
                "observed_order": abs(float(conv.observed_order)), "gci_fine": float(conv.gci_fine)}
    have = sorted({c for family in STATION_FILES[tmr_model] for c in station_table(tmr_model, family)})
    raise KeyError(f"{code!r}: no {tmr_model} results at x = 0.97 from that code (have: {', '.join(have)})")


def compare(model: str, cf: float) -> dict[str, Any]:
    """A measured ``cf`` against every code the TMR publishes at the station, with the differences listed.

    The error is reported against each code separately, because the codes answer different questions: the compressible
    pair is "the reference" the issue asks about, and the incompressible ones are the comparison that takes the Mach
    number out of the difference before any model effect is credited with it.
    """
    tmr_model = _tmr_model(model)
    measured = float(cf)
    references: dict[str, dict[str, Any]] = {}
    for family in STATION_FILES[tmr_model]:
        for code in station_table(tmr_model, family):
            references[code] = reference_at_station(tmr_model, code)
    if "CFL3D" not in references:
        raise KeyError(f"{tmr_model}: the compressible reference has no CFL3D entry (have: {', '.join(references)})")

    def rel(base: float) -> float:
        return float((measured - base) / base) if np.isfinite(base) and base else float("nan")

    headline = references["CFL3D"]
    by_code = {code: rel(ref["cf_fine"]) for code, ref in references.items()}
    return {"model": model, "tmr_model": tmr_model, "station": STATION, "cf_measured": measured,
            "reference": headline, "references": references,
            "cf_reference": headline["cf_fine"], "cf_reference_converged": headline["cf_converged"],
            "relative_error": rel(headline["cf_fine"]),
            "relative_error_converged": rel(headline["cf_converged"]),
            "relative_error_by_code": by_code,
            "relative_error_incompressible": {code: by_code[code] for code, ref in references.items()
                                              if ref["family"] == "incompressible"},
            "differences": differences(tmr_model)}
