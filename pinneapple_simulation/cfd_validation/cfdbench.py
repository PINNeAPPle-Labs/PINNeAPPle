"""The CFDBench cavity family, read off its own archive and scored against the same table our solver is (#272).

CFDBench (Luo, Chen and Zhang, 2023, arXiv:2310.05963) is a large machine-learning benchmark, and the issue that asked
for this case set asks for its numbers beside ours. Its data are on HuggingFace (``chen-yingfa/CFDBench``, licence
``apache-2.0``, read from the dataset card) as one zip per problem:

    https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cavity.zip     # 786 MB, the other three too

and the loader that reads it is :mod:`pinneapple_data.datasets.cfdbench`, which is what this module goes through: a
number produced by a second reader of the archive would measure the reader as much as the data.

What "the same four problems" means here
----------------------------------------
The archive publishes ``cavity``, ``tube``, ``dam`` and ``cylinder``. Only the cavity has a table of *answers* in this
repository (Ghia's, in :mod:`~pinneapple_simulation.cfd_validation.reference`), so only the cavity can be scored and
put in a column beside our own numbers. The tube has something weaker, and it is transcribed and used here: the paper
publishes the *operating parameters* of its cases (Table 3), which :func:`tube_against_table` holds the disk against —
that fixes which case is which, and it is all the tube can support, because the flow is a water-air two-phase one for
which the paper publishes no physical solution. The dam and the cylinder have neither: they are loaded by the same
adapter and have nothing to be wrong against here. That is recorded as it is rather than as four rows of a table.

The two things a reader of this data has to get right
-----------------------------------------------------
* **the Reynolds number includes the density.** ``case.json`` gives ``vel_top``, ``width``, ``density`` and
  ``viscosity``; the case is ``Re = rho U L / mu``. The ``prop`` family varies the density *and* the viscosity, so a
  reader that computes ``U L / nu`` reads the whole family as one Reynolds number per viscosity: measured here, the
  cases that read as ``Re = 100`` that way span 13 % to 44 % against the same column of the table, which is what a
  mislabelled Reynolds number looks like. With the density in, the cases that read ``Re = 100`` agree with each other
  to the last digit.
* **the field is a transient run on a 64 x 64 grid.** The archive stores the fields interpolated to 64 x 64 and the
  cavity cases keep only the last few frames, so the four stations nearest the moving lid (``y >= 0.9531``) — the
  layer a 64-cell grid cannot resolve — are where the difference against the table is concentrated. Both readings are
  reported: the whole column, and the column without those four stations, because quoting only the first would blame
  the data for the grid and quoting only the second would hide it.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .cavity import profile_error
from .reference import CFDBENCH_TUBE, GHIA_U, GHIA_V

__all__ = ["ARCHIVE", "PROBLEM", "PROBLEMS", "SUBSETS", "NEAR_LID", "reynolds", "open_cases", "families", "cases_at",
           "sample", "score", "study", "tube_against_table"]

#: The four problems the archive publishes, and the archive of each (the cylinder's is three zips).
PROBLEMS = {
    "cavity": "https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/cavity.zip",
    "tube": "https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/tube.zip",
    "dam": "https://huggingface.co/datasets/chen-yingfa/CFDBench/resolve/main/dam.zip",
    "cylinder": "https://huggingface.co/datasets/chen-yingfa/CFDBench/tree/main/cylinder",
}

#: The archive of the family that can be scored here, and the one the other three sit in.
ARCHIVE = PROBLEMS["cavity"]
PROBLEM = "cavity"
SUBSETS = ("bc", "geo", "prop")

#: The stations of Ghia's ``u`` profile that sit in the last 5 % of the cavity, where the archive's 64-cell grid stops
#: resolving the layer the lid drags along. The second reading of every score leaves them out.
NEAR_LID = 0.95


def reynolds(case: Any) -> float:
    """``Re = rho U L / mu`` of one case, from the four numbers ``case.json`` records.

    The density is in the formula because the ``prop`` family varies it: dropping it (``U L / nu``) turns six different
    flows into one Reynolds number, and the difference shows up as error against the table rather than as a wrong
    label. ``width`` is the cavity's side, which is the length scale of the case.
    """
    p = case.params()
    for need in ("vel_top", "width", "density", "viscosity"):
        if not p.get(need):
            raise ValueError(f"{case.summary()}: case.json has no usable '{need}' (it has {sorted(p)})")
    return float(p["density"]) * float(p["vel_top"]) * float(p["width"]) / float(p["viscosity"])


def open_cases(root: str | os.PathLike[str], problem: str = PROBLEM) -> list[Any]:
    """Every case of one problem under an unpacked archive, through :mod:`pinneapple_data.datasets.cfdbench`.

    The zips unpack to ``bc/``, ``geo/`` and ``prop/`` at the top level, which is the shape the adapter's
    ``open_cases`` reads when the directory it is given *is* one problem. Raises with the URL when the directory is
    missing, because a missing download is the failure this module can do nothing about.
    """
    from pinneapple_data.datasets import cfdbench as adapter

    root = os.fspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} is not a directory: unzip {PROBLEMS.get(problem, ARCHIVE)} into it first")
    return list(adapter.open_cases(root, problem=problem))


def families(root: str | os.PathLike[str]) -> dict[str, dict[str, Any] | str]:
    """Which problems of the archive are unpacked under ``root``, and how much of each one the loader opens.

    The record this returns is the evidence for the part of the issue that cannot be scored: every family the same
    reader opens, with its case count, its subsets and the shape of its fields — and ``"not unpacked"`` for the ones
    that are not here, which is a statement about this machine rather than about the data. The cavity is the one with a
    published table in this repository, so it is the one :func:`study` scores; the others are reported as loaded and
    unscored.
    """
    out: dict[str, dict[str, Any] | str] = {}
    base = os.fspath(root)
    for name in PROBLEMS:
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            out[name] = "not unpacked"
            continue
        cases = open_cases(path, problem=name)
        shapes = {tuple(case.arrays()["u"].shape) for case in cases}
        frames = sorted({s[0] for s in shapes if len(s) == 3})
        grids = sorted({tuple(s[1:]) for s in shapes if len(s) == 3})
        out[name] = {"cases": len(cases), "subsets": sorted({case.subset for case in cases}),
                     "frames": [frames[0], frames[-1]] if frames else [],
                     "grid": [list(g) for g in grids] if len(grids) > 1 else (list(grids[0]) if grids else None)}
    return out


def cases_at(cases: list[Any], reynolds_number: float, *, tolerance: float = 1e-6) -> list[Any]:
    """The cases of the square cavity at one Reynolds number, sorted.

    The square is required because the table is the unit square: the ``geo`` family varies the aspect ratio, and a
    2 x 1 cavity compared with Ghia's profiles would report the geometry as model error.
    """
    out = []
    for case in cases:
        p = case.params()
        if abs(float(p.get("width", 0.0)) - float(p.get("height", -1.0))) > 1e-12:
            continue
        if abs(reynolds(case) - float(reynolds_number)) <= tolerance * max(1.0, float(reynolds_number)):
            out.append(case)
    return sorted(out, key=lambda c: c.summary())


def sample(case: Any) -> dict[str, np.ndarray | float | int]:
    """The case's last frame, sampled where the table publishes its stations, in the table's own normalisation.

    The archive's velocities carry the case's own lid speed (``vel_top``), which is not one: everything is divided by
    it so the profile can be compared with a table that is written for ``U = 1``, and the frame index and the largest
    change between the last two frames are returned with it, because "the last frame" is only the steady answer if the
    run had reached one and that is the reader's to see.
    """
    params = case.params()
    scale = float(params["vel_top"])
    arrays = case.arrays()
    u = np.asarray(arrays["u"], dtype=float)
    v = np.asarray(arrays["v"], dtype=float)
    if u.ndim != 3 or u.shape != v.shape:
        raise ValueError(f"{case.summary()}: u {u.shape} and v {v.shape} are not two equal (time, height, width) "
                         "arrays -- 0-d or 2-d data is not the layout the archive publishes")
    frame = u[-1] / scale
    velocity_previous = u[-2] / scale if u.shape[0] > 1 else frame
    w = v[-1] / scale
    ny, nx = frame.shape
    # the centreline is between the two middle cells: the archive's grid is cell-centred, so x = 0.5 falls on a face
    yc, xc = (np.arange(ny) + 0.5) / ny, (np.arange(nx) + 0.5) / nx
    j, i = nx // 2, ny // 2
    u_mid = 0.5 * (frame[:, j - 1] + frame[:, j])
    v_mid = 0.5 * (w[i - 1, :] + w[i, :])
    u_ref, v_ref = GHIA_U[100], GHIA_V[100]                 # the stations are the same at every column of the table
    y_st, x_st = np.asarray(u_ref.points), np.asarray(v_ref.points)
    return {"y": y_st, "u": np.interp(y_st, yc, u_mid), "x": x_st, "v": np.interp(x_st, xc, v_mid),
            "frame": int(u.shape[0] - 1), "drift": float(np.abs(frame - velocity_previous).max()),
            "scale": scale, "shape": [int(ny), int(nx)]}


def score(case: Any, reynolds_number: float) -> dict[str, Any]:
    """One case as a row: its errors against the published column, and the ``u`` reading without the lid stations."""
    got = sample(case)
    re = int(round(float(reynolds_number)))
    rows: dict[str, Any] = {"case": case.summary(), "reynolds": float(reynolds_number),
                            "shape": got["shape"], "frame": got["frame"], "drift": got["drift"],
                            "lid_speed": got["scale"],
                            "y": [float(v) for v in got["y"]], "u": [float(v) for v in got["u"]],
                            "x": [float(v) for v in got["x"]], "v": [float(v) for v in got["v"]],
                            "errors": {"u": profile_error(GHIA_U[re], np.asarray(got["u"])),
                                       "v": profile_error(GHIA_V[re], np.asarray(got["v"]))}}
    # the same u reading without the stations in the layer the archive's grid cannot resolve: the two numbers
    # together say how much of the difference is the data and how much is the 64-cell grid it was interpolated to
    ref = GHIA_U[re]
    keep = np.asarray([p < NEAR_LID for p in ref.points])
    keep[0] = keep[-1] = False
    d = (np.asarray(got["u"], dtype=float) - np.asarray(ref.values, dtype=float))[keep]
    rms = float(np.sqrt(np.mean(np.asarray(ref.values, dtype=float)[keep] ** 2)))
    rows["errors"]["u_without_lid"] = {"l2_relative": float(np.sqrt(np.mean(d ** 2)) / rms), "points": int(keep.sum())}
    return rows


def _archive_root(root: str | os.PathLike[str]) -> str:
    """The directory the families are under, given either that directory or one unpacked family inside it."""
    base = os.fspath(root)
    return os.path.dirname(base.rstrip(os.sep)) if os.path.basename(base.rstrip(os.sep)) in PROBLEMS else base


def study(root: str | os.PathLike[str], reynolds_numbers: tuple[float, ...] = (100.0, 400.0, 1000.0)) -> dict[str, Any]:
    """The cavity family at the Reynolds numbers of the table: every square case at each, scored against it.

    Returns the record the case's ``results/cfdbench.json`` holds. The cases at one Reynolds number are duplicates of
    each other in the archive (the same flow reached through different ``density``/``viscosity``/``vel_top``
    combinations), which is visible in the rows: identical profiles, and the same error to the last digit.
    """
    cases = open_cases(root)
    record: dict[str, Any] = {"archive": ARCHIVE, "problem": PROBLEM, "citation": GHIA_U[100].citation,
                              "cases_on_disk": len(cases),
                              "families": families(_archive_root(root)),
                              "what": "the archive's own fields, read by pinneapple_data.datasets.cfdbench, sampled "
                                      "where the table publishes its stations and scored with the solver's ruler",
                              "near_lid": {"from": NEAR_LID, "why": "the four stations of the u profile nearest the "
                                                                    "moving lid: a 64-cell grid does not resolve the "
                                                                    "layer there, and the difference against the "
                                                                    "table is concentrated in them"},
                              "rows": []}
    for re in reynolds_numbers:
        selected = cases_at(cases, float(re))
        record["rows"].append({"reynolds": float(re), "found": len(selected),
                               "cases": [score(case, float(re)) for case in selected]})
    return record


#: Which field of ``case.json`` carries which parameter of the paper's Table 3. The archive names the tube's diameter
#: ``height`` and its length ``width``; the ``prop`` family's ``density`` is the index, which is the finding below.
_TUBE_FIELDS = {
    "bc": lambda p: {"bc": p.get("vel_in")},
    "prop": lambda p: {"prop/density": p.get("density"), "prop/viscosity": p.get("viscosity")},
    "geo": lambda p: {"geo/diameter": p.get("height"),
                      "geo/ratio": (p.get("width") / p.get("height")) if p.get("height") else None},
}


def tube_against_table(root: str | os.PathLike[str], *, tolerance: float = 1e-9) -> dict[str, Any]:
    """The tube family held against the operating-parameter table its paper publishes (Table 3 of CFDBench).

    The cavity is the one problem of this archive with a table of answers to be scored against; the tube has none, so
    what can be established about it is weaker and worth stating exactly: that the cases read off the disk **are** the
    cases the paper says were generated. That is a real check and not a formality — it is what catches a reader that
    is subtly wrong about the data, and here it catches one thing in the data itself.

    Every published value is looked for in the archive and every value in the archive is looked for in the table, for
    each of the three subsets. The result carries both lists, so a mismatch reads as what it is: a value published and
    absent, a value present and unpublished, or a count that disagrees with the paper's own. A value counts as found
    when it is within ``tolerance`` of the published one, which is the float noise of a parameter written by a
    generator as ``0.30000000000000004``.
    """
    cases = open_cases(root, problem="tube")
    published = CFDBENCH_TUBE.sets
    seen: dict[str, set[float]] = {k: set() for k in published}
    counts = {subset: 0 for subset in SUBSETS}
    unreadable: list[str] = []
    for case in cases:
        params = case.params()
        counts[case.subset] = counts.get(case.subset, 0) + 1
        for key, value in _TUBE_FIELDS.get(case.subset, lambda p: {})(params).items():
            if value is None:
                unreadable.append(f"{case.summary()}: no value for {key}")
                continue
            seen[key].add(round(float(value), 9))

    def near(value: float, values: set[float]) -> bool:
        return any(abs(value - v) <= tolerance for v in values)

    parameters = []
    for key, want in sorted(published.items()):
        got = seen[key]
        missing = [v for v in want if not near(v, got)]
        extra = sorted(v for v in got if not near(v, want))
        parameters.append({"parameter": key, "published": [float(v) for v in want],
                           "found": [float(v) for v in sorted(got)],
                           "published_but_absent": [float(v) for v in missing],
                           "present_but_unpublished": extra,
                           "agrees": not missing and not extra})

    # the paper's Table 6 counts, so a family that is short of cases says so instead of quietly matching a subset
    expected_cases = {"bc": 50, "prop": 100, "geo": 25}
    return {
        "case": CFDBENCH_TUBE.case, "citation": CFDBENCH_TUBE.citation,
        "archive": PROBLEMS["tube"], "note": CFDBENCH_TUBE.note,
        "what": "the tube family read off the archive and held against the operating-parameter table the paper "
                "publishes for it; there is no table of answers for this problem, so this checks that the cases are "
                "the ones the paper says were generated, not that they are right",
        "cases": counts, "cases_published": expected_cases,
        "unreadable": unreadable,
        "parameters": parameters,
        "agrees": all(p["agrees"] for p in parameters) and not unreadable,
        "disagrees_on": [p["parameter"] for p in parameters if not p["agrees"]],
    }

