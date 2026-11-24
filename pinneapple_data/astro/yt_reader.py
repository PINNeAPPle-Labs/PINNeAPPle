"""The output of the astrophysics codes, as ``PhysicalSample``, through `yt` (#515).

    from pinneapple_data.astro import yt_reader

    s = yt_reader.load("cosmo_sim/DD0042/output_0042")            # yt decides the format: Enzo, FLASH, GADGET ...
    s = yt_reader.load("sedov/", fields=["density", ("gas", "temperature")], level=2)
    s = yt_reader.load("snapshot_000.hdf5", to="points")          # particles stay particles
    s = yt_reader.load("output_0042", region=((0, 0, 0), (1, 1, 1)))   # a box, in the dataset's units
    yt_reader.list_fields("output_0042")                          # what can be asked for, with units

`yt` (Turk et al., ApJS 2011, BSD-3-Clause) reads more than thirty formats -- Enzo, FLASH, RAMSES, GADGET, AREPO,
SWIFT, Athena/Athena++, Cholla, ART, generic HDF5 -- so one reader covers grid, AMR, particle and moving-mesh data.
It is an **optional** dependency: nothing is imported until :func:`load` or :func:`list_fields` is called, and a
machine without it gets an ImportError that says how to install it, never a broken ``import pinneapple_data``.

    pip install yt

``state`` is an ``xarray.Dataset`` on a uniform grid (``domain={"type": "grid"}``: the AMR is resampled to ``level``,
which is what a surrogate or the physics audit expects) or a dict of point arrays (``domain={"type": "points"}``:
particle data is not resampled, because the particles *are* the data). ``schema["units"]`` carries yt's units for
every field, ``provenance`` the code, the time, the box and the level, and ``extras["yt_dataset"]`` the live yt
dataset for anything this reader does not cover.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["available", "load", "list_fields", "grid_to_dataset", "points_to_dataset", "YT_INSTALL_HINT"]

YT_INSTALL_HINT = 'the yt reader needs yt: pip install yt (or pip install "pinneapple[astro]")'

Field = Union[str, Tuple[str, str]]


def available() -> Tuple[bool, str]:
    """Whether yt can be imported, and the reason when it cannot (the sentence a caller can print)."""
    try:
        import yt  # noqa: F401
    except Exception as exc:                                   # ImportError, or a broken install of its own deps
        return False, f"{YT_INSTALL_HINT} ({type(exc).__name__}: {exc})"
    return True, ""


def _open(path, dataset_type: Optional[str] = None):
    """The yt dataset behind ``path``: opened from the file or the directory, or the object itself when it is already
    a dataset (which is how a test builds one in memory with ``yt.load_uniform_grid``)."""
    if hasattr(path, "arbitrary_grid") and hasattr(path, "field_list"):
        return path
    return _yt().load(str(path), dataset_type=dataset_type or None)


def _yt():
    try:
        import yt
    except Exception as exc:
        raise ImportError(YT_INSTALL_HINT) from exc
    return yt


def _name(field: Field) -> str:
    """``("gas", "density")`` -> ``gas_density``: one column name for the xarray dataset."""
    return field if isinstance(field, str) else "_".join(str(f) for f in field)


def _to_numpy(obj) -> Tuple[np.ndarray, str]:
    """A yt array (or plain array) as ``(numpy, unit)``: yt wraps its arrays in ``unyt_array``, which carries the unit."""
    unit = ""
    for attr in ("units", "unit"):
        u = getattr(obj, attr, None)
        if u is not None:
            unit = str(u)
            break
    arr = np.asarray(obj)
    if hasattr(obj, "units"):                                    # unyt: .d converts to a plain ndarray of floats
        try:
            arr = np.asarray(obj.d)
        except Exception:                                        # a non-numeric field: keep what np.asarray gave
            pass
    return arr, unit


def grid_to_dataset(fields: Dict[str, Tuple[np.ndarray, str]], *, left_edge: Sequence[float],
                    right_edge: Sequence[float], dims: Sequence[int], time: Optional[float] = None,
                    unit_system: str = "cgs") -> "Any":
    """``{name: (array, unit)}`` on a uniform grid -> ``xarray.Dataset`` with the cell centres as coordinates.

    yt's grid arrays carry (x, y, z) in that order -- measured on yt 4.4.4 with a grid whose density is a function of
    x alone and whose temperature is a function of z alone: ``grid[field][:, 0, 0]`` walked x and ``[0, 0, :]`` walked
    z, for both ``covering_grid`` and ``arbitrary_grid``. No transpose is applied, and the test keeps it that way: a
    dataset whose axes are silently transposed is a plot that looks plausible and is wrong.
    """
    import xarray as xr

    coords: Dict[str, np.ndarray] = {}
    for i, axis in enumerate("xyz"):                              # cell centres, so a slice lands on the middle
        n = max(int(dims[i]), 1)
        coords[axis] = float(left_edge[i]) + (np.arange(n) + 0.5) * (float(right_edge[i]) - float(left_edge[i])) / n
    data_vars = {}
    for name, (arr, _unit) in fields.items():
        a = np.asarray(arr, float)
        shape = (coords["x"].size, coords["y"].size, coords["z"].size)
        if a.size == int(np.prod(shape)):
            a = a.reshape(shape)                       # (x, y, z), the order yt writes and this module assumes
        data_vars[name] = (tuple("xyz")[:a.ndim], a)
    ds = xr.Dataset(data_vars, coords={k: v for k, v in coords.items() if k in coords and len(coords) == 3})
    ds.attrs["unit_system"] = unit_system
    if time is not None:
        ds.attrs["time"] = float(time)
    return ds


class PointState(dict):
    """The ``state`` of a points sample: a plain dict of arrays that carries the unit system it was read in."""

    unit_system: str = ""


def points_to_dataset(fields: Dict[str, Tuple[np.ndarray, str]], *,
                      unit_system: str = "cgs") -> Dict[str, np.ndarray]:
    """Particle data as a flat ``{"x", "y", "z", <field>: values}`` dict: one row per particle, one column per field.

    The ``unit_system`` is recorded on the dict's class through :class:`PointState` so the sample still says which unit
    system its numbers are in after the xarray wrapper is gone (a points sample is a dict, not a Dataset)."""
    out = PointState({name: np.asarray(arr, float).ravel() for name, (arr, _unit) in fields.items()})
    out.unit_system = unit_system
    return out


def _unit_system(ds) -> str:
    """The unit system the dataset was opened in (``"cgs"`` for most frontends, ``"code"`` for a code-unit dataset)."""
    for attr in ("unit_system", "units"):
        u = getattr(ds, attr, None)
        name = getattr(u, "name", None)
        if isinstance(name, str):
            return name
        for sub in ("unit_system", "name"):
            v = getattr(u, sub, None)
            if isinstance(v, str):
                return v
    return "cgs"


def _one_field(have: List[Field], f: Field) -> Field:
    """The dataset's field for the one asked for, by ``(code, name)`` or by name alone.

    ``("gas", "density")`` is how yt's own docs spell a fluid field, but a frontend may call that code ``"stream"`` or
    ``"enzo"``: the name alone is what the reader matches on when the code given is not one the dataset has."""
    if f in have:
        return f
    name = str(f[1]) if isinstance(f, tuple) else str(f)
    cands = [g for g in have if str(g[1] if isinstance(g, tuple) else g) == name]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise KeyError(f"the dataset has no field {f!r}; ask list_fields() (have: "
                       f"{[str(g) for g in have][:20]})")
    raise KeyError(f"{f!r} is ambiguous in this dataset: {[str(g) for g in cands]}")


def _resolve_fields(ds, fields: Optional[Iterable[Field]]) -> List[Field]:
    """The fields to read: those asked for, checked against what the dataset has; else the density, the temperature,
    the velocity and the magnetic field when present -- the four an audit of a simulation needs."""
    have = full_field_list(ds)
    if fields is None:
        wanted: List[Field] = []
        for want in ("density", "temperature", "velocity_x", "magnetic_field_x", "pressure"):
            for f in have:
                if str(f[1] if isinstance(f, tuple) else f) == want and f not in wanted:
                    wanted.append(f)
                    break
        return wanted or have[:4]
    out: List[Field] = []
    for f in fields:
        key = _one_field(have, f)
        if key not in out:
            out.append(key)
    return out


def full_field_list(ds) -> List[Field]:
    """Every field yt can read from this dataset, the ``(code, name)`` pairs the callers pass back."""
    return list(ds.field_list)


def list_fields(path: str, *, dataset_type: Optional[str] = None) -> List[Dict[str, str]]:
    """The fields the dataset has, with their units: ``[{"field": "gas_density", "code": "gas", "unit": "g/cm**3"}]``.

    Import, open and discard -- the same work :func:`load` does before it reads anything, exposed because choosing the
    fields is the step that needs the answer.
    """
    ds = _open(path, dataset_type)
    out = []
    for f in ds.field_list:
        try:
            unit = str(ds._get_field_info(f).units)
        except Exception:                                   # a field yt describes lazily: the unit is unknown
            unit = ""
        out.append({"field": _name(f), "code": f[0] if isinstance(f, tuple) else "", "unit": unit})
    return out


def load(path: str, fields: Optional[Iterable[Field]] = None, *, level: int = 0,
         region: Optional[Tuple[Sequence[float], Sequence[float]]] = None, dims: Optional[Sequence[int]] = None,
         dataset_type: Optional[str] = None, to: str = "grid", units: bool = True,
         physics_audit: bool = False) -> "Any":
    """One yt dataset (Enzo, FLASH, RAMSES, GADGET, AREPO, SWIFT, Athena++, Athena, Cholla, generic HDF5 ...) as a
    :class:`~pinneapple_data.physical_sample.PhysicalSample`.

    path          : the dataset file or the directory yt opens, or an already-open yt dataset
    fields        : field names (``("gas", "density")`` or ``"density"``); by default density, temperature,
                    velocity, magnetic field and pressure when the dataset has them
    level         : AMR level of the uniform resampling (grid data is returned at this level's resolution)
    region        : ``(left_corner, right_corner)`` in the dataset's length unit; the whole box by default
    dims          : the grid shape to sample at; the level's covering grid by default
    dataset_type  : force a yt frontend by name (blank: let yt decide)
    to            : ``"grid"`` (uniform grid, ``domain={"type": "grid"}``) or ``"points"`` (particles and cell
                    centres as points, in ``domain={"type": "points"}``)
    units         : carry yt's units into ``schema["units"]`` (default; ``False`` drops them)
    physics_audit : also compute mass and energy totals as ``extras["physics_audit"]`` (a magnitude a reader can check
                    against the code's own log)

    Nothing is imported from yt until this function runs: a machine without yt can still import
    :mod:`pinneapple_data.astro`.
    """
    from pinneapple_data.physical_sample import PhysicalSample

    yt = _yt()
    if not hasattr(yt, "load"):
        raise ImportError(YT_INSTALL_HINT)
    ds = _open(path, dataset_type)
    names = _resolve_fields(ds, fields)
    if to not in ("grid", "points"):
        raise ValueError(f"to={to!r}; use 'grid' (uniform grid) or 'points' (particles)")

    left = np.asarray(ds.domain_left_edge, dtype=float)
    right = np.asarray(ds.domain_right_edge, dtype=float)
    if region is not None:
        left, right = np.asarray(region[0], dtype=float), np.asarray(region[1], dtype=float)
        if left.shape != (3,) or right.shape != (3,):
            raise ValueError(f"region must be two 3-vectors, got {left.shape} and {right.shape}")
    shape = tuple(int(d) for d in (dims or (np.asarray(ds.domain_dimensions, dtype=int) * 2 ** int(level))))
    try:
        time = float(ds.current_time.in_units("s").d)
    except Exception:
        try:
            time = float(ds.current_time)
        except Exception:
            time = None

    values: Dict[str, Tuple[np.ndarray, str]] = {}
    if to == "grid":
        grid = ds.arbitrary_grid(left, right, shape)
        for f in names:
            arr = grid[f]
            data, unit = _to_numpy(arr)
            values[_name(f)] = (np.asarray(data, float).reshape(shape), unit)
        state = grid_to_dataset(values, left_edge=left, right_edge=right, dims=shape, time=time,
                                unit_system=_unit_system(ds))
        domain = {"type": "grid", "level": int(level), "dims": list(shape)}
    else:
        ad = ds.all_data()
        for f in names:
            data, unit = _to_numpy(ad[f])
            values[_name(f)] = (np.asarray(data, float).ravel(), unit)
        for axis in "xyz":
            key = ("io", f"particle_position_{axis}")
            if key in ds.field_list:
                data, unit = _to_numpy(ad[key])
                values[axis] = (np.asarray(data, float).ravel(), unit)
        state = points_to_dataset(values, unit_system=_unit_system(ds))
        domain = {"type": "points", "n": int(len(next(iter(values.values()))[0])) if values else 0}

    provenance: Dict[str, Any] = {
        "reader": "yt", "yt_version": getattr(yt, "__version__", ""),
        "source": str(getattr(ds, "filename", "") or path),
        "dataset_type": str(getattr(ds, "dataset_type", "") or ""),
        "time": time, "level": int(level), "dims": list(shape),
        "left_edge": left.tolist(), "right_edge": right.tolist(),
        "length_unit": str(getattr(ds, "length_unit", "") or ""),
        "fields": {k: {"unit": u} for k, (_a, u) in values.items() if u},
    }
    schema: Dict[str, Any] = {
        "units": {k: u for k, (_a, u) in values.items() if u} if units else {},
        "variables": {k: {"unit": u if units else "", "kind": ("coordinates" if k in "xyz" else "field")}
                      for k, (_a, u) in values.items()},
    }
    extras: Dict[str, Any] = {"yt_dataset": ds}
    if physics_audit:
        extras["physics_audit"] = _audit(ds, values, to=to, shape=shape, level=level)
    return PhysicalSample(state=state, schema=schema, domain=domain, provenance=provenance, extras=extras)


def _audit(ds, values: Dict[str, Tuple[np.ndarray, str]], *, to: str, shape: Sequence[int],
           level: int) -> Dict[str, Any]:
    """Mass and energy totals over the volume sampled, in the units the fields carry: the two numbers a reader can
    compare with the code's own log. Divergence of B (MHD) and virial checks need the derivatives of the mesh and are
    not computed here -- they are the next step of #515, not a silent zero."""
    out: Dict[str, Any] = {"level": int(level), "points": int(np.prod(shape)) if to == "grid" else None}
    for key in ("density", "total_energy", "kinetic_energy", "thermal_energy", "pressure"):
        for name, (arr, unit) in values.items():
            if name.endswith(key):
                out[f"{name}_sum"] = float(np.nansum(arr))
                out[f"{name}_max"] = float(np.nanmax(arr)) if arr.size else None
                out[f"{name}_unit"] = unit
                break
    out["note"] = ("mass and energy are sums of the sampled values, not integrals over the cell volumes: multiply by "
                   "the cell volume (extras['yt_dataset'] has cell_volume) for a physical total")
    return out
