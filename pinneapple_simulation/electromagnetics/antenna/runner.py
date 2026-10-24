"""Drive an external parametric simulator script, one process per sample (and per frequency), and read its tables.

:class:`CLISimulator` runs any command-line script that takes a design and frequencies and writes its result
(``result.json`` with ``{"status": "ok" | "invalid", "rows": [...]}`` by default, or anything a ``parse`` callable
reads). Each call is a separate process with a timeout, so a crash, a memory blow-up or a hung solve costs one sample,
not the study. Lessons measured while running EMerge on a 4-core, 15 GB machine, built in as defaults:

* EMerge's PARDISO backend can die with SIGBUS when MKL is not usable: on SIGBUS the sample is retried once with
  ``fallback_args`` (for :mod:`.ava` the default is already SuperLU; ``--pardiso`` opts in);
* three frequencies in one process with SuperLU went past 15 GB and were SIGKILLed: ``per_frequency=True`` runs one
  frequency per process (each frequency also gets its own mesh sized to its wavelength);
* ``grid.model_S`` (the rational S-parameter model) fails with a single frequency: :mod:`.ava` reads ``grid.S``.

Every run records its exit status, wall time and peak memory (``ru_maxrss`` of that child, from ``os.wait4``).

The table format, one row per (design, frequency): ``tag``, the design parameters, ``f_GHz``, ``S11_dB``,
``D_bore_dBi``, ``G_bore_dBi``, ``Greal_bore_dBi``, ``D_peak_dBi``, ``Greal_peak_dBi``, ``rad_eff`` (extra columns
are kept). :func:`read_table` reads it from CSV, TSV or whitespace-separated text; :func:`write_table` writes CSV.
"""
from __future__ import annotations

import csv
import io
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["RESULT_COLUMNS", "RunRecord", "CLISimulator", "ResultTable", "read_table", "write_table",
           "ava_simulator"]

RESULT_COLUMNS = ("f_GHz", "S11_dB", "D_bore_dBi", "G_bore_dBi", "Greal_bore_dBi", "D_peak_dBi", "Greal_peak_dBi",
                  "rad_eff")


@dataclass
class RunRecord:
    """One process: which design and frequencies, how it ended, what it cost."""
    tag: str
    freqs: list[float]
    status: str                 # ok | invalid | timeout | killed (SIGKILL, usually out of memory) | sigbus | failed
    returncode: int | None
    wall_s: float
    max_rss_mb: float
    rows: list[dict[str, Any]] = field(default_factory=list)
    message: str = ""
    retried: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "rows"} | {"n_rows": len(self.rows)}


def _default_parse(out_dir: str) -> tuple[str, list[dict[str, Any]], str]:
    p = os.path.join(out_dir, "result.json")
    if not os.path.exists(p):
        return "failed", [], "no result.json written"
    with open(p) as fh:
        res = json.load(fh)
    return str(res.get("status", "ok")), list(res.get("rows", [])), str(res.get("error", ""))


class CLISimulator:
    """Run ``argv + [params_flag, json, freq_flag, f..., out_flag, dir, tag_flag, tag]`` per design.

    ``argv``: the command (e.g. ``[sys.executable, "script.py"]``); ``per_frequency``: one process per frequency;
    ``timeout``: seconds per process; ``fallback_args``: appended for one retry after a SIGBUS; ``parse(out_dir)``
    returns ``(status, rows, message)``; ``env``: extra environment variables (thread counts, log levels);
    ``extra_args``: arguments added to every command (e.g. ``["--scan", "0", "30"]``); ``resume``: reuse the
    ``result.json`` an earlier (interrupted) run left in the same folder when its status is ``ok`` and the
    parameters it was run with (``runner_params.json``, written by the runner) match, instead of solving again
    (recorded with ``message="reused"``).
    Rows get the design's ``tag`` and parameters prepended, so they form a :class:`ResultTable`.
    """

    def __init__(self, argv: Sequence[str], workdir: str, *, per_frequency: bool = True, timeout: float = 3600.0,
                 params_flag: str = "--params", freq_flag: str = "--freq", out_flag: str = "--out",
                 tag_flag: str | None = "--tag", fallback_args: Sequence[str] = (),
                 parse: Callable[[str], tuple[str, list[dict[str, Any]], str]] | None = None,
                 env: Mapping[str, str] | None = None, log: Callable[[str], None] | None = None,
                 extra_args: Sequence[str] = (), resume: bool = False):
        self.argv, self.workdir = list(argv), os.path.abspath(workdir)
        self.per_frequency, self.timeout = per_frequency, float(timeout)
        self.params_flag, self.freq_flag, self.out_flag, self.tag_flag = params_flag, freq_flag, out_flag, tag_flag
        self.fallback_args = list(fallback_args)
        self.parse = parse or _default_parse
        self.env = dict(env or {})
        self.extra_args = list(extra_args)
        self.resume = bool(resume)
        self.log = log or (lambda s: None)
        self.records: list[RunRecord] = []

    # ------------------------------------------------------------------------------------------- one process
    def _cmd(self, params: Mapping[str, Any], freqs: Sequence[float], out: str, tag: str, extra: Sequence[str]):
        cmd = self.argv + [self.params_flag, json.dumps(dict(params)), self.freq_flag, *[f"{f:g}" for f in freqs],
                           self.out_flag, out]
        if self.tag_flag:
            cmd += [self.tag_flag, tag]
        return cmd + self.extra_args + list(extra)

    def _run(self, cmd: list[str], out: str) -> tuple[int | None, float, float, str]:
        os.makedirs(out, exist_ok=True)
        env = os.environ | self.env
        t0 = time.perf_counter()
        with open(os.path.join(out, "stdout.log"), "w") as so:
            proc = subprocess.Popen(cmd, stdout=so, stderr=subprocess.STDOUT, env=env, cwd=out,
                                    start_new_session=True)
            while True:
                pid, st, ru = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    proc.returncode = os.waitstatus_to_exitcode(st)
                    return proc.returncode, time.perf_counter() - t0, ru.ru_maxrss / 1024.0, ""
                if time.perf_counter() - t0 > self.timeout:
                    os.killpg(proc.pid, signal.SIGKILL)
                    _, st, ru = os.wait4(proc.pid, 0)
                    return None, time.perf_counter() - t0, ru.ru_maxrss / 1024.0, f"timeout after {self.timeout:g} s"
                time.sleep(0.2)

    def _reusable(self, out: str, params: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        p, q = os.path.join(out, "result.json"), os.path.join(out, "runner_params.json")
        if not (self.resume and os.path.exists(p) and os.path.exists(q)):
            return None
        try:
            with open(p) as fh:
                res = json.load(fh)
            with open(q) as fh:
                saved = json.load(fh)
        except (OSError, ValueError):
            return None
        same = res.get("status") == "ok" and all(
            k in saved and np.isclose(float(saved[k]), float(v)) if isinstance(v, (int, float)) else saved.get(k) == v
            for k, v in params.items())
        return list(res.get("rows", [])) if same else None

    def run_one(self, params: Mapping[str, Any], freqs: Sequence[float], tag: str) -> RunRecord:
        """One process for ``freqs`` of design ``params`` (retried once with ``fallback_args`` after a SIGBUS)."""
        sub = tag if len(freqs) != 1 else f"{tag}/f{freqs[0]:g}"
        out = os.path.join(self.workdir, sub)
        old = self._reusable(out, params)
        if old is not None:
            rec = RunRecord(tag, list(freqs), "ok", 0, 0.0, 0.0, [{"tag": tag, **dict(params)} | r for r in old],
                            "reused")
            self.records.append(rec)
            self.log(f"{sub}: reused")
            return rec
        retried = False
        extra: list[str] = []
        while True:
            rc, wall, rss, msg = self._run(self._cmd(params, freqs, out, tag, extra), out)
            with open(os.path.join(out, "runner_params.json"), "w") as fh:      # what this folder was solved for
                json.dump(dict(params), fh)
            if rc is None:
                status = "timeout"
            elif rc == -signal.SIGKILL:
                status, msg = "killed", "SIGKILL (out of memory?)"
            elif rc == -signal.SIGBUS:
                status, msg = "sigbus", "SIGBUS (PARDISO/MKL?)"
            else:
                status, rows, pmsg = self.parse(out)
                msg = msg or pmsg
                if rc != 0 and status == "ok":
                    status = "failed"
            if status == "sigbus" and self.fallback_args and not retried:
                retried, extra = True, list(self.fallback_args)
                self.log(f"{sub}: SIGBUS, retrying with {' '.join(extra)}")
                continue
            break
        rows = rows if status == "ok" else []
        prefix = {"tag": tag, **dict(params)}
        rec = RunRecord(tag, list(freqs), status, rc, round(wall, 2), round(rss, 1),
                        [prefix | r for r in rows], msg, retried)
        self.records.append(rec)
        self.log(f"{sub}: {status} in {wall:.0f} s, {rss:.0f} MB")
        return rec

    def run(self, designs: Iterable[Mapping[str, Any]], freqs: Sequence[float],
            tags: Sequence[str] | None = None) -> ResultTable:
        """Every design at every frequency; designs whose script reports ``invalid`` are skipped (recorded)."""
        rows: list[dict[str, Any]] = []
        for i, d in enumerate(designs):
            tag = tags[i] if tags is not None else f"d{i:03d}"
            groups = [[f] for f in freqs] if self.per_frequency else [list(freqs)]
            for g in groups:
                rec = self.run_one(d, g, tag)
                rows += rec.rows
                if rec.status == "invalid":
                    break
        return ResultTable(rows)

    def summary(self) -> dict[str, Any]:
        recs = self.records
        by: dict[str, int] = {}
        for r in recs:
            by[r.status] = by.get(r.status, 0) + 1
        ok = [r for r in recs if r.status == "ok" and r.message != "reused"]
        return {"processes": len(recs), "status": by, "reused": sum(r.message == "reused" for r in recs),
                "wall_s_total": round(sum(r.wall_s for r in recs), 1),
                "wall_s_median_ok": float(np.median([r.wall_s for r in ok])) if ok else None,
                "max_rss_mb": max((r.max_rss_mb for r in recs), default=None)}


def ava_simulator(workdir: str, **kw: Any) -> CLISimulator:
    """A :class:`CLISimulator` on the open AVA model of :mod:`.ava` (SuperLU, one frequency per process).

    The child runs ``ava.py`` as a script (it has no package imports), so it loads NumPy and EMerge only."""
    from . import ava
    argv = [sys.executable, ava.__file__]
    kw.setdefault("env", {"EMERGE_STD_LOGLEVEL": "WARNING"})
    return CLISimulator(argv, workdir, **kw)


# ------------------------------------------------------------------------------------------------------ tables
class ResultTable:
    """Rows of (design, frequency); columns are the union of the rows' keys."""

    def __init__(self, rows: Iterable[Mapping[str, Any]]):
        self.rows = [dict(r) for r in rows]

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def columns(self) -> list[str]:
        cols: list[str] = []
        for r in self.rows:
            cols += [k for k in r if k not in cols]
        return cols

    def col(self, name: str) -> np.ndarray:
        return np.array([_float(r.get(name)) for r in self.rows], dtype=float)

    @property
    def tags(self) -> list[str]:
        out: list[str] = []
        for r in self.rows:
            if str(r.get("tag")) not in out:
                out.append(str(r.get("tag")))
        return out

    @property
    def freqs(self) -> np.ndarray:
        return np.unique(self.col("f_GHz"))

    def param_names(self) -> list[str]:
        """Design-parameter columns: everything before ``f_GHz`` except ``tag``."""
        cols = self.columns
        stop = cols.index("f_GHz") if "f_GHz" in cols else len(cols)
        return [c for c in cols[:stop] if c != "tag"]

    def design(self, tag: str) -> dict[str, Any]:
        r = next(r for r in self.rows if str(r.get("tag")) == tag)
        return {k: r[k] for k in self.param_names() if k in r}

    def where(self, fn: Callable[[dict[str, Any]], bool]) -> ResultTable:
        return ResultTable(r for r in self.rows if fn(r))

    def __add__(self, other: ResultTable) -> ResultTable:
        return ResultTable(self.rows + other.rows)


def _float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _convert(v: str) -> Any:
    v = v.strip()
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if v.lstrip("+-").isdigit() else f


def read_table(path: str) -> ResultTable:
    """Read a results table (CSV, TSV or whitespace separated, one header line, ``#`` comments skipped)."""
    with open(path) as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return ResultTable([])
    head = lines[0]
    if "," in head:
        reader = csv.DictReader(io.StringIO("".join(lines)))
    elif "\t" in head:
        reader = csv.DictReader(io.StringIO("".join(lines)), delimiter="\t")
    else:
        names = head.split()
        reader = (dict(zip(names, ln.split(), strict=False)) for ln in lines[1:])
    return ResultTable({k.strip(): _convert(v) for k, v in row.items() if k is not None} for row in reader)


def write_table(table: ResultTable | Iterable[Mapping[str, Any]], path: str) -> str:
    t = table if isinstance(table, ResultTable) else ResultTable(table)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=t.columns)
        w.writeheader()
        for r in t.rows:
            w.writerow(r)
    return path
