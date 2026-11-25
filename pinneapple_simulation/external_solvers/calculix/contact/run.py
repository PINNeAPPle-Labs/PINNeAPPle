"""One contact run, end to end: build the deck, solve it with the ``ccx`` PrePoMax ships, read the results back through
:mod:`pinneapple_data.cae` and compare them with the closed form of :mod:`.verify`.

Nothing here decides on its own whether a run is good. It produces the deck, the solver's own files, the numbers as
read and the comparisons; the caller writes the report and the evidence record from those. The result files are read
by the data layer and not here, so a value that a report shows and a value a verification compares come from the same
fixed columns.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pinneapple_data import cae

from . import cases as C


class SolverMissing(RuntimeError):
    """No ``ccx`` to run: the deck is still written and the run says so instead of guessing a result."""


def ccx_binary(explicit: str | None = None) -> str | None:
    """Where the solver is: the setting when it names a file, else ``ccx`` on ``PATH``. None when there is none."""
    for candidate in (explicit, os.environ.get("CCX_BIN"), "ccx"):
        if not candidate:
            continue
        found = shutil.which(candidate)
        if found:
            return found
    return None


def version(binary: str) -> str | None:
    """The CalculiX version, read from the banner ``ccx -v`` prints (it exits 0 on every version we know of)."""
    try:
        out = subprocess.run([binary, "-v"], capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (out.stdout + out.stderr).splitlines():
        if "Version" in line:
            found = re.search(r"Version\s+(\S+)", line)
            return found.group(1) if found else line.strip()
    return None


def solve(deck_path: Path, binary: str, timeout: float, workdir: Path) -> dict[str, Any]:
    """Run ``ccx`` on the deck in its own directory and read back whatever it wrote.

    The solver reads its argument as a file stem in the working directory, writes ``.dat``/``.frd``/``.sta`` beside it,
    and asks on stdin for another file when the deck ends early — hence the closed stdin: without it a malformed deck
    would hang instead of failing.
    """
    deck_path, workdir = Path(deck_path), Path(workdir)
    t0 = time.perf_counter()
    cmd = [binary, "-i", deck_path.stem]
    try:
        proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, env={**os.environ, "OMP_NUM_THREADS": "1"})
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:  # the solver is still running: nothing it wrote is a result
        (workdir / "ccx.log").write_text((e.stdout or "") if isinstance(e.stdout, str) else "", encoding="utf-8")
        return {"status": "timeout", "detail": f"the solver ran longer than {timeout:g} s", "seconds": timeout,
                "exit_code": None, "path": deck_path}
    elapsed = time.perf_counter() - t0
    log = workdir / "ccx.log"
    log.write_text((stdout or "") + (stderr or ""), encoding="latin-1")
    status = cae.read_status(_read(workdir / f"{deck_path.stem}.sta"))
    verdict = cae.read_log(log.read_bytes(), status, produced=(workdir / f"{deck_path.stem}.dat").is_file())
    if code != 0 and verdict.ok:
        verdict = cae.Verdict("failed", f"the solver exited with {code}")
    return {"status": verdict.status, "detail": verdict.detail, "seconds": round(elapsed, 2), "exit_code": code,
            "path": deck_path, "status_file": status, "converged": status.converged}


def _read(path: Path) -> bytes:
    """The bytes of a file the solver may not have written (an empty file reads as a file with nothing in it)."""
    return path.read_bytes() if path.is_file() else b""


def read_results(deck_path: Path, times: dict[str, float] | None = None) -> dict[str, Any]:
    """Everything a verification reads out of one run, through :mod:`pinneapple_data.cae`.

    ``contact_until`` holds the contact state at each step time a case asks for, which is how a two-step case reads
    what the preload did apart from what the pull did.
    """
    deck_path = Path(deck_path)
    workdir, stem = deck_path.parent, deck_path.stem
    frd = _read(workdir / f"{stem}.frd")
    dat = _read(workdir / f"{stem}.dat")
    return {"totals": cae.read_totals(dat) if dat else {},
            "contact": cae.contact(frd) if frd else {},
            "contact_until": {name: cae.contact(frd, until=t) for name, t in (times or {}).items()} if frd else {},
            "displacement": cae.displacement(frd) if frd else {},
            "stress": cae.stress(frd) if frd else {},
            "printed_displacements": cae.read_displacements(dat) if dat else {},
            "status": cae.read_status(_read(workdir / f"{stem}.sta")),
            "verdict": cae.read_log(_read(workdir / "ccx.log"),
                                    cae.read_status(_read(workdir / f"{stem}.sta")),
                                    produced=(workdir / f"{stem}.dat").is_file())}


def run_case(case_id: str | None, params: dict[str, Any] | None, out_dir: Path, binary: str | None = None,
             timeout: float = 600.0, cases: Mapping[str, C.Case] | None = None) -> dict[str, Any]:
    """Build, solve, read and verify one case. ``out_dir`` receives the deck and the solver's files.

    The return value is what a report and an evidence record are written from: the resolved parameters, the closed
    form, the deck, the solver, the measured numbers as read, and the checks. No Markdown is produced here — that is
    the caller's, and it must not be able to change a number the checks saw.
    """
    case, resolved, built = C.build(case_id, params, cases)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    deck_path = out / f"{case.id}.inp"
    deck_path.write_text(built.deck, encoding="utf-8")
    result: dict[str, Any] = {"case": case.id, "title": case.title, "params": resolved, "analytic": built.analytic,
                              "notes": list(built.notes), "interface": list(built.interface),
                              "coords": built.coords,
                              "deck": {"name": deck_path.name, "bytes": deck_path.stat().st_size},
                              "checks": [], "solver": {"binary": None, "available": False}}
    solver = binary or ccx_binary()
    if solver is None:
        result["error"] = ("no CalculiX solver (ccx) in this deployment: the deck is written, but nothing was solved. "
                           "PrePoMax distributes ccx; set CCX_BIN to point at it.")
        return result
    result["solver"] = {"binary": solver, "version": version(solver), "available": True}
    outcome = solve(deck_path, solver, timeout, out)
    parsed = read_results(deck_path, built.times)
    parsed["verdict"] = outcome["status"]
    checks = [_verdict_check(outcome)]
    if outcome["status"] in ("ok", "not_converged"):
        checks += case.checks(resolved, built, parsed)
    result.update({"ok": outcome["status"] == "ok", "status": outcome["status"], "detail": outcome["detail"],
                   "seconds": outcome["seconds"], "checks": checks, "contact": parsed.get("contact") or {},
                   "measured": measured(parsed, built)})
    if outcome["status"] != "ok":
        result["error"] = outcome["detail"]
    return result


def _verdict_check(outcome: dict[str, Any]) -> Any:
    from . import verify as V

    ok = outcome["status"] == "ok"
    increments = getattr(outcome.get("status_file"), "increments", 0)
    return V.Check("solver.convergiu", V.PASS if ok else V.FAIL, value=1.0 if ok else 0.0, reference=1.0,
                   detail=f"{outcome['status']}: {outcome['detail']}; {increments} incremento(s) em "
                          f"{outcome['seconds']} s", source=".sta e log do CalculiX")


def measured(parsed: dict[str, Any], built: C.Built) -> dict[str, Any]:
    """The numbers the run produced, as read — never adjusted towards the reference."""
    interface = set(built.interface) if built.interface else None
    rows = [r for n, r in (parsed.get("contact") or {}).items() if interface is None or n in interface]
    pressures = [r["pressure"] for r in rows if r["pressure"] > 0.0]
    totals = {name: {"fx": t.values[0], "fy": t.values[1], "fz": t.values[2], "time": t.time}
              for name, t in (parsed.get("totals") or {}).items()}
    return {"reactions": totals, "contact_nodes": len(pressures),
            "peak_pressure": max(pressures) if pressures else None,
            "max_shear": max((abs(r["shear1"]) for r in rows), default=None),
            "max_slip": max((abs(r["slip1"]) for r in rows), default=None),
            "coulomb_ratio": cae.coulomb_ratio(parsed.get("contact") or {}, built.interface or None)}
