"""The files a CalculiX run writes beside the deck (.dat, .sta) and the solver's own log: the numbers a verification
compares, read from what the solver printed — never from the deck that asked for them.

``.dat`` is the file ``*NODE PRINT`` and ``*EL PRINT`` fill. Two of its blocks matter to a verification, and they are
different questions:

* ``total force (fx,fy,fz) for set <NSET> and time <t>`` — the summed reaction of a node set, one line per increment:
  this is the equilibrium check (the load applied is the load the support carries) and the measured friction force
  (``|F_t| = mu*N``) without integrating a pressure field;
* ``displacements (vx,vy,vz) for set <NSET> and time <t>`` — the printed nodal displacements of a set.

Note what is **not** here: the per-node ``forces`` block of the same file is a nodal field, and its numbers start with a
node id where the total's do not — reading one as the other is how a "reaction" ends up being a single node's share.

``.sta`` and the log answer a third question, whether the run is worth reading at all. ``no convergence`` printed by
``ccx`` means nothing on its own (it prints it for every iteration that needs another pass and then prints
``convergence`` for the same increment); the two messages that settle a run are ``too many attempts`` and
``Job finished``.

The ``.dat`` metadata view (largest displacement, min/max of a printed result) is
``pinneapple_data.simulation_metadata.calculix.parse_dat``; this module reads the totals that view does not carry.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# " total force (fx,fy,fz) for set BFIX and time  0.2500000E+00"
_TOTAL = re.compile(r"^\s*total force \((fx,fy,fz)\) for set (\S+) and time\s+(" + _FLOAT + r")", re.I)
_DISP = re.compile(r"^\s*displacements \((vx,vy,vz)\) for set (\S+) and time\s+(" + _FLOAT + r")", re.I)


def _text(src: bytes | str) -> str:
    return src.decode("latin-1", "replace") if isinstance(src, (bytes, bytearray)) else src


# --------------------------------------------------------------------------------------------------------- .dat
@dataclass
class Totals:
    """The summed reaction of a node set at the last increment, and the history of it over the run."""
    set_name: str
    values: tuple[float, float, float]          # (fx, fy, fz) of the last increment
    time: float
    history: list[tuple[float, tuple[float, float, float]]] = field(default_factory=list)

    def at(self, time: float, tol: float = 1e-9) -> tuple[float, float, float] | None:
        """The reaction at the increment whose time is ``time`` (the last one at or before it)."""
        rows = [v for t, v in self.history if t <= time + tol]
        return rows[-1] if rows else None


def read_totals(dat: bytes | str) -> dict[str, Totals]:
    """``set name -> total reaction``, with the per-increment history of the same set.

    Only ``total force`` lines are read: the per-node ``forces`` block of the same file starts with a node id and
    would be read as a first component if the pattern did not demand the word ``total``.
    """
    lines = _text(dat).splitlines()
    out: dict[str, Totals] = {}
    for i, line in enumerate(lines):
        m = _TOTAL.match(line)
        if not m:
            continue
        # the numbers are on the next line, printed after a blank one
        row = next((ln for ln in lines[i + 1:i + 4] if ln.strip()), "")
        nums = [float(v) for v in re.findall(_FLOAT, row)[:3]]
        if len(nums) != 3:
            continue
        name, t = m.group(2), float(m.group(3))
        values = (nums[0], nums[1], nums[2])
        hist = out[name].history if name in out else []
        hist.append((t, values))
        out[name] = Totals(name, values, t, hist)
    return out


def read_displacements(dat: bytes | str) -> dict[str, dict[int, tuple[float, float, float]]]:
    """``set name -> node -> displacement``, of the last increment printed for that set."""
    lines = _text(dat).splitlines()
    out: dict[str, dict[int, tuple[float, float, float]]] = {}
    i = 0
    while i < len(lines):
        m = _DISP.match(lines[i])
        if not m:
            i += 1
            continue
        nodes: dict[int, tuple[float, float, float]] = {}
        for line in lines[i + 1:]:
            if not line.strip():
                if nodes:                                              # the block ends at the blank line
                    break
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 4 or not parts[0].lstrip("+-").isdigit():
                break
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
        out[m.group(2)] = nodes
        i += 1
    return out


# --------------------------------------------------------------------------------------------------------- .sta
@dataclass
class Status:
    """What the increment summary records: how many increments were written, how many attempts they cost, and how
    far the step time got. ``converged`` is the only claim made here — an increment that was written."""
    increments: int = 0
    attempts: int = 0
    last_time: float = 0.0

    @property
    def converged(self) -> bool:
        return self.increments > 0 and self.last_time > 0


def read_status(sta: bytes | str) -> Status:
    """The ``.sta`` summary ``ccx`` appends: a row is one convergent increment, its third column the attempts it took
    (``2U`` marks an attempt that was cut back)."""
    rows = 0
    attempts = 0
    last = 0.0
    for line in _text(sta).splitlines():
        parts = line.split()
        if len(parts) != 7 or not all(re.fullmatch(r"[-+]?\d+", x) for x in parts[:4]):
            continue
        try:
            attempts += int(parts[2])
            last = max(last, float(parts[4]))
        except ValueError:
            continue
        rows += 1
    return Status(increments=rows, attempts=attempts, last_time=last)


# --------------------------------------------------------------------------------------------------------- log
FINISHED = "Job finished"
GIVE_UP = ("too many attempts", "too many cutbacks")   # the solver cut back until it stopped: no equilibrium


@dataclass
class Verdict:
    """How the run ended, and the line of the log that says so."""
    status: str            # ok | failed | not_converged | unavailable
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def read_log(log: bytes | str, status: Status | None = None, produced: bool = True) -> Verdict:
    """Classify a ``ccx`` run the way the solver reports it, never by the exit code alone.

    ``no convergence`` on its own means nothing: ``ccx`` prints it for every iteration that needs another pass and
    then prints ``convergence`` for the same increment, so a normal run contains both. The two messages that settle a
    run are ``too many attempts`` (the step gave up — for a contact problem that is usually the physical answer, a
    joint that could not hold) and ``Job finished`` (it reached ``*END STEP``). A run that finished but wrote no
    result file has not converged, whatever the exit code says.
    """
    text = _text(log)
    low = text.lower()
    if any(g in low for g in GIVE_UP):
        return Verdict("not_converged", "the solver cut the increment back until it gave up (too many attempts): "
                                        "the interface found no equilibrium in that step")
    errors = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("*ERROR")]
    if errors:
        return Verdict("failed", errors[0][:400])
    if FINISHED.lower() not in low:
        return Verdict("not_converged", "the run did not reach *END STEP")
    if status is not None and not status.converged:
        return Verdict("not_converged", "the .sta records no completed increment")
    if not produced:
        return Verdict("not_converged", "the run produced no result file")
    return Verdict("ok", "job finished")


def finite(*values: float) -> bool:
    """Whether every value is a finite number — the guard before a comparison is written down."""
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
