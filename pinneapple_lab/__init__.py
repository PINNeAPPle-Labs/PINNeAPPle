"""PINNeAPPle Lab: run many experiments reproducibly and grow a database of results and datasets.

An experiment is a class with parameters and a ``run(ctx)`` method. The runner gives every run an id from the
experiment, its version and its parameters, records its inputs, outputs, metrics, validation checks,
figures and datasets in one folder, keeps its status (running, completed, failed, failed_validation) and indexes
everything in SQLite, so runs can be queried, compared, exported as training sets and summarised in a catalogue.

    from pinneapple_lab import Experiment, register, run, sweep, LabStore

    @register
    class Oscillator(Experiment):
        name = "oscillator"
        params = {"zeta": 0.1, "omega": 2.0, "dt": 0.01}

        def run(self, ctx):
            t, x = simulate(**ctx.params)
            ctx.output("trajectory", x)
            ctx.metric("l2_error", err)
            ctx.check("accurate", err < 1e-3)
            ds = ctx.dataset("trajectories")
            ds.add(t=t, x=x, label=ctx.params["zeta"])

    run("oscillator", {"zeta": 0.3})                          # cached: same params -> same run
    sweep("oscillator", grid={"zeta": [0.05, 0.1, 0.5]}, n_jobs=3)
    LabStore().catalog()                                       # CATALOG.md with tables and thumbnails

Command line: ``python -m pinneapple_lab list | run NAME -p k=v | sweep NAME -g k=v1,v2 | status | report | export``.

Built-in experiments (``pinneapple_lab.experiments``) span ODEs, PDEs solved by X-TFC, black-hole hydrodynamics
and lattice-Boltzmann flow; new ones only need a class.
"""
from .context import RunContext
from .runner import RunResult, run, sweep
from .spec import Experiment, available, get, register
from .store import LabStore, default_root

__all__ = ["Experiment", "LabStore", "RunContext", "RunResult", "available", "default_root", "get", "register",
           "run", "sweep"]
