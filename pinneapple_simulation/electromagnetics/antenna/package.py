"""Read an external study package from a local folder and re-run one of its designs.

A package is a folder with a results table (one row per (design, frequency), the columns of :mod:`.runner`) and,
optionally, the parametric command-line script that produced it. Nothing from it is copied into the repository:
it is read where it is. An optional ``package.json`` says which files to use and how to call the script::

    {"table": "results.csv",
     "script": "ava_cli.py",
     "argv": ["{python}", "{script}"],           # the command before the design / frequency flags
     "flags": {"params": "--params", "freq": "--freq", "out": "--out", "tag": "--tag"},
     "length_col": "Lb"}

Without it the first ``*.csv`` / ``*.tsv`` / ``*.txt`` table and the first ``*.py`` script are used with the flags of
:mod:`.ava` (``--params JSON --freq GHz --out DIR --tag TAG``, ``result.json`` in the output folder).
"""
from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from .runner import CLISimulator, ResultTable, read_table

__all__ = ["Package", "load_package"]

_TABLE_EXT = (".csv", ".tsv", ".txt", ".dat")


@dataclass
class Package:
    root: str
    table_path: str
    table: ResultTable
    script: str | None = None
    argv: list[str] = field(default_factory=list)
    flags: dict[str, str] = field(default_factory=dict)
    length_col: str = "Lb"

    def simulator(self, workdir: str, **kw: Any) -> CLISimulator:
        """A :class:`.runner.CLISimulator` on the package's script (one frequency per process)."""
        if not self.script:
            raise FileNotFoundError(f"{self.root}: the package has no script to re-run")
        argv = [a.format(python=sys.executable, script=self.script) for a in self.argv]
        fl = {"params": "--params", "freq": "--freq", "out": "--out", "tag": "--tag"} | self.flags
        return CLISimulator(argv, workdir, params_flag=fl["params"], freq_flag=fl["freq"], out_flag=fl["out"],
                            tag_flag=fl.get("tag"), **kw)

    def describe(self) -> dict[str, Any]:
        t = self.table
        return {"root": self.root, "table": os.path.basename(self.table_path), "rows": len(t),
                "designs": len(t.tags), "frequencies_GHz": [float(f) for f in t.freqs],
                "parameters": t.param_names(), "script": os.path.basename(self.script) if self.script else None}


def load_package(root: str) -> Package:
    """The package in folder ``root`` (see the module docstring)."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no package folder {root}")
    cfg: dict[str, Any] = {}
    p = os.path.join(root, "package.json")
    if os.path.exists(p):
        with open(p) as fh:
            cfg = json.load(fh)
    if "table" in cfg:
        table_path = os.path.join(root, cfg["table"])
    else:
        cands = sorted(f for f in glob.glob(os.path.join(root, "*")) if f.lower().endswith(_TABLE_EXT))
        if not cands:
            raise FileNotFoundError(f"{root}: no results table (*.csv, *.tsv, *.txt)")
        table_path = cands[0]
    script = os.path.join(root, cfg["script"]) if cfg.get("script") else \
        next(iter(sorted(glob.glob(os.path.join(root, "*.py")))), None)
    table = read_table(table_path)
    missing = [c for c in ("tag", "f_GHz") if c not in table.columns]
    if missing:
        raise ValueError(f"{table_path}: missing columns {missing}")
    return Package(root, table_path, table, script, list(cfg.get("argv", ["{python}", "{script}"])),
                   dict(cfg.get("flags", {})), str(cfg.get("length_col", "Lb")))
