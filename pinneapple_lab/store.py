"""The lab's database: run folders plus a SQLite index, queries, dataset export and the catalogue."""
from __future__ import annotations

import glob
import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from typing import Any

import numpy as np

from .dataset import read_dataset

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, experiment TEXT NOT NULL, version TEXT, params TEXT, status TEXT,
    started REAL, finished REAL, seconds REAL, checks_total INTEGER, checks_failed INTEGER,
    metrics TEXT, git TEXT, error TEXT, tags TEXT
);
CREATE TABLE IF NOT EXISTS metrics (run_id TEXT, name TEXT, value REAL, PRIMARY KEY (run_id, name));
CREATE TABLE IF NOT EXISTS datasets (run_id TEXT, name TEXT, n_samples INTEGER, path TEXT, PRIMARY KEY (run_id, name));
CREATE INDEX IF NOT EXISTS runs_experiment ON runs (experiment);
"""


def default_root() -> str:
    """``$PINNEAPPLE_LAB`` or ``./lab``."""
    return os.environ.get("PINNEAPPLE_LAB", os.path.abspath("lab"))


class LabStore:
    """Experiment database rooted at ``root`` (runs in ``root/runs/<experiment>/<run_id>``, index ``root/index.sqlite``)."""

    def __init__(self, root: str | None = None):
        self.root = os.path.abspath(root or default_root())
        os.makedirs(os.path.join(self.root, "runs"), exist_ok=True)
        self.db = os.path.join(self.root, "index.sqlite")
        with closing(self._conn()) as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db, timeout=60)
        c.row_factory = sqlite3.Row
        return c

    def run_dir(self, experiment: str, run_id: str) -> str:
        return os.path.join(self.root, "runs", experiment, run_id)

    # -- writing -----------------------------------------------------------
    def index_run(self, record: dict[str, Any], metrics: dict[str, Any], datasets: dict[str, dict[str, Any]]) -> None:
        checks = record.get("validation", {})
        with closing(self._conn()) as c, c:
            c.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                record["run_id"], record["experiment"], record.get("version"), json.dumps(record.get("params", {}),
                                                                                         sort_keys=True),
                record.get("status"), record.get("started"), record.get("finished"), record.get("seconds"),
                checks.get("total", 0), checks.get("failed", 0), json.dumps(metrics), record.get("git"),
                record.get("error"), json.dumps(record.get("tags", []))))
            c.execute("DELETE FROM metrics WHERE run_id = ?", (record["run_id"],))
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    c.execute("INSERT INTO metrics VALUES (?,?,?)", (record["run_id"], k, float(v)))
            c.execute("DELETE FROM datasets WHERE run_id = ?", (record["run_id"],))
            for name, card in datasets.items():
                c.execute("INSERT INTO datasets VALUES (?,?,?,?)", (record["run_id"], name, card.get("n_samples", 0),
                                                                    card.get("path", "")))

    def reindex(self) -> int:
        """Rebuild the index from the run folders (e.g. after copying runs from another machine)."""
        n = 0
        for p in glob.glob(os.path.join(self.root, "runs", "*", "*", "run.json")):
            d = os.path.dirname(p)
            with open(p) as f:
                rec = json.load(f)
            metrics = json.load(open(os.path.join(d, "metrics.json"))) if os.path.exists(os.path.join(d, "metrics.json")) else {}
            cards = {}
            for cp in glob.glob(os.path.join(d, "datasets", "*", "card.json")):
                card = json.load(open(cp))
                card["path"] = os.path.relpath(os.path.dirname(cp), self.root)
                cards[card["name"]] = card
            self.index_run(rec, metrics, cards)
            n += 1
        return n

    # -- reading -----------------------------------------------------------
    def runs(self, experiment: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM runs WHERE 1=1", []
        if experiment:
            q += " AND experiment = ?"
            args.append(experiment)
        if status:
            q += " AND status = ?"
            args.append(status)
        with closing(self._conn()) as c:
            rows = [dict(r) for r in c.execute(q + " ORDER BY started", args)]
        for r in rows:
            r["params"] = json.loads(r["params"] or "{}")
            r["metrics"] = json.loads(r["metrics"] or "{}")
            r["tags"] = json.loads(r["tags"] or "[]")
        return rows

    def get(self, run_id: str) -> dict[str, Any]:
        rows = [r for r in self.runs() if r["run_id"] == run_id]
        if not rows:
            raise KeyError(run_id)
        r = rows[0]
        r["dir"] = self.run_dir(r["experiment"], run_id)
        return r

    def table(self, experiment: str):
        """Runs of one experiment as a pandas DataFrame: params and metrics as columns."""
        import pandas as pd
        rows = []
        for r in self.runs(experiment):
            row = {"run_id": r["run_id"], "status": r["status"], "seconds": r["seconds"],
                   "checks_failed": r["checks_failed"]}
            row.update({f"param.{k}": v for k, v in r["params"].items()})
            row.update({f"metric.{k}": v for k, v in r["metrics"].items() if not isinstance(v, (list, dict))})
            rows.append(row)
        return pd.DataFrame(rows)

    def status(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        with closing(self._conn()) as c:
            for r in c.execute("SELECT experiment, status, COUNT(*) n FROM runs GROUP BY experiment, status"):
                out.setdefault(r["experiment"], {})[r["status"]] = r["n"]
        return out

    def datasets(self, experiment: str | None = None, name: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT d.*, r.experiment, r.status FROM datasets d JOIN runs r USING (run_id) WHERE 1=1"
        args: list[Any] = []
        if experiment:
            q += " AND r.experiment = ?"
            args.append(experiment)
        if name:
            q += " AND d.name = ?"
            args.append(name)
        with closing(self._conn()) as c:
            return [dict(r) for r in c.execute(q, args)]

    def samples(self, experiment: str, name: str, *, status: str = "completed",
                with_params: bool = True) -> Iterator[dict[str, Any]]:
        """Iterate over every sample of dataset ``name`` across the runs of ``experiment`` (default: completed
        runs only, so a failed or unvalidated run does not leak into a training set)."""
        runs = {r["run_id"]: r for r in self.runs(experiment, status=status)}
        for d in self.datasets(experiment, name):
            if d["run_id"] not in runs:
                continue
            for s in read_dataset(os.path.join(self.root, d["path"])):
                s["_run_id"] = d["run_id"]
                if with_params:
                    s.update({f"param.{k}": v for k, v in runs[d["run_id"]]["params"].items()})
                yield s

    def export_dataset(self, experiment: str, name: str, out: str, *, status: str = "completed") -> dict[str, Any]:
        """Concatenate a dataset over runs into one ``.npz`` (arrays stacked per key where shapes agree, plus
        the scalar columns) and a ``.card.json`` next to it. Returns the card."""
        cols: dict[str, list] = {}
        n = 0
        for s in self.samples(experiment, name, status=status):
            for k, v in s.items():
                cols.setdefault(k, [None] * n).append(v)
            for k in cols:
                if len(cols[k]) < n + 1:
                    cols[k].append(None)
            n += 1
        arrays, skipped = {}, []
        for k, vals in cols.items():
            if any(v is None for v in vals):
                skipped.append(k)
                continue
            try:
                arrays[k] = np.stack([np.asarray(v) for v in vals])
            except ValueError:
                skipped.append(k)
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        np.savez_compressed(out, **{k.replace(".", "__"): v for k, v in arrays.items()})
        card = {"experiment": experiment, "dataset": name, "n_samples": n, "keys": {k: list(v.shape) for k, v in arrays.items()},
                "skipped_keys": skipped, "status_filter": status, "created": time.time(),
                "runs": sorted({s for s in cols.get("_run_id", []) if s})}
        with open(os.path.splitext(out)[0] + ".card.json", "w") as f:
            json.dump(card, f, indent=1)
        return card

    # -- catalogue -----------------------------------------------------------
    def datasets_catalog(self, path: str | None = None) -> str:
        """Write DATASETS.md: every dataset with its description, sample count over completed runs, schema
        (shapes, dtypes, ranges, units) and the commands that rebuild and export it. Shards are not versioned;
        the runs are cached by parameters, so the sweep command regenerates exactly what is listed."""
        path = path or os.path.join(self.root, "DATASETS.md")
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for d in self.datasets():
            if d["status"] == "completed":
                groups.setdefault((d["experiment"], d["name"]), []).append(d)
        total = sum(x["n_samples"] for v in groups.values() for x in v)
        lines = ["# PINNeAPPle Lab datasets", "",
                 f"{len(groups)} datasets, {total} samples from completed (validated) runs. Arrays are not in git: "
                 "rebuild a dataset with its sweep command (cached runs are skipped), then export it as one file.", ""]
        for (exp, name), ds in sorted(groups.items()):
            card: dict[str, Any] = {}
            try:
                with open(os.path.join(self.root, ds[0]["path"], "card.json")) as f:
                    card = json.load(f)
            except OSError:
                pass
            params = sorted({k for r in self.runs(exp, status="completed") for k in r["params"]})
            lines += [f"## `{exp}` / `{name}`", "", card.get("description", ""), "",
                      f"{sum(x['n_samples'] for x in ds)} samples from {len(ds)} runs; every sample also carries the "
                      f"run parameters ({', '.join(params)}).", "",
                      "| field | kind | shape / type | range | units |", "|---|---|---|---|---|"]
            units = card.get("units", {})
            for k, s in card.get("schema", {}).items():
                shape = s.get("shape", s.get("type", ""))
                rng = (f"{s['min']:.4g} .. {s['max']:.4g}" if "min" in s and s["min"] is not None
                       else ", ".join(map(str, s.get("values", [])[:6])))
                lines.append(f"| {k} | {s.get('kind', '')} | {shape} {s.get('dtype', '')} | {rng} | {units.get(k, '')} |")
            runs = self.runs(exp, status="completed")
            vary = {k: sorted({json.dumps(r["params"].get(k)) for r in runs}) for k in params}
            grid = "".join(f" -g {k}={','.join(v.strip(chr(34)) for v in vals)}"
                           for k, vals in vary.items() if len(vals) > 1)
            if any(len(v) > 12 for v in vary.values()):                # a Latin-hypercube sweep, not a grid
                grid = f" -n {len(runs)}"
            lines += ["", f"Schema ranges are those of one run (`{ds[0]['run_id']}`).", "", "```bash",
                      f"python -m pinneapple_lab sweep {exp}{grid}",
                      f"python -m pinneapple_lab export {exp} {name} {exp}_{name}.npz", "```", ""]
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def catalog_html(self, path: str | None = None, **kw) -> str:
        """Self-contained HTML catalogue (filters, run cards, a sheet with each run's full record)."""
        from .html import write_html
        return write_html(self, path, **kw)

    def catalog(self, path: str | None = None, *, max_rows: int = 30, thumbs: int = 2) -> str:
        """Write a Markdown catalogue: per experiment, a status summary, a table of runs (parameters, key
        metrics, checks) and thumbnails of the first figures."""
        from .spec import get
        path = path or os.path.join(self.root, "CATALOG.md")
        lines = ["# PINNeAPPle Lab catalogue", "",
                 f"{sum(sum(v.values()) for v in self.status().values())} runs, generated {time.strftime('%Y-%m-%d %H:%M')}.", ""]
        lines += ["| experiment | runs | completed | failed validation | failed | datasets (samples) |", "|---|---|---|---|---|---|"]
        stat = self.status()
        for exp in sorted(stat):
            ds = self.datasets(exp)
            dsum = {}
            for d in ds:
                dsum[d["name"]] = dsum.get(d["name"], 0) + d["n_samples"]
            st = stat[exp]
            lines.append(f"| [{exp}](#{exp}) | {sum(st.values())} | {st.get('completed', 0)} | "
                         f"{st.get('failed_validation', 0)} | {st.get('failed', 0)} | "
                         + (", ".join(f"{k} ({v})" for k, v in dsum.items()) or "-") + " |")
        for exp in sorted(stat):
            try:
                cls = get(exp)
                desc = cls.description
            except KeyError:
                desc = ""
            runs = self.runs(exp)
            lines += ["", f"## {exp}", "", desc, ""]
            pkeys = sorted({k for r in runs for k in r["params"]})
            mkeys = sorted({k for r in runs for k, v in r["metrics"].items() if isinstance(v, (int, float))})[:6]
            lines.append("| run | status | " + " | ".join(pkeys + mkeys) + " | checks |")
            lines.append("|" + "---|" * (3 + len(pkeys) + len(mkeys)))
            for r in runs[-max_rows:]:
                cells = [str(r["params"].get(k, "")) for k in pkeys]
                cells += [f"{r['metrics'][k]:.4g}" if isinstance(r["metrics"].get(k), (int, float)) else "" for k in mkeys]
                ok = f"{r['checks_total'] - r['checks_failed']}/{r['checks_total']}"
                rel = os.path.relpath(self.run_dir(exp, r["run_id"]), os.path.dirname(path))
                lines.append(f"| [{r['run_id'][-12:]}]({rel}) | {r['status']} | " + " | ".join(cells) + f" | {ok} |")
            shown = 0
            for r in reversed(runs):
                if shown >= thumbs:
                    break
                figs = sorted(glob.glob(os.path.join(self.run_dir(exp, r["run_id"]), "figures", "*.png")))
                if figs:
                    lines += ["", f"![{exp}]({os.path.relpath(figs[0], os.path.dirname(path))})"]
                    shown += 1
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path
