"""The node that checks a flow against the committed results of the example it reproduces (``examples/<name>``), and
the helpers the example-domain nodes share (object artifacts, the checkout's ``examples`` folder).

``examples.compare`` takes the metrics of the nodes upstream of it and, for each comparison it is given, looks up the
reference number in the example's own files: a key path into a committed results file
(``results/takeoff.json:nominal.light_ga.distance_35ft_m``, list items selected by field values with
``runs[aircraft=a320,seed=0].dnz_cg_rms``), a metric of its manifest (``manifest:distance_35ft_a320``) or a metric of
another upstream node that reads a committed file the example keeps in a solver format (``node:committed.tip_uz_m``),
or a cell of a markdown table in the guide the example's numbers are published in when the example commits no
results file (``doc:docs/core_concepts/free_surface.md|+2°|Peak k``). A comparison
passes when ``|flow - reference| <= atol + rtol |reference|``; the tolerance is part of the flow document, next to a
note saying where it comes from, or is a number of the example itself (``atol_ref``, e.g. the spread over the seeds
of a stochastic study). The checks are recorded on the output port, so the run's experiment manifest
carries them.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Any

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact, is_artifact
from ..types import register_port_type

register_port_type("anim_scene", "animated studio scenes (pinneapple_tools.visualization.studio Scene + Animation, "
                                 "object artifact) for the viewer and render nodes")
register_port_type("clip", "rendered video clips and stills (MP4 + PNG) with their labels")

_SEG = re.compile(r"^([^\[\]]*)(?:\[([^\]]+)\])?$")


def save_object(ctx, name: str, obj: Any) -> dict[str, Any]:
    """A Python object (flight histories, wind fields, studio scenes) as a pickle artifact of this node."""
    p = ctx.path(f"{name}.pkl")
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return artifact(p, "pkl")


def load_object(v: Any) -> Any:
    """The object of an artifact written by :func:`save_object` (only artifacts a node of this run wrote)."""
    if not (is_artifact(v) and v.get("kind") == "pkl"):
        raise StepFailed("expected an object artifact written by an upstream node")
    p = Path(v["$artifact"])
    if not p.is_file():
        raise StepFailed(f"artifact {p} is missing (the run folder was removed?)")
    with open(p, "rb") as f:
        return pickle.load(f)  # noqa: S301 - written by a node of this run (save_object), never by a flow document


def examples_root() -> Path | None:
    """The checkout's ``examples`` folder (``$PINNEAPPLE_REPO/examples`` or next to the packages), None when this is
    an installed wheel without it."""
    env = os.environ.get("PINNEAPPLE_REPO")
    base = Path(env) if env else Path(__file__).resolve().parents[2]
    p = base / "examples"
    return p if p.is_dir() else None


def example_dir(name: str) -> Path:
    root = examples_root()
    if root is None:
        raise StepFailed("the examples folder is not here (an installed wheel?): set PINNEAPPLE_REPO to a checkout")
    p = (root / name).resolve()
    if p.parent != root.resolve() or not p.is_dir():
        raise StepFailed(f"unknown example {name!r}")
    return p


def _walk(obj: Any, path: str) -> Any:
    for part in [s for s in path.split(".") if s] if path else []:
        m = _SEG.match(part)
        if not m:
            raise KeyError(part)
        key, sel = m.groups()
        if key:
            obj = obj[int(key)] if isinstance(obj, list) else obj[key]
        if sel:
            if sel.lstrip("-").isdigit():
                obj = obj[int(sel)]
                continue
            want = dict(kv.split("=", 1) for kv in sel.split(","))
            hits = [o for o in obj if all(str(o.get(k)) == v for k, v in want.items())]
            if len(hits) != 1:
                raise KeyError(f"[{sel}] selects {len(hits)} items")
            obj = hits[0]
    return obj


def _split_dotted(folder: Path, ref: str) -> tuple[Path, str]:
    """``results/a.json:x.y`` -> (file, "x.y"); keys may contain dots (``energy_rel_error_dt0.02``), so the key path
    of a results file is split on ``.`` only where the result exists."""
    f, _, path = ref.partition(":")
    p = (folder / f).resolve()
    if folder.resolve() not in p.parents or not p.is_file():
        raise StepFailed(f"reference file {f!r} is not a file of the example")
    return p, path


class Written(float):
    """A reference number with the text it was written as (``1.40`` keeps its two decimals)."""
    text: str = ""


def _written(text: str) -> Written:
    w = Written(float(text))
    w.text = text
    return w


_NUM = re.compile(r"[-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def doc_table_value(ref: str) -> float:
    """``doc:<path under docs/>|<row key>|<column header>``: the first number in that cell of a markdown table (the
    row whose first cell is the key; ``−`` and ``×`` are read as minus and nothing)."""
    from ..examples import ROOT
    parts = ref.removeprefix("doc:").split("|")
    if len(parts) != 3:
        raise KeyError(f"doc reference {ref!r}: doc:<path>|<row key>|<column header>")
    path, row, col = parts
    p = (ROOT / path).resolve()
    if (ROOT / "docs").resolve() not in p.parents or not p.is_file():
        raise StepFailed(f"doc reference {path!r} is not a file under docs/")
    header = None
    for line in p.read_text().splitlines():
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        if cells[0] == row and col in header:
            m = _NUM.search(cells[header.index(col)].replace("×", ""))
            if m:
                return _written(m.group(0).replace("−", "-"))
    raise KeyError(f"no number in row {row!r}, column {col!r} of {path}")


def _repo_file(rel: str) -> Path:
    from ..examples import ROOT
    p = (ROOT / rel).resolve()
    if ROOT.resolve() not in p.parents or not p.is_file():
        raise StepFailed(f"{rel!r} is not a file of the checkout")
    return p


def reference_value(folder: Path, ref: str) -> float:
    """The number ``ref`` names in an example folder (see the module docstring)."""
    if ref.startswith("doc:"):
        return doc_table_value(ref)
    if ref.startswith("text:"):                     # text:<file>|<regex with one group>: a number in committed text
        rel, _, rx = ref[5:].partition("|")
        m = re.search(rx, _repo_file(rel).read_text())
        if not m:
            raise KeyError(f"{rx!r} not found in {rel}")
        return _written(m.group(1))
    if ref.startswith("repo:"):                     # repo:<json file of the checkout>:<key path> (e.g. a model card)
        rel, _, path = ref[5:].partition(":")
        p = _repo_file(rel)
        return reference_value(p.parent, f"{p.name}:{path}")
    if ref.startswith("manifest:"):
        name = ref.split(":", 1)[1]
        man = json.loads((folder / "experiment.json").read_text())
        hit = [m for m in man.get("metrics", []) if m.get("name") == name]
        if not hit:
            raise KeyError(f"no metric {name!r} in the manifest")
        return float(hit[0]["value"])
    p, path = _split_dotted(folder, ref)
    data = json.loads(p.read_text())
    try:
        return float(_walk(data, path))
    except (KeyError, IndexError, TypeError):
        pass
    # keys with dots: try the longest key that exists at each level
    obj, rest = data, path
    while rest:
        if isinstance(obj, dict):
            k = next((k for k in sorted(obj, key=len, reverse=True) if rest == k or rest.startswith(k + ".")), None)
            if k is None:
                raise KeyError(rest)
            obj, rest = obj[k], rest[len(k) + 1:]
        else:
            head, _, rest = rest.partition(".")
            obj = _walk(obj, head)
    return float(obj)


def rounding_half_unit(x: float) -> float:
    """Half a unit of the last digit of ``x`` as Python writes it (0.00147 -> 5e-6); 0 for a number written with 12
    or more significant digits (not rounded)."""
    r = (getattr(x, "text", "") or repr(float(x))).lower()
    mant, _, exp = r.partition("e")
    digits = mant.lstrip("-").replace(".", "").lstrip("0")
    if len(digits) >= 12 or not math.isfinite(x):
        return 0.0
    dec = len(mant.split(".")[1]) if "." in mant else 0
    if mant.endswith(".0") and not getattr(x, "text", ""):
        dec = 0
    return 0.5 * 10.0 ** (-dec + (int(exp) if exp else 0))


def metric_value(reports: dict[str, dict[str, Any]], name: str) -> float:
    """``node.metric`` from the reports of the upstream nodes (a metric with its ``how`` is ``{"value": ...}``)."""
    nid, _, key = name.partition(".")
    rep = reports.get(nid)
    if rep is None:
        raise KeyError(f"no upstream node {nid!r}")
    ms = rep.get("metrics") or {}
    if key not in ms:
        raise KeyError(f"node {nid!r} has no metric {key!r}")
    v = ms[key]
    v = v.get("value") if isinstance(v, dict) else v
    return float(v)


@node("examples.compare", "Compare with the example's results", category="Examples",
      inputs=[Port("metrics", "any", multiple=True, help="the nodes whose metrics are compared (any outputs)")],
      outputs=[Port("metrics", "metrics")],
      params=[Param("example", "str", "", help="folder under examples/ (e.g. flight_dynamics)"),
              Param("comparisons", "list", [],
                    help="[{flow: 'node.metric', ref: 'results/x.json:a.b' | 'manifest:name', rtol, atol, atol_ref}]; "
                         "atol_ref: a number of the example (its seed spread) used as the absolute tolerance"),
              Param("rtol", "float", 1e-6, minimum=0.0, help="default relative tolerance"),
              Param("atol", "float", 0.0, minimum=0.0, help="default absolute tolerance"),
              Param("tolerance_source", "text", "", help="where the tolerance comes from (shown in the report)"),
              Param("rounded", "bool", True, help="a committed number written with few digits matches within half a "
                                                  "unit of its last digit (the example rounded it)"),
              Param("strict", "bool", False, help="fail the node when a comparison does not match")],
      calls="pinneapple_lab.manifest (the example's experiment.json) and the example's committed results files")
def compare(ctx, inputs, params):
    """Each comparison: the flow's measured number against the example's committed one, within the tolerance."""
    folder = example_dir(params["example"])
    rows, checks = [], []
    for c in params["comparisons"]:
        rtol = float(c.get("rtol", params["rtol"]))
        atol = float(c.get("atol", params["atol"]))
        try:
            got = metric_value(ctx.upstream_reports, c["flow"]) * float(c.get("scale", 1.0))   # scale: units (e.g. % )
        except (KeyError, TypeError, ValueError) as exc:
            rows.append([c["flow"], c["ref"], None, None, None, f"no flow value: {exc}"])
            checks.append({"name": c["flow"], "passed": False, "kind": "reference", "detail": f"missing: {exc}"})
            continue
        try:
            ref = (metric_value(ctx.upstream_reports, c["ref"][5:]) if c["ref"].startswith("node:")
                   else reference_value(folder, c["ref"]))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise StepFailed(f"reference {c['ref']!r}: {exc}") from None
        if c.get("atol_ref"):                    # the example's own spread (e.g. over seeds) as the tolerance
            atol = abs(reference_value(folder, c["atol_ref"]))
        if c.get("rounded", params["rounded"]):
            atol = max(atol, rounding_half_unit(ref))
        diff = abs(got - ref)
        ok = bool(math.isfinite(got) and diff <= atol + rtol * abs(ref))
        rel = diff / abs(ref) if ref else diff
        rows.append([c["flow"], c["ref"], got, ref, rel, "match" if ok else "differs"])
        checks.append({"name": c["flow"], "passed": ok, "kind": "reference", "value": got,
                       "detail": f"reference {ref:.10g} ({c['ref']}), relative difference {rel:.3g}, "
                                 f"tolerance atol {atol:g} + rtol {rtol:g}"})
    ctx.table(f"flow against examples/{params['example']} (committed results)",
              ["flow metric", "reference", "flow", "example", "relative difference", "outcome"], rows)
    n_ok = sum(c["passed"] for c in checks)
    ctx.metric("comparisons", len(checks))
    ctx.metric("matched", n_ok)
    if params["tolerance_source"]:
        ctx.metric("tolerance_source", params["tolerance_source"])
    ctx.summary(f"{n_ok} of {len(checks)} numbers match examples/{params['example']}")
    if params["strict"] and n_ok < len(checks):
        raise StepFailed(f"{len(checks) - n_ok} comparison(s) differ from examples/{params['example']}")
    return {"metrics": {"comparisons": len(checks), "matched": n_ok, "rows": rows, "checks": checks}}


@node("examples.flatten", "Numbers of an upstream result", category="Examples",
      inputs=[Port("value", "any", help="a result dict (e.g. the metrics output of a node that returns it)")],
      outputs=[Port("metrics", "metrics")],
      params=[Param("max_items", "int", 400, minimum=1, maximum=10000)],
      calls="no library call: re-records the numbers an upstream node returned (for examples.compare)")
def flatten(ctx, inputs, params):
    """Every number of a nested result as a metric (``a.b.0.c``), so ``examples.compare`` can check the result of a
    node that returns its numbers in an output instead of recording them; strings and booleans are kept too."""
    n = 0

    def walk(v, key):
        nonlocal n
        if n >= int(params["max_items"]):
            return
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{key}.{k}" if key else str(k))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                walk(x, f"{key}.{i}")
        elif isinstance(v, (bool, int, float, str)) and key:
            ctx.metric(key, v)
            n += 1
    walk(inputs["value"], "")
    ctx.summary(f"{n} values")
    return {"metrics": ctx.report["metrics"]}
