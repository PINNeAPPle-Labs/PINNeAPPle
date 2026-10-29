"""Track D: measured ergonomics, facts only.

* runner lines: non-blank, non-comment lines of each framework's runner per track, without the module docstring
  (the neutral shared code in pinneapple_tools.benchmark_suite.crossframework is not counted for anyone);
* install size: bytes of the files the framework's own distribution installed (``importlib.metadata`` records),
  dependencies not included;
* import time: wall time of ``import <module>`` in a fresh interpreter, median of 3.

    python -m benchmarks.vs_frameworks.measure_ergonomics      # writes results/D.json
"""
from __future__ import annotations

import ast
import io
import json
import statistics
import subprocess
import sys
import tokenize
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMEWORKS = {
    "pinneapple": {"dists": ["pinneapple"], "module": "pinneapple", "runners": {"A": "track_a_pinneapple",
                   "B": "track_b_pinneapple", "C": "track_c_pinneapple"}},
    "deepxde": {"dists": ["deepxde"], "module": "deepxde", "runners": {"A": "track_a_deepxde", "C": "track_c_deepxde"}},
    "physicsnemo_sym": {"dists": ["nvidia-physicsnemo.sym"], "module": "physicsnemo.sym",
                        "runners": {"A": "track_a_physicsnemo_sym", "C": "track_c_physicsnemo_sym"}},
    "physicsnemo": {"dists": ["nvidia-physicsnemo"], "module": "physicsnemo.models.fno",
                    "runners": {"B": "track_b_physicsnemo"}},
    "neuraloperator": {"dists": ["neuraloperator"], "module": "neuralop", "runners": {"B": "track_b_neuraloperator"}},
}


def code_lines(path: Path) -> int:
    src = path.read_text()
    tree = ast.parse(src)
    skip = set()
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None),
                                                                        ast.Constant):
        skip.update(range(tree.body[0].lineno, tree.body[0].end_lineno + 1))
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                        tokenize.ENDMARKER):
            continue
        for ln in range(tok.start[0], tok.end[0] + 1):
            if ln not in skip:
                lines.add(ln)
    return len(lines)


def source_tree_mb() -> float:
    """PINNeAPPle is installed in editable mode here, so its record lists no files: the size of its ``pinneapple*``
    package directories in the checkout (without ``__pycache__``) is used instead (what a wheel would carry)."""
    root = HERE.parents[1]
    total = 0
    for pkg in root.glob("pinneapple*"):
        if pkg.is_dir() and (pkg / "__init__.py").exists():
            total += sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    return total / 1e6


def install_mb(dists) -> float:
    if dists == ["pinneapple"]:
        return source_tree_mb()
    total = 0
    for d in dists:
        dist = metadata.distribution(d)
        for f in dist.files or []:
            p = Path(dist.locate_file(f))
            if p.is_file():
                total += p.stat().st_size
    return total / 1e6


def import_s(module: str) -> float:
    code = f"import time; t = time.perf_counter(); import {module}; print(time.perf_counter() - t)"
    ts = []
    for _ in range(3):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=HERE.parents[1],
                             env={"DDE_BACKEND": "pytorch", "PATH": "/usr/bin:/bin"})
        ts.append(float(out.stdout.strip().splitlines()[-1]))
    return statistics.median(ts)


def main():
    res = {}
    for fw, spec in FRAMEWORKS.items():
        res[fw] = {"loc": {t: code_lines(HERE / f"{m}.py") for t, m in spec["runners"].items()},
                   "install_mb": install_mb(spec["dists"]), "import_s": import_s(spec["module"]),
                   "dists": {d: metadata.version(d) for d in spec["dists"]}, "module": spec["module"]}
        print(fw, res[fw])
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "D.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
