"""Flag reusable logic that is written in apps/ or examples/ instead of being imported from the library.

Rule (AGENTS.md, CONTRIBUTING.md): reusable logic goes in the library; apps and examples only compose library calls.
Two static (AST) checks, no code is run:

1. **Local helper modules.** A file in ``apps/`` or ``examples/`` that imports another ``.py`` file of the same tree
   (``from _common import ...``, ``from .engine import ...``, ``import plots`` next to it). Shared helpers belong in a
   ``pinneapple*`` package. An app's own package (``apps/<app>/<app>/``: routes, schemas) is glue and is allowed;
   so is ``apps/_shared/appkit.py`` (login, static files).
2. **Duplicates of library helpers.** A top-level function or class whose name is one of the helpers the library
   already provides (``DUPLICATES`` below: JSON sanitisers, autograd derivatives, error metrics, Latin hypercube,
   colour bars, ...). The message names the library API to use instead.

``scripts/app_helpers_baseline.txt`` lists what existed when the check was added (see
``docs/dev/LIBRARY_CONSOLIDATION.md``). The check fails for anything NEW, and ``tests/test_app_helpers.py`` also
fails for a baseline line that no longer occurs, so the debt can only shrink.

Run: ``python scripts/check_app_helpers.py`` (exit 1 lists the new findings);
``python scripts/check_app_helpers.py --update`` rewrites the baseline (only to remove fixed lines in a PR that
fixes them; adding lines needs a reason in the PR).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("apps", "examples")
SKIP_PARTS = {"_out", "_runs", "__pycache__", "node_modules", "static", "scratch"}
BASELINE = ROOT / "scripts" / "app_helpers_baseline.txt"

# name of a local definition -> the library API that already does it
DUPLICATES: dict[str, str] = {
    "_clean": "pinneapple_core.jsonable.to_jsonable",
    "_jsonable": "pinneapple_core.jsonable.to_jsonable",
    "jsonable": "pinneapple_core.jsonable.to_jsonable",
    "to_jsonable": "pinneapple_core.jsonable.to_jsonable",
    "_grad": "pinneapple_physics.pinn_solver.compiler.autograd_ops.grad",
    "grad": "pinneapple_physics.pinn_solver.compiler.autograd_ops.grad",
    "laplacian": "pinneapple_physics.pinn_solver.compiler.autograd_ops.laplacian",
    "rel_l2": "pinneapple_physics.metrics.relative_l2",
    "l2_rel": "pinneapple_physics.metrics.relative_l2",
    "relative_l2": "pinneapple_physics.metrics.relative_l2",
    "rel_l2_error": "pinneapple_physics.metrics.relative_l2",
    "rmse": "pinneapple_physics.metrics.rmse",
    "compute_metrics": "pinneapple_physics.metrics.summary",
    "lhs": "pinneapple_data.parameter_sampling / pinneapple_neural.architectures.rom.parametric.latin_hypercube",
    "_lhs": "pinneapple_data.parameter_sampling / pinneapple_neural.architectures.rom.parametric.latin_hypercube",
    "lhs_sample": "pinneapple_data.parameter_sampling / pinneapple_neural.architectures.rom.parametric.latin_hypercube",
    "_lhs_sample": "pinneapple_data.parameter_sampling / pinneapple_neural.architectures.rom.parametric.latin_hypercube",
    "latin_hypercube": "pinneapple_neural.architectures.rom.parametric.latin_hypercube",
    "gci": "pinneapple_lab.uq.gci",
    "colorbar": "pinneapple_tools.visualization.studio.colormap.colorbar",
    "pick_device": "pinneapple.best_device",
    "get_device": "pinneapple.best_device",
    "set_seed": "pinneapple_neural.trainer.audit.set_seed",
    "sod_exact": "pinneapple_simulation.numerical_solvers.compressible_fv.sod_exact",
    "von_mises": "pinneapple_simulation.numerical_solvers.solid_fem.von_mises",
    "unwrap_pred": "pinneapple_neural.utils.as_tensor",
    "_unwrap": "pinneapple_neural.utils.as_tensor",
    "_dark_fig": "pinneapple_tools.visualization.style.use_cfd_style(dark=True)",
    "_dark": "pinneapple_tools.visualization.style.use_cfd_style(dark=True)",
    "dark_ax": "pinneapple_tools.visualization.style.use_cfd_style(dark=True)",
    "_setup_dark": "pinneapple_tools.visualization.style.use_cfd_style(dark=True)",
    "block_bootstrap": "pinneapple_analysis.uncertainty.resampling.block_bootstrap",
    "jackknife": "pinneapple_analysis.uncertainty.resampling.jackknife",
    # October apps (#1226)
    "_perimeter": "pinneapple_simulation.meshless.Domain2D.perimeter / estimate_nodes",
    "point_cloud_from_table": "pinneapple_simulation.meshless.point_cloud_from_table",
    "domain_from_spec": "pinneapple_simulation.meshless.domain_from_spec",
    "default_spacings": "pinneapple_simulation.meshless.default_spacings",
    "solutions_csv": "pinneapple_simulation.meshless.MeshlessStudy.solutions_csv",
    "cached_view": "pinneapple_tools.visualization.studio.cached_view",
    "_cached_view": "pinneapple_tools.visualization.studio.cached_view",
    "save_view": "pinneapple_tools.visualization.studio.save_view",
    "prune_views": "pinneapple_tools.visualization.studio.prune_views",
    "downsample": "pinneapple_core.series.downsample",
    "_downsample": "pinneapple_core.series.downsample",
    "_downsample_line": "pinneapple_core.series.downsample / bucket_edges",
    "_down": "pinneapple_core.series.downsample",
    "find_case_root": "pinneapple_simulation.external_solvers.openfoam.find_case_root",
    "_find_case_root": "pinneapple_simulation.external_solvers.openfoam.find_case_root",
    "_case_root": "pinneapple_simulation.external_solvers.openfoam.find_case_root",
    "_unzip_case": "pinneapple_simulation.external_solvers.openfoam.extract_case",
    "design_size": "pinneapple_data.doe.design_size",
    "pack_folder": "pinneapple_simulation.external_solvers.campaign_cases.pack_folder",
}


def _skip(path: Path) -> bool:
    return any(p in SKIP_PARTS or p.startswith("_runs") for p in path.relative_to(ROOT).parts)


def _is_app_package(path: Path) -> bool:
    """apps/<app>/<app>/... is the app's own server package (routes, schemas): glue, allowed to import itself."""
    parts = path.relative_to(ROOT).parts
    return len(parts) >= 3 and parts[0] == "apps" and parts[1] == parts[2]


def _local_target(path: Path, node: ast.AST) -> Path | None:
    """The .py file in apps/ or examples/ that an import statement resolves to, if any."""
    names: list[tuple[str, int]] = []
    if isinstance(node, ast.ImportFrom):
        if node.module is None:                                  # from . import x
            names = [(a.name, node.level) for a in node.names]
        else:
            names = [(node.module, node.level)]
    elif isinstance(node, ast.Import):
        names = [(a.name, 0) for a in node.names]
    for name, level in names:
        base = path.parent
        for _ in range(max(level - 1, 0)):
            base = base.parent
        rel = Path(*name.split("."))
        for cand in (base / rel.with_suffix(".py"), base / rel / "__init__.py"):
            if cand.is_file() and cand != path:
                return cand
    return None


def findings_for(path: Path) -> list[tuple[str, str]]:
    """(baseline key, message) for every finding in one file."""
    rel = path.relative_to(ROOT).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    out: list[tuple[str, str]] = []
    if not _is_app_package(path):
        seen: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                target = _local_target(path, node)
                if target is None or _skip(target) or target.name == "appkit.py":
                    continue
                trel = target.relative_to(ROOT).as_posix()
                if trel in seen:
                    continue
                seen.add(trel)
                out.append((f"{rel} -> {trel}",
                            f"{rel}:{node.lineno}: imports the local helper module {trel}; "
                            "put reusable logic in a pinneapple* package and import it from there"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in DUPLICATES:
            out.append((f"{rel}::{node.name}",
                        f"{rel}:{node.lineno}: defines {node.name}(), which the library already has: "
                        f"use {DUPLICATES[node.name]}"))
    return out


def scan() -> dict[str, str]:
    found: dict[str, str] = {}
    for d in DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            if _skip(path):
                continue
            for key, msg in findings_for(path):
                found.setdefault(key, msg)
    return found


def load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    return {ln.strip() for ln in BASELINE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")}


def main(argv: list[str]) -> int:
    found = scan()
    if "--update" in argv:
        header = ("# Existing duplicates of library helpers in apps/ and examples/ (scripts/check_app_helpers.py).\n"
                  "# This list may only shrink: fix an entry by moving the logic into the library, then delete it.\n")
        BASELINE.write_text(header + "".join(f"{k}\n" for k in sorted(found)), encoding="utf-8")
        print(f"wrote {len(found)} entries to {BASELINE.relative_to(ROOT)}")
        return 0
    base = load_baseline()
    new = sorted(k for k in found if k not in base)
    for k in new:
        print(found[k])
    stale = sorted(base - set(found))
    if stale:
        print(f"\n{len(stale)} baseline entries are fixed; delete them from {BASELINE.relative_to(ROOT)}:")
        print("\n".join(f"  {k}" for k in stale))
    if new:
        print(f"\n{len(new)} new finding(s). Reusable logic goes in the library; apps and examples only compose "
              "library calls (docs/dev/LIBRARY_CONSOLIDATION.md).")
    return 1 if new else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
