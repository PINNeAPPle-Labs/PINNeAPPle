#!/usr/bin/env python3
"""
check_all_models.py — sanity-check every arena architecture with a fast N-epoch run on synthetic data.

For each architecture x problem it checks that:
  - the model builds from the PINNeAPPle registry (``pinneapple_arena.pointwise.instantiate_model``; an
    architecture the registry can not build for pointwise data is a FAIL, never a stand-in MLP),
  - a warm-up forward pass runs,
  - the physics loss loads and gives a finite value,
  - training completes the epochs with a finite loss.
Trajectory / snapshot models (Koopman, POD, DMD, HAVOK, ARIMA) are SKIP: not applicable to pointwise data.

The data is random (a smoke test of building and training, not of accuracy).

Usage (PINNeAPPle importable):
    python apps/pinn_arena/scripts/check_all_models.py --epochs 1 --skip-3d
    python apps/pinn_arena/scripts/check_all_models.py --problems burgers_1d --arch vanilla_pinn fno
    python apps/pinn_arena/scripts/check_all_models.py --device cuda --log report.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

# ── ANSI colours (disabled on Windows without ANSI support) ──────────────────
_USE_COLOR = sys.platform != "win32" or os.environ.get("TERM")

def _c(text: str, *codes: str) -> str:
    if not _USE_COLOR:
        return text
    return "".join(codes) + text + "\033[0m"

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# ── Problem definitions ───────────────────────────────────────────────────────
# Each entry: (problem_id, in_dim, out_dim, label)
# in_dim must match the problem's coordinate dimensions.
ALL_PROBLEMS = [
    ("burgers_1d",        2, 1, "1D  burgers_1d   (x,t → u)"),
    ("heat_equation_2d",  3, 1, "2D  heat_eq_2d   (x,y,t → u)"),
    ("navier_stokes_3d",  4, 3, "3D  ns_3d        (x,y,z,t → u,v,p)"),
]

N_TRAIN = 300
N_VAL   = 100
N_COL   = 150   # collocation points for physics loss


# ── Tiny synthetic dataset ────────────────────────────────────────────────────

def _synth(in_dim: int, out_dim: int, n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, in_dim)).astype(np.float32)
    y = rng.random((n, out_dim)).astype(np.float32)
    return X, y


# ── Physics loss check ────────────────────────────────────────────────────────

def _check_physics(problem_id: str, model, in_dim: int, device: str) -> dict:
    """Try to compile and evaluate the physics loss. Returns status dict."""
    try:
        import torch

        from pinneapple_arena.pointwise import PhysicsLossGenerator

        gen = PhysicsLossGenerator(problem_id, {})
        loss_fn = gen.compile_losses()
        if loss_fn is None:
            return {"ok": False, "value": None, "note": "no physics loss for this problem"}

        X_col, _ = _synth(in_dim, 1, N_COL)
        X_t = torch.tensor(X_col, dtype=torch.float32, device=device).requires_grad_(True)

        val, info = loss_fn(model, {"X": X_t})
        finite = bool(torch.isfinite(val).item())
        return {
            "ok": finite,
            "value": round(float(val.item()), 6),
            "note": "finite" if finite else "NaN/Inf",
        }
    except Exception as exc:
        return {"ok": False, "value": None, "note": f"{type(exc).__name__}: {exc}"}


# ── Core test function ────────────────────────────────────────────────────────

def test_architecture(
    arch_id: str,
    problem_id: str,
    in_dim: int,
    out_dim: int,
    n_epochs: int,
    device: str,
) -> dict:
    """
    Build, warm-up, and train `arch_id` for `n_epochs` on synthetic data.
    Returns a result dict with status, error, physics info, and timing.
    """
    result: dict = {
        "arch":        arch_id,
        "problem":     problem_id,
        "status":      "FAIL",
        "error":       None,
        "physics_ok":  False,
        "physics_val": None,
        "physics_note": "",
        "epochs_done": 0,
        "final_loss":  None,
        "duration_s":  0.0,
    }
    t0 = time.perf_counter()

    try:
        import torch

        from pinneapple_arena.pointwise import (
            ArchitectureNotApplicable,
            PhysicsLossGenerator,
        )
        from pinneapple_arena.pointwise import instantiate_model as _instantiate_model
        from pinneapple_arena.pointwise import model_forward as _model_forward
        from pinneapple_arena.pointwise import train_closed_form_model as _train_closed_form_model
        from pinneapple_arena.pointwise import train_torch_model as _train_torch_model

        # ── 1. Instantiate model ──────────────────────────────────────────────
        try:
            model = _instantiate_model(arch_id, in_dim, out_dim)
        except ArchitectureNotApplicable as exc:
            result["status"] = "SKIP"
            result["error"] = str(exc)
            result["physics_note"] = "not applicable"
            result["duration_s"] = round(time.perf_counter() - t0, 2)
            return result

        # Closed-form readouts (ELM, hybrid RBF, ESN) are fitted, not gradient-trained.
        if getattr(model, "closed_form", False):
            X_train, y_train = _synth(in_dim, out_dim, N_TRAIN)
            X_val,   y_val   = _synth(in_dim, out_dim, N_VAL, seed=99)
            out = _train_closed_form_model(model, X_train, y_train, X_val, y_val)
            val_loss = out["best_val_loss"]
            result["physics_note"] = "closed-form fit — no physics loss"
            result["final_loss"] = round(val_loss, 6)
            if np.isfinite(val_loss):
                result["status"] = "PASS"
            else:
                result["error"] = "validation loss is NaN/Inf"
            result["duration_s"] = round(time.perf_counter() - t0, 2)
            return result

        # ── 2. Warmup forward (triggers lazy init for Noether wrappers) ───────
        model_dev = model.to(device) if hasattr(model, "to") else model
        with torch.no_grad():
            try:
                dummy = torch.zeros(2, in_dim, dtype=torch.float32, device=device)
                _model_forward(model_dev, dummy)
            except Exception:
                pass  # warmup errors are non-fatal; caught again in train step

        # ── 3. Physics loss check ─────────────────────────────────────────────
        phys = _check_physics(problem_id, model_dev, in_dim, device)
        result["physics_ok"]   = phys["ok"]
        result["physics_val"]  = phys["value"]
        result["physics_note"] = phys["note"]

        # ── 4. Training data ──────────────────────────────────────────────────
        X_train, y_train = _synth(in_dim, out_dim, N_TRAIN)
        X_val,   y_val   = _synth(in_dim, out_dim, N_VAL, seed=99)
        X_col,   _       = _synth(in_dim, out_dim, N_COL, seed=77)

        physics_loss_fn = None
        try:
            physics_loss_fn = PhysicsLossGenerator(problem_id, {}).compile_losses()
        except Exception:
            pass

        # ── 5. Train ──────────────────────────────────────────────────────────
        train_out = _train_torch_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_col=X_col,
            physics_loss_fn=physics_loss_fn,
            n_epochs=n_epochs,
            lr=1e-3,
            device=device,
            patience=n_epochs + 1,  # disable early-stopping during sanity check
        )

        history = train_out.get("loss_history", [])
        result["epochs_done"] = len(history)
        result["final_loss"]  = round(history[-1], 6) if history else None

        if history and all(np.isfinite(v) for v in history[-min(3, len(history)):]):
            result["status"] = "PASS"
        else:
            result["error"]  = "loss history empty or NaN/Inf"

    except Exception as exc:
        result["status"] = "FAIL"
        result["error"]  = f"{type(exc).__name__}: {str(exc)[:120]}"

    result["duration_s"] = round(time.perf_counter() - t0, 2)
    return result


# ── Report helpers ────────────────────────────────────────────────────────────

_COL_ARCH   = 36
_COL_STATUS = 10
_COL_PHYS   = 28

def _fmt_status(s: str) -> str:
    if s == "PASS":
        return _c("PASS", GREEN, BOLD)
    if s == "SKIP":
        return _c("SKIP", YELLOW)
    return _c("FAIL", RED, BOLD)

def _fmt_phys(ok: bool, val, note: str) -> str:
    if ok:
        return _c(f"✓ {val:.6g}", GREEN)
    return _c(f"✗ {note[:22]}", RED)

def _row(r: dict) -> str:
    arch   = r["arch"][:_COL_ARCH].ljust(_COL_ARCH)
    status = _fmt_status(r["status"])
    phys   = _fmt_phys(r["physics_ok"], r["physics_val"] or 0, r["physics_note"])
    time_s = f"{r['duration_s']:5.1f}s"
    err    = f"  {_c(r['error'][:70], DIM, RED)}" if r["error"] else ""
    return f"  {arch} {status:<6}  phys={phys:<30} {time_s}{err}"


def _print_section_header(label: str) -> None:
    print()
    print(_c("═" * 80, CYAN))
    print(_c(f"  {label}", BOLD + CYAN))
    print(_c("─" * 80, CYAN))


def _print_summary(results: list[dict], log_path: str) -> None:
    total   = len(results)
    passed  = sum(1 for r in results if r["status"] == "PASS")
    failed  = sum(1 for r in results if r["status"] == "FAIL")
    skipped = sum(1 for r in results if r["status"] == "SKIP")
    phys_ok = sum(1 for r in results if r["physics_ok"])

    print()
    print(_c("═" * 80, BOLD))
    print(_c("  SUMMARY", BOLD))
    print(_c("─" * 80, BOLD))
    print(f"  Architectures tested : {total}")
    print(f"  {_c('PASS', GREEN, BOLD)}                  : {passed}")
    print(f"  {_c('FAIL', RED, BOLD)}                  : {failed}")
    print(f"  {_c('SKIP', YELLOW)}                  : {skipped}")
    print(f"  Physics loss OK      : {phys_ok} / {total}")
    print(_c("─" * 80, BOLD))

    if failed:
        print(_c("\n  Failed architectures:", RED, BOLD))
        failed_by_problem: dict[str, list] = {}
        for r in results:
            if r["status"] == "FAIL":
                failed_by_problem.setdefault(r["problem"], []).append(r)
        for prob, rs in failed_by_problem.items():
            print(f"\n  [{prob}]")
            for r in rs:
                print(f"    • {r['arch']:<38} {r['error']}")

    if log_path:
        print(f"\n  Full log: {log_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check all PINNeAPPle Arena architectures"
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="Training epochs per architecture (default: 2)",
    )
    parser.add_argument(
        "--problems", nargs="+",
        default=[p[0] for p in ALL_PROBLEMS],
        help="Problem IDs to test (default: all three)",
    )
    parser.add_argument(
        "--arch", nargs="+", default=None,
        help="Subset of architecture IDs (default: all 93)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device (default: cpu; pass 'cuda' for GPU)",
    )
    parser.add_argument(
        "--skip-3d", action="store_true",
        help="Skip navier_stokes_3d (faster run on memory-constrained machines)",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop after the first failed architecture",
    )
    parser.add_argument("--log", default=None, help="write the per-architecture log to this file")
    args = parser.parse_args()

    # ── Load arch catalog ─────────────────────────────────────────────────────
    try:
        from pinneapple_arena.pointwise import ARCHITECTURES_CATALOG
    except Exception as exc:
        print(_c(f"Cannot load ARCHITECTURES_CATALOG: {exc}", RED, BOLD))
        sys.exit(1)

    arch_ids: list[str] = list(ARCHITECTURES_CATALOG.keys())
    if args.arch:
        unknown = [a for a in args.arch if a not in ARCHITECTURES_CATALOG]
        if unknown:
            print(_c(f"  Unknown architectures (ignored): {unknown}", YELLOW))
        arch_ids = [a for a in args.arch if a in ARCHITECTURES_CATALOG]

    # ── Resolve problems ──────────────────────────────────────────────────────
    prob_map = {p[0]: p for p in ALL_PROBLEMS}
    problems_to_run: list[tuple] = []
    for pid in args.problems:
        if args.skip_3d and "3d" in pid.lower():
            print(_c(f"  Skipping {pid} (--skip-3d)", YELLOW))
            continue
        if pid in prob_map:
            problems_to_run.append(prob_map[pid])
        else:
            print(_c(f"  Unknown problem '{pid}', skipping", YELLOW))

    if not problems_to_run:
        print(_c("No valid problems selected. Exiting.", RED))
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log

    log_lines: list[str] = [
        "PINNeAPPle Arena — Architecture sanity test",
        f"Run at: {ts}",
        f"Epochs: {args.epochs}  Device: {args.device}",
        f"Problems: {[p[0] for p in problems_to_run]}",
        f"Architectures: {len(arch_ids)}",
        "",
    ]

    n = len(arch_ids)
    abort = False

    for prob_id, in_dim, out_dim, label in problems_to_run:
        _print_section_header(f"Problem: {label}")
        log_lines.append(f"\n{'='*60}\nProblem: {label}\n{'='*60}")

        for idx, arch_id in enumerate(arch_ids):
            if abort:
                break

            # Progress indicator (overwritten each line)
            print(f"  [{idx+1:2d}/{n}] {arch_id:<40}…", end="\r", flush=True)

            r = test_architecture(arch_id, prob_id, in_dim, out_dim, args.epochs, args.device)
            all_results.append(r)

            row = _row(r)
            print(row)
            log_lines.append(
                f"  {arch_id:<36} {r['status']:<6}  "
                f"phys_ok={r['physics_ok']}  phys_val={r['physics_val']}  "
                f"{r['duration_s']:.1f}s  error={r['error']}"
            )

            if args.fail_fast and r["status"] == "FAIL":
                print(_c("\n  --fail-fast: stopping after first failure.", YELLOW))
                abort = True

        if abort:
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(all_results, log_path)

    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
