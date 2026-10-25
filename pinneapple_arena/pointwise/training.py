"""Training and scoring of one architecture on a pointwise field problem (the PINNeAPPle Arena protocol).

Every arena problem is a field regression: coordinates ``(N, in_dim)`` in, field values ``(N, out_dim)`` out. A
model is built from the PINNeAPPle registry (``pp.build_model``) and, when it takes grids or sequences, wrapped by
``pinneapple_arena.pointwise.adapters``. It is trained on reference samples with an optional PDE residual on
separate collocation points (``pinneapple_arena.pointwise.losses``), early-stopped on a validation split, and
scored on a held-out test split (``field_metrics``, ``prediction_quality``).

No stand-in is ever trained under an architecture's name: when the registry cannot build an architecture for this
data, ``instantiate_model`` raises ``ArchitectureUnavailable`` (or ``ArchitectureNotApplicable``) and the caller
reports the architecture as failed, with the reason.

    from pinneapple_arena.pointwise import training as T
    in_dim, out_dim = T.problem_dims("burgers_1d")
    model = T.instantiate_model("vanilla_pinn", in_dim, out_dim)
    out = T.train_torch_model(model, X_tr, y_tr, X_val, y_val, X_col, loss_fn, n_epochs=200, lr=1e-3, device="cpu")
    metrics = T.field_metrics(T.predict(model, X_test), y_test)
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from pinneapple_arena.pointwise.adapters import adapt_model, check_applicable
from pinneapple_arena.pointwise.architectures import registry_key
from pinneapple_physics import metrics as _metrics

try:
    import psutil as _psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    _psutil = None

HAS_TORCH = True
logger = logging.getLogger(__name__)


class ArchitectureUnavailable(RuntimeError):
    """The registry could not build the architecture for this problem's data (unknown id, or a constructor that
    needs inputs a pointwise problem does not have). The architecture is reported as failed, never replaced."""


def best_device() -> str:
    """cuda > mps > cpu, as a string (``pinneapple_neural.trainer.best_device``)."""
    from pinneapple_neural.trainer import best_device as _best
    return str(_best()).split(":")[0]


def device_info() -> dict:
    """The compute device and, for CUDA, its name, memory and CUDA version."""
    device = best_device()
    info: dict = {"device": device, "gpu_available": device != "cpu"}
    if device == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            info["cuda_version"] = torch.version.cuda
        except Exception:  # noqa: BLE001
            pass
    elif device == "mps":
        info["gpu_name"] = "Apple Silicon (MPS)"
    return info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def field_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Error metrics of a prediction against its reference, pooled over all points and fields
    (``pinneapple_physics.metrics`` on the flattened arrays): ``l2_relative``, ``l1`` (mean absolute error),
    ``rmse``, ``max_error``, ``r2`` and ``mape`` (percent, over the points where the reference is not ~0)."""
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"prediction has {y_pred.size} values, the reference {y_true.size}")
    diff = y_pred - y_true
    l2_rel = float(_metrics.relative_l2(y_pred, y_true))
    l1 = float(_metrics.mae(y_pred, y_true))
    rmse = float(_metrics.rmse(y_pred, y_true))
    max_err = float(_metrics.max_abs(y_pred, y_true))
    r2 = float(_metrics.r2(y_pred, y_true))

    # MAPE: skip near-zero true values to avoid division artifacts
    nz = np.abs(y_true) > 1e-10
    mape = float(np.mean(np.abs(diff[nz] / y_true[nz])) * 100) if nz.any() else None

    return {
        "l2_relative": l2_rel,
        "l1": l1,
        "rmse": rmse,
        "max_error": max_err,
        "r2": r2,
        "mape": round(mape, 4) if mape is not None and np.isfinite(mape) else None,
    }


def prediction_quality(y_pred: np.ndarray, y_true: np.ndarray, metrics: dict) -> dict:
    """
    Detect degenerate model outputs and produce a prediction quality score (a heuristic of this arena, not a
    physics check: it flags constant, zero, NaN and worse-than-the-mean predictions).

    Returns a dict with:
      quality_score  – float 0–1 (higher = better)
      quality_level  – "excellent" | "good" | "fair" | "poor" | "unreliable"
      quality_flags  – list of short strings describing detected issues
      pred_std_ratio – pred_std / true_std (ideal ≈ 1.0)
    """
    pred_flat = np.asarray(y_pred, dtype=np.float64).ravel()
    true_flat = np.asarray(y_true, dtype=np.float64).ravel()

    flags: list[str] = []

    # ── Critical failures ────────────────────────────────────────────────────
    has_nan_inf = bool(not np.all(np.isfinite(pred_flat)))
    if has_nan_inf:
        flags.append("nan_or_inf")

    # ── Variance-based degeneracy ─────────────────────────────────────────────
    pred_std = float(np.std(pred_flat))
    true_std = float(np.std(true_flat)) if np.std(true_flat) > 0 else 1.0
    std_ratio = pred_std / (true_std + 1e-12)

    if not has_nan_inf:
        if pred_std < 1e-8:
            flags.append("constant_output")
        elif std_ratio < 0.02:
            flags.append("near_constant_output")

    # ── Trivial zero prediction ───────────────────────────────────────────────
    pred_mean_abs = float(np.mean(np.abs(pred_flat)))
    true_mean_abs = float(np.mean(np.abs(true_flat))) + 1e-12
    if not has_nan_inf and pred_mean_abs / true_mean_abs < 0.01 and true_mean_abs > 1e-6:
        flags.append("zero_output")

    # ── Metrics-based quality ─────────────────────────────────────────────────
    r2  = float(metrics.get("r2", metrics.get("r2_score", 0.0)) or 0.0)
    l2  = float(metrics.get("l2_relative", metrics.get("l2_relative_error", 1.0)) or 1.0)

    if not has_nan_inf:
        if r2 < 0.0:
            flags.append("below_mean_baseline")
        if l2 > 1.0:
            flags.append("high_relative_error")

    # ── Composite score ───────────────────────────────────────────────────────
    if has_nan_inf:
        score = 0.0
    elif "constant_output" in flags or "zero_output" in flags:
        score = 0.04
    elif "near_constant_output" in flags:
        score = 0.12
    else:
        r2_clamped = max(0.0, min(1.0, r2))
        l2_score   = max(0.0, 1.0 - min(2.0, l2))
        score      = 0.5 * r2_clamped + 0.5 * l2_score
        if "below_mean_baseline" in flags:
            score *= 0.35
        elif "high_relative_error" in flags:
            score *= 0.55

    score = round(float(np.clip(score, 0.0, 1.0)), 3)

    if score >= 0.75:
        level = "excellent"
    elif score >= 0.55:
        level = "good"
    elif score >= 0.35:
        level = "fair"
    elif score >= 0.10:
        level = "poor"
    else:
        level = "unreliable"

    return {
        "quality_score": score,
        "quality_level": level,
        "quality_flags": flags,
        "pred_std_ratio": round(float(std_ratio), 3),
    }


# ---------------------------------------------------------------------------
# Model dimensionality helpers
# ---------------------------------------------------------------------------

def problem_in_dim(problem_id: str, parameters: dict | None = None) -> int:
    """Returns input dimensionality (number of coordinates) for the problem."""
    _DIM_MAP = {
        "burgers_1d": 2,
        "navier_stokes_2d": 2,
        "lid_driven_cavity": 2,
        "bernoulli_flow": 1,
        "stokes_flow": 1,
        "darcy_flow": 1,
        "channel_poiseuille": 1,
        "linear_elasticity_2d": 2,
        "beam_bending": 1,
        "poisson_electrostatics": 2,
        "helmholtz_wave": 2,
        "laplace_potential": 2,
        "schrodinger_stationary": 1,
        "schrodinger_timedep": 2,
        "heat_equation_1d": 2,
        "heat_equation_2d": 3,
        "wave_equation": 2,
        "kdv_equation": 2,
        "klein_gordon": 2,
        "thermoelasticity_2d": 2,
        "navier_stokes_3d": 3,
        "crash_analysis": 2,
        "gross_pitaevskii": 1,
        "magnetostatics_2d": 2,
        "linear_elasticity_3d": 3,
        # 2D showcase — cylinder_flow_2d uses (x, y, t) → 3 inputs
        "cylinder_flow_2d":    3,
        "airfoil_naca0012_2d": 2,
        "shallow_water_2d":    2,
        # 3D multi-physics problems
        "taylor_green_vortex":  3,
        "darcy_flow_3d":        3,
        "stokes_sphere":        3,
        "hartmann_mhd":         3,
        "chip_heat_3d":         3,
        "helmholtz_3d_cavity":  3,
    }
    return _DIM_MAP.get(problem_id, int((parameters or {}).get("dim", 2)))


def problem_out_dim(problem_id: str) -> int:
    """Returns output dimensionality (number of fields) for the problem."""
    _OUT_MAP = {
        "burgers_1d": 1,
        "navier_stokes_2d": 3,
        "lid_driven_cavity": 2,   # (u, v) from the stream-function solution
        "bernoulli_flow": 2,
        "stokes_flow": 1,
        "darcy_flow": 2,
        "channel_poiseuille": 1,
        "linear_elasticity_2d": 2,
        "beam_bending": 2,
        "poisson_electrostatics": 1,
        "helmholtz_wave": 1,
        "laplace_potential": 1,
        "schrodinger_stationary": 1,
        "schrodinger_timedep": 1,
        "heat_equation_1d": 1,
        "heat_equation_2d": 1,
        "wave_equation": 1,
        "kdv_equation": 1,
        "klein_gordon": 1,
        "thermoelasticity_2d": 3,
        "navier_stokes_3d": 4,
        "crash_analysis": 1,
        "gross_pitaevskii": 1,
        "magnetostatics_2d": 1,
        "linear_elasticity_3d": 3,
        # 2D showcase — cylinder_flow_2d outputs (u, v, p) → 3 outputs
        "cylinder_flow_2d":    3,
        "airfoil_naca0012_2d": 1,  # Cp
        "shallow_water_2d":    1,  # η/h₀
        # 3D multi-physics problems
        "taylor_green_vortex":  4,
        "darcy_flow_3d":        4,
        "stokes_sphere":        4,
        "hartmann_mhd":         4,
        "chip_heat_3d":         1,
        "helmholtz_3d_cavity":  1,
    }
    return _OUT_MAP.get(problem_id, 1)


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

# Architectures that use operator-style (channels) arguments
_OPERATOR_ARCHS = {
    "fno", "gno", "deeponet", "multiscale_deeponet", "pino", "uno", "ts_fno", "afno",
}


def instantiate_model(arch_id: str, in_dim: int, out_dim: int) -> Any:
    """Build ``arch_id`` from the PINNeAPPle registry (``pp.build_model``) for ``(N, in_dim) -> (N, out_dim)`` data,
    wrapped by the pointwise / closed-form adapters where it takes grids or sequences.

    Raises ``ArchitectureNotApplicable`` for trajectory / snapshot models and ``ArchitectureUnavailable`` when the
    registry cannot build it; nothing else is trained in its place."""
    check_applicable(arch_id)
    import pinneapple as pp
    key = registry_key(arch_id)
    try:
        if arch_id in _OPERATOR_ARCHS:
            model = pp.build_model(key, in_channels=in_dim, out_channels=out_dim)
        else:
            model = pp.build_model(key, in_dim=in_dim, out_dim=out_dim, hidden_dim=64, n_layers=4)
        return adapt_model(arch_id, model, in_dim, out_dim)
    except Exception as exc:  # noqa: BLE001 - any constructor error means "not available for this data"
        msg = str(exc)
        if len(msg) > 300:                      # an unknown key lists the whole registry
            msg = msg[:300] + "..."
        raise ArchitectureUnavailable(
            f"pp.build_model({key!r}, in_dim={in_dim}, out_dim={out_dim}) failed: {type(exc).__name__}: {msg}"
        ) from exc


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def model_forward(model: Any, X_t: Any) -> Any:
    """Unified forward pass that handles diverse model return types."""
    try:
        out = model(X_t)
    except TypeError:
        # Some operator-style models (e.g. GNO) require coords as a keyword-only argument
        try:
            out = model(X_t, coords=X_t)
        except TypeError:
            inputs = [X_t[:, i:i+1] for i in range(X_t.shape[1])]
            out = model(*inputs)
    if hasattr(out, "y"):
        return out.y
    if isinstance(out, tuple):
        return out[0]
    return out


def train_torch_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_col: np.ndarray | None,      # separate collocation points for PDE residual
    physics_loss_fn: Any | None,
    n_epochs: int,
    lr: float,
    device: str,
    progress_callback: Callable | None = None,
    patience: int = 30,               # early-stopping: val checks without improvement
) -> dict[str, Any]:
    """
    Train a torch model with:
    - Supervised data loss on X_train / y_train
    - Optional PDE residual loss on SEPARATE X_col (physics-informed)
    - Cosine annealing LR schedule
    - Early stopping on held-out X_val / y_val
    - Best-weights restore before returning
    """
    import torch.nn.functional as F

    model = model.to(device)

    # Trigger lazy parameter initialization (e.g. _SymmetryWrapper defers _build() to first forward), and check the
    # model is a pointwise field model: (N, in_dim) -> (N, out_dim). A model that returns another shape would be
    # trained on a broadcast loss that means nothing, so it is refused instead.
    out_dim = int(np.asarray(y_train).reshape(len(X_train), -1).shape[1])
    with torch.no_grad():
        _dummy = torch.zeros(3, X_train.shape[1], dtype=torch.float32, device=device)
        _shape = tuple(model_forward(model, _dummy).shape)
    if _shape != (3, out_dim):
        raise ValueError(f"{type(model).__name__} maps (3, {X_train.shape[1]}) points to shape {_shape}, not "
                         f"(3, {out_dim}): it is not a pointwise field model, so the arena can not train it")

    params = list(model.parameters())
    if not params:
        raise ValueError(
            f"{type(model).__name__} has no trainable parameters after initialization. "
            "A required dependency may not be installed or the model failed to initialise."
        )

    # ── Separate optimizer groups for InversePINN learnable PDE params ────────
    # inverse_params (learnable ν, k, etc.) receive far larger gradients than
    # network weights because they appear in second-derivative residuals.
    # With a single global optimizer + global clip_grad_norm_, the large inverse
    # param gradients steal the entire clip budget, starving network weight
    # updates → periodic loss spikes.  Fix: dedicate a separate group with a
    # much lower LR and its own gradient clip.
    _has_inv = hasattr(model, "inverse_params") and bool(list(model.inverse_params.parameters()))
    if _has_inv:
        _net_params = [p for n, p in model.named_parameters() if "inverse_params" not in n]
        _inv_params = [p for n, p in model.named_parameters() if "inverse_params" in n]
        optimizer = torch.optim.Adam([
            {"params": _net_params, "lr": lr, "weight_decay": 1e-5},
            {"params": _inv_params, "lr": lr * 0.1},   # 10× slower for PDE params
        ])
    else:
        _net_params = params
        _inv_params = []
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 100.0
    )

    X_t     = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t     = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val,   dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val,   dtype=torch.float32, device=device)
    X_col_t = (
        torch.tensor(X_col, dtype=torch.float32, device=device)
        if X_col is not None and len(X_col) > 0
        else None
    )

    loss_history: list[float] = []
    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0
    val_freq = max(1, n_epochs // 50)   # validate ~50 times during training
    stopped_epoch = n_epochs

    # ── Extended metrics tracking ─────────────────────────────────────────────
    _n_pde_evaluations = 0
    _train_start_ts = time.time()
    _time_to_convergence_s: float | None = None
    _convergence_epoch: int | None = None
    _initial_val_loss: float | None = None
    if HAS_TORCH and device == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    _proc_mem_before = 0
    _tmem_started = False
    if HAS_PSUTIL:
        try:
            _proc_mem_before = _psutil.Process().memory_info().rss
        except Exception:
            pass
    else:
        try:
            import tracemalloc
            tracemalloc.start()
            _tmem_started = True
        except Exception:
            pass

    # ── Adaptive physics loss weighting ──────────────────────────────────────
    # Fixed weight causes periodic spikes: the physics residual gradient scale
    # can suddenly dominate the data gradient, destabilising training.
    # Fix: linear warmup over 12% of training + EMA magnitude normalisation so
    # the physics contribution stays proportional to the data loss.
    warmup_end  = max(1, int(n_epochs * 0.12))
    _data_ema   = 1.0   # running EMA of data_loss magnitude
    _phys_ema   = 1.0   # running EMA of phys_loss magnitude
    _EMA_ALPHA  = 0.02  # EMA decay (≈ 50-epoch memory)
    _PHYS_TARGET_RATIO = 0.10   # physics should contribute ~10% of total loss

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        y_pred = model_forward(model, X_t)
        data_loss = F.mse_loss(y_pred, y_t)
        total_loss = data_loss

        # Physics loss on separate interior collocation points (not training data)
        if physics_loss_fn is not None and X_col_t is not None:
            try:
                phys_loss, _ = physics_loss_fn(model, {"X": X_col_t})
                _n_pde_evaluations += 1
                if torch.isfinite(phys_loss) and float(phys_loss.item()) > 0:
                    d_val = float(data_loss.item())
                    p_val = float(phys_loss.item())

                    # Update EMA of each loss scale
                    _data_ema = (1 - _EMA_ALPHA) * _data_ema + _EMA_ALPHA * d_val
                    _phys_ema = (1 - _EMA_ALPHA) * _phys_ema + _EMA_ALPHA * p_val

                    # Linear warmup: physics weight ramps from 0 → target
                    warmup = min(1.0, epoch / warmup_end)

                    # Adaptive scale: keep physics contribution ≈ target ratio of data
                    scale = (_data_ema / max(_phys_ema, 1e-10)) * _PHYS_TARGET_RATIO
                    # Clamp to a sane range — never let physics overpower data loss
                    scale = max(1e-5, min(scale, 2.0))

                    total_loss = total_loss + warmup * scale * phys_loss
            except Exception:
                pass

        if not torch.isfinite(total_loss):
            logger.warning("NaN/Inf total loss at epoch %d — stopping early", epoch)
            stopped_epoch = epoch
            break

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(_net_params, max_norm=1.0)
        if _inv_params:
            # Tight per-group clip: prevents PDE param overshoots without
            # consuming the network's gradient budget
            torch.nn.utils.clip_grad_norm_(_inv_params, max_norm=0.05)
        optimizer.step()
        scheduler.step()

        loss_history.append(float(total_loss.item()))

        # Validation-based early stopping
        if epoch % val_freq == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model_forward(model, X_val_t)
                val_loss = float(F.mse_loss(val_pred, y_val_t).item())
            model.train()

            if _initial_val_loss is None:
                _initial_val_loss = val_loss
            if (_time_to_convergence_s is None and _initial_val_loss is not None
                    and _initial_val_loss > 1e-10 and val_loss < _initial_val_loss * 0.1):
                _time_to_convergence_s = time.time() - _train_start_ts
                _convergence_epoch = epoch

            if val_loss < best_val_loss * 0.9999:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "Early stopping at epoch %d (val_loss=%.4e)", epoch, val_loss
                    )
                    stopped_epoch = epoch
                    break

        if progress_callback is not None and epoch % max(1, n_epochs // 20) == 0:
            frac = (epoch + 1) / n_epochs   # per-arch fraction 0→1; caller remaps to global range
            progress_callback(frac, f"Epoch {epoch}/{n_epochs}", loss_history[:])

    # Restore best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        y_pred_np = model_forward(model, X_t).cpu().numpy()

    # ── Post-training memory stats ────────────────────────────────────────────
    _peak_gpu_memory_mb: float | None = None
    _peak_cpu_memory_mb: float | None = None
    if HAS_TORCH and device == "cuda":
        try:
            _peak_gpu_memory_mb = round(
                torch.cuda.max_memory_allocated() / 1024 ** 2, 1
            )
        except Exception:
            pass
    if HAS_PSUTIL:
        try:
            _peak_cpu_memory_mb = round(
                max(_psutil.Process().memory_info().rss - _proc_mem_before, 0) / 1024 ** 2, 1
            )
        except Exception:
            pass
    elif _tmem_started:
        try:
            import tracemalloc
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _peak_cpu_memory_mb = round(peak / 1024 ** 2, 1)
        except Exception:
            pass

    return {
        "loss_history": loss_history,
        "y_pred": y_pred_np,
        "best_val_loss": best_val_loss,
        "stopped_at_epoch": stopped_epoch,
        "n_pde_evaluations": _n_pde_evaluations,
        "time_to_convergence_s": _time_to_convergence_s,
        "convergence_epoch": _convergence_epoch,
        "peak_gpu_memory_mb": _peak_gpu_memory_mb,
        "peak_cpu_memory_mb": _peak_cpu_memory_mb,
    }


def train_closed_form_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    progress_callback: Callable | None = None,
) -> dict[str, Any]:
    """Train a ClosedFormAdapter model (ELM, hybrid RBF, ESN readouts) by fit()."""
    model.fit(X_train, y_train)
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
        val_pred = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
    val_loss = float(np.mean((val_pred - np.asarray(y_val).reshape(val_pred.shape)) ** 2))
    if progress_callback:
        progress_callback(1.0, "Closed-form readout fitted")
    return {"loss_history": [], "y_pred": y_pred, "best_val_loss": val_loss, "stopped_at_epoch": 0}


# ---------------------------------------------------------------------------
# Inverse problem runner
# ---------------------------------------------------------------------------

def run_inverse_solver(
    model: Any,
    problem_id: str,
    inverse_cfg: dict,
    coords: np.ndarray,
    solution: np.ndarray,
    device: str,
) -> dict:
    """
    Wrap a trained/fresh model with PINNeAPPle's InverseProblemSolver and
    call invert() to recover unknown PDE parameters.

    Returns a dict with 'identified_params' and solver metadata.
    """
    try:
        from pinneapple_neural.inverse.inverse_solver import InverseProblemSolver
        from pinneapple_neural.inverse.observation_operator import PointObsOperator
    except ImportError:
        # The arena was written against pinneapple_neural.inverse, which this library does not have; the inverse
        # solver here is pinneapple_analysis.inverse_problems (a different API) and is not wired in yet.
        return {"error": "parameter identification is not wired to pinneapple_analysis.inverse_problems yet; "
                         "no parameters were identified"}

    import torch

    params      = inverse_cfg.get("params", [])
    init_guess  = inverse_cfg.get("initial_guesses", {})
    n_obs       = int(inverse_cfg.get("n_obs", 200))
    noise_std   = float(inverse_cfg.get("noise_std", 0.01))
    method      = inverse_cfg.get("method", "adam")
    n_iters     = int(inverse_cfg.get("n_iters", 1000))
    lambda_reg  = float(inverse_cfg.get("lambda_reg", 1e-4))
    noise_model_name = inverse_cfg.get("noise_model", "gaussian")

    if not params:
        return {"error": "No parameters specified for inverse problem"}

    # Sample observation points from available data
    n_available = len(coords)
    obs_idx = np.random.choice(n_available, min(n_obs, n_available), replace=False)
    X_obs = coords[obs_idx]
    y_obs = solution[obs_idx]
    if noise_std > 0:
        y_obs = y_obs + np.random.normal(0, noise_std, y_obs.shape).astype(np.float32)

    try:
        obs_op = PointObsOperator(
            torch.tensor(X_obs, dtype=torch.float32, device=device),
            torch.tensor(y_obs, dtype=torch.float32, device=device),
        )

        solver = InverseProblemSolver(
            model=model,
            observation_operator=obs_op,
            params=params,
            initial_guesses=init_guess,
            method=method,
            n_iters=n_iters,
            lambda_reg=lambda_reg,
            noise_model=noise_model_name,
            device=device,
        )

        result = solver.invert()
        return {
            "identified_params": result.get("params", {}),
            "solver_method": method,
            "n_iters": n_iters,
            "n_obs": n_obs,
            "noise_model": noise_model_name,
            "converged": result.get("converged", False),
            "final_residual": result.get("final_residual", None),
        }
    except Exception as exc:
        logger.warning("Inverse solver failed: %s", exc)
        return {"error": str(exc)}


def predict(model: Any, X: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, float]:
    """``(prediction, milliseconds)`` of a trained model on ``X`` (no gradients)."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    t0 = time.time()
    with torch.no_grad():
        out = model_forward(model, X_t).cpu().numpy()
    return out, round((time.time() - t0) * 1000, 2)


def rank_models(results: list[dict]) -> list[dict]:
    """Completed results ranked by relative L2 error (ascending, ``rank`` 1..n); failed ones after, ``rank`` None."""
    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] != "completed"]
    completed.sort(key=lambda r: r["metrics"].get("l2_relative", float("inf")))
    for i, r in enumerate(completed):
        r["rank"] = i + 1
    for r in failed:
        r["rank"] = None
    return completed + failed
