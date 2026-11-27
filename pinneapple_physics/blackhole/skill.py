"""Skill of the black-hole weather forecaster against persistence, climatology and DMD, with physics checks.

    from pinneapple_physics.blackhole.skill import evaluate_skill
    report = evaluate_skill({"residual": (model, codec)}, train, val, test, frame_dt=20.0, cell_volume=vol)

``train`` / ``val`` / ``test``: lists (or {seed: ...}) of whole trajectories of log10 density frames (T, H, W) in the
forecast window (``pinneapple_physics.blackhole.dataset.load_trajectories``).

* Rollouts of ``blocks`` blocks (5 frames each) from start times every ``start_every`` frames of every test
  trajectory; the same starts for every forecast.
* Baselines: **persistence** (last input frame), **climatology** (time mean of the training trajectories) and
  **DMD** (rank-r dynamic mode decomposition of the training trajectories,
  ``pinneapple_neural.architectures.rom.dmd``; rank chosen on the validation trajectories; the part of the state
  outside the DMD subspace is carried as persistence), iterated from the last input frame.
* Error: MAE of the normalised log density in the window, the forecaster's scale. Per lead, the difference model
  minus persistence and the skill score get 95 % intervals from a moving-block bootstrap over start times
  (``pinneapple_analysis.verification.forecast_skill``, blocks of ``block_M``). The **skill horizon** is the last
  lead up to which the model beats persistence with 95 % confidence at every lead.
* Physics checks (no ground truth needed for the first three): the window mass changes no faster than in training
  (``MassEnvelope``); the forecast is finite and its density positive and inside the training range; the inward mass
  flow implied by the forecast (minus the change of the mass enclosed within r -- the net accretion that outward
  angular-momentum transport drives) agrees in sign and correlates with the truth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from pinneapple_analysis.verification.forecast_skill import skill_vs_lead

from .forecast import DensityCodec, MassEnvelope, rollout

__all__ = ["DMDBaseline", "window_mass", "implied_inflow", "acc_by_lead", "first_below", "evaluate_skill"]

K = 5


class DMDBaseline:
    """Rank-r DMD of frame sequences (``pinneapple_neural`` ``DynamicModeDecomposition``, mean-centred), iterated
    from one frame; the component of that frame outside the DMD subspace is kept frozen (persistence)."""

    def __init__(self, trajectories: Sequence[np.ndarray], rank: int):
        from pinneapple_neural.architectures.rom.dmd import DynamicModeDecomposition
        x0 = np.concatenate([t[:-1].reshape(len(t) - 1, -1) for t in trajectories]).astype(np.float64)
        x1 = np.concatenate([t[1:].reshape(len(t) - 1, -1) for t in trajectories]).astype(np.float64)
        self.dmd = DynamicModeDecomposition(r=rank).fit_pairs(torch.from_numpy(x0), torch.from_numpy(x1))
        self.shape = trajectories[0].shape[1:]
        self.rank = rank

    def forecast(self, last: np.ndarray, n: int) -> np.ndarray:
        x0 = torch.from_numpy(last.reshape(1, -1).astype(np.float64))
        traj = self.dmd.rollout(x0, n)[0]                          # (n + 1, D), row 0 = projection of x0
        U, mu = self.dmd.basis_, self.dmd.mean_
        perp = (x0 - mu) - ((x0 - mu) @ U) @ U.t()
        return (traj[1:] + perp).numpy().reshape(n, *self.shape).astype(np.float32)


def window_mass(x: np.ndarray, codec: DensityCodec, vol: np.ndarray) -> np.ndarray:
    """Mass in the window for normalised frames (..., H, W)."""
    return (10.0 ** (x.astype(np.float64) * (codec.log_max - codec.log_min) + codec.log_min) * vol).sum((-2, -1))


def implied_inflow(x: np.ndarray, codec: DensityCodec, vol: np.ndarray, dt: float, shells: Sequence[int]) -> np.ndarray:
    """Net inward mass flow through the shells (radial indices) implied by the change of the enclosed mass,
    -dM(<r)/dt between consecutive frames: (frames - 1, len(shells)). Ignores flow through the polar edges."""
    rho = 10.0 ** (x.astype(np.float64) * (codec.log_max - codec.log_min) + codec.log_min)
    m_r = np.cumsum((rho * vol).sum(-1), -1)[:, list(shells)]
    return -np.diff(m_r, axis=0) / dt


def acc_by_lead(frames: np.ndarray, truth: np.ndarray, clim: np.ndarray) -> np.ndarray:
    """Anomaly correlation with the truth at every lead, anomalies taken from the time-mean flow ``clim``.

    This is the anomaly correlation of weather forecasting, and the quantity the trust horizon of the docs is written
    on (the first lead where it falls below 0.6); it is reported next to the skill score, which is what
    ``skill_vs_lead`` measures against persistence."""
    a = frames - clim
    b = truth - clim
    a = a - a.mean((-2, -1), keepdims=True)
    b = b - b.mean((-2, -1), keepdims=True)
    num = (a * b).sum((-2, -1))
    den = np.sqrt((a * a).sum((-2, -1)) * (b * b).sum((-2, -1)))
    return num / np.maximum(den, 1e-30)


def first_below(leads: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """The last lead before ``values`` first falls below ``threshold`` (the usefulness horizon; ``leads[-1]`` when it
    never does)."""
    bad = np.nonzero(np.asarray(values) < threshold)[0]
    return float(leads[bad[0] - 1]) if len(bad) and bad[0] > 0 else (0.0 if len(bad) else float(leads[-1]))


def _as_list(t) -> list[np.ndarray]:
    return list(t.values()) if isinstance(t, Mapping) else list(t)


def evaluate_skill(models: Mapping[str, tuple[torch.nn.Module, DensityCodec]], train, val, test, *, frame_dt: float,
                   cell_volume: np.ndarray, blocks: int = 20, start_every: int = 5, block_M: float = 1000.0,
                   dmd_ranks: Sequence[int] = (5, 10, 20, 40, 80), log=print) -> dict:
    codec = next(iter(models.values()))[1]
    for name, (_, c) in models.items():
        if (c.log_min, c.log_max) != (codec.log_min, codec.log_max):
            raise ValueError(f"{name}: another normalisation than the first model")

    def enc(lg):
        return ((np.asarray(lg, np.float32) - codec.log_min) / (codec.log_max - codec.log_min)).astype(np.float32)

    train_x, val_x = [enc(t) for t in _as_list(train)], [enc(t) for t in _as_list(val)]
    test_x = {k: enc(v) for k, v in (test.items() if isinstance(test, Mapping) else enumerate(test))}
    vol, dt = cell_volume, float(frame_dt)
    n_lead = blocks * K
    clim = np.concatenate(train_x).mean(0)

    def starts_of(x):
        return list(range(K - 1, len(x) - n_lead, start_every))

    dmd_val = {}
    for r in dmd_ranks:
        d = DMDBaseline(train_x, r)
        dmd_val[int(r)] = float(np.mean([np.abs(d.forecast(x[s], n_lead) - x[s + 1:s + 1 + n_lead]).mean()
                                         for x in val_x for s in starts_of(x)]))
    rank = min(dmd_val, key=dmd_val.get)
    dmd = DMDBaseline(train_x, rank)
    log(f"DMD validation MAE by rank {dmd_val} -> rank {rank}")

    env = MassEnvelope.fit([window_mass(x, codec, vol) for x in train_x], dt)
    shells = [vol.shape[0] // 4, vol.shape[0] // 2]
    errors = {n: [] for n in ["persistence", "climatology", "dmd", *models]}
    accs = {n: [] for n in ["persistence", "climatology", "dmd", *models]}
    mass_err = {n: [] for n in ["persistence", "dmd", *models]}
    phys = {n: {"violations": 0, "frames": 0, "min": np.inf, "max": -np.inf, "nonfinite": 0, "out": 0, "px": 0,
                "pred": [], "true": []} for n in models}
    n_starts = 0
    for x in test_x.values():
        for s in starts_of(x):
            n_starts += 1
            truth, last = x[s + 1:s + 1 + n_lead], x[s]
            fc = {"persistence": np.repeat(last[None], n_lead, 0), "climatology": np.repeat(clim[None], n_lead, 0),
                  "dmd": dmd.forecast(last, n_lead)}
            for name, (model, _) in models.items():
                fc[name] = rollout(model, x[s - K + 1:s + 1], blocks)
            m_true = window_mass(truth, codec, vol)
            for n, f in fc.items():
                errors[n].append(np.abs(f - truth).mean((1, 2)))
                accs[n].append(acc_by_lead(f, truth, clim))
                if n in mass_err:
                    mass_err[n].append(np.abs(np.log(window_mass(f, codec, vol) / m_true)))
            m0 = float(window_mass(last[None], codec, vol)[0])
            infl_true = implied_inflow(np.concatenate([last[None], truth]), codec, vol, dt, shells).mean(0)
            for name in models:
                f, p = fc[name], phys[name]
                p["nonfinite"] += int((~np.isfinite(f)).sum())
                p["min"], p["max"] = min(p["min"], float(f.min())), max(p["max"], float(f.max()))
                p["out"] += int(((f < -0.05) | (f > 1.05)).sum())
                p["px"] += f.size
                v = env.violations(window_mass(f, codec, vol), dt, m0)
                p["violations"] += int(v.sum())
                p["frames"] += len(v)
                p["pred"].append(implied_inflow(np.concatenate([last[None], f]), codec, vol, dt, shells).mean(0))
                p["true"].append(infl_true)
    errors = {n: np.array(v) for n, v in errors.items()}
    accs = {n: np.array(v).mean(0) for n, v in accs.items()}
    mass_err = {n: np.array(v) for n, v in mass_err.items()}
    block = max(1, int(round(block_M / (start_every * dt))))
    leads = dt * (1 + np.arange(n_lead))
    report = {"n_starts": n_starts, "trajectories": sorted(test_x), "frame_dt": dt, "leads_M": leads.tolist(),
              "bootstrap_block_starts": block, "dmd_rank": rank, "dmd_val_mae_by_rank": dmd_val,
              "mean_mae": {n: e.mean(0).tolist() for n, e in errors.items()},
              "mean_acc": {n: a.tolist() for n, a in accs.items()}, "models": {}}
    for name in models:
        res = skill_vs_lead(errors[name], errors["persistence"], leads, block)
        nb = n_lead // K                                         # judged on whole 100 M blocks
        eb = errors[name][:, :nb * K].reshape(n_starts, nb, K).mean(2)
        pb = errors["persistence"][:, :nb * K].reshape(n_starts, nb, K).mean(2)
        res_b = skill_vs_lead(eb, pb, K * dt * (1 + np.arange(nb)), block, seed=100)
        vs = {b: skill_vs_lead(errors[name], errors[b], leads, block, seed=200).to_dict() for b in ("climatology", "dmd")}
        p = phys[name]
        ip, it = np.array(p["pred"]), np.array(p["true"])
        report["models"][name] = {
            "vs_persistence": res.to_dict(), "vs_persistence_blocks": res_b.to_dict(), "vs": vs,
            "skill_horizon_M": res.horizon, "skill_horizon_blocks_M": res_b.horizon,
            "acc_by_lead": accs[name].tolist(), "acc_horizon_M": first_below(leads, accs[name], 0.6),
            "skill_horizon_vs_dmd_M": vs["dmd"]["horizon"], "skill_horizon_vs_climatology_M": vs["climatology"]["horizon"],
            "mass_log_error": mass_err[name].mean(0).tolist(),
            "mass_log_error_persistence": mass_err["persistence"].mean(0).tolist(),
            "physics": {
                "mass_envelope_violation_fraction": p["violations"] / max(p["frames"], 1),
                "mass_envelope_rate_max": env.rate_max,
                "nonfinite_values": p["nonfinite"],
                "x_range_forecast": [p["min"], p["max"]],
                # training range is [0, 1]: fraction of forecast pixels more than 0.05 outside it
                "x_out_of_range_fraction": p["out"] / max(p["px"], 1),
                "inflow_shells": shells,
                "inflow_sign_agreement": float((np.sign(ip) == np.sign(it)).mean()),
                "inflow_corr": [float(np.corrcoef(ip[:, i], it[:, i])[0, 1]) for i in range(ip.shape[1])],
                "inflow_mean_pred": ip.mean(0).tolist(), "inflow_mean_true": it.mean(0).tolist()},
        }
        ss = res.skill_score
        log(f"{name}: skill horizon {res.horizon:.0f} M (frames), {res_b.horizon:.0f} M (100 M blocks); "
            f"ACC horizon {report['models'][name]['acc_horizon_M']:.0f} M; "
            f"SS at {leads[K - 1]:.0f} M {ss[K - 1, 0]:+.3f} [{ss[K - 1, 1]:+.3f}, {ss[K - 1, 2]:+.3f}], "
            f"at {leads[-1]:.0f} M {ss[-1, 0]:+.3f} [{ss[-1, 1]:+.3f}, {ss[-1, 2]:+.3f}]")
    return report
