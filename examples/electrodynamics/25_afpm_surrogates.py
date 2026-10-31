"""Surrogates of an axial-flux PM machine in 3-D with calibrated uncertainty and physics checks.

48 designs from a space-filling design over four variables of the default ``AFPMMachine`` (magnet arc 0.6-0.9 of the
pole pitch, slot opening 0.3-0.7 of the slot pitch, gap 0.7-1.5 mm, magnet thickness 3-6 mm), each checked in full
3-D and estimated in quasi-3D on the coarse grid of the Axial-Flux Motor Designer
(``machines3d.afpm_surrogate.run``, about 55 s per design). The designs with magnets thicker than 5.5 mm are held
out (extrapolation); the rest are split into train, validation and test. Surrogate families (GP, MLP ensemble,
quadratic response surface) are compared by ``pinneapple_autolab.study.benchmark``: 5-fold cross-validation,
cross-conformal 90 % intervals, physics checks (``afpm_surrogate.physics_checks``) and a written selection rule.
Each selected surrogate is then certified (``afpm_surrogate.certify``): a trust decision from its physics checks,
its held-out test and extrapolation errors and interval coverage, every check recorded in a VeriPhysics evidence
ledger (``results/25_afpm_surrogates.ledger.json``).

    python examples/electrodynamics/25_afpm_surrogates.py      # about 45 min; the campaign resumes
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pinneapple_autolab.study import benchmark as B
from pinneapple_autolab.study import dataset as D
from pinneapple_autolab.study.campaign import plan_doe, run_campaign
from pinneapple_simulation.electromagnetics.machines3d import afpm_surrogate as S

OUT = Path(__file__).parent / "results"
CAMPAIGN = OUT / "25_afpm_campaign"
N_DOE = 48
HOLDOUT_THICKNESS = 0.0055
TARGETS = ("cogging_pkpk_3d", "linkage_3d", "ratio_cogging", "ratio_linkage")


def main() -> dict:
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    plan = plan_doe(S.space(), N_DOE, "maximin", seed=0)
    recs = run_campaign(S.simulator(), plan["cases"], str(CAMPAIGN), purpose="doe",
                        log=lambda s: print(s, flush=True))
    names = list(S.VARIABLES)
    bounds = {k: (lo, hi) for k, (lo, hi, *_) in S.VARIABLES.items()}
    ds = D.build_dataset(recs, names, list(S.OUTPUTS), bounds=bounds,
                         units={k: u for k, (_, _, u, _) in S.VARIABLES.items()},
                         provenance={"holdout_rule": f"magnet_thickness > {HOLDOUT_THICKNESS} m -> extrapolation"})
    j = names.index("magnet_thickness")
    ds = ds.with_split(D.region_split(ds, holdout=lambda X: X[:, j] > HOLDOUT_THICKNESS, val=0.15, test=0.15,
                                      seed=0))
    measured = {t: (float(ds.y(t).min()), float(ds.y(t).max())) for t in ("ratio_cogging", "ratio_linkage")}
    board = B.benchmark(ds, TARGETS, None, physics=S.physics_checks(measured), k=5, seed=0,
                        log=lambda s: print(s, flush=True))
    sel = board.select()
    from pinneapple_veriphysics.ledger import EvidenceLedger
    ledger = EvidenceLedger()
    certs = {}
    for t in TARGETS:
        if sel["choice"].get(t):
            c = S.certify(board, ds, t, ledger=ledger)
            c.pop("ledger")
            certs[t] = c
            print(f"{t} ({c['family']}): {c['decision']}", flush=True)
    (OUT / "25_afpm_surrogates.ledger.json").write_text(json.dumps(ledger.to_dict(), indent=1))
    out = {"variables": {k: list(v) for k, v in S.VARIABLES.items()}, "n_doe": N_DOE,
           "doe_coverage": plan["coverage"], "doe_random_reference": plan["random_reference"],
           "n_ok": int(sum(r.ok for r in recs)), "split": {k: len(v) for k, v in ds.split.items()},
           "seconds_per_design": float(np.mean([r.cost_s for r in recs if r.ok])),
           "measured_ratios": measured, "board": board.table(), "physics": board.physics, "selection": sel,
           "certification": certs,
           "protocol": board.protocol, "seconds": time.perf_counter() - t0, "date": time.strftime("%Y-%m-%d")}
    (OUT / "25_afpm_surrogates.json").write_text(json.dumps(out, indent=1, default=float))
    figure(out, ds, board)
    return out


def figure(out: dict, ds, board) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = {"cogging_pkpk_3d": "cogging pk-pk, 3-D (N m)", "linkage_3d": "tooth flux linkage, 3-D (Wb)",
              "ratio_cogging": "cogging, quasi-3D / 3-D", "ratio_linkage": "flux linkage, quasi-3D / 3-D"}
    fig, axs = plt.subplots(1, len(TARGETS), figsize=(4.2 * len(TARGETS), 4.2), facecolor="#fcfcfb")
    U = ds.unit_x()
    for a, t in zip(axs, TARGETS, strict=True):
        fam = out["selection"]["choice"].get(t)
        if fam is None:
            a.set_title(f"{t}: no eligible family")
            continue
        pred = board.predictor(t, fam)
        for split, c, mk in (("test", "#2a78d6", "o"), ("extrapolation", "#eb6834", "s")):
            idx = ds.idx(split)
            if not len(idx):
                continue
            mu, hw = pred(U[idx])
            y = ds.y(t)[idx]
            a.errorbar(y, mu, yerr=hw, fmt=mk, color=c, ms=5, lw=1, capsize=2, label=f"{split} ({len(idx)})")
        lo, hi = float(ds.y(t).min()), float(ds.y(t).max())
        a.plot([lo, hi], [lo, hi], color="#52514e", lw=0.8)
        a.set(xlabel=f"3-D solve: {labels[t]}", ylabel="surrogate, 90 % interval", title=f"{t} ({fam})")
        a.legend(frameon=False, fontsize=8)
        a.set_facecolor("#fcfcfb")
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "25_afpm_surrogates.png", dpi=150)


if __name__ == "__main__":
    r = main()
    print(f"done in {r['seconds']:.0f} s")
