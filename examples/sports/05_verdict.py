"""The verdict table of the football studies: where physics helped, with the measured gain against the best
data-only model at equal data, and where it did not. Reads the results of 01-04, writes results/verdict.md and
results/verdict.json, and records the experiment manifest (metrics with how they were measured, checks).

    python examples/sports/05_verdict.py
"""
from __future__ import annotations

import json
from pathlib import Path

from pinneapple_core.jsonable import to_jsonable
from pinneapple_lab import experiment

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def _load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _ci(d):
    return f"{d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def verdict_rows(ball, pc, kin, xg) -> list[dict]:
    rows = []
    if ball:
        ga = ball["gravity_audit"]
        rows.append({"case": "Ball flight: data consistency (SkillCorner 3-D ball, 10 Hz)",
                     "physics": "drag ODE with free gravity, fitted per flight",
                     "baseline": "none (a data-quality test)",
                     "measured": f"g = {ga['g_drag_median']:.2f} m/s² [{ga['g_drag_ci95'][0]:.2f}, "
                                 f"{ga['g_drag_ci95'][1]:.2f}] over {ga['n_used']} flights (simulated flights, same fit: "
                                 f"{ball['synthetic_no_spin_audit']['g_drag_median']:.2f})",
                     "verdict": "physics HELPED as a check: the broadcast ball height is not physically consistent "
                                "(apparent gravity far below 9.81 at every launch speed, so not backspin lift); "
                                "drag and spin cannot be identified from it"})
        h = ball["holdout"]
        best_data = min(("poly2", "ml"), key=lambda k: h[k]["landing_err_median"])
        best_phys = min(("vacuum", "drag", "drag_free_g"), key=lambda k: h[k]["landing_err_median"])
        rows.append({"case": "Ball flight: landing point from the first half of the flight",
                     "physics": f"{best_phys} (best of vacuum / drag with g = 9.81 / drag with free g); hybrid trees",
                     "baseline": f"{best_data} (best of quadratic extrapolation / boosted trees)",
                     "measured": f"median landing error: physics {h[best_phys]['landing_err_median']:.2f} m, "
                                 f"hybrid {h['hybrid']['landing_err_median']:.2f} m, data-only "
                                 f"{h[best_data]['landing_err_median']:.2f} m ({h['n_test_flights']} test flights)",
                     "verdict": _better(h[best_phys]["landing_err_median"], h[best_data]["landing_err_median"],
                                        h["hybrid"]["landing_err_median"])})
        idt = ball["identifiability"]
        rows.append({"case": "Ball flight from 2-D tracking (Metrica-type, no height)",
                     "physics": "Cramer-Rao bound of the drag-Magnus fit",
                     "baseline": "the same flight seen in 3-D",
                     "measured": f"std of vertical launch speed {idt['3d']['std']['vz0']:.2f} m/s (3-D) vs "
                                 f"{idt['2d']['std']['vz0']:.3g} m/s (2-D)",
                     "verdict": "ill-posed in 2-D: elevation, spin and drag trade off; do not fit them without height"})
    if pc:
        rows_pc = []
        for k in ("compare_1_to_2", "compare_2_to_1"):
            c = pc[k]
            rows_pc.append(f"{k[-4:].replace('_to_', '->')}: fitted pitch control "
                           f"{c['scores']['pc_reaction']['log_loss']:.4f} vs {c['best_data_only']} "
                           f"{c['scores'][c['best_data_only']]['log_loss']:.4f}, diff "
                           f"{_ci(c['paired_vs_best_data_only']['pc_reaction'])}")
        sig = [pc[k]["paired_vs_best_data_only"]["pc_reaction"]["ci95"][1] < 0 for k in ("compare_1_to_2",
                                                                                        "compare_2_to_1")]
        eff = _curve(pc["efficiency_1_to_2"]["rows"])
        n_small = min(n for (_, n) in eff if n >= 100)
        small = (f"with {n_small} training passes: fitted pitch control {eff[('pc_reaction', n_small)]:.4f}, "
                 f"logistic {eff[('logistic', n_small)]:.4f}, trees {eff[('gbm', n_small)]:.4f}; unfitted pitch "
                 f"control (0 passes) {pc['compare_1_to_2']['scores']['pc_default']['log_loss']:.4f}")
        rows.append({"case": "Pass completion (Metrica, 2 matches, ~2000 passes)",
                     "physics": "Spearman pitch control, 5 physical parameters fitted",
                     "baseline": "logistic regression / boosted trees on geometric features",
                     "measured": "; ".join(rows_pc) + "; " + small,
                     "verdict": ("physics HELPED" if all(sig) else
                                 "physics HELPED with little data (lower log-loss than every data-only model up to a "
                                 "few hundred passes, unfitted or fitted) and is better calibrated; at full data it "
                                 "matches the best data-only model (gain not significant in both directions)")})
    if kin:
        syn = kin["synthetic_smoothing"]
        parts = []
        for fps, s in sorted({(r["fps"], r["noise_m"]) for r in syn}, reverse=True):
            k = next(r for r in syn if r["fps"] == fps and r["noise_m"] == s and r["method"] == "kalman_rts")
            sg = min((r for r in syn if r["fps"] == fps and r["noise_m"] == s and r["method"].startswith("savitzky")),
                     key=lambda r: r["mp_rmse"])
            parts.append(f"{fps:g} Hz/{s * 100:.0f} cm: {k['mp_rmse']:.1f} vs {sg['mp_rmse']:.1f} W/kg")
        rows.append({"case": "Speed and metabolic power from noisy tracking (simulated, known truth)",
                     "physics": "constant-acceleration Kalman-RTS, noise by maximum likelihood",
                     "baseline": "best Savitzky-Golay window (chosen with the truth: an optimistic baseline)",
                     "measured": "metabolic-power RMSE " + "; ".join(parts),
                     "verdict": "physics HELPED (lower error in every case, calibrated intervals); the gain is "
                                "largest against an untuned window. Caveat: the simulated motion follows a model "
                                "close to the filter's"})
        ml = kin["match_load"]
        rows.append({"case": "Metabolic power on Metrica (real, no ground truth)",
                     "physics": "equivalent-slope model on Kalman-RTS states",
                     "baseline": "the same on raw differences / Savitzky-Golay",
                     "measured": f"noise fitted {ml['noise_fit']['sigma_pos'] * 100:.1f} cm; team mean "
                                 + ", ".join(f"{k} {v['mean_mp_team']:.2f} W/kg" for k, v in ml["methods"].items()),
                     "verdict": "processing uncertainty is reported, not a win: the provider data are already "
                                "smooth, so the choice of filter moves the load only a few per cent"})
        for H in ("1s", "2s"):
            rr = kin["prediction"][H]["rows"]
            cv = next(r for r in rr if r["model"] == "constant_velocity")
            ph = next(r for r in rr if r["model"] == "physics_limited")
            full = {r["model"]: r for r in rr if r["n_train"] == max(x["n_train"] for x in rr)}
            small = {r["model"]: r for r in rr if r["n_train"] == min(x["n_train"] for x in rr if x["n_train"] > 0)
                     and r["seed"] == 0}
            rows.append({"case": f"Player position {H} ahead (Metrica, train match 1, test match 2)",
                         "physics": "constant velocity / decaying acceleration; trees on the constant-velocity "
                                    "residual (+ reachable-set projection)",
                         "baseline": "boosted trees on the past second",
                         "measured": f"mean error: CV {cv['fde_mean']:.2f} m, physics {ph['fde_mean']:.2f} m; "
                                     f"all data: trees {full['gbm']['fde_mean']:.2f} m, residual "
                                     f"{full['gbm_residual_cv']['fde_mean']:.2f} m; "
                                     f"{small['gbm']['n_train']} samples: trees {small['gbm']['fde_mean']:.2f} m, "
                                     f"residual {small['gbm_residual_cv']['fde_mean']:.2f} m; unreachable trees "
                                     f"predictions {full['gbm']['unreachable'] * 100:.2f} %",
                         "verdict": _pred_verdict(full, small)})
    if xg:
        s = xg["scores"]
        p = xg["paired_vs_standard_logistic"]
        rows.append({"case": "Expected goals (StatsBomb 360, test Euro 2024)",
                     "physics": "intercept-time features: open goal fraction, keeper margin, blockers",
                     "baseline": "standard logistic xG (distance, angle, body part, context)",
                     "measured": f"log-loss standard {s['standard_logistic']['log_loss']:.4f}, +physics "
                                 f"{s['standard+physics_logistic']['log_loss']:.4f} (diff "
                                 f"{_ci(p['standard+physics_logistic'])}), +freeze geometry "
                                 f"{s['standard+freeze_geometry_logistic']['log_loss']:.4f}; StatsBomb xG "
                                 f"{s['statsbomb_xg_reference']['log_loss']:.4f}",
                     "verdict": ("physics HELPED" if p["standard+physics_logistic"]["ci95"][1] < 0 else
                                 "physics did NOT help: no measurable gain over shot geometry with ~2600 training "
                                 "shots; case dropped")})
    return rows


def _curve(rows):
    acc: dict = {}
    for r in rows:
        acc.setdefault((r["model"], r["n_train"]), []).append(r["log_loss"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _better(phys, data, hybrid):
    best = min(phys, data, hybrid)
    if best == data:
        return "physics did NOT help: data-only extrapolation is as good or better on this data"
    who = "physics" if best == phys else "hybrid (trees + physics features)"
    return f"{who} best: {100 * (1 - best / data):.0f} % lower median landing error than the best data-only model"


def _pred_verdict(full, small):
    g, r = full["gbm"]["fde_mean"], full["gbm_residual_cv"]["fde_mean"]
    gs, rs = small["gbm"]["fde_mean"], small["gbm_residual_cv"]["fde_mean"]
    out = []
    out.append(f"physics prior HELPED with little data ({100 * (1 - rs / gs):.0f} % lower error)" if rs < gs else
               "no gain with little data")
    out.append(f"{'matched' if abs(r - g) / g < 0.02 else ('helped' if r < g else 'hurt')} with all data "
               f"({100 * (1 - r / g):+.1f} %)")
    return "; ".join(out)


def main():
    ball, pc, kin, xg = (_load(n) for n in ("ball_flight", "pitch_control", "player_kinematics", "expected_goals"))
    rows = verdict_rows(ball, pc, kin, xg)
    md = ["| Case | Physics | Data-only baseline | Measured (held-out) | Verdict |", "|---|---|---|---|---|"]
    md += [f"| {r['case']} | {r['physics']} | {r['baseline']} | {r['measured']} | {r['verdict']} |" for r in rows]
    (RES / "verdict.md").write_text("\n".join(md) + "\n")
    (RES / "verdict.json").write_text(json.dumps(to_jsonable(rows), indent=1))
    with experiment("examples/sports", title="Football: where physics helps on public tracking and event data",
                    description="Ball flight (drag, Magnus, gravity; inverse problem with UQ; data-consistency "
                                "audit), Spearman pitch control vs data-only pass models, Kalman smoothing and "
                                "metabolic power, trajectory prediction with physical limits, and physics features "
                                "for expected goals, each against data-only baselines at equal data. Data: "
                                "SkillCorner open data, Metrica Sports sample data, StatsBomb open data.",
                    domain="sports", methods=["inverse problem (ODE)", "Kalman-RTS", "pitch control",
                                              "gradient boosting"],
                    modules=["pinneapple_systems.sports.football." + m for m in
                             ("ball", "ball_study", "data", "kinematics", "kin_study", "metrics", "pitch_control",
                              "pc_study", "plots", "xg")],
                    links={"guide": "docs/guides/football_physics.md"}, register=False) as ex:
        ex.manifest.scripts[:] = ["01_ball_flight.py", "02_pitch_control.py", "03_player_kinematics.py",
                                  "04_expected_goals.py", "05_verdict.py"]
        if ball:
            ga = ball["gravity_audit"]
            ex.log_metric("ball_g_fitted_median", ga["g_drag_median"], "m/s^2",
                          "median gravity of the drag fit with free g over SkillCorner aerial flights "
                          "(results/ball_flight.json)")
            ex.check("ball_height_physically_consistent", ga["g_drag_ci95"][0] <= 9.81 <= ga["g_drag_ci95"][1],
                     detail="false = the data's ball height fails the gravity check", kind="physics")
            ex.log_metric("ball_cd_mae_synthetic", ball["synthetic"]["cd_mae"], "",
                          "drag-coefficient error of the drag-Magnus fit on simulated kicks, 0.2 m noise")
            ex.check("ball_cd_interval_coverage", value=ball["synthetic"]["cd_coverage95"], min=0.85,
                     label="95 % intervals of C_D hold the truth (simulated)")
            for m in ("poly2", "ml", "drag", "drag_free_g", "hybrid"):
                ex.log_metric(f"ball_landing_err_median_{m}", ball["holdout"][m]["landing_err_median"], "m",
                              "median horizontal landing error, held-out SkillCorner matches, half the flight seen")
        if pc:
            for k in ("pc_default", "pc_reaction", "logistic", "gbm", "hybrid"):
                ex.log_metric(f"pass_logloss_{k}", pc["compare_1_to_2"]["scores"][k]["log_loss"], "",
                              "test log-loss on Metrica match 2 after fitting on match 1")
        if kin:
            k = next(r for r in kin["synthetic_smoothing"] if r["method"] == "kalman_rts" and r["fps"] == 10.0
                     and r["noise_m"] == 0.2)
            ex.log_metric("kalman_mp_rmse_10hz_20cm", k["mp_rmse"], "W/kg",
                          "metabolic-power RMSE of the Kalman-RTS smoother on simulated runs")
            ex.check("kalman_velocity_coverage", value=k["vel_coverage95"], min=0.9, max=0.99,
                     label="Kalman 95 % velocity intervals cover the truth")
        if xg:
            for k in ("standard_logistic", "standard+physics_logistic", "statsbomb_xg_reference"):
                ex.log_metric(f"xg_logloss_{k}", xg["scores"][k]["log_loss"], "", "test log-loss on Euro 2024 shots")
        for f in sorted((HERE / "figures").glob("*.png")):
            ex.add_file(str(f))
        for f in ("verdict.md", "verdict.json", "ball_flight.json", "pitch_control.json", "player_kinematics.json",
                  "expected_goals.json"):
            if (RES / f).exists():
                ex.add_file(str(RES / f), "data")
    print("\n".join(md))


if __name__ == "__main__":
    main()
