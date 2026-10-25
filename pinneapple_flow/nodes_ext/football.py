"""Football nodes: the ball-flight inverse problem and the player-tracking smoothing check of ``examples/sports``
(simulated data, no download), and pass-outcome pitch control on a Metrica sample match (downloads it once)."""
from __future__ import annotations

from ..registry import Param, Port, node


@node("football.ball_flight_inverse", "Ball flight: drag, spin and gravity from tracked positions",
      category="Inverse problems",
      outputs=[Port("metrics", "metrics")],
      params=[Param("n", "int", 20, minimum=4, maximum=400, help="simulated flights"),
              Param("noise", "float", 0.2, minimum=0.01, maximum=2.0, help="position noise of the tracking [m]"),
              Param("seed", "int", 0, minimum=0)],
      calls="pinneapple_systems.sports.football.ball_study.synthetic_recovery + ball_study.identifiability")
def football_ball_flight_inverse(ctx, inputs, params):
    """Simulated kicks (drag crisis, random spin) observed with noise are fitted back with the drag-Magnus ODE:
    errors of launch speed and drag coefficient, coverage of the Laplace 95 % intervals, and the Cramer-Rao
    comparison of 3-D against 2-D tracking (no ball height: launch elevation and spin are not identifiable)."""
    from pinneapple_systems.sports.football import ball_study
    r = ball_study.synthetic_recovery(int(params["n"]), noise=float(params["noise"]), seed=int(params["seed"]))
    ctx.metric("speed_mae", r["speed_mae"], "launch speed error, drag-Magnus fit [m/s]")
    ctx.metric("cd_mae", r["cd_mae"], "drag coefficient error")
    ctx.metric("cd_coverage95", r["cd_coverage95"], "share of 95 % intervals holding the true C_D")
    ident = ball_study.identifiability()
    ctx.metric("vz0_std_3d", ident["3d"]["std"]["vz0"], "Cramer-Rao std of the vertical launch speed, 3-D tracking")
    ctx.metric("vz0_std_2d", ident["2d"]["std"]["vz0"], "the same without the ball height")
    ctx.summary(f"C_D error {r['cd_mae']:.3f} (coverage {r['cd_coverage95']:.2f}); vertical launch speed std "
                f"{ident['3d']['std']['vz0']:.2f} m/s in 3-D vs {ident['2d']['std']['vz0']:.3g} m/s in 2-D")
    return {"metrics": ctx.report["metrics"]}


@node("football.tracking_smoothing", "Player tracking: Kalman smoothing and metabolic power",
      category="Analysis",
      outputs=[Port("metrics", "metrics")],
      params=[Param("noise", "float", 0.2, minimum=0.01, maximum=1.0, help="position noise [m]"),
              Param("fps", "float", 10.0, minimum=5.0, maximum=50.0, help="sampling rate [Hz]")],
      calls="pinneapple_systems.sports.football.kin_study.synthetic_smoothing (kinematics.kalman_smooth, "
            "kinematics.metabolic_power)")
def football_tracking_smoothing(ctx, inputs, params):
    """Simulated runs with known speed and acceleration, observed with noise: speed and metabolic-power error of
    raw differences, Savitzky-Golay and the constant-acceleration Kalman-RTS smoother (noise fitted by maximum
    likelihood), and the coverage of the smoother's 95 % velocity intervals."""
    from pinneapple_systems.sports.football import kin_study
    rows = kin_study.synthetic_smoothing(noise=(float(params["noise"]),), fps=(float(params["fps"]),), n=10)
    for r in rows:
        ctx.metric(f"speed_rmse_{r['method']}", r["speed_rmse"], "m/s against the known truth")
        ctx.metric(f"mp_rmse_{r['method']}", r["mp_rmse"], "metabolic power, W/kg")
    k = next(r for r in rows if r["method"] == "kalman_rts")
    ctx.metric("kalman_vel_coverage95", k["vel_coverage95"], "share of 95 % velocity intervals holding the truth")
    ctx.summary(f"Kalman-RTS speed RMSE {k['speed_rmse']:.3f} m/s, coverage {k['vel_coverage95']:.2f}")
    return {"metrics": ctx.report["metrics"]}


@node("football.pass_pitch_control", "Pass outcome by pitch control on Metrica tracking",
      category="Analysis",
      outputs=[Port("metrics", "metrics")],
      params=[Param("train_game", "int", 1, minimum=1, maximum=2, help="Metrica sample game to fit on")],
      calls="pinneapple_systems.sports.football.pitch_control.passes_from_metrica + pc_study.compare "
            "(data.metrica_match)")
def football_pass_pitch_control(ctx, inputs, params):
    """Spearman-type pitch control (fitted on one Metrica sample match) against logistic regression and boosted
    trees on geometric features, scored on the passes of the other match (log-loss, Brier, AUC, calibration)."""
    from pinneapple_systems.sports.football import data, pc_study, pitch_control
    a = int(params["train_game"])
    try:
        tr = pitch_control.passes_from_metrica(data.metrica_match(a))
        te = pitch_control.passes_from_metrica(data.metrica_match(3 - a))
    except OSError as exc:
        ctx.not_run(f"Metrica sample data not reachable ({exc})")
    r = pc_study.compare(tr, te)
    for k, s in r["scores"].items():
        ctx.metric(f"log_loss_{k}", s["log_loss"], f"test passes of game {3 - a}")
    ctx.summary(f"best data-only {r['best_data_only']}; pitch control fitted log-loss "
                f"{r['scores']['pc_reaction']['log_loss']:.4f}")
    return {"metrics": ctx.report["metrics"]}
