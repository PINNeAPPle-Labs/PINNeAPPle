"""Who the vehicle is, learned in flight -- and how long the answer holds (#335).

    python examples/multirotor_flight/expert_identification.py           # the whole study (~3 min)
    python examples/multirotor_flight/expert_identification.py --quick   # 14 s per flight instead of 20 (~2 min)

The first example of this folder (:mod:`flight_safety`) measures the disagreement between a *declared* vehicle and the
one flying. This one asks the three questions that come before and after that:

1. **What does the vehicle weigh, how hard is it to turn, and how much does the air push back?** Identified in flight
   by recursive least squares on the same 6-DoF model, from what a flight controller has -- the commands it sent, the
   IMU, the GPS velocity -- starting from a declaration that is wrong by a factor of two, and measured against the
   truth of the simulator: how far off the answer is, and how many seconds it took to get there.
2. **What is the nominal model missing?** A steady wind and the ground effect are not faults, and a detector cannot
   tell them from one. A small correction is learned on flights outside the nominal and measured on flights it never
   saw -- including the version of it that does not work, because that is the result too.
3. **How long does the model answer for?** With the online expert the parameter uncertainty is a number, and the
   horizon it implies -- *this model answers for N seconds before its residual crosses the fault threshold* -- is
   measured on flights with different excitation, and against faults that grow.

Every flight here runs at 200 Hz (a flight controller's estimator rate, and a fifth of the cost of the 500 Hz campaign
of the other script) for 20 s, with a different seed per flight: a flight is never used both to teach and to test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
DT = 0.005
DURATION = 20.0
FAULT_TIME = 8.0


def _out(*parts: str) -> str:
    path = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(path) or OUT, exist_ok=True)
    return path


# ----------------------------------------------------------------------------------------------------- 1. the expert
def _trajectories():
    from pinneapple_physics.multirotor.control import Trajectory

    return {
        # straight and slow: the rotors carry the weight and nothing else moves
        "hover": Trajectory("hover", altitude=5.0),
        # a circle at 2.5 m/s: the airspeed identifies the drag along the axis the vehicle flies along
        "circle": Trajectory("circle", altitude=5.0, radius=4.0, period_s=10.0),
        # the maneuver: a figure-eight with an oscillating heading, so roll, pitch and yaw all see a torque
        "weave": Trajectory("weave", altitude=5.0, radius=3.0, period_s=8.0),
    }


def identify(*, seeds: dict[str, int]) -> tuple[dict, dict]:
    """Run the expert over one flight of each excitation and report the error against the truth."""
    from pinneapple_physics.multirotor.estimate import OnlineExpert
    from pinneapple_physics.multirotor.sim import fly

    truth = {"mass_kg": 1.5, "inertia": (0.011, 0.011, 0.021), "drag_area": (0.030, 0.030, 0.080)}
    report, histories, flights = {}, {}, {}
    for name, trajectory in _trajectories().items():
        flight = fly(duration=DURATION, dt=DT, seed=seeds[name], trajectory=trajectory)
        history = OnlineExpert().run(flight)
        flights[name], histories[name] = flight, history
        last = history.at(DURATION - 0.01)
        row = {"seed": seeds[name],
               "track": {
                   "mass_kg": [float(history.at(t).mass_kg) for t in (1.0, 2.0, 5.0, 10.0, 19.99)],
                   "time_s": [1.0, 2.0, 5.0, 10.0, 19.99]},
               "settle_s": history.settle_time(tolerance=0.05),
               "mass_kg": float(last.mass_kg), "mass_std_kg": float(last.mass_std_kg),
               "inertia": [float(v) for v in last.inertia], "inertia_std": [float(v) for v in last.inertia_std],
               "drag_area": [float(v) for v in last.drag_area], "drag_std": [float(v) for v in last.drag_std],
               "model_sigma_m_s2": float(last.model_sigma_m_s2),
               "speed_max_m_s": float(np.max(np.linalg.norm(flight.velocity, axis=1))),
               "rate_max_rad_s": float(np.max(np.linalg.norm(flight.gyro, axis=1)))}
        row["mass_error"] = float(row["mass_kg"] / truth["mass_kg"] - 1.0)
        row["inertia_error"] = [float(v / t - 1.0) for v, t in zip(row["inertia"], truth["inertia"], strict=False)]
        row["drag_error"] = [float(v / t - 1.0) for v, t in zip(row["drag_area"], truth["drag_area"], strict=False)]
        report[name] = row
        print(f"  {name:7s} m={row['mass_kg']:6.4f} ({100 * row['mass_error']:+5.2f} %) settle {row['settle_s']} s | "
              f"I {np.round(row['inertia'], 5)} ({100 * np.array(row['inertia_error'])[0]:+5.1f} % x) | "
              f"CdA {np.round(row['drag_area'], 4)} | sigma {row['model_sigma_m_s2']:.3f}", flush=True)
    return report, {"histories": histories, "flights": flights, "truth": truth}


# ----------------------------------------------------------------------------------------- 2. the learned residual
def _bob(vertical: float, altitude: float, period: float):
    """A hover that oscillates through the ground-effect skirt: the only flight on which the ground effect is visible."""
    from pinneapple_physics.multirotor.control import Trajectory

    class _Bob(Trajectory):
        def at(self, t):
            position, velocity, yaw = super().at(t)
            w = 2 * math.pi / self.period_s
            position, velocity = position.copy(), velocity.copy()
            position[2] += -self.vertical * math.sin(w * t)
            velocity[2] = -self.vertical * w * math.cos(w * t)
            return position, velocity, yaw

    trajectory = _Bob("hover", altitude=altitude, period_s=period)
    trajectory.vertical = float(vertical)
    return trajectory


def learn_residual(*, seeds) -> dict:
    """Fit the missing physics on some flights, and measure it on flights it never saw."""
    from pinneapple_physics.multirotor.detect import calibrate
    from pinneapple_physics.multirotor.estimate import ResidualCorrection
    from pinneapple_physics.multirotor.faults import Fault
    from pinneapple_physics.multirotor.sim import fly

    def rms(flight, residual, *, after=3.0):
        mask = flight.time > after
        return float(np.sqrt(np.mean(np.sum((residual[mask][:, :3] - calibration.mean[:3]) ** 2, axis=1))))

    def gust(speed, seed):
        return fly((Fault("gust", 2.0, speed, direction=(1.0, 0.3, 0.0)),), duration=DURATION, dt=DT, seed=seed,
                   trajectory=_bob(0.0, 5.0, 8.0))

    from pinneapple_physics.multirotor.control import Trajectory

    def calm(seed):
        return fly(duration=DURATION, dt=DT, seed=seed, trajectory=Trajectory("hover", altitude=5.0))

    calibration = calibrate([calm(0), calm(1)])

    # --- the ground effect: taught on three flights through the skirt, measured on two it never saw
    training = [fly(duration=DURATION, dt=DT, seed=seeds["ge_a"], trajectory=_bob(0.20, 0.25, 6.0)),
                fly(duration=DURATION, dt=DT, seed=seeds["ge_b"], trajectory=_bob(0.30, 0.30, 5.0)),
                fly(duration=DURATION, dt=DT, seed=seeds["ge_c"], trajectory=_bob(0.10, 0.20, 7.0))]
    model = ResidualCorrection.fit(training)
    ground_effect = {"gains": model.to_dict()["ground_effect_gain"],
                     "total_gain": model.to_dict()["ground_effect_total"],
                     "declared_gain": 0.15, "fit_residual_m_s2": model.residual_m_s2,
                     "train_residual_m_s2": [rms(f, f.residual) for f in training], "cases": {}}
    for name, flight in (("unseen amplitude 0.22 m", _bob(0.22, 0.22, 5.5)),
                         ("unseen amplitude 0.15 m", _bob(0.15, 0.20, 8.0)),
                         ("calm at 5 m, no ground effect", None)):
        run = fly(duration=DURATION, dt=DT, seed=seeds[name], trajectory=flight) if flight is not None else calm(seeds[name])
        before, after = rms(run, run.residual), rms(run, model.apply(run, wind=None))
        ground_effect["cases"][name] = {"before_m_s2": before, "after_m_s2": after,
                                        "ratio": after / max(before, 1e-9)}
        print(f"  ground effect {name:30s} {before:6.3f} -> {after:6.3f} m/s^2 "
              f"(-{100 * (1 - after / before):4.1f} %)", flush=True)

    # --- the wind: the same correction, and the version of it that does not transfer
    windy = [gust(6.0, seeds["gust_a"]), gust(10.0, seeds["gust_b"])]
    pooled = ResidualCorrection.fit(windy, pool_wind=True)
    wind = {"pooled_world_force_m_s2": [float(v) for v in pooled.pooled_wind], "cases": {}}
    for name, run in (("seen 6 m/s, another seed", gust(6.0, seeds["gust_c"])),
                      ("seen 10 m/s, another seed", gust(10.0, seeds["gust_d"])),
                      ("unseen 8 m/s", gust(8.0, seeds["gust_e"])),
                      ("unseen 3 m/s", gust(3.0, seeds["gust_f"])),
                      ("calm, no wind at all", calm(seeds["gust_g"]))):
        before = rms(run, run.residual)
        fixed = rms(run, pooled.apply(run, wind=pooled.pooled_wind))
        causal = rms(run, model.apply(run))
        wind["cases"][name] = {"before_m_s2": before, "pooled_m_s2": fixed, "causal_m_s2": causal,
                               "pooled_ratio": fixed / max(before, 1e-9), "causal_ratio": causal / max(before, 1e-9)}
        print(f"  wind {name:28s} {before:6.3f} | pooled {fixed:6.3f} ({fixed / before:4.2f}x) | "
              f"causal {causal:6.3f} ({causal / before:4.2f}x)", flush=True)
    return {"ground_effect": ground_effect, "wind": wind, "calibration": calibration}


# ---------------------------------------------------------------------------------------------- 3. the horizon
def horizons(*, seeds, model_sigma: dict[str, float]) -> dict:
    """How many seconds the model answers for, as the fault grows and as the excitation changes."""
    from pinneapple_physics.multirotor.control import Trajectory
    from pinneapple_physics.multirotor.detect import calibrate, diagnose
    from pinneapple_physics.multirotor.estimate import OnlineExpert, confidence_horizon
    from pinneapple_physics.multirotor.faults import Fault
    from pinneapple_physics.multirotor.sim import fly

    def calm(seed):
        return fly(duration=DURATION, dt=DT, seed=seed, trajectory=Trajectory("hover", altitude=5.0))

    calibration = calibrate([calm(0), calm(1)])
    # the level a fault has to reach to be reported at all: the signature of the smallest payload the campaign carries,
    # g * 0.10 / 1.5 = 0.65 m/s^2, measured rather than declared
    threshold = 9.80665 * 0.10 / 1.5
    instants = (FAULT_TIME + 0.5, FAULT_TIME + 1.0, FAULT_TIME + 1.5, FAULT_TIME + 2.0, FAULT_TIME + 4.0)
    report = {"threshold_m_s2": threshold, "instants_s": list(instants), "ramps": {}, "step": {},
              "excitation": {}}

    from pinneapple_physics.multirotor.detect import ThresholdDetector, tune

    healthy = [calm(0), calm(1)]
    detector = tune(ThresholdDetector, calibration, healthy)

    # --- a fault that grows: a payload ramping in over six seconds, at three sizes
    ramping = {}
    for magnitude in (0.15, 0.30, 0.60):
        flight = fly((Fault("payload", FAULT_TIME, magnitude, ramp_s=6.0),), duration=DURATION, dt=DT,
                     seed=seeds["ramp"], trajectory=Trajectory("hover", altitude=5.0))
        history = OnlineExpert().run(flight)
        row = {}
        for t in instants:
            horizon = confidence_horizon(flight, history, calibration, index=history.index(t),
                                         threshold_m_s2=threshold)
            row[f"{t:.1f}"] = {"seconds": horizon.seconds, "level_m_s2": horizon.level_m_s2,
                               "rate_m_s2_s": horizon.rate_m_s2_s, "capped": horizon.capped}
        ramping[magnitude] = flight, history
        verdict = diagnose(flight, detector.run(flight), calibration,
                           horizon=confidence_horizon(flight, history, calibration,
                                                      index=history.index(FAULT_TIME + 4.0),
                                                      threshold_m_s2=threshold))
        report["ramps"][f"{magnitude:.2f} kg"] = {"horizon": row, "diagnosis": verdict["label"],
                                                  "diagnosis_horizon_s": verdict.get("horizon_s")}
        print(f"  ramp +{magnitude:.2f} kg  N = " +
              "  ".join(f"{row[f'{t:.1f}']['seconds']:5.1f} s" for t in instants) +
              f"   (verdict: {verdict['label']}, horizon {verdict.get('horizon_s')} s)", flush=True)

    # --- a step: the model does not drift into the fault, it is already past it
    step = fly((Fault("payload", FAULT_TIME, 0.30),), duration=DURATION, dt=DT, seed=seeds["step"],
               trajectory=Trajectory("hover", altitude=5.0))
    step_history = OnlineExpert().run(step)
    for label, t in (("before", FAULT_TIME - 0.5), ("after", FAULT_TIME + 2.0)):
        horizon = confidence_horizon(step, step_history, calibration, index=step_history.index(t),
                                     threshold_m_s2=threshold)
        report["step"][label] = {"seconds": horizon.seconds, "level_m_s2": horizon.level_m_s2}
    print(f"  step  +0.30 kg  N = {report['step']['before']['seconds']:.1f} s before, "
          f"{report['step']['after']['seconds']:.1f} s after", flush=True)

    # --- the same flight, judged by the model a hover taught and the model a maneuver taught
    judged, judged_history = ramping[min(ramping)]
    for name, sigma in model_sigma.items():
        horizon = confidence_horizon(judged, None, calibration, index=judged_history.index(FAULT_TIME + 1.5),
                                     threshold_m_s2=threshold, sigma_m_s2=sigma)
        report["excitation"][name] = {"sigma_m_s2": sigma, "seconds": horizon.seconds,
                                      "level_m_s2": horizon.level_m_s2}
        print(f"  judged by a model identified on a {name:6s} flight (sigma {sigma:.3f} m/s^2): "
              f"N = {horizon.seconds:5.1f} s", flush=True)
    return report


# ---------------------------------------------------------------------------------------- 4. the external logs
EXTERNAL_LOGS = {
    "why": "the issue asks the estimator to be run on real flight logs (PX4 Flight Review, the CMU ALFA dataset). "
           "Neither could be obtained from this machine, and no run is reported rather than a fabricated one",
    "probes": [
        {"url": "https://review.px4.io/", "http": 200, "what": "the service is alive"},
        {"url": "https://review.px4.io/browse", "http": 200,
         "what": "returns a page with no log ids: Flight Review has no public index, a log is reachable only by the id "
                 "its owner publishes, and there is no id to use"},
        {"url": "https://theairlab.org/alfa-dataset/", "http": 200,
         "what": "ALFA: 1.7 GB zipped (12.5 GB unzipped) of ROS .bag files from an ArduPilot Pixhawk over MAVLink 2.0, "
                 "behind a Box shared link"},
        {"url": "https://api.box.com/2.0/shared_items", "http": 401,
         "what": "an anonymous shared-link query to the Box API needs an app token"},
        {"url": "https://kilthub.cmu.edu/api/datasets/:persistentId/?persistentId=doi:10.1184/R1/12707963",
         "http": 202, "what": "the Dataverse record behind the DOI answers the API with an empty body (bot challenge)"},
    ],
    "what_a_test_would_need": [
        "time-aligned accelerometer, gyro, attitude and GPS velocity at a known rate -- ALFA's .bag and a PX4 .ulg both "
        "have them, and this estimator's interface (OnlineExpert.update) takes exactly those plus the rotor speed",
        "the rotor speed the thrust is computed from. This simulator's ESC reports it; a log carries a PWM command and "
        "not the speed, so a PWM -> omega curve has to be identified first, and that curve is the largest single error "
        "term the expert has (a 1 % speed error is a 2 % thrust error, which is the whole vertical drag signal)",
        "the mass and the inertia of the airframe the log was flown on, which neither dataset publishes per flight; "
        "without them nothing can be scored, only cross-checked",
        "an interactive download (a browser or an authenticated client) and a .bag/.ulg reader, neither of which this "
        "environment has",
    ],
}


# -------------------------------------------------------------------------------------------------------- the report
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="fly 14 s per flight instead of 20 (the ramps and the horizons still land)")
    ap.add_argument("--out", default=_out("results", "expert_identification.json"))
    args = ap.parse_args(argv)
    started = time.time()
    if args.quick:
        # the seeds do not change: a shorter flight is a different measurement, not a different draw of the same one,
        # so the numbers of a quick run must not be compared with the committed ones
        global DURATION
        DURATION = 14.0
    base = 300
    seeds = {name: base + i for i, name in enumerate(("hover", "circle", "weave"))}
    seeds.update({name: base + 10 + i for i, name in enumerate(("ge_a", "ge_b", "ge_c"))})
    seeds.update({name: base + 20 + i for i, name in enumerate(("gust_a", "gust_b", "gust_c", "gust_d", "gust_e",
                                                                "gust_f", "gust_g"))})
    seeds.update({"ramp": base + 40, "step": base + 41})
    seeds["unseen amplitude 0.22 m"] = base + 50
    seeds["unseen amplitude 0.15 m"] = base + 51
    seeds["calm at 5 m, no ground effect"] = base + 52

    print("1. the expert, from a declaration that is 2x too heavy:", flush=True)
    identification, flown = identify(seeds=seeds)
    print("2. the residual the nominal model is missing:", flush=True)
    residual = learn_residual(seeds=seeds)
    print("3. the confidence horizon:", flush=True)
    horizon = horizons(seeds=seeds,
                       model_sigma={"hover": identification["hover"]["model_sigma_m_s2"],
                                    "weave": identification["weave"]["model_sigma_m_s2"]})

    checks = _checks(identification, residual, horizon)
    result = {"issue": 335, "dt_s": DT, "duration_s": DURATION, "fault_time_s": FAULT_TIME, "seeds": seeds,
              "identification": identification, "learned_residual": _plain(residual),
              "horizon": horizon, "external_logs": EXTERNAL_LOGS, "checks": checks,
              "truth": {"mass_kg": 1.5, "inertia": [0.011, 0.011, 0.021], "drag_area": [0.030, 0.030, 0.080]},
              "seconds": time.time() - started,
              "machine": {"platform": platform.platform(), "python": platform.python_version(),
                          "processor": platform.processor() or platform.machine(), "cpus": os.cpu_count()}}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"wrote {args.out}", flush=True)
    _figures(identification, flown, residual)
    return 0 if all(c["passed"] for c in checks) else 1


def _plain(value):
    """Only what a figure and a reader need: the calibration object is dropped from the record."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items() if k != "calibration"}
    return value


def _checks(identification, residual, horizon) -> list[dict]:
    """The claims of the report, held to the numbers that produced them."""
    def passed(name, condition, detail, kind):
        return {"name": name, "passed": bool(condition), "kind": kind, "label": detail}

    hover = identification["hover"]
    weave = identification["weave"]
    mass_worst = max(abs(row["mass_error"]) for row in identification.values())
    inertia_worst = max(abs(e) for e in weave["inertia_error"])
    vertical = abs(weave["drag_error"][2])
    ge = residual["ground_effect"]["cases"]
    wind = residual["wind"]["cases"]
    calm = wind["calm, no wind at all"]
    unseen = wind["unseen 8 m/s"]
    ramps = horizon["ramps"]
    first, second, last = (horizon["instants_s"][i] for i in (0, 1, -1))
    ordered = [ramps[k]["horizon"][f"{first:.1f}"]["seconds"] for k in ramps]
    # within one flight the horizon is non-increasing as the fault grows, and the smallest fault has to lose its
    # horizon in the seconds after the onset -- a strictly smaller number, not the same zero twice
    monotone = all(all(ramps[k]["horizon"][f"{a:.1f}"]["seconds"] >= ramps[k]["horizon"][f"{b:.1f}"]["seconds"]
                       for a, b in zip(horizon["instants_s"], horizon["instants_s"][1:], strict=False))
                   for k in ramps)
    shortest = ramps[min(ramps)]["horizon"]
    return [
        passed("mass_is_identified_in_every_flight", mass_worst < 0.02,
               f"worst mass error over hover, circle and weave is {100 * mass_worst:.2f} %", "numerical"),
        passed("a_hover_does_not_identify_the_inertia",
               hover["inertia_std"][0] / hover["inertia"][0] > 0.2 and inertia_worst < 0.10,
               f"the hover's inertia is +-{100 * hover['inertia_std'][0] / hover['inertia'][0]:.0f} % "
               f"(and {100 * hover['inertia_error'][0]:+.0f} % wrong); the weave's is "
               f"{100 * inertia_worst:.1f} %", "numerical"),
        passed("the_vertical_drag_is_not_identified", vertical > 0.10,
               f"the drag along the thrust axis comes out {100 * weave['drag_area'][2] / 0.08:.0f} % of the truth: a "
               f"flight that does not fly vertically cannot identify it", "reference"),
        passed("the_ground_effect_transfers_to_unseen_flights",
               all(case["ratio"] < 0.8 for name, case in ge.items() if name.startswith("unseen")),
               "held-out flights through the skirt: " +
               ", ".join(f"{k} {v['ratio']:.2f}x" for k, v in ge.items()), "numerical"),
        passed("the_ground_effect_leaves_a_calm_flight_alone", ge["calm at 5 m, no ground effect"]["ratio"] < 1.05,
               f"a flight at 5 m is unchanged ({ge['calm at 5 m, no ground effect']['ratio']:.2f}x)", "numerical"),
        passed("the_causal_wind_transfers_to_unseen_speeds", unseen["causal_ratio"] < 0.4,
               f"the 8 m/s gust it never saw: {unseen['before_m_s2']:.2f} -> {unseen['causal_m_s2']:.2f} m/s^2 "
               f"({unseen['causal_ratio']:.2f}x)", "numerical"),
        passed("a_pooled_wind_does_not_transfer", calm["pooled_ratio"] > 2.0,
               f"one constant world force fitted across the gusts makes a calm flight {calm['pooled_ratio']:.1f}x "
               f"worse: the wind is a state, not a coefficient", "numerical"),
        passed("the_causal_wind_leaves_a_calm_flight_alone", calm["causal_ratio"] < 1.05,
               f"a calm flight is unchanged ({calm['causal_ratio']:.2f}x)", "numerical"),
        passed("the_horizon_shortens_as_the_fault_grows",
               ordered[0] >= ordered[1] >= ordered[2] and monotone and
               shortest[f"{second:.1f}"]["seconds"] < shortest[f"{first:.1f}"]["seconds"],
               "half a second after the onset: " + ", ".join(f"{k} {v:.1f} s" for k, v in
                                                             zip(ramps, ordered, strict=False)) +
               f"; the smallest of them is down to {shortest[f'{second:.1f}']['seconds']:.1f} s one second later and "
               f"{shortest[f'{last:.1f}']['seconds']:.1f} s four seconds after that", "numerical"),
        passed("a_step_fault_expires_the_model", horizon["step"]["after"]["seconds"] < 1.0 <
               horizon["step"]["before"]["seconds"],
               f"the model answers for {horizon['step']['before']['seconds']:.1f} s before a 0.30 kg step and "
               f"{horizon['step']['after']['seconds']:.1f} s after it", "numerical"),
        passed("a_poorly_excited_model_answers_for_less", horizon["excitation"]["hover"]["seconds"] <
               horizon["excitation"]["weave"]["seconds"] and horizon["excitation"]["hover"]["sigma_m_s2"] >
               3.0 * horizon["excitation"]["weave"]["sigma_m_s2"],
               f"the same flight, judged by a model a hover taught, answers for "
               f"{horizon['excitation']['hover']['seconds']:.1f} s; by one a maneuver taught, "
               f"{horizon['excitation']['weave']['seconds']:.1f} s", "numerical"),
    ]


def _figures(identification, flown, residual) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = flown["truth"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    for name, history in flown["histories"].items():
        axes[0].plot(history.time, 100.0 * (history.mass / truth["mass_kg"] - 1.0), lw=1.0, label=name)
    axes[0].axhline(0.0, color="#999999", lw=0.8)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("mass error (%)")
    axes[0].set_ylim(-5.0, 55.0)          # the declaration it starts from (+100 %) is off the top: what matters is
    axes[0].set_title("mass, from a declaration 2x too heavy")   # how fast it comes down, and a hover does it in one sample
    axes[0].legend(frameon=False, fontsize=8)

    for name, history in flown["histories"].items():
        axes[1].plot(history.time, 100.0 * (history.inertia[:, 2] / truth["inertia"][2] - 1.0), lw=1.0, label=name)
    axes[1].axhline(0.0, color="#999999", lw=0.8)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("yaw inertia error (%)")
    axes[1].set_ylim(-25.0, 130.0)
    axes[1].set_title("inertia: only the maneuver identifies it")

    for name, history in flown["histories"].items():
        axes[2].plot(history.time, 100.0 * (history.drag[:, 0] / truth["drag_area"][0] - 1.0), lw=1.0, label=name)
    axes[2].axhline(0.0, color="#999999", lw=0.8)
    axes[2].axhline(100.0, color="#a23a00", lw=0.8, ls="--")
    axes[2].annotate("the declared value", xy=(12.0, 104.0), fontsize=8, color="#a23a00")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("horizontal drag error (%)")
    axes[2].set_ylim(-25.0, 215.0)
    axes[2].set_title("drag area, against what was declared")
    fig.savefig(_out("figures", "expert_identification.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
