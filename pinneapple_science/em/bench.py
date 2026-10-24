"""Bench experiments (practice): a virtual laboratory with instruments that add noise, calibration error and
resolution, a fit with its uncertainty, and the comparison with the value the bench was built with.

* :class:`Mu0Solenoid` -- measure mu0 from B(I) at the centre of a solenoid with a Hall probe and an ammeter
  (3-D field of the real helix, Earth's field as an offset), the dimensions read with a caliper.
* :class:`RCDischarge` -- measure a capacitance from the discharge of an RC circuit on an oscilloscope.

Each ends with a GUM uncertainty budget (type A from the fit, type B from the instrument specifications) and a
Monte Carlo of the whole bench (fresh instruments each time) that measures how often the stated expanded
uncertainty covers the true value: the uncertainty claim is itself checked.
"""
from __future__ import annotations

import math

import numpy as np

from pinneapple_simulation.electromagnetics import statics as S
from pinneapple_simulation.electromagnetics.constants import MU0

from .. import plots as P
from ..core import Experiment, Result
from ..instruments import Ammeter, FieldProbe, Instrument, Oscilloscope, curve_fit, linear_fit
from ..registry import register

AREA = "physics.electromagnetism"


@register
class Mu0Solenoid(Experiment):
    name = "em.bench_mu0_solenoid"
    title = "Bench: measuring mu0 with a solenoid"
    area = AREA
    topic = "magnetostatics (laboratory)"
    kind = "practice"
    dims = ("3D",)
    description = ("A wound solenoid, a current supply, an ammeter and a Hall probe at the centre. The current is "
                   "stepped, B read (averaged), B(I) fitted; mu0 = slope sqrt(L^2 + 4 R^2) / N with L and R from a "
                   "caliper. The field is the 3-D Biot-Savart field of the actual helix plus Earth's field. The "
                   "result carries its uncertainty budget and is compared with CODATA 2022.")
    references = ("JCGM 100:2008 (GUM)", "CODATA 2022: mu0 = 1.25663706127(20)e-6 N/A^2 (Mohr et al., "
                  "Rev. Mod. Phys. 97, 025002 (2025))", "Griffiths, 4th ed., prob. 5.11 (finite solenoid)")
    params = {"turns": 500, "length": 0.30, "radius": 0.025, "I_max": 2.0, "points": 11, "repeats": 10,
              "B_earth": 30e-6, "probe_noise": 5e-6, "probe_gain_u": 5e-3, "probe_offset_u": 2e-6,
              "probe_resolution": 1e-6, "ammeter_noise": 1e-3, "ammeter_gain_u": 1e-3, "ammeter_resolution": 1e-3,
              "caliper_u": 2e-4, "seed": 7, "monte_carlo": 300}
    quick = {"monte_carlo": 60, "turns": 200}

    MU0_CODATA = 1.25663706127e-6
    MU0_CODATA_U = 0.00000000020e-6

    def _bench(self, p, seed, B_per_A):
        rng = np.random.default_rng(seed)
        probe = FieldProbe(noise=p["probe_noise"], gain_u=p["probe_gain_u"], offset_u=p["probe_offset_u"],
                           resolution=p["probe_resolution"], seed=int(rng.integers(1 << 31)))
        amm = Ammeter(noise=p["ammeter_noise"], gain_u=p["ammeter_gain_u"], resolution=p["ammeter_resolution"],
                      seed=int(rng.integers(1 << 31)))
        caliper = Instrument("caliper", "m", noise=p["caliper_u"], seed=int(rng.integers(1 << 31)))
        I_set = np.linspace(0, p["I_max"], p["points"])
        I_true = I_set * (1 + 2e-3 * rng.standard_normal(len(I_set)))       # the supply is not exact either
        I_read = amm.read(I_true, n=p["repeats"])
        B_read = probe.read(B_per_A * I_true + p["B_earth"], n=p["repeats"])
        sB = probe.sigma(B_read, n=p["repeats"])
        fit = linear_fit(I_read, B_read, sigma_y=sB)
        L = caliper.read(p["length"])
        R = caliper.read(p["radius"])
        geo = math.sqrt(L * L + 4 * R * R)
        mu = fit.slope * geo / p["turns"]
        # GUM budget (relative standard uncertainties)
        u = {"fit slope (type A)": fit.slope_u / fit.slope,
             "Hall probe gain (type B)": p["probe_gain_u"],
             "ammeter gain (type B)": p["ammeter_gain_u"],
             "length L (caliper)": (L / geo ** 2) * p["caliper_u"],
             "radius R (caliper)": (4 * R / geo ** 2) * p["caliper_u"]}
        uc = math.sqrt(sum(v * v for v in u.values()))
        return dict(mu=mu, uc=uc, budget=u, fit=fit, I=I_read, B=B_read, sB=sB, L=L, R=R)

    def _run(self, res: Result, dim: str, p):
        st, en = S.helix_segments(p["radius"], p["length"], p["turns"], n_per_turn=48)
        B_per_A = float(S.biot_savart(np.zeros((1, 3)), st, en, 1.0)[0, 2])
        b = self._bench(p, p["seed"], B_per_A)
        mu, uc = b["mu"], b["uc"]
        U = 2 * uc * mu
        res.value("mu0_measured", mu, "N/A^2", "slope of B(I) x sqrt(L^2 + 4 R^2) / N", uncertainty=uc * mu)
        res.value("expanded_uncertainty_k2", U, "N/A^2", "k = 2, GUM root-sum-square of the budget")
        res.value("earth_field_intercept", b["fit"].intercept, "T", "intercept of the B(I) fit (Earth's field + probe offset)")
        res.value("reduced_chi2", b["fit"].reduced_chi2, "", "of the weighted fit")
        for k, v in b["budget"].items():
            res.value(f"u_rel[{k}]", v, "", "relative standard uncertainty, budget line")
        res.compare("mu0 (bench) vs CODATA 2022", mu, self.MU0_CODATA, tol=U / mu, kind="codata", unit="N/A^2",
                    source="CODATA 2022 (Mohr et al., Rev. Mod. Phys. 97, 025002, 2025)",
                    how="compatibility test |mu0 - mu0_CODATA| <= U (k = 2); the tolerance is the stated U")
        # does the stated uncertainty cover the truth? Monte Carlo of the whole bench
        n = p["monte_carlo"]
        hits, z = 0, []
        for s in range(n):
            r = self._bench(p, 1000 + s, B_per_A)
            z.append((r["mu"] - MU0) / (r["uc"] * r["mu"]))
            hits += abs(r["mu"] - MU0) <= 2 * r["uc"] * r["mu"]
        cov = hits / n
        lo, hi = _wilson(hits, n)
        res.value("coverage_k2", cov, "", f"fraction of {n} independent benches whose k=2 interval holds mu0")
        res.check("stated uncertainty covers the truth about 95 % of the time", lo <= 0.9545 <= hi,
                  value=cov, detail=f"{hits}/{n}, Wilson 95 % interval {lo:.3f}-{hi:.3f}; "
                                    f"std of the normalised errors {np.std(z):.2f} (1 if the budget is right)")
        res.check("the helix is close to a current sheet at the centre (systematic below 0.1 %)",
                  abs(B_per_A / float(S.finite_solenoid_on_axis_B(0.0, p["radius"], p["length"], p["turns"], 1.0)) - 1)
                  < 1e-3, detail="Biot-Savart of the helix vs Griffiths prob. 5.11")
        res.data("bench", I=b["I"], B=b["B"], sigma_B=b["sB"])
        res.data("monte_carlo", normalised_error=z)
        fig, ax = P.figure(3, width=3.9)
        ax[0].errorbar(b["I"], b["B"] * 1e3, yerr=2 * b["sB"] * 1e3, fmt="o", color=P.COMPUTED, ms=5, capsize=2,
                       label="Hall probe readings (2 sigma)")
        ii = np.linspace(0, p["I_max"], 50)
        ax[0].plot(ii, (b["fit"].slope * ii + b["fit"].intercept) * 1e3, color=P.REFERENCE, lw=1.5,
                   label="weighted fit")
        ax[0].set(xlabel="I (A), ammeter", ylabel="B (mT)", title=f"solenoid, N = {p['turns']}")
        ax[0].legend(fontsize=7.5)
        names = list(b["budget"])
        vals = [b["budget"][k] * 1e3 for k in names]
        ax[1].barh(range(len(names)), vals, color=P.COMPUTED, height=0.55)
        ax[1].set_yticks(range(len(names)), [nm.split(" (")[0] for nm in names], fontsize=7.5)
        ax[1].set(xlabel="relative standard uncertainty (x 1e-3)", title="uncertainty budget")
        ax[2].hist(z, bins=24, color=P.COMPUTED, alpha=0.85, density=True, label=f"{n} virtual benches")
        zz = np.linspace(-4, 4, 200)
        ax[2].plot(zz, np.exp(-zz ** 2 / 2) / math.sqrt(2 * math.pi), color=P.REFERENCE, lw=1.5, label="N(0, 1)")
        ax[2].set(xlabel="(mu0_meas - mu0) / u_c", ylabel="density", title=f"coverage of U (k=2): {100 * cov:.1f} %")
        ax[2].legend(fontsize=7.5)
        P.badge(ax[0], f"mu0 = ({mu * 1e6:.4f} +- {U * 1e6:.4f}) e-6", abs(mu - MU0) <= U, loc="lower right")
        fig.tight_layout()
        res.figure("bench_mu0", fig)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


@register
class RCDischarge(Experiment):
    name = "em.bench_rc_discharge"
    title = "Bench: capacitance from an RC discharge on an oscilloscope"
    area = AREA
    topic = "circuits (laboratory)"
    kind = "practice"
    dims = ("0D",)
    description = ("A capacitor (nominal 470 nF, +-10 %) discharges through a resistor measured with an "
                   "ohmmeter. An 8-bit oscilloscope with noise and a time-base error records V(t); an exponential "
                   "fit gives tau and C = tau / R with its uncertainty, compared with the capacitor's true value "
                   "(the bench's ground truth, hidden from the fit).")
    references = ("JCGM 100:2008 (GUM)", "Nilsson & Riedel, Electric Circuits, 10th ed., sec. 7.2")
    params = {"R": 10e3, "C_nominal": 470e-9, "V0": 5.0, "ohmmeter_u": 1e-3, "scope_rate": 1e5,
              "scope_duration": 25e-3, "scope_range": 6.0, "scope_bits": 8, "scope_noise": 0.02,
              "scope_timebase_u": 5e-4, "seed": 3, "monte_carlo": 300}
    quick = {"monte_carlo": 60}

    def _bench(self, p, seed):
        rng = np.random.default_rng(seed)
        C_true = p["C_nominal"] * (1 + 0.1 / math.sqrt(3) * rng.uniform(-math.sqrt(3), math.sqrt(3)))
        R_true = p["R"] * (1 + 0.01 * rng.standard_normal())
        ohm = Instrument("ohmmeter", "ohm", gain_u=p["ohmmeter_u"], seed=int(rng.integers(1 << 31)))
        R_meas = ohm.read(R_true)
        scope = Oscilloscope(p["scope_rate"], p["scope_duration"], p["scope_range"], p["scope_bits"],
                             noise=p["scope_noise"], timebase_u=p["scope_timebase_u"], seed=int(rng.integers(1 << 31)))
        tau = R_true * C_true
        t, v = scope.capture(lambda tt: p["V0"] * np.exp(-tt / tau))
        f = curve_fit(lambda tt, a, ta, c: a * np.exp(-tt / ta) + c, t, v, [v[0], 0.004, 0.0], sigma_y=scope.sigma())
        tau_m, tau_u = f.params[1], f.errors[1]
        C = tau_m / R_meas
        budget = {"fit tau (type A)": tau_u / tau_m, "ohmmeter (type B)": p["ohmmeter_u"],
                  "time base (type B)": p["scope_timebase_u"]}
        uc = math.sqrt(sum(x * x for x in budget.values()))
        return dict(C=C, uc=uc, C_true=C_true, budget=budget, t=t, v=v, fit=f, R=R_meas)

    def _run(self, res: Result, dim: str, p):
        b = self._bench(p, p["seed"])
        C, uc = b["C"], b["uc"]
        U = 2 * uc * C
        res.value("C_measured", C, "F", "tau (exponential fit) / R (ohmmeter)", uncertainty=uc * C)
        res.value("expanded_uncertainty_k2", U, "F", "k = 2")
        res.value("tau_fit", b["fit"].params[1], "s", "exponential fit with offset, scope samples")
        for k, v in b["budget"].items():
            res.value(f"u_rel[{k}]", v, "", "relative standard uncertainty, budget line")
        res.compare("C (bench) vs the capacitor's true value", C, b["C_true"], tol=U / C, kind="ground_truth",
                    unit="F", source="the value the virtual bench was built with (drawn within the 10 % tolerance, "
                    "hidden from the fit)", how="compatibility |C - C_true| <= U (k = 2)")
        res.compare("C vs the nominal label (470 nF +- 10 %)", C, p["C_nominal"], tol=0.1, kind="ground_truth",
                    unit="F", source="manufacturer's tolerance band")
        n = p["monte_carlo"]
        hits, z = 0, []
        for s in range(n):
            r = self._bench(p, 5000 + s)
            z.append((r["C"] - r["C_true"]) / (r["uc"] * r["C"]))
            hits += abs(r["C"] - r["C_true"]) <= 2 * r["uc"] * r["C"]
        cov = hits / n
        lo, hi = _wilson(hits, n)
        res.value("coverage_k2", cov, "", f"fraction of {n} independent benches whose k=2 interval holds C_true")
        res.check("stated uncertainty covers the truth about 95 % of the time", lo <= 0.9545 <= hi,
                  value=cov, detail=f"{hits}/{n}, Wilson 95 % interval {lo:.3f}-{hi:.3f}; std of the normalised "
                                    f"errors {np.std(z):.2f}")
        fig, ax = P.figure(2, width=4.5)
        ax[0].plot(b["t"] * 1e3, b["v"], ".", color=P.COMPUTED, ms=2, label="oscilloscope samples (8 bit)")
        a_, ta, c_ = b["fit"].params
        ax[0].plot(b["t"] * 1e3, a_ * np.exp(-b["t"] / ta) + c_, color=P.REFERENCE, lw=1.5, label="fit")
        ax[0].set(xlabel="t (ms)", ylabel="V (V)", title=f"RC discharge, tau = {ta * 1e3:.4f} ms")
        ax[0].legend(fontsize=8)
        ax[1].hist(z, bins=24, color=P.COMPUTED, alpha=0.85, density=True, label=f"{n} virtual benches")
        zz = np.linspace(-4, 4, 200)
        ax[1].plot(zz, np.exp(-zz ** 2 / 2) / math.sqrt(2 * math.pi), color=P.REFERENCE, lw=1.5, label="N(0, 1)")
        ax[1].set(xlabel="(C_meas - C_true) / u_c", ylabel="density", title=f"coverage of U (k=2): {100 * cov:.1f} %")
        ax[1].legend(fontsize=8)
        P.badge(ax[0], f"C = {C * 1e9:.2f} +- {U * 1e9:.2f} nF (true {b['C_true'] * 1e9:.2f})",
                abs(C - b["C_true"]) <= U, loc="upper right")
        fig.tight_layout()
        res.figure("bench_rc", fig)
        res.data("monte_carlo", normalised_error=z)
