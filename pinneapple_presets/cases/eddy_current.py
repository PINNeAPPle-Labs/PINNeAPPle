"""Eddy-current inspection of aircraft skin: probe impedance, depth of penetration and thinning sensitivity."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_physics.closed_form.eddy_coil import MU0, coil_impedance
from pinneapple_simulation.numerical_solvers.eddy_current_fdm import (
    annular_current_source,
    eddy_current_density,
    solve_axisymmetric_eddy_current,
)

from ..core import Output, Param, Preset, check

DD = "Dodd & Deeds, J. Appl. Phys. 39(6):2829-2838 (1968), solução analítica bobina sobre placa"
IACS = 58.0e6                                                          # S/m, 100 % IACS


class EddyCurrentInspection(Preset):
    name = "eddy_current_skin_inspection"
    title = "Inspeção por correntes parasitas de chapa de alumínio aeronáutico: impedância e afinamento"
    sector = "Eletromagnetismo / ensaios não destrutivos (aeronáutica, óleo e gás)"
    problem = ("Sonda absoluta tipo pancake (bobina com núcleo de ar) sobre chapa de Al 2024-T3. Variação de "
               "impedância pela solução analítica de Dodd & Deeds, conferida pelo FDM axissimétrico da lib; "
               "profundidade de penetração; sensibilidade ao afinamento por corrosão frente ao lift-off.")
    question = ("Nesta frequência a sonda distingue 10 % de perda de espessura da variação de lift-off do operador? "
                "Qual frequência de ensaio separa melhor os dois sinais, e quanto de espessura se lê de uma medição "
                "em várias frequências?")
    params = (
        Param("f", "Hz", 10e3, "premissa: ensaio de corrosão em chapa fina, 1-100 kHz típico (ASM Handbook vol. 17)",
              100, 1e6),
        Param("sigma", "S/m", 0.30 * IACS, "Al 2024-T3: ~30 % IACS (ASM / MIL-HDBK-5): 17,4 MS/m", 0.5e6, 60e6),
        Param("thickness", "m", 0.002, "premissa: revestimento de fuselagem de 2,0 mm", 0.0002, 0.02),
        Param("liftoff", "m", 0.0005, "premissa: 0,5 mm (camada de tinta + desgaste da face da sonda)", 0.0, 0.003),
        Param("r1", "m", 0.002, "premissa: bobina pancake, raio interno 2 mm", 0.0002, 0.02),
        Param("r2", "m", 0.004, "raio externo 4 mm", 0.0005, 0.03),
        Param("height", "m", 0.002, "altura da bobina 2 mm", 0.0002, 0.02),
        Param("turns", "-", 200.0, "premissa: 200 espiras", 1, 5000),
        Param("loss", "-", 0.10, "afinamento a detectar: 10 % da espessura (critério típico de corrosão)", 0.01, 0.5),
        Param("liftoff_noise", "m", 5e-5, "premissa: variação de lift-off do operador ±0,05 mm (define a direção do "
              "lift-off no plano de impedância)", 1e-6, 5e-4),
        Param("noise_frac", "-", 0.002, "premissa: ruído do instrumento 0,2 % de |dZ|", 1e-4, 0.05),
        Param("angle_min", "deg", 20.0, "premissa: separação de fase mínima entre afinamento e lift-off para girar o "
              "lift-off para a horizontal e ler o afinamento na vertical (prática de ajuste de fase)", 5, 60),
    )
    outputs = (
        Output("dR", "ohm", "variação de resistência da sonda"), Output("dX", "ohm", "variação de reatância"),
        Output("R_norm", "-", "dR / X0 (plano de impedância normalizado)"), Output("X_norm", "-", "(X0 + dX) / X0"),
        Output("delta", "m", "profundidade padrão de penetração"), Output("t_over_delta", "-", "espessura / delta"),
        Output("signal_loss", "ohm", "|dZ| causado pelo afinamento"),
        Output("signal_liftoff", "ohm", "|dZ| causado pelo ruído de lift-off"),
        Output("phase_angle_deg", "deg", "ângulo entre os sinais de afinamento e de lift-off no plano de impedância"),
        Output("separation", "-", "componente do sinal de afinamento perpendicular ao lift-off / ruído do instrumento"),
    )
    validity = ("Bobina de núcleo de ar, chapa plana infinita não magnética (mu_r = 1) com ar abaixo, regime "
                "harmônico quase estático (f << ressonância da bobina), sem trincas (um defeito localizado precisa do "
                "FDM ou de um modelo 3-D).")
    pipeline = ("geometria: bobina retangular (r1, r2, altura, lift-off), chapa de espessura c",
                "modelo: potencial vetor A_theta, regime harmônico, sigma da liga (% IACS)",
                "solver: coil_impedance (Dodd & Deeds, quadratura de Gauss em log alpha); FDM axissimétrico para "
                "conferir",
                "pós: plano de impedância, delta, sinal de afinamento vs lift-off",
                "decisão: ângulo entre afinamento e lift-off >= mínimo e componente perpendicular >= 3x ruído")
    library = ("pinneapple_physics.closed_form.eddy_coil.coil_impedance",
               "pinneapple_simulation.numerical_solvers.eddy_current_fdm.solve_axisymmetric_eddy_current")
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("phase_angle_deg", "separation", "t_over_delta")
    limitations = ("O ângulo entre afinamento e lift-off é o ângulo entre duas linhas, dobrado em 0-90°. O sinal do "
                   "afinamento gira cerca de 2 t/delta rad atrás do lift-off conforme a frequência sobe (atraso de "
                   "fase da parede de fundo), então o ângulo dobrado sobe e desce em dente de serra acima de ~4 kHz. "
                   "Isso é físico: o atraso desdobrado é suave e linear em t/delta (figura, eixo da direita). Acima "
                   "de ~12 kHz o sinal de afinamento cai abaixo de 3x o ruído e essas frequências não servem "
                   "(faixa cinza).",)
    missing = {"surrogate": "a solução analítica leva ~10 ms; um surrogate não economiza tempo",
               "reason": "não há caso de ensaio não destrutivo no pinneapple_reason"}

    def _z(self, p, **over):
        q = {**p, **over}
        return coil_impedance(q["f"], r1=q["r1"], r2=q["r2"], l1=q["liftoff"], l2=q["liftoff"] + q["height"],
                              turns=q["turns"], sigma=q["sigma"], thickness=q["thickness"])

    def solve(self, p):
        z = self._z(p)
        zl = self._z(p, thickness=p["thickness"] * (1 - p["loss"]))["dZ"]
        zo = self._z(p, liftoff=p["liftoff"] + p["liftoff_noise"])["dZ"]
        s_loss, s_lift = zl - z["dZ"], zo - z["dZ"]
        perp = abs((s_loss * np.conj(s_lift)).imag) / max(abs(s_lift), 1e-30)
        lag = float(np.angle(s_loss / s_lift))
        ang = float(np.degrees(abs(lag)))
        ang = min(ang, 180 - ang)  # angle between the two signal lines, in [0, 90]
        return {**{k: z[k] for k in ("dR", "dX", "R_norm", "X_norm", "delta", "X0")},
                "t_over_delta": p["thickness"] / z["delta"], "signal_loss": abs(s_loss), "signal_liftoff": abs(s_lift),
                "phase_angle_deg": ang, "phase_lag_rad": lag, "signal_perp": perp,
                "separation": perp / (p["noise_frac"] * abs(z["dZ"]))}

    def checks(self, p, out):
        return [check("passive load: dR >= 0 (eddy losses)", out["dR"] >= 0, value=out["dR"], limit=0.0),
                check("conductor lowers the inductance: dX <= 0", out["dX"] <= 0, value=out["dX"], limit=0.0),
                check("coil geometry r2 > r1", p["r2"] > p["r1"]),
                check("quasi-static: probe much smaller than the wavelength", 2 * p["r2"] < 3e8 / p["f"] / 100,
                      value=2 * p["r2"], limit=3e8 / p["f"] / 100)]

    def decide(self, p, out):
        ok = out["separation"] >= 3 and out["phase_angle_deg"] >= p["angle_min"]
        return {"verdict": "OK" if ok else "NOT OK",
                "reasons": [f"delta = {1e3 * out['delta']:.2f} mm, t/delta = {out['t_over_delta']:.2f}",
                            f"sinal do afinamento de {100 * p['loss']:.0f} %: {1e3 * out['signal_loss']:.2f} mohm, "
                            f"lift-off ±{1e6 * p['liftoff_noise']:.0f} um: {1e3 * out['signal_liftoff']:.2f} mohm, "
                            f"ângulo de fase entre eles {out['phase_angle_deg']:.0f}°",
                            f"componente perpendicular / ruído = {out['separation']:.1f} (mín. 3); ângulo "
                            f"{out['phase_angle_deg']:.0f}° (mín. {p['angle_min']:.0f}°)"]}

    # ------------------------------------------------------------------------------------------- validation
    def fdm(self, p, h: float, box: float = 0.05):
        """The same probe and plate solved by the library's axisymmetric FDM; returns dZ, X0 and the fields."""
        r = np.arange(0, box + h / 2, h)
        z = np.arange(-box, box + h / 2, h)
        J = annular_current_source(r, z, (p["r1"] + p["r2"]) / 2, p["liftoff"] + p["height"] / 2, p["height"],
                                   (p["r2"] - p["r1"]) / 2, p["turns"], 1.0)
        RR, ZZ = np.meshgrid(r, z, indexing="ij")
        mu = MU0 * np.ones_like(RR)
        w = 2 * math.pi * p["f"]
        Z, As = [], []
        for s in (0.0, p["sigma"]):
            sg = np.where((ZZ < 0) & (ZZ >= -p["thickness"] - 1e-12), s, 0.0)
            A = solve_axisymmetric_eddy_current(r, z, w, mu, sg, J)
            Z.append(1j * w * np.sum(2 * math.pi * RR * A * J) * h * h)
            As.append((A, sg))
        return {"dZ": Z[1] - Z[0], "X0": Z[0].imag, "r": r, "z": z, "A": As[1][0], "sigma_map": As[1][1]}

    def validate(self):
        p = self.defaults()
        o = self.run().outputs
        dz = complex(o["dR"], o["dX"])
        rows = []
        res = {}
        for h in (2e-4, 1e-4):
            f = self.fdm(p, h)
            res[h] = f["dZ"]
        rows.append({"case": "dZ da sonda: Dodd & Deeds vs FDM axissimétrico da lib (h = 0,1 mm)", "output": "dZ",
                     "unit": "ohm", "computed": abs(dz), "reference": abs(res[1e-4]),
                     "rel_error": abs(dz - res[1e-4]) / abs(res[1e-4]), "tol": 0.02,
                     "pass": bool(abs(dz - res[1e-4]) / abs(res[1e-4]) <= 0.02), "kind": "independent_solver",
                     "source": "solve_axisymmetric_eddy_current (diferenças finitas, A = 0 a 0,05 m); a analítica é "
                               + DD, "params": {}})
        # thick plate limit: a plate of 10 delta equals the half-space
        hs = coil_impedance(p["f"], r1=p["r1"], r2=p["r2"], l1=p["liftoff"], l2=p["liftoff"] + p["height"],
                            turns=p["turns"], sigma=p["sigma"])
        th = self._z(p, thickness=10 * o["delta"])
        rows.append({"case": "placa de 10 delta = semi-espaço", "output": "dZ", "unit": "ohm", "computed": abs(th["dZ"]),
                     "reference": abs(hs["dZ"]), "rel_error": abs(th["dZ"] - hs["dZ"]) / abs(hs["dZ"]), "tol": 1e-6,
                     "pass": bool(abs(th["dZ"] - hs["dZ"]) / abs(hs["dZ"]) <= 1e-6), "kind": "analytic",
                     "source": DD + " (limite c -> infinito)", "params": {}})
        # air-coil inductance against Wheeler's short-coil formula (independent, 1928)
        a = (p["r1"] + p["r2"]) / 2 / 0.0254
        b, c = p["height"] / 0.0254, (p["r2"] - p["r1"]) / 0.0254
        Lw = 0.8 * a ** 2 * p["turns"] ** 2 / (6 * a + 9 * b + 10 * c) * 1e-6
        rows.append({"case": "indutância da bobina no ar: Dodd & Deeds vs Wheeler (multicamada)", "output": "L0",
                     "unit": "H", "computed": o["X0"] / (2 * math.pi * p["f"]), "reference": Lw,
                     "rel_error": abs(o["X0"] / (2 * math.pi * p["f"]) - Lw) / Lw, "tol": 0.05,
                     "pass": bool(abs(o["X0"] / (2 * math.pi * p["f"]) - Lw) / Lw <= 0.05), "kind": "analytic",
                     "source": "Wheeler, Proc. IRE 16 (1928): L = 0,8 a2 N2/(6a + 9b + 10c) uH em polegadas, ~1 %",
                     "params": {}})
        conv = {"parameter": "passo h do FDM [m]", "values": [2e-4, 1e-4],
                "dZ_abs": [abs(res[2e-4]), abs(res[1e-4])],
                "rel_change_last": abs(res[1e-4] - res[2e-4]) / abs(res[1e-4]),
                "error_vs_analytic": [abs(res[k] - dz) / abs(dz) for k in (2e-4, 1e-4)],
                "pass": abs(res[1e-4] - dz) < abs(res[2e-4] - dz)}
        return {"preset": self.name, "references": rows, "convergence": conv,
                "all_pass": all(r["pass"] for r in rows) and conv["pass"]}

    # ------------------------------------------------------------------------------------------- extras
    def optimize(self, **kw):
        """Test frequency with the largest phase angle between the thinning and lift-off signals among those whose
        perpendicular thinning signal is at least 3x the instrument noise, scanned 1-200 kHz; the pick re-run."""
        fs = np.geomspace(1e3, 2e5, 120)
        runs = [self.run(**{**kw, "f": float(f)}).outputs for f in fs]
        sep = [o["separation"] for o in runs]
        # the thinning signal turns ~2 t/delta rad behind lift-off as f rises (back-wall phase lag); unwrapped here, the
        # folded line angle in [0, 90] deg is what separates them, so it saws as the lag passes multiples of 90 deg
        lag = np.degrees(np.unwrap([o["phase_lag_rad"] for o in runs])).tolist()
        ang = [o["phase_angle_deg"] if o["separation"] >= 3 else -1 for o in runs]
        k = int(np.argmax(ang))
        ver = self.run(**{**kw, "f": float(fs[k])})
        return {"objective": "frequência de maior ângulo afinamento/lift-off com sinal perpendicular >= 3x ruído",
                "f_best_hz": float(fs[k]), "angle_best_deg": runs[k]["phase_angle_deg"], "separation_best": sep[k], "t_over_delta_best": ver.outputs["t_over_delta"],
                "angle_at_given_f": self.run(**kw).outputs["phase_angle_deg"], "verified_ok":
                    ver.decision["verdict"] == "OK", "scan_hz": fs.tolist(), "scan_separation": sep,
                "scan_angle_deg": [o["phase_angle_deg"] for o in runs], "scan_phase_lag_deg": lag,
                "scan_t_over_delta": [o["t_over_delta"] for o in runs]}

    def invert(self, true_t: float = 0.0017, true_lo: float = 0.0006, noise: float = 0.005, seed: int = 0, **kw):
        """Remaining thickness and lift-off from a multi-frequency measurement (2, 5, 10, 20, 50 kHz), 0.5 % noise
        on each impedance component. Synthetic truth: 1.70 mm left of 2.0 mm, lift-off 0.6 mm."""
        p = self.resolve(**kw)
        fs = np.array([2e3, 5e3, 10e3, 20e3, 50e3])

        def meas(t, lo):
            out = []
            for f in fs:
                z = self._z(p, f=float(f), thickness=t, liftoff=lo)
                out += [z["dR"], z["dX"]]
            return np.array(out)

        rng = np.random.default_rng(seed)
        y0 = meas(true_t, true_lo)
        sig = noise * np.abs(y0)
        y = y0 + rng.normal(0, 1, len(y0)) * sig
        x = np.array([p["thickness"], p["liftoff"]])
        for _ in range(40):
            r0 = (meas(*x) - y) / sig
            J = np.column_stack([(meas(*(x + e)) - meas(*(x - e))) / (2 * e.sum()) / sig
                                 for e in (np.array([1e-6, 0]), np.array([0, 1e-6]))])
            step = np.linalg.lstsq(J, -r0, rcond=None)[0]
            x = x + step
            if np.abs(step).max() < 1e-10:
                break
        cov = np.linalg.inv(J.T @ J)
        sd = np.sqrt(np.diag(cov))
        return {"description": "Espessura remanescente e lift-off a partir da impedância medida em 5 frequências (2 a "
                               "50 kHz), ruído de 0,5 % em cada componente. Verdade sintética: 1,70 mm (de 2,0) e "
                               "lift-off de 0,6 mm.", "parameter": "espessura, lift-off", "unit": "m",
                "true": [true_t, true_lo], "recovered": x.tolist(),
                "ci95": [[float(x[i] - 1.96 * sd[i]), float(x[i] + 1.96 * sd[i])] for i in range(2)],
                "rel_error": float(abs(x[0] - true_t) / true_t),
                "true_inside_ci95": bool(np.all(np.abs(x - [true_t, true_lo]) <= 1.96 * sd)),
                "noise": f"{100 * noise} %", "how": "Gauss-Newton ponderado sobre Dodd & Deeds, covariância (J^T J)^-1"}

    def variants(self):
        return {"1 kHz": {"f": 1e3}, "3 kHz": {"f": 3e3}, "10 kHz (base)": {}, "100 kHz": {"f": 1e5},
                "titânio Ti-6Al-4V (1 % IACS)": {"sigma": 0.0104 * IACS},
                "inox 304 (2,4 % IACS)": {"sigma": 0.024 * IACS}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p = result.params
        fs = np.geomspace(500, 5e5, 30)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for frac, lab in ((1.0, "t nominal"), (1 - p["loss"], f"-{100 * p['loss']:.0f} %"), (0.5, "-50 %")):
            zz = [self._z(p, f=float(f), thickness=p["thickness"] * frac) for f in fs]
            ax[0].plot([z["R_norm"] for z in zz], [z["X_norm"] for z in zz], "-o", ms=2, label=lab)
        lo = np.linspace(0, 2e-3, 20)
        zz = [self._z(p, liftoff=float(x)) for x in lo]
        ax[0].plot([z["R_norm"] for z in zz], [z["X_norm"] for z in zz], "k--", label="lift-off 0-2 mm")
        ax[0].set(xlabel="dR / X0", ylabel="X / X0", title="Plano de impedância normalizado")
        ax[0].legend(fontsize=7)
        o = self.optimize(**{k: v for k, v in p.items() if k != "f"})

        hz, sep = np.array(o["scan_hz"]), np.array(o["scan_separation"])
        ax[1].semilogx(hz, o["scan_angle_deg"], label="ângulo entre as linhas (0-90°)")
        ax[1].axhline(p["angle_min"], color="r", ls="--", lw=1, label=f"mínimo {p['angle_min']:.0f}°")
        bad = sep < 3
        ax[1].fill_between(hz, 0, 90, where=bad, color="0.85", step="mid", label="afinamento < 3x ruído")
        ax[1].axvline(o["f_best_hz"], color="g", lw=1, label=f"escolhida {o['f_best_hz'] / 1e3:.1f} kHz")
        ax[1].axvline(p["f"], color="k", lw=0.8, ls=":", label=f"dada {p['f'] / 1e3:.0f} kHz")
        ax[1].set(xlabel="frequência [Hz]", ylabel="ângulo afinamento / lift-off [graus]", ylim=(0, 92),
                  title="Escolha da frequência (atraso de fase ~ 2 t/delta, dobrado em 0-90°)")
        tw = ax[1].twinx()
        tw.semilogx(hz, -np.array(o["scan_phase_lag_deg"]), color="C1", lw=0.8, ls="--")
        tw.set_ylabel("atraso de fase desdobrado [graus]", color="C1")
        ax[1].legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        path = os.path.join(out_dir, "eddy.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from ..viz3d import Scene, Surface, box
        p = result.params
        f = self.fdm(p, 2e-4, box=0.02)
        r, z, A, sg = f["r"], f["z"], f["A"], f["sigma_map"]
        J = np.abs(eddy_current_density(A, 2 * math.pi * p["f"], sg))
        k = int(np.argmin(np.abs(z - (-1e-4))))                    # just under the plate surface
        Jr = J[:, k] / 1e6
        # plate top surface as a disc of radius 15 mm coloured by |J|, the coil as a ring above it
        rr = np.linspace(0, 0.015, 60)
        th = np.linspace(0, 2 * np.pi, 72)
        R, T = np.meshgrid(rr, th, indexing="ij")
        V = np.c_[(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel(), np.zeros(R.size)]
        idx = np.arange(R.size).reshape(R.shape)
        q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        F = np.r_[q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
        plate = Surface("plate_top", V * 1e3, F, "aluminium", {"J": np.interp(R.ravel(), r, Jr)})
        bv, bf = box([-15, -15, -1e3 * p["thickness"]], [15, 15, -0.01])
        under = Surface("plate", bv, bf, "aluminium", {"J": np.zeros(len(bv))})
        ring = []
        for zz in (p["liftoff"], p["liftoff"] + p["height"]):
            for rad in (p["r1"], p["r2"]):
                ring.append((rad, zz))
        Rc, Tc = np.meshgrid(np.linspace(p["r1"], p["r2"], 6), th, indexing="ij")
        top = np.c_[(Rc * np.cos(Tc)).ravel(), (Rc * np.sin(Tc)).ravel(), np.full(Rc.size, p["liftoff"] + p["height"])]
        idc = np.arange(Rc.size).reshape(Rc.shape)
        qc = np.stack([idc[:-1, :-1], idc[1:, :-1], idc[1:, 1:], idc[:-1, 1:]], -1).reshape(-1, 4)
        coil = Surface("coil", top * 1e3, np.r_[qc[:, [0, 1, 2]], qc[:, [0, 2, 3]]], "copper",
                       {"J": np.zeros(len(top))})
        return Scene([plate, under, coil], title=self.title,
                     labels={"J": "|J| de correntes parasitas na superfície [A/mm2 por A na bobina]"})
