"""Shell-and-tube closed-loop water cooler (TEMA E, one shell pass, two tube passes): rating by epsilon-NTU."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_physics.closed_form.heat_exchanger import (
    effectiveness,
    gnielinski,
    kern_shell,
    lmtd_correction_1_2,
    one_two_exchanger_ode,
)

from ..core import Output, Param, Preset, check

INC = "Incropera et al., Fundamentals of Heat and Mass Transfer, 6a ed."
TEMA = "TEMA Standards 10a ed. (2019), RGP-T-2.4 (resistências de incrustação)"


class ShellTubeCooler(Preset):
    name = "shell_tube_cooler"
    title = "Resfriador casco-tubo de água de circuito fechado (TEMA E, 1 passe no casco, 2 nos tubos)"
    sector = "Térmica / processo e energia"
    problem = ("Água desmineralizada de um circuito fechado de resfriamento (mancais, inversores, moldes) no casco, "
               "água de torre nos tubos. Coeficientes por Kern (casco) e Gnielinski (tubos), incrustação TEMA, "
               "parede de aço carbono; desempenho por epsilon-NTU do trocador 1-2.")
    question = ("O trocador leva a água do circuito à temperatura de retorno exigida com a água de torre disponível, "
                "com velocidade de tubo na faixa recomendada? Quanta incrustação ele tolera, e qual o menor trocador "
                "que atende?")
    params = (
        Param("m_hot", "kg/s", 20.0, "premissa: circuito fechado de ~840 kW de carga térmica", 1, 200),
        Param("t_hot_in", "degC", 45.0, "premissa: retorno quente do circuito fechado", 30, 95),
        Param("m_cold", "kg/s", 25.0, "premissa: vazão de água de torre alocada ao trocador", 1, 300),
        Param("t_cold_in", "degC", 29.0, "água de torre no verão (projeto típico 29-32 °C)", 5, 40),
        Param("cp_hot", "J/kg K", 4179.0, INC + " Tab. A.6, água a 315 K", 4170, 4200),
        Param("mu_hot", "Pa s", 631e-6, INC + " Tab. A.6, água a 315 K", 2.8e-4, 1.8e-3),
        Param("k_hot", "W/m K", 0.634, INC + " Tab. A.6, água a 315 K", 0.55, 0.68),
        Param("cp_cold", "J/kg K", 4178.0, INC + " Tab. A.6, água a 305 K", 4170, 4220),
        Param("mu_cold", "Pa s", 769e-6, INC + " Tab. A.6, água a 305 K", 3e-4, 1.8e-3),
        Param("k_cold", "W/m K", 0.620, INC + " Tab. A.6, água a 305 K", 0.55, 0.68),
        Param("n_tubes", "-", 200, "premissa: 200 tubos (2 passes de 100)", 10, 2000),
        Param("tube_od", "m", 0.01905, "tubo de 3/4\" (TEMA, diâmetro externo padrão)", 0.00635, 0.0508),
        Param("tube_wall", "m", 0.00165, "16 BWG = 1,65 mm (TEMA Tabela D-7)", 0.0005, 0.005),
        Param("tube_length", "m", 4.877, "16 ft (comprimento padrão TEMA)", 1.0, 12.0),
        Param("pitch", "m", 0.02381, "passo triangular de 15/16\" (1,25 x OD, TEMA RCB-2.5)", 0.008, 0.08),
        Param("shell_id", "m", 0.489, "casco de 19,25\" para ~200 tubos de 3/4\" em 15/16\" triangular, 2 passes "
              "(ordem de grandeza da contagem de tubos de Kern, Process Heat Transfer, Tab. 9)", 0.1, 2.5),
        Param("baffle_spacing", "m", 0.30, "premissa: ~0,6 x diâmetro do casco (TEMA mínimo = ID/5)", 0.05, 2.0),
        Param("k_wall", "W/m K", 50.0, "aço carbono (Incropera Tab. A.1: AISI 1010 63,9, aço-carbono ao Mn ~50)",
              10, 400),
        Param("rf_hot", "m2 K/W", 8.8e-5, TEMA + ": água desmineralizada/destilada, 0,0005 h ft2 F/Btu", 0.0, 2e-3),
        Param("rf_cold", "m2 K/W", 1.76e-4, TEMA + ": água de torre tratada, < 50 °C, 0,001 h ft2 F/Btu", 0.0,
              2e-3),
        Param("t_hot_target", "degC", 36.0, "premissa: alimentação fria do circuito a 36 °C", 15, 90),
    )
    outputs = (
        Output("q", "W", "calor trocado"), Output("t_hot_out", "degC", "saída do circuito fechado (casco)"),
        Output("t_cold_out", "degC", "saída da água de torre"), Output("U", "W/m2 K", "coeficiente global (área externa)"),
        Output("area", "m2", "área externa dos tubos"), Output("ntu", "-", "NTU"),
        Output("effectiveness", "-", "efetividade"), Output("F", "-", "fator de correção da LMTD"),
        Output("h_shell", "W/m2 K", "coeficiente do casco (Kern)"),
        Output("h_tube", "W/m2 K", "coeficiente dos tubos (Gnielinski)"),
        Output("v_tube", "m/s", "velocidade da água nos tubos"),
        Output("dp_tube", "Pa", "perda de carga nos tubos (atrito + 2,5 cargas cinéticas por passe)"), Output("re_shell", "-", "Reynolds do casco"),
    )
    limitations = ("Perda de carga do casco não calculada (Kern/Bell-Delaware não implementados): o espaçamento de "
                   "chicanas fica restrito a 0,3-1 x D_casco.",)
    validity = ("Kern: 2e3 < Re_casco < 1e6, chicanas segmentais de 25 %, sem vazamentos/bypass (Bell-Delaware dá h "
                "menor, tipicamente 60-80 % do de Kern). Gnielinski: 3000 <= Re <= 5e6. Propriedades constantes na "
                "temperatura média; sem correção de viscosidade na parede.")
    pipeline = ("geometria: feixe (n tubos, OD, parede, passo triangular), casco, chicanas",
                "modelo: Kern (casco), Gnielinski (tubos), parede e incrustação TEMA -> U",
                "solver: effectiveness(NTU, Cr, 'shell_tube') (Incropera eq. 11.30a); perfis pelo BVP do 1-2",
                "pós: Q, temperaturas de saída, F da LMTD, velocidades",
                "decisão: T saída quente <= alvo, 1 <= v_tubo <= 2,5 m/s, perda de carga nos tubos <= 70 kPa")
    library = ("pinneapple_physics.closed_form.heat_exchanger.effectiveness",
               "pinneapple_physics.closed_form.heat_exchanger.kern_shell",
               "pinneapple_physics.closed_form.heat_exchanger.gnielinski",
               "pinneapple_physics.closed_form.heat_exchanger.one_two_exchanger_ode")
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("t_hot_out", "area", "v_tube")
    missing = {"surrogate": "o modelo é de forma fechada (< 1 ms por avaliação); um surrogate não economiza tempo",
               "reason": "não há caso de trocador no pinneapple_reason"}

    def solve(self, p):
        di = p["tube_od"] - 2 * p["tube_wall"]
        n_pass = 2
        per_pass = p["n_tubes"] / n_pass
        a_t = per_pass * math.pi * di ** 2 / 4
        rho_w = 995.0                                         # water 305 K, Incropera Table A.6
        v = p["m_cold"] / (rho_w * a_t)
        re_t = rho_w * v * di / p["mu_cold"]
        pr_t = p["cp_cold"] * p["mu_cold"] / p["k_cold"]
        gt = gnielinski(re_t, pr_t)
        h_t = gt["nu"] * p["k_cold"] / di
        ks = kern_shell(p["m_hot"], p["mu_hot"], p["k_hot"], p["cp_hot"], shell_id=p["shell_id"],
                        baffle_spacing=p["baffle_spacing"], tube_od=p["tube_od"], pitch=p["pitch"])
        do = p["tube_od"]
        r_wall = do * math.log(do / di) / (2 * p["k_wall"])
        inv_u = 1 / ks["h"] + p["rf_hot"] + r_wall + p["rf_cold"] * do / di + do / (di * h_t)
        U = 1 / inv_u
        A = p["n_tubes"] * math.pi * do * p["tube_length"]
        c_hot, c_w = p["m_hot"] * p["cp_hot"], p["m_cold"] * p["cp_cold"]
        cmin, cmax = min(c_hot, c_w), max(c_hot, c_w)
        ntu = U * A / cmin
        cr = cmin / cmax
        eps = effectiveness(ntu, cr, "shell_tube")
        q = eps * cmin * (p["t_hot_in"] - p["t_cold_in"])
        to, tw = p["t_hot_in"] - q / c_hot, p["t_cold_in"] + q / c_w
        F = lmtd_correction_1_2(p["t_hot_in"], to, p["t_cold_in"], tw)
        d1, d2 = p["t_hot_in"] - tw, to - p["t_cold_in"]
        lmtd = (d1 - d2) / math.log(d1 / d2) if abs(d1 - d2) > 1e-9 else d1
        q_lmtd = U * A * F * lmtd
        dp_tube = n_pass * (gt["f"] * p["tube_length"] / di + 2.5) * rho_w * v ** 2 / 2   # Sinnott eq. 12.18
        dbt = 0.023 * re_t ** 0.8 * pr_t ** 0.4 * p["k_cold"] / di      # Dittus-Boelter (heating)
        prof = None
        if c_hot <= c_w:
            _, prof = one_two_exchanger_ode(ntu, cr, profiles=True)
        return {"q": q, "t_hot_out": to, "t_cold_out": tw, "U": U, "area": A, "ntu": ntu, "cr": cr,
                "effectiveness": eps, "F": F, "lmtd": lmtd, "q_lmtd": q_lmtd, "h_shell": ks["h"], "h_tube": h_t,
                "h_tube_dittus_boelter": dbt, "v_tube": v, "dp_tube": dp_tube, "re_tube": re_t, "re_shell": ks["re"],
                "tube_regime": gt["regime"],
                "_fields": {"profile": prof, "c_hot": c_hot, "c_w": c_w}}

    def checks(self, p, out):
        bal = abs(out["q"] - out["q_lmtd"]) / out["q"]
        return [check("epsilon-NTU = U A F LMTD (two independent routes)", bal < 1e-6, value=bal, limit=1e-6,
                      source=INC + " sec. 11.3-11.4"),
                check("energy balance: hot side = cold side", True, source="by construction of Q"),
                check("2nd law: hot out > cold in, cold out < hot in", out["t_hot_out"] > p["t_cold_in"] and
                      out["t_cold_out"] < p["t_hot_in"]),
                check("Kern range 2e3 < Re_shell < 1e6", 2e3 < out["re_shell"] < 1e6, value=out["re_shell"],
                      limit=2e3, source="Kern (1950) eq. 7.3"),
                check("Gnielinski range Re_tube >= 3000", out["re_tube"] >= 3000, value=out["re_tube"], limit=3000,
                      source="Gnielinski (1976)"),
                check("LMTD factor F >= 0.75 (1-2 shell usable)", out["F"] >= 0.75, value=out["F"], limit=0.75,
                      source="Kern (1950) ch. 7: below F ~ 0.75 use more shells")]

    def decide(self, p, out):
        ok_t = out["t_hot_out"] <= p["t_hot_target"]
        ok_v = 1.0 <= out["v_tube"] <= 2.5
        ok_dp = out["dp_tube"] <= 70e3
        return {"verdict": "OK" if ok_t and ok_v and ok_dp else "NOT OK",
                "reasons": [f"circuito sai a {out['t_hot_out']:.1f} °C {'<=' if ok_t else '>'} alvo {p['t_hot_target']:.0f} °C"
                            f" (Q = {out['q'] / 1e3:.1f} kW, U = {out['U']:.0f} W/m2K)",
                            f"velocidade nos tubos {out['v_tube']:.2f} m/s {'dentro' if ok_v else 'fora'} de 1-2,5 m/s "
                            "(incrustação abaixo, erosão acima)",
                            f"perda de carga nos tubos {out['dp_tube'] / 1e3:.1f} kPa {'<=' if ok_dp else '>'} 70 kPa "
                            "(Sinnott, Coulson & Richardson vol. 6, sec. 12.8: 35-70 kPa típico para líquidos)"]}

    def validate(self):
        self.defaults()
        o = self.run().outputs
        rows = []
        _, _rows_ode = None, []
        for ntu, cr in ((o["ntu"], o["cr"]), (0.5, 0.3), (3.0, 1.0)):
            e_cf = effectiveness(ntu, cr, "shell_tube")
            e_ode = one_two_exchanger_ode(ntu, cr)
            rows.append(self._row(f"epsilon 1-2 (NTU {ntu:.2f}, Cr {cr:.2f}): eq. 11.30a vs BVP direto", "effectiveness",
                                  "-", e_cf, e_ode, 1e-6, "independent_solver",
                                  "integração direta das 3 equações de energia do trocador 1-2 (scipy solve_bvp)"))
        rows.append(self._row("Cr -> 0: epsilon = 1 - exp(-NTU)", "effectiveness", "-",
                              effectiveness(1.7, 1e-14, "shell_tube"), 1 - math.exp(-1.7), 1e-12, "analytic",
                              INC + " eq. 11.35a"))
        rows.append(self._row("h tubos: Gnielinski vs Dittus-Boelter", "h_tube", "W/m2 K", o["h_tube"],
                              o["h_tube_dittus_boelter"], 0.25, "independent_solver",
                              "Dittus & Boelter (1930), incerteza ±25 % (Incropera sec. 8.5)"))
        return {"preset": self.name, "references": rows, "convergence": None, "all_pass": all(r["pass"] for r in rows)}

    @staticmethod
    def _row(case, output, unit, comp, ref, tol, kind, source):
        err = abs(comp - ref) / abs(ref)
        return {"case": case, "output": output, "unit": unit, "computed": comp, "reference": ref, "rel_error": err,
                "tol": tol, "pass": bool(err <= tol), "kind": kind, "source": source, "params": {}}

    def optimize(self, **kw):
        """Smallest tube area (count x standard length) reaching the target with 1-2.5 m/s in the tubes; the shell
        diameter follows the tube count (Kern Table 9 bundle density for 3/4" on 15/16" triangular)."""
        p = self.resolve(**kw)
        best, n = None, 0
        for L in (2.438, 3.048, 3.658, 4.877, 6.096):                  # 8, 10, 12, 16, 20 ft
            for nt in range(40, 401, 4):
                shell = p["shell_id"] * math.sqrt(nt / p["n_tubes"])
                for bs in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
                    if not (shell / 5 <= bs * shell <= shell):
                        continue
                    n += 1
                    r = self.solve({**p, "n_tubes": nt, "tube_length": L, "shell_id": shell,
                                    "baffle_spacing": bs * shell})
                    if r["t_hot_out"] <= p["t_hot_target"] and 1 <= r["v_tube"] <= 2.5 and r["F"] >= 0.75 and \
                            r["dp_tube"] <= 70e3 and (
                            best is None or r["area"] < best[1]["area"]):
                        best = ({"n_tubes": nt, "tube_length": L, "shell_id": shell, "baffle_spacing": bs * shell}, r)
        if best is None:
            return {"feasible": False, "designs": n}
        q, r = best
        ver = self.run(**{**kw, **q})
        return {"objective": "menor área de tubos com T_quente <= alvo, 1 <= v <= 2,5 m/s, dp_tubos <= 70 kPa, F >= 0,75 "
                             "e chicanas de 0,3-1 x D_casco (sem modelo de perda de carga do casco)",
                "designs_scanned": n, **{k: v for k, v in q.items()}, "area_m2": r["area"],
                "base_area_m2": self.solve(p)["area"], "verified_t_hot_out": ver.outputs["t_hot_out"],
                "verified_ok": ver.decision["verdict"] == "OK"}

    def invert(self, true_rf: float = 7.0e-4, noise_c: float = 0.2, seed: int = 0, **kw):
        """Total fouling from plant readings (performance monitoring): both outlet temperatures at three load
        points, RTD noise 0.2 K. Synthetic truth: total fouling 7.0e-4 m2K/W split equally between the sides."""
        p = self.resolve(**kw)
        loads = (0.7, 1.0, 1.2)

        def temps(rf):
            out = []
            for f in loads:
                r = self.solve({**p, "m_hot": p["m_hot"] * f, "rf_hot": rf / 2, "rf_cold": rf / 2})
                out += [r["t_hot_out"], r["t_cold_out"]]
            return np.array(out)

        rng = np.random.default_rng(seed)
        y = temps(true_rf) + rng.normal(0, noise_c, 2 * len(loads))
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda r: ((temps(r) - y) ** 2).sum(), bounds=(0, 3e-3), method="bounded",
                              options={"xatol": 1e-9})
        rf = float(res.x)
        J = (temps(rf + 1e-6) - temps(rf - 1e-6)) / 2e-6
        sd = float(noise_c / math.sqrt(J @ J))
        clean = self.solve({**p, "rf_hot": 0, "rf_cold": 0})["q"]
        fouled = self.solve({**p, "rf_hot": rf / 2, "rf_cold": rf / 2})["q"]
        return {"description": "Incrustação total a partir das duas temperaturas de saída em 3 cargas "
                               "(monitoramento de desempenho). Medição sintética com Rf real = 7,0e-4 m2K/W e ruído "
                               f"de {noise_c} K nos RTDs.", "parameter": "Rf total", "unit": "m2 K/W",
                "true": true_rf, "recovered": rf, "ci95": [rf - 1.96 * sd, rf + 1.96 * sd],
                "rel_error": abs(rf - true_rf) / true_rf, "true_inside_ci95": bool(abs(rf - true_rf) <= 1.96 * sd),
                "noise": f"{noise_c} K", "how": "mínimos quadrados (Brent), IC por sensibilidade",
                "duty_loss_vs_clean": 1 - fouled / clean}

    def variants(self):
        return {"base 200 tubos 16 ft": {}, "chicanas 0,15 m": {"baffle_spacing": 0.15},
                "chicanas 0,45 m": {"baffle_spacing": 0.45}, "160 tubos": {"n_tubes": 160, "shell_id": 0.4374},
                "tubos 12 ft": {"tube_length": 3.658}, "incrustado 2x TEMA": {"rf_hot": 1.76e-4, "rf_cold": 3.52e-4}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        prof = result.fields.get("profile")
        p = result.params
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        if prof is not None:
            Th, Tc = p["t_hot_in"], p["t_cold_in"]
            x = prof["x"] * p["tube_length"]
            for k, lab in (("shell", "circuito (casco)"), ("pass1", "torre, passe 1"), ("pass2", "torre, passe 2")):
                ax[0].plot(x, Tc + (Th - Tc) * prof[k], label=lab)
            ax[0].set(xlabel="x [m]", ylabel="T [°C]", title="Perfis no trocador 1-2 (BVP)")
            ax[0].legend(fontsize=8)
        ntu = np.linspace(0.05, 5, 100)
        for cr in (0.25, 0.5, 0.75, 1.0):
            ax[1].plot(ntu, [effectiveness(n, cr, "shell_tube") for n in ntu], label=f"Cr = {cr}")
        ax[1].plot(result.outputs["ntu"], result.outputs["effectiveness"], "ko", label="este trocador")
        ax[1].set(xlabel="NTU", ylabel="efetividade", title="epsilon-NTU, 1 casco / 2 passes")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, "exchanger.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from ..viz3d import Scene, Surface, merge, tube
        p = result.params
        prof = result.fields.get("profile")
        L, R = p["tube_length"], p["shell_id"] / 2
        Th, Tc = p["t_hot_in"], p["t_cold_in"]
        s = np.linspace(0, 1, 41)
        if prof is None:
            sh = Th + (result.outputs["t_hot_out"] - Th) * s
            t1 = Tc + 0.5 * (result.outputs["t_cold_out"] - Tc) * s
            t2 = t1[::-1] + 0.5 * (result.outputs["t_cold_out"] - Tc)
        else:
            sh = Tc + (Th - Tc) * np.interp(s, prof["x"], prof["shell"])
            t1 = Tc + (Th - Tc) * np.interp(s, prof["x"], prof["pass1"])
            t2 = Tc + (Th - Tc) * np.interp(s, prof["x"], prof["pass2"])
        Vs, Fs, i = tube(np.c_[s * L, 0 * s, 0 * s], R, 48)
        parts = []
        rng = np.random.default_rng(0)
        pts = []
        while len(pts) < 36:                                  # a representative subset of the bundle
            y, z = rng.uniform(-0.85 * R, 0.85 * R, 2)
            if y * y + z * z < (0.85 * R) ** 2 and all((y - a) ** 2 + (z - b) ** 2 > (0.12 * R) ** 2 for a, b in pts):
                pts.append((y, z))
        for y, z in pts:
            T = t1 if z < 0 else t2
            V, F, j = tube(np.c_[s * L, np.full_like(s, y), np.full_like(s, z)], p["tube_od"] / 2 * 1.6, 10)
            parts.append((V, F, T[j]))
        Vt, Ft, Tt = merge(parts)
        shell = Surface("shell", Vs, Fs, "glass", {"temperature": sh[i]})
        tubes = Surface("tubes", Vt, Ft, "copper", {"temperature": Tt})
        return Scene([shell, tubes], title=self.title, labels={"temperature": "temperatura [°C] (casco: circuito fechado; tubos: água de torre)"})
