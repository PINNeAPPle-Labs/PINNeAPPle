"""Centrifugal pump lifting from a sump: duty point, NPSH available vs required and the cavitation margin."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_physics.closed_form.pump_npsh import (
    G,
    atmosphere,
    duty_point,
    fit_quadratic,
    iapws97_psat,
    npsh_available,
    pipe_loss,
    water_density,
)

from ..core import Output, Param, Preset, check

HI = "ANSI/HI 9.6.1-2017 (margem NPSHa/NPSH3 recomendada 1,1-2,5 conforme a aplicação)"
IF97 = "IAPWS R7-97(2012), IF97 eq. 30 e Tabela 35 (valores de verificação)"
CURVE_Q = (0.0, 50.0, 100.0, 120.0)            # m3/h
CURVE_H = (54.0, 52.0, 44.0, 39.0)             # m
CURVE_N = (50.0, 100.0, 120.0)
CURVE_N3 = (2.0, 3.5, 4.6)                     # NPSH3, m


class PumpNPSH(Preset):
    name = "pump_npsh_cavitation"
    title = "Bomba centrífuga em poço: ponto de operação, NPSH disponível e margem contra cavitação"
    sector = "Fluidos / bombeamento (saneamento, processo, utilidades)"
    problem = ("Bomba centrífuga de sucção axial acima de um poço aberto (altura de sucção), recalcando para um "
               "reservatório. Ponto de operação pela interseção das curvas da bomba e do sistema, NPSH disponível com "
               "pressão de vapor IAPWS-IF97 e atmosfera ISO 2533, contra o NPSH3 da bomba com a margem da ANSI/HI.")
    question = ("A bomba opera sem cavitar nesta instalação (altitude, temperatura, altura de sucção, tubulação)? "
                "Qual a maior altura de sucção ou temperatura admissível, e que diâmetro de sucção resolve?")
    params = (
        Param("altitude", "m", 800.0, "premissa: estação de bombeamento a 800 m de altitude", 0, 4000),
        Param("t_water", "degC", 40.0, "premissa: água de processo / retorno morno a 40 °C", 1, 120),
        Param("z_suction", "m", -4.0, "nível do poço 4 m abaixo do eixo da bomba (altura de sucção; + = afogada)",
              -9, 30),
        Param("h_static", "m", 30.0, "premissa: desnível geométrico até o reservatório de recalque", 0, 50),
        Param("d_suction", "m", 0.1541, "DN150 Sch 40, diâmetro interno 154,1 mm (ASME B36.10M)", 0.04, 0.6),
        Param("l_suction", "m", 12.0, "premissa: 12 m de tubo de sucção", 1, 200),
        Param("k_suction", "-", 3.0, "válvula de pé com crivo ~2,0-2,5 + curva 90° 0,3 + entrada 0,5 (Crane TP-410 "
              "ordem de grandeza)", 0, 20),
        Param("d_discharge", "m", 0.1281, "DN125 Sch 40, diâmetro interno 128,2 mm (ASME B36.10M)", 0.04, 0.6),
        Param("l_discharge", "m", 300.0, "premissa: 300 m de linha de recalque", 1, 5000),
        Param("k_discharge", "-", 10.0, "premissa: retenção + gaveta + curvas + saída", 0, 60),
        Param("roughness", "m", 4.5e-5, "aço comercial 0,045 mm (White, Fluid Mechanics, Tab. 6.1)", 0, 2e-3),
        Param("npsh_ratio_min", "-", 1.3, HI + "; 1,3 premissa para água limpa em serviço contínuo", 1.0, 2.5),
        Param("curve_scale", "-", 1.0, "1 = curva genérica de bomba 80-200 a 2900 rpm (pontos de catálogo da ficha); "
              "afinidade: Q ~ s, H ~ s^2, NPSH ~ s^2", 0.6, 1.2),
    )
    outputs = (
        Output("q", "m3/h", "vazão no ponto de operação"), Output("head", "m", "altura manométrica no ponto"),
        Output("npsha", "m", "NPSH disponível"), Output("npsh3", "m", "NPSH3 da bomba no ponto"),
        Output("ratio", "-", "NPSHa / NPSH3"), Output("margin", "m", "NPSHa - NPSH3"),
        Output("v_suction", "m/s", "velocidade na sucção"), Output("max_lift", "m", "maior altura de sucção com a "
                                                                                       "margem exigida"),
        Output("max_temp", "degC", "maior temperatura da água com a margem exigida"),
    )
    validity = ("Água limpa, regime permanente, curvas da bomba interpoladas entre 0 e 120 m3/h (extrapolação "
                "acusada), sem vórtice na sucção (submergência suficiente, ANSI/HI 9.8), pressão atmosférica padrão.")
    pipeline = ("geometria: poço, tubo de sucção, bomba, linha de recalque",
                "modelo: curvas H(Q) e NPSH3(Q) por mínimos quadrados dos pontos de catálogo; perdas Darcy-Colebrook",
                "solver: duty_point (bisseção) e npsh_available (IAPWS-IF97, ISO 2533)",
                "pós: NPSHa, NPSH3, razão e margem; maior altura de sucção e temperatura admissíveis",
                "decisão: NPSHa / NPSH3 >= razão mínima da ANSI/HI 9.6.1")
    library = ("pinneapple_physics.closed_form.pump_npsh", "pinneapple_physics.closed_form.pipe_flow.friction_factor")
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("npsha", "ratio", "q")
    reason_case = "pumped_main"
    missing = {"surrogate": "forma fechada (< 1 ms por avaliação): um surrogate não economiza tempo"}

    def _curves(self, s):
        q = np.array(CURVE_Q) * s / 3600
        h = np.array(CURVE_H) * s * s
        hq = fit_quadratic(q, h)
        nq = fit_quadratic(np.array(CURVE_N) * s / 3600, np.array(CURVE_N3) * s * s)
        return hq, nq, q.max()

    def _state(self, p, q):
        w = water_density(p["t_water"])
        ls = pipe_loss(q, p["d_suction"], p["l_suction"], p["k_suction"], p["roughness"], w["rho"], w["mu"])
        ld = pipe_loss(q, p["d_discharge"], p["l_discharge"], p["k_discharge"], p["roughness"], w["rho"], w["mu"])
        return ls, ld

    def solve(self, p):
        hq, nq, qmax = self._curves(p["curve_scale"])
        sysloss = lambda q: sum(x["h"] for x in self._state(p, q))  # noqa: E731
        q = duty_point(hq, p["h_static"], sysloss, qmax * 1.0)
        ls, ld = self._state(p, q)
        patm = atmosphere(p["altitude"])
        na = npsh_available(patm, p["t_water"], p["z_suction"], ls["h"])
        n3 = float(nq[0] + nq[1] * q + nq[2] * q * q)
        head = float(hq[0] + hq[1] * q + hq[2] * q * q)
        need = p["npsh_ratio_min"] * n3
        max_lift = -(p["z_suction"] - (na["npsha"] - need))          # the lift that leaves exactly the margin
        lo, hi = 1.0, 120.0
        f = lambda t: npsh_available(patm, t, p["z_suction"], self._state({**p, "t_water": t}, q)[0]["h"])["npsha"] \
            - need  # noqa: E731
        if f(lo) < 0:
            tmax = float("nan")
        elif f(hi) > 0:
            tmax = hi
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
            tmax = 0.5 * (lo + hi)
        return {"q": q * 3600, "head": head, "npsha": na["npsha"], "npsh3": n3, "ratio": na["npsha"] / n3,
                "margin": na["npsha"] - n3, "v_suction": ls["v"], "max_lift": max_lift, "max_temp": tmax,
                "h_suction_loss": ls["h"], "h_discharge_loss": ld["h"], "p_atm": patm, "p_vapour": na["pv"],
                "rho": na["rho"], "re_suction": ls["re"], "system_balance": head - p["h_static"] - ls["h"] - ld["h"],
                "_fields": {"hq": hq, "nq": nq, "qmax": qmax}}

    def checks(self, p, out):
        return [check("duty point: pump head = static + losses", abs(out["system_balance"]) < 1e-6,
                      value=out["system_balance"], limit=1e-6),
                check("duty point inside the catalogue curve (no extrapolation)", out["q"] <= 120 * p["curve_scale"],
                      value=out["q"], limit=120 * p["curve_scale"], source="catalogue points of the data sheet"),
                check("suction velocity <= 2.4 m/s (HI 9.6.6 guidance range)", out["v_suction"] <= 2.4,
                      value=out["v_suction"], limit=2.4, source="ANSI/HI 9.6.6"),
                check("liquid below boiling at the sump (p_v < p_atm)", out["p_vapour"] < out["p_atm"],
                      value=out["p_vapour"], limit=out["p_atm"], critical=True)]

    def decide(self, p, out):
        ok = out["ratio"] >= p["npsh_ratio_min"]
        return {"verdict": "OK" if ok else "NOT OK",
                "reasons": [f"Q = {out['q']:.1f} m3/h, H = {out['head']:.1f} m",
                            f"NPSHa {out['npsha']:.2f} m / NPSH3 {out['npsh3']:.2f} m = {out['ratio']:.2f} "
                            f"{'>=' if ok else '<'} {p['npsh_ratio_min']}",
                            f"altura de sucção máxima {out['max_lift']:.2f} m; temperatura máxima "
                            + (f"{out['max_temp']:.1f} °C" if out["max_temp"] == out["max_temp"] else
                               "nenhuma (a margem falta mesmo com água a 1 °C)")]}

    def validate(self):
        rows = []
        for T, ref in ((300.0, 0.353658941e-2), (500.0, 0.263889776e1), (600.0, 0.123443146e2)):
            rows.append(self._row(f"pressão de saturação a {T:.0f} K", "p_sat", "MPa", iapws97_psat(T - 273.15) / 1e6,
                                  ref, 1e-8, "standard", IF97))
        for z, ref in ((1000.0, 89874.6), (2000.0, 79495.2)):
            rows.append(self._row(f"pressão atmosférica a {z:.0f} m", "p_atm", "Pa", atmosphere(z), ref, 1e-5,
                                  "standard", "ISO 2533:1975 / ICAO Doc 7488 (atmosfera padrão), tabela"))
        p = self.defaults()
        q = self.run().outputs["q"] / 3600
        w = water_density(p["t_water"])
        ls = pipe_loss(q, p["d_suction"], p["l_suction"], p["k_suction"], p["roughness"], w["rho"], w["mu"])
        re, rr = ls["re"], p["roughness"] / p["d_suction"]
        f_h = (-1.8 * math.log10((rr / 3.7) ** 1.11 + 6.9 / re)) ** -2
        h_h = (f_h * p["l_suction"] / p["d_suction"] + p["k_suction"]) * ls["v"] ** 2 / (2 * G)
        rows.append(self._row("perda na sucção: Colebrook vs Haaland", "h_suction_loss", "m", ls["h"], h_h, 0.02,
                              "independent_solver", "Haaland (1983), explícita, ±2 % de Colebrook"))
        return {"preset": self.name, "references": rows, "convergence": None, "all_pass": all(r["pass"] for r in rows)}

    @staticmethod
    def _row(case, output, unit, comp, ref, tol, kind, source):
        err = abs(comp - ref) / abs(ref)
        return {"case": case, "output": output, "unit": unit, "computed": comp, "reference": ref, "rel_error": err,
                "tol": tol, "pass": bool(err <= tol), "kind": kind, "source": source, "params": {}}

    def optimize(self, **kw):
        """Cheapest fix of the suction: the smallest standard suction size (Sch 40) that gives the ANSI/HI margin
        at the given lift, else the highest pump setting (largest lift) with the current pipe."""
        sizes = {"DN100": 0.1023, "DN125": 0.1282, "DN150": 0.1541, "DN200": 0.2027, "DN250": 0.2545}
        rows = {}
        best = None
        for k, d in sizes.items():
            r = self.run(**{**kw, "d_suction": d})
            rows[k] = {"ratio": r.outputs["ratio"], "npsha": r.outputs["npsha"], "q": r.outputs["q"]}
            if best is None and r.decision["verdict"] == "OK":
                best = k
        base = self.run(**kw).outputs
        out = {"objective": "menor diâmetro de sucção (Sch 40) com NPSHa/NPSH3 >= razão mínima; se nenhum basta, "
                            "baixar a bomba até a altura de sucção que deixa a margem",
               "smallest_ok_suction": best, "table": rows}
        if best is None:
            z = -base["max_lift"] + 0.05                          # 5 cm of construction margin
            ver = self.run(**{**kw, "z_suction": z})
            out.update({"pump_lowered_to_lift_m": -z, "verified_ratio": ver.outputs["ratio"],
                        "verified_ok": ver.decision["verdict"] == "OK"})
        return out

    def invert(self, true_k: float = 9.0, noise_kpa: float = 1.0, seed: int = 0, **kw):
        """Suction loss coefficient (a clogging strainer) from the suction-flange gauge read at four valve settings
        (flows from a magnetic flowmeter), gauge noise 1 kPa. Synthetic truth K = 9 instead of the clean 3."""
        p = self.resolve(**kw)
        qs = np.array([40.0, 60.0, 80.0, 100.0]) / 3600
        w = water_density(p["t_water"])
        patm = atmosphere(p["altitude"])

        def gauge(k):
            out = []
            for q in qs:
                ls = pipe_loss(q, p["d_suction"], p["l_suction"], k, p["roughness"], w["rho"], w["mu"])
                out.append(patm + w["rho"] * G * (p["z_suction"] - ls["h"]) - 0.5 * w["rho"] * ls["v"] ** 2)
            return np.array(out)

        rng = np.random.default_rng(seed)
        y = gauge(true_k) + rng.normal(0, noise_kpa * 1e3, len(qs))
        J = (gauge(1.0) - gauge(0.0))                            # p is linear in K
        k = float(J @ (y - gauge(0.0)) / (J @ J))
        sd = float(noise_kpa * 1e3 / math.sqrt(J @ J))
        return {"description": "Coeficiente de perda da sucção (crivo entupindo) a partir do manômetro no flange de "
                               "sucção em 4 aberturas da válvula de recalque (vazão por medidor magnético). Verdade "
                               f"sintética K = {true_k} (limpo: 3); ruído de {noise_kpa} kPa.",
                "parameter": "K_suction", "true": true_k, "recovered": k, "ci95": [k - 1.96 * sd, k + 1.96 * sd],
                "rel_error": abs(k - true_k) / true_k, "true_inside_ci95": bool(abs(k - true_k) <= 1.96 * sd),
                "noise": f"{noise_kpa} kPa", "how": "mínimos quadrados lineares (p é afim em K), IC exato",
                "npsh_ratio_with_recovered_k": self.run(**{**kw, "k_suction": k}).outputs["ratio"]}

    def variants(self):
        return {"base 800 m, 40 °C, sucção 4 m": {}, "nível do mar": {"altitude": 0.0},
                "2500 m de altitude": {"altitude": 2500.0}, "água a 70 °C": {"t_water": 70.0},
                "bomba afogada (+2 m)": {"z_suction": 2.0}, "sucção DN100": {"d_suction": 0.1023}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        F, p = result.fields, result.params
        q = np.linspace(1e-4, F["qmax"], 100)
        hq, nq = F["hq"], F["nq"]
        sysh = [p["h_static"] + sum(x["h"] for x in self._state(p, qq)) for qq in q]
        patm = atmosphere(p["altitude"])
        na = [npsh_available(patm, p["t_water"], p["z_suction"], self._state(p, qq)[0]["h"])["npsha"] for qq in q]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(q * 3600, hq[0] + hq[1] * q + hq[2] * q * q, label="bomba H(Q)")
        ax[0].plot(q * 3600, sysh, label="sistema")
        ax[0].plot(result.outputs["q"], result.outputs["head"], "ko")
        ax[0].set(xlabel="Q [m3/h]", ylabel="H [m]", title="Ponto de operação")
        ax[0].legend(fontsize=8)
        ax[1].plot(q * 3600, na, label="NPSH disponível")
        n3 = nq[0] + nq[1] * q + nq[2] * q * q
        ax[1].plot(q * 3600, n3, label="NPSH3")
        ax[1].plot(q * 3600, p["npsh_ratio_min"] * n3, "--", label=f"{p['npsh_ratio_min']} x NPSH3")
        ax[1].axvline(result.outputs["q"], color="k", lw=0.8)
        ax[1].set(xlabel="Q [m3/h]", ylabel="NPSH [m]", title="Margem contra cavitação")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, "pump.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from ..viz3d import Scene, Surface, box, tube
        p, o = result.params, result.outputs
        rho, patm = o["rho"], o["p_atm"]
        zs = p["z_suction"]                                   # liquid level relative to the pump axis
        n = 40
        s = np.linspace(0, 1, n)
        # suction pipe: vertical from 1 m below the level up to the pump axis, then horizontal 1.5 m
        z0 = zs - 1.0
        P = np.r_[np.c_[np.zeros(n), np.zeros(n), z0 + (0 - z0) * s], np.c_[1.5 * s[1:], np.zeros(n - 1),
                                                                            np.zeros(n - 1)]]
        L = np.r_[(0 - z0) * s, (0 - z0) + 1.5 * s[1:]]
        h_loss = o["h_suction_loss"] * L / L[-1]
        v2 = (o["v_suction"]) ** 2 / (2 * G)
        p_abs = (patm + rho * G * (zs - P[:, 2] - h_loss - v2)) / 1e3
        V, F, i = tube(P, 0.25, 20)
        pipe = Surface("suction", V, F, "steel", {"pressure": p_abs[i]})
        sv, sf = box([-2, -2, zs - 3], [2, 2, zs])
        sump = Surface("sump", sv, sf, "water", {"pressure": np.full(len(sv), patm / 1e3)})
        pv, pf = box([1.5, -0.6, -0.6], [2.7, 0.6, 0.6])
        pump = Surface("pump", pv, pf, "dark", {"pressure": np.full(len(pv), p_abs[-1])})
        return Scene([sump, pipe, pump], title=self.title,
                     labels={"pressure": f"pressão absoluta [kPa] (vapor: {o['p_vapour'] / 1e3:.1f} kPa)"})
