"""Plate-fin heat sink in ducted forced air for a power module (resistance network + 3-D finite-volume base)."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_design.thermal.heatsink.engine import DesignInput, OperatingInput, evaluate_design
from pinneapple_physics.closed_form import plate_fin_heatsink as pfh

from ..core import Output, Param, Preset, SurrogateSpec, check

TEERTSTRA = ("Teertstra, Yovanovich & Culham, Analytical forced convection modeling of plate fin heat sinks, "
             "IEEE SEMI-THERM XV (1999) / J. Electronics Manufacturing 10(4) 2000")
STEPHAN = "Stephan (1959) Nusselt de entrada para placas paralelas + epsilon-NTU do aquecimento do ar (Shah & London)"
LEE = "Lee, Song, Au & Moran (1995), resistência de espalhamento"


class PlateFinHeatsink(Preset):
    name = "plate_fin_heatsink_forced"
    title = "Dissipador de aletas planas com ventilação forçada (módulo de potência)"
    sector = "Térmica / eletrônica de potência"
    problem = ("Dissipador de alumínio extrudado com aletas retas, ar forçado canalizado (shroud), fonte de calor "
               "concentrada (módulo/transistor 30 x 30 mm) no centro da base. Rede de resistências com correlações "
               "publicadas e base resolvida por volumes finitos 3-D.")
    question = ("A temperatura da fonte fica abaixo do limite (85 °C) com a vazão de ar disponível? Qual o "
                "dissipador mais leve que atende, e quanto pesa a incerteza da correlação de convecção?")
    params = (
        Param("power", "W", 60.0, "premissa: perdas de um módulo MOSFET/IGBT de ~1 kW (eficiência ~94 %)", 1, 300),
        Param("t_ambient", "degC", 35.0, "premissa: ar interno de gabinete eletrônico (40 °C máx. típico de projeto)",
              -20, 70),
        Param("velocity", "m/s", 3.0, "premissa: ventilador axial de 80 mm canalizado (1-5 m/s típico)", 0.5, 8.0),
        Param("base_width", "m", 0.10, "perfil extrudado comercial 100 x 100 mm (premissa de catálogo)", 0.03, 0.3),
        Param("base_depth", "m", 0.10, "comprimento na direção do ar, 100 mm", 0.03, 0.3),
        Param("base_thickness", "m", 0.005, "base de 5 mm (perfis extrudados típicos 3-10 mm)", 0.002, 0.02),
        Param("n_fins", "-", 15, "15 aletas (passo ~7 mm, típico de extrudado para ar forçado)", 4, 40),
        Param("fin_thickness", "m", 0.0015, "1,5 mm (limite prático de extrusão de Al ~1 mm)", 0.0008, 0.005),
        Param("fin_height", "m", 0.035, "35 mm", 0.01, 0.08),
        Param("material", "-", "al6063", "Al 6063-T5, k = 209 W/m K (Incropera Tabela A.1 / ASM)",
              choices=tuple(pfh.MATERIALS)),
        Param("source_size", "m", 0.030, "encapsulamento de 30 x 30 mm (módulo de potência pequeno)", 0.005, 0.2),
        Param("tim", "K m2/W", 1.0e-5, "pasta térmica fina, R'' = 10 K mm2/W (5-20 típico)", 1e-6, 1e-4),
        Param("t_limit", "degC", 85.0, "premissa: limite de temperatura de case/base do componente", 40, 175),
        Param("h_scale", "-", 1.0, "multiplicador do h (1 = correlação; < 1 = desvio de ar pelo lado, sem shroud)",
              0.3, 1.5),
    )
    outputs = (
        Output("t_source_max_c", "degC", "temperatura máxima na fonte (FVM 3-D da base + TIM)"),
        Output("t_source_pessimistic_c", "degC", "a mesma com h 15 % menor (incerteza típica da correlação)"),
        Output("r_total_k_w", "K/W", "resistência fonte-ar"),
        Output("pressure_drop_pa", "Pa", "perda de carga do ar nas aletas"),
        Output("h_w_m2k", "W/m2 K", "coeficiente de convecção médio"),
        Output("fin_efficiency", "-", "eficiência da aleta"),
        Output("mass_g", "g", "massa do dissipador"),
        Output("air_rise_c", "K", "aquecimento do ar (balanço de energia)"),
    )
    validity = ("Escoamento laminar canalizado entre aletas (Re_Dh < 2300), aletas finas (t << b), base isotrópica, "
                "sem radiação (conservador), fonte retangular única. Correlações de Teertstra: 0,1 < Re* < 100.")
    pipeline = ("geometria: HeatSinkGeometry / DesignInput (perfil extrudado parametrizado)",
                "modelo: correlação de Teertstra (h), eficiência de aleta (Incropera), espalhamento (Lee 1995)",
                "solver: evaluate_design: rede de resistências + base por volumes finitos 3-D (malha 48 e 28)",
                "pós: T máx da fonte com banda de ±15 % no h, perda de carga (Muzychka-Yovanovich), massa",
                "decisão: T pessimista <= limite; otimização de massa com o solver")
    library = ("pinneapple_design.thermal.heatsink.engine.evaluate_design",
               "pinneapple_physics.closed_form.plate_fin_heatsink.evaluate")
    surrogate_spec = SurrogateSpec(inputs=("velocity", "n_fins", "fin_height", "power"),
                                   outputs=("t_source_max_c", "pressure_drop_pa"), n_train=80, n_test=25,
                                   bounds={"velocity": (1.0, 6.0), "n_fins": (8, 26), "fin_height": (0.02, 0.06),
                                           "power": (20.0, 120.0)}, log=("velocity",),
                                   log_outputs=("pressure_drop_pa",))
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("t_source_max_c", "pressure_drop_pa", "mass_g")
    reason_case = "power_board"
    missing = {}
    limitations = ("Sem dado experimental público acessível desta geometria: a validação usa os limites analíticos da "
                   "correlação, uma correlação independente (Stephan + epsilon-NTU) e o FVM contra a teoria de "
                   "espalhamento.", "Escoamento canalizado: sem shroud, o desvio lateral do ar reduz o desempenho "
                   "em 20-50 % (use h_scale ou a inversão abaixo).")

    def _di(self, p):
        return (DesignInput(p["base_width"] * 1e3, p["base_depth"] * 1e3, p["base_thickness"] * 1e3,
                            int(round(p["n_fins"])), p["fin_thickness"] * 1e3, p["fin_height"] * 1e3, p["material"]),
                OperatingInput(p["power"], p["t_ambient"], p["velocity"], p["source_size"] * 1e3,
                               p["source_size"] * 1e3, tim_k_mm2_w=p["tim"] * 1e6, t_limit_c=p["t_limit"]))

    def solve(self, p):
        if p["h_scale"] != 1.0:                      # closed form only (evaluate_design has no h multiplier)
            d, o = self._di(p)
            r = pfh.evaluate(d.geometry(), o.operating(), h_scale=p["h_scale"])
            return {"t_source_max_c": r["t_source_max_c"], "t_source_pessimistic_c": float("nan"),
                    "r_total_k_w": r["r_total_k_w"], "pressure_drop_pa": r["pressure_drop_pa"],
                    "h_w_m2k": r["h_w_m2k"], "fin_efficiency": r["fin_efficiency"], "mass_g": 1e3 * r["mass_kg"],
                    "air_rise_c": p["power"] / (r["airflow_m3_h"] / 3600 * pfh.air_properties(p["t_ambient"])["rho"]
                                                * pfh.AIR_CP), "re_dh": r["dimensionless"]["re_dh"],
                    "re_star": r["dimensionless"]["re_star"], "energy_balance": 0.0, "checks_engine": [],
                    "_fields": {}}
        d, o = self._di(p)
        r = evaluate_design(d, o)
        k = r["kpis"]
        g = d.geometry()
        air = pfh.air_properties(p["t_ambient"])
        mdot = k["airflow_m3_h"] / 3600 * air["rho"]
        dim = r["details"]["dimensionless"]
        return {"t_source_max_c": k["t_source_max_c"], "t_source_pessimistic_c": k["t_source_band_c"][1],
                "r_total_k_w": k["r_total_k_w"], "pressure_drop_pa": k["pressure_drop_pa"], "h_w_m2k": k["h_w_m2k"],
                "fin_efficiency": k["fin_efficiency"], "mass_g": k["mass_g"],
                "air_rise_c": p["power"] / (mdot * pfh.AIR_CP), "re_dh": dim["re_dh"], "re_star": dim["re_star"],
                "fin_gap_mm": k["fin_gap_mm"], "airflow_m3_h": k["airflow_m3_h"],
                "checks_engine": r["checks"], "_fields": {"field3d": r["field3d"], "geometry": g}}

    def checks(self, p, out):
        c = [check("laminar channel flow (Re_Dh < 2300)", out["re_dh"] < 2300, value=out["re_dh"], limit=2300,
                   source=TEERTSTRA),
             check("Teertstra range 0.1 < Re* < 100", 0.1 < out["re_star"] < 100, value=out["re_star"], limit=100,
                   source=TEERTSTRA),
             check("air outlet colder than the source (2nd law)", p["t_ambient"] + out["air_rise_c"] <
                   out["t_source_max_c"], value=p["t_ambient"] + out["air_rise_c"], limit=out["t_source_max_c"])]
        for ch in out.get("checks_engine", []):
            c.append(check(f"engine: {ch['name']}", ch["status"] == "pass" if ch["status"] in ("pass", "fail")
                           else None, value=ch["value"] if isinstance(ch["value"], (int, float)) else None,
                           source=ch.get("law", ""), reason=ch.get("detail", "")))
        return c

    def complete(self, p, out):
        return {**out, "t_source_pessimistic_c": float("nan")}

    def decide(self, p, out):
        t = out["t_source_pessimistic_c"]
        t = out["t_source_max_c"] if math.isnan(t) else t
        ok = t <= p["t_limit"]
        return {"verdict": "OK" if ok else "NOT OK",
                "reasons": [f"T fonte {out['t_source_max_c']:.1f} °C (h -15 %: {t:.1f} °C) "
                            f"{'<=' if ok else '>'} limite {p['t_limit']:.0f} °C",
                            f"perda de carga {out['pressure_drop_pa']:.1f} Pa, massa {out['mass_g']:.0f} g"]}

    # ------------------------------------------------------------------------------------------- validation
    def validate(self):
        """No public measurement of this geometry: independent correlation, analytic limits, FVM vs theory."""
        p = self.defaults()
        d, o = self._di(p)
        g, op = d.geometry(), o.operating()
        rows = []
        for v in (1.0, 3.0, 6.0):
            r = pfh.evaluate(g, pfh.OperatingPoint(op.power_w, op.t_ambient_c, v, op.source_width, op.source_depth,
                                                   op.tim_resistance_k_m2_w))
            air = pfh.air_properties(op.t_ambient_c + 0.5 * (r["t_fin_root_c"] - op.t_ambient_c))
            b, H, L = g.fin_gap, g.fin_height, g.base_depth
            dh = 2 * b
            re = v * g.base_width / ((g.n_fins - 1) * b) * dh / air["nu"]
            xs = L / (dh * re * air["pr"])
            h = (7.55 + 0.024 * xs ** -1.14 / (1 + 0.0358 * air["pr"] ** 0.17 * xs ** -0.64)) * air["k"] / dh
            k = pfh.MATERIALS[g.material]["k"]
            lc = H + g.fin_thickness / 2
            ml = math.sqrt(2 * h / (k * g.fin_thickness)) * lc
            ha = h * (math.tanh(ml) / ml * 2 * lc * L * g.n_fins + (g.base_width - g.n_fins * g.fin_thickness) * L)
            mcp = air["rho"] * v * g.base_width * H * air["cp"]
            r_ind = 1 / (mcp * (1 - math.exp(-ha / mcp)))
            comp = r["resistances_k_w"]["fins_convection"]
            rows.append(self._row(f"R convecção das aletas a {v:g} m/s vs correlação independente", "r_fins", "K/W",
                                  comp, r_ind, 0.08, "independent_solver", STEPHAN))
        # fully developed limit of the Teertstra correlation: Nu -> Re* Pr / 2
        gs = pfh.HeatSinkGeometry(g.base_width, 0.30, g.base_thickness, 30, g.fin_thickness, g.fin_height)
        air = pfh.air_properties(op.t_ambient_c)
        fc = pfh.forced_convection(gs, 0.3, air)
        rows.append(self._row("limite plenamente desenvolvido Nu = Re* Pr / 2 (Re* pequeno)", "nu_b", "-",
                              fc["nu_b"], fc["re_star"] * air["pr"] / 2, 0.02, "analytic", TEERTSTRA))
        # base FVM against spreading theory
        r = evaluate_design(d, o)
        rfvm = r["resistances_k_w"]["spreading_and_base"]
        rlee = pfh.evaluate(g, op)["resistances_k_w"]["spreading_and_base"]
        rows.append(self._row("espalhamento + base: FVM 3-D vs teoria", "r_spreading", "K/W", rfvm, rlee, 0.05,
                              "analytic", LEE))
        # grid convergence of the FVM hotspot
        conv = None
        ts = []
        for n in (24, 36, 48, 64):
            from pinneapple_design.thermal.heatsink.engine import base_theta
            f = base_theta(g, o, r["resistances_k_w"]["fins_convection"], n=n)
            ts.append(float(f["theta_max"]))
        rel = abs(ts[-1] - ts[-2]) / ts[-1]
        conv = {"parameter": "FVM grid n x n", "values": [24, 36, 48, 64], "theta_max": ts, "rel_change_last": rel,
                "pass": rel < 0.01}
        return {"preset": self.name, "references": rows, "convergence": conv,
                "all_pass": all(x["pass"] for x in rows) and conv["pass"]}

    @staticmethod
    def _row(case, output, unit, comp, ref, tol, kind, source):
        err = abs(comp - ref) / abs(ref)
        return {"case": case, "output": output, "unit": unit, "computed": comp, "reference": ref, "rel_error": err,
                "tol": tol, "pass": bool(err <= tol), "kind": kind, "source": source, "params": {}}

    def references(self):
        return []

    # ------------------------------------------------------------------------------------------- extras
    def optimize(self, **kw):
        """Lightest heat sink (fin count, height, thickness on a grid) meeting the pessimistic limit and
        dp <= 25 Pa (a small axial fan's useful pressure); the surrogate's choice is re-run on the solver."""
        p = self.resolve(**kw)
        grid = [(n, hgt, t) for n in range(8, 27, 2) for hgt in (0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06)
                for t in (0.001, 0.0015, 0.002)]
        cands = []
        for n, hgt, t in grid:
            q = {**p, "n_fins": n, "fin_height": hgt, "fin_thickness": t}
            if (q["base_width"] - n * t) / (n - 1) < 0.002:
                continue
            d, o = self._di(q)
            r = pfh.evaluate(d.geometry(), o.operating(), h_scale=0.85)       # pessimistic band, closed form
            if r["t_source_max_c"] <= p["t_limit"] and r["pressure_drop_pa"] <= 25:
                cands.append((r["mass_kg"], (n, hgt, t)))
        out = {"objective": "massa mínima com T(h -15 %) <= limite e perda de carga <= 25 Pa",
               "search": f"grade de {len(grid)} geometrias (n aletas x altura x espessura) na forma fechada (espalhamento "
                         "de Lee); as mais leves são conferidas no solver (FVM 3-D) até uma passar"}
        ver, rejected = None, 0
        for _mass, (n, hgt, t) in sorted(cands):
            r = self.run(**{**kw, "n_fins": n, "fin_height": hgt, "fin_thickness": t})
            if r.decision["verdict"] == "OK":
                ver = (n, hgt, t, r)
                break
            rejected += 1
        if ver is None:
            return {**out, "feasible": False}
        n, hgt, t, r = ver
        out.update({"n_fins": n, "fin_height_m": hgt, "fin_thickness_m": t, "mass_g": r.outputs["mass_g"],
                    "verified_t_pessimistic_c": r.outputs["t_source_pessimistic_c"],
                    "verified_dp_pa": r.outputs["pressure_drop_pa"], "verified_ok": True,
                    "closed_form_picks_rejected_by_solver": rejected, "base_mass_g": self.run(**kw).outputs["mass_g"]})
        from ..surrogate import load_surrogate
        sur = load_surrogate(self.name)
        if sur is not None:
            best_s = None
            for n, hgt, t in grid:
                if t != p["fin_thickness"] or not (sur.bounds["n_fins"][0] <= n <= sur.bounds["n_fins"][1]) or not (
                        sur.bounds["fin_height"][0] <= hgt <= sur.bounds["fin_height"][1]):
                    continue
                pr, _ = sur.predict_one({**p, "n_fins": n, "fin_height": hgt})
                d, o = self._di({**p, "n_fins": n, "fin_height": hgt})
                m = 1e3 * pfh.evaluate(d.geometry(), o.operating())["mass_kg"]
                if pr["t_source_max_c"] + 0.15 * (pr["t_source_max_c"] - p["t_ambient"]) <= p["t_limit"] and \
                        pr["pressure_drop_pa"] <= 25 and (best_s is None or m < best_s[1]):
                    best_s = ((n, hgt), m, pr["t_source_max_c"])
            if best_s:
                (n, hgt), m, ts = best_s
                vs = self.run(**{**kw, "n_fins": n, "fin_height": hgt})
                out.update({"surrogate_choice": {"n_fins": n, "fin_height_m": hgt, "fin_thickness_m": p["fin_thickness"],
                                                 "mass_g": m},
                            "surrogate_t_pred_c": ts, "solver_t_at_surrogate_choice_c": vs.outputs["t_source_max_c"],
                            "surrogate_vs_solver_diff_c": ts - vs.outputs["t_source_max_c"],
                            "surrogate_choice_ok_on_solver": vs.decision["verdict"] == "OK"})
        return out

    def invert(self, true_h_scale: float = 0.72, noise_c: float = 0.3, seed: int = 0, **kw):
        """Effective convection multiplier (air bypass of an unshrouded sink) from thermocouple readings of the
        source at four power levels; synthetic truth h_scale = 0.72 plus 0.3 K noise (type-T thermocouple class 1
        is +-0.5 K)."""
        p = self.resolve(**kw)
        d, o = self._di(p)
        g = d.geometry()
        powers = np.array([20.0, 40.0, 60.0, 80.0])

        def temps(hs):
            return np.array([pfh.evaluate(g, pfh.OperatingPoint(q, p["t_ambient"], p["velocity"], p["source_size"],
                                                                p["source_size"], p["tim"]), h_scale=hs)
                             ["t_source_max_c"] for q in powers])

        rng = np.random.default_rng(seed)
        y = temps(true_h_scale) + rng.normal(0, noise_c, len(powers))
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda s: ((temps(s) - y) ** 2).sum(), bounds=(0.3, 1.5), method="bounded",
                              options={"xatol": 1e-6})
        s = float(res.x)
        J = (temps(s + 1e-4) - temps(s - 1e-4)) / 2e-4
        sd = float(noise_c / math.sqrt(J @ J))
        return {"description": "Multiplicador efetivo do h (desvio de ar de um dissipador sem shroud) a partir de "
                               "termopares na fonte em 4 potências (20-80 W). Medição sintética do solver com "
                               f"h_scale real = {true_h_scale} e ruído de {noise_c} K.",
                "parameter": "h_scale", "true": true_h_scale, "recovered": s, "ci95": [s - 1.96 * sd, s + 1.96 * sd],
                "rel_error": abs(s - true_h_scale) / true_h_scale,
                "true_inside_ci95": bool(abs(s - true_h_scale) <= 1.96 * sd), "noise": f"{noise_c} K (1 sigma)",
                "how": "mínimos quadrados (Brent) sobre o modelo; IC 95 % pela sensibilidade dT/dh_scale"}

    def variants(self):
        return {"8 aletas 35 mm": {"n_fins": 8}, "15 aletas 35 mm (base)": {},
                "22 aletas 35 mm": {"n_fins": 22}, "15 aletas 55 mm": {"fin_height": 0.055},
                "22 aletas 25 mm 1 mm": {"n_fins": 22, "fin_height": 0.025, "fin_thickness": 0.001},
                "cobre C110, 15 aletas": {"material": "cu110"}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f = result.fields.get("field3d")
        if not f:
            return []
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        b = f["base"]
        im = ax[0].imshow(np.asarray(b["bottom_c"]), origin="lower", cmap="jet",
                          extent=[0, b["width"], 0, b["depth"]])
        fig.colorbar(im, ax=ax[0], label="T [°C]")
        ax[0].set(xlabel="x [mm]", ylabel="y (ar) [mm]", title="Face inferior da base (FVM 3-D)")
        vs = np.linspace(0.8, 7, 25)
        p = result.params
        d, o = self._di(p)
        tt = [pfh.evaluate(d.geometry(), pfh.OperatingPoint(p["power"], p["t_ambient"], v, p["source_size"],
                                                            p["source_size"], p["tim"]))["t_source_max_c"] for v in vs]
        dp = [pfh.evaluate(d.geometry(), pfh.OperatingPoint(p["power"], p["t_ambient"], v, p["source_size"],
                                                            p["source_size"], p["tim"]))["pressure_drop_pa"] for v in vs]
        ax[1].plot(vs, tt, "C3", label="T fonte")
        ax[1].axhline(p["t_limit"], color="C3", ls="--", lw=1)
        ax[1].set(xlabel="velocidade do ar [m/s]", ylabel="T fonte [°C]", title="Curva do dissipador")
        a2 = ax[1].twinx()
        a2.plot(vs, dp, "C0", label="Δp")
        a2.set_ylabel("Δp [Pa]", color="C0")
        fig.tight_layout()
        path = os.path.join(out_dir, "heatsink.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from ..viz3d import Scene, Surface, box, merge
        f = result.fields.get("field3d")
        if not f:
            d, o = self._di(result.params)
            f = evaluate_design(d, o)["field3d"]
        b, fins, prof = f["base"], f["fins"], f["fin_profile"]
        W, Dp, tb = b["width"], b["depth"], b["thickness"]
        xs, ys = np.asarray(b["x"]), np.asarray(b["y"])
        top, bot = np.asarray(b["top_c"]), np.asarray(b["bottom_c"])
        nx, ny = 40, 40
        V, F = box([0, 0, 0], [W, Dp, tb], (nx, ny, 2))
        from scipy.interpolate import RegularGridInterpolator
        it = RegularGridInterpolator((ys, xs), top, bounds_error=False, fill_value=None)
        ib = RegularGridInterpolator((ys, xs), bot, bounds_error=False, fill_value=None)
        zf = V[:, 2] / tb
        T = (1 - zf) * ib(np.c_[V[:, 1], V[:, 0]]) + zf * it(np.c_[V[:, 1], V[:, 0]])
        parts = [(V, F, T)]
        z = np.asarray(prof["z"])
        ratio = np.asarray(prof["theta_ratio"])
        Ta = f["t_ambient_c"]
        root = np.asarray(fins["root_c"])
        for i, xc in enumerate(fins["x_centers"]):
            t = fins["thickness"]
            Vf, Ff = box([xc - t / 2, 0, tb], [xc + t / 2, Dp, tb + fins["height"]], (1, 24, 12))
            ry = np.interp(Vf[:, 1], ys, root[i])
            Tf = Ta + (ry - Ta) * np.interp(Vf[:, 2] - tb, z, ratio)
            parts.append((Vf, Ff, Tf))
        V, F, T = merge(parts)
        V = V / 1000.0 * 100                                      # mm -> display units (1 dm = 10)
        s = f["source"]
        sv, sf = box([(s["x"] - s["w"] / 2) / 10, (s["y"] - s["d"] / 2) / 10, -0.4],
                     [(s["x"] + s["w"] / 2) / 10, (s["y"] + s["d"] / 2) / 10, 0.0])
        return Scene([Surface("heatsink", V, F, "aluminium", {"temperature": T}),
                      Surface("source", sv, sf, "dark", {"temperature": np.full(len(sv), f["hotspot"]["t_c"])})],
                     title=self.title, labels={"temperature": "temperatura [°C]"})
