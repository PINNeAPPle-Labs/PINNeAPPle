"""Wing / blade section at Re 4e6: lift, drag and moment from the verified zoo surrogate of OpenFOAM RANS."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_design.aero.geometry import naca4, properties, surfaces, valid
from pinneapple_design.aero.panel import panel_solve

from ..core import Output, Param, Preset, check

ZOO = "airfoil_cst_rans_mlp"
NACA2412 = naca4("2412")
CARD = ("zoo airfoil_cst_rans_mlp: ensemble de 5 MLPs treinados em OpenFOAM simpleFoam k-omega SST, Re 4e6; cartão "
        "com hold-out, grade verificada (GCI), XFOIL e NASA TMR")
TAT = "teoria do aerofólio fino (Anderson, Fundamentals of Aerodynamics, cap. 4): alpha_L0 = -(1/pi) int dz/dx (cos t - 1) dt"


def thin_airfoil_zero_lift(code: str) -> float:
    """alpha_L0 (deg) of a NACA 4-digit camber line by thin-airfoil theory (numerical quadrature)."""
    m, p = int(code[0]) / 100, int(code[1]) / 10
    if m == 0:
        return 0.0
    t = np.linspace(0, np.pi, 200001)
    x = 0.5 * (1 - np.cos(t))
    dz = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    return float(np.degrees(-np.trapezoid(dz * (np.cos(t) - 1), t) / np.pi))


class AirfoilSection(Preset):
    name = "airfoil_section_re4e6"
    title = "Perfil de asa / pá a Re 4e6: sustentação, arrasto e momento (surrogate verificado de RANS)"
    sector = "Aerodinâmica / aeronaves leves, VANTs, pás de turbina eólica"
    problem = ("Seção de asa ou pá descrita por 6 parâmetros CST (3 de espessura, 3 de arqueamento), escoamento "
               "subsônico incompressível a Re = 4e6. Cl, Cd e Cm pelo modelo do zoo (ensemble de MLPs treinado em "
               "OpenFOAM RANS k-omega SST), com o desvio do ensemble; o método de painéis de Hess-Smith dá a "
               "referência invíscida.")
    question = ("Qual a sustentação, o arrasto e a eficiência L/D deste perfil no ângulo de ataque de operação? Que "
                "perfil maximiza L/D com espessura estrutural >= 12 % e Cm >= -0,12?")
    params = (
        Param("T0", "-", float(NACA2412[0]), "CST de espessura, ajuste do NACA 2412 (pinneapple_design.aero.naca4)",
              0.10, 0.24),
        Param("T1", "-", float(NACA2412[1]), "idem", 0.08, 0.22),
        Param("T2", "-", float(NACA2412[2]), "idem", 0.08, 0.22),
        Param("M0", "-", float(NACA2412[3]), "CST de arqueamento, ajuste do NACA 2412", -0.03, 0.08),
        Param("M1", "-", float(NACA2412[4]), "idem", -0.02, 0.14),
        Param("M2", "-", float(NACA2412[5]), "idem", -0.04, 0.12),
        Param("alpha_deg", "deg", 4.0, "premissa: cruzeiro de aeronave leve (Cl ~0,6)", -2.0, 14.0),
        Param("chord", "m", 1.5, "premissa: corda média de aeronave leve (V ~ 40 m/s dá Re 4e6)", 0.1, 5.0),
        Param("velocity", "m/s", 40.0, "premissa; Re = V c / nu, nu do ar a 15 °C = 1,46e-5 m2/s (ISA)", 5, 120),
    )
    outputs = (
        Output("Cl", "-", "coeficiente de sustentação (RANS, ensemble)"), Output("Cd", "-", "arrasto"),
        Output("Cm", "-", "momento em c/4"), Output("LD", "-", "Cl/Cd"),
        Output("Cl_std", "-", "desvio do ensemble"), Output("Cl_inviscid", "-", "Cl pelo método de painéis"),
        Output("thickness", "-", "espessura máxima / corda"), Output("reynolds", "-", "Re da condição dada"),
        Output("lift_per_span", "N/m", "sustentação por metro de envergadura"),
    )
    validity = ("Re = 4e6 (a condição dada é avisada se o Re diferir mais de 25 %), Mach < 0,3, ângulo -2 a 14° (sem "
                "estol profundo), formas dentro da caixa CST do treino; transição livre não modelada (RANS "
                "totalmente turbulento).")
    pipeline = ("geometria: CST de 6 parâmetros (pinneapple_design.aero.geometry), propriedades e validade da forma",
                "modelo: RANS k-omega SST (OpenFOAM) aprendido pelo zoo; painéis de Hess-Smith (invíscido)",
                "solver: pp.zoo.load('airfoil_cst_rans_mlp').predict + panel_solve",
                "pós: Cl, Cd, Cm, L/D, incerteza do ensemble, sustentação por envergadura",
                "decisão: forma válida, dentro do domínio, L/D e Cm aceitáveis")
    library = ("pinneapple_hub.zoo (airfoil_cst_rans_mlp)", "pinneapple_design.aero.panel.panel_solve",
               "pinneapple_design.aero.geometry")
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("Cl", "Cd", "LD")
    missing = {"surrogate": "o solver deste preset já é um surrogate verificado do zoo (erros de hold-out no cartão "
                            "e medidos aqui); o OpenFOAM não está instalado neste container para conferir ótimos",
               "reason": "não há caso aerodinâmico no pinneapple_reason"}
    limitations = ("Os ótimos são conferidos com o mesmo surrogate (ensemble, domínio) e com o método de painéis, não "
                   "com uma nova rodada de OpenFOAM (não instalado aqui).",)

    _model = None

    def model(self):
        if PresetCache.model is None:
            import pinneapple as pp
            PresetCache.model = pp.zoo.load(ZOO)
        return PresetCache.model

    @staticmethod
    def shape(p):
        return np.array([p["T0"], p["T1"], p["T2"], p["M0"], p["M1"], p["M2"]])

    def solve(self, p):
        sh = self.shape(p)
        out = self.model().predict(**{k: p[k] for k in ("T0", "T1", "T2", "M0", "M1", "M2")},
                                   alpha_deg=[p["alpha_deg"]])
        pr = properties(sh)
        ok, why = valid(sh)
        pan = panel_solve(sh, p["alpha_deg"], n_per_side=200, close_te=True)
        nu = 1.46e-5
        re = p["velocity"] * p["chord"] / nu
        q = 0.5 * 1.225 * p["velocity"] ** 2
        Cl = float(out["Cl"][0])
        return {"Cl": Cl, "Cd": float(out["Cd"][0]), "Cm": float(out["Cm"][0]), "LD": float(out["LD"][0]),
                "Cl_std": float(out["Cl_std"][0]), "Cd_std": float(out["Cd_std"][0]),
                "Cl_inviscid": float(pan.cl_kutta_joukowski), "Cl_inviscid_pressure": float(pan.cl_pressure),
                "thickness": pr["thickness"], "camber": pr["camber"], "reynolds": re, "lift_per_span": q * p["chord"] * Cl,
                "shape_valid": ok, "shape_issue": why,
                "_fields": {"panel": pan, "shape": sh}}

    def checks(self, p, out):
        return [check("shape valid (no crossing / waviness)", out["shape_valid"], reason=out["shape_issue"]),
                check("Reynolds within 25 % of the trained 4e6", abs(out["reynolds"] / 4e6 - 1) <= 0.25,
                      value=out["reynolds"], limit=4e6, source=CARD),
                check("ensemble spread small (Cl std < 0.01)", out["Cl_std"] < 0.01, value=out["Cl_std"], limit=0.01),
                check("viscous Cl below inviscid Cl (boundary-layer decambering)", out["Cl"] < out["Cl_inviscid"]
                      if out["Cl_inviscid"] > 0.1 else None, value=out["Cl"], limit=out["Cl_inviscid"]),
                check("panel method: Kutta-Joukowski lift = pressure integral", abs(out["Cl_inviscid"] -
                      out["Cl_inviscid_pressure"]) < 0.01 * max(abs(out["Cl_inviscid"]), 0.1),
                      value=out["Cl_inviscid_pressure"], limit=out["Cl_inviscid"], source="Hess-Smith")]

    def decide(self, p, out):
        ok = out["shape_valid"] and out["Cm"] >= -0.12
        return {"verdict": "OK" if ok else "NOT OK",
                "reasons": [f"Cl {out['Cl']:.3f} ± {out['Cl_std']:.3f}, Cd {out['Cd']:.5f}, L/D {out['LD']:.1f}, "
                            f"Cm {out['Cm']:.3f} (limite -0,12)",
                            f"espessura {100 * out['thickness']:.1f} %, sustentação {out['lift_per_span']:.0f} N/m"]}

    def validate(self):
        m = self.model()
        rows = []
        for split, src in (("naca_reference", "OpenFOAM RANS, 36 rodadas NACA 0012/2412/4412/4415 fora do treino "
                                              "(dados do zoo)"),
                           ("xfoil_naca2412", "XFOIL 6.99 com transição forçada, NACA 2412, 9 ângulos (dados do zoo)")):
            X, Y = m.reference(split)
            names = m.input_names
            pr = m.predict(**{n: X[:, i] for i, n in enumerate(names[:-1])}, alpha_deg=X[:, -1])
            mae = float(np.mean(np.abs(pr["Cl"] - Y[:, 0])))
            rows.append({"case": f"Cl vs {split} ({len(Y)} pontos): erro = MAE absoluto em Cl", "output": "Cl", "unit": "-",
                         "computed": mae, "reference": 0.0, "rel_error": mae, "tol": 0.03 if "xfoil" in split else 0.01,
                         "pass": bool(mae <= (0.03 if "xfoil" in split else 0.01)), "kind": "independent_solver",
                         "source": src + "; erro absoluto médio em Cl, medido aqui"})
            cd = float(np.mean(np.abs(pr["Cd"] - Y[:, 1]) / Y[:, 1]))
            rows.append({"case": f"Cd vs {split}: erro = erro relativo médio", "output": "Cd", "unit": "-", "computed": cd,
                         "reference": 0.0, "rel_error": cd, "tol": 0.08 if "xfoil" in split else 0.02,
                         "pass": bool(cd <= (0.08 if "xfoil" in split else 0.02)), "kind": "independent_solver",
                         "source": src})
        # thin-airfoil theory: zero-lift angle of the NACA 2412 from the surrogate's polar
        al = np.linspace(-2, 4, 13)
        cl = m.predict(**{k: float(v) for k, v in zip(("T0", "T1", "T2", "M0", "M1", "M2"), NACA2412, strict=False)}, alpha_deg=al)["Cl"]
        a0 = float(np.polyfit(cl, al, 1)[1])
        tat = thin_airfoil_zero_lift("2412")
        rows.append({"case": "ângulo de sustentação nula do NACA 2412", "output": "alpha_L0", "unit": "deg",
                     "computed": a0, "reference": tat, "rel_error": abs(a0 - tat) / abs(tat), "tol": 0.10,
                     "pass": bool(abs(a0 - tat) / abs(tat) <= 0.10), "kind": "analytic", "source": TAT})
        slope = float(np.polyfit(al, cl, 1)[0])
        pan = [panel_solve(NACA2412, a, n_per_side=200, close_te=True).cl_kutta_joukowski for a in (0.0, 4.0)]
        ps = (pan[1] - pan[0]) / 4
        rows.append({"case": "inclinação dCl/dalpha: RANS vs painéis (viscoso ~5-10 % menor)", "output": "Cl_alpha",
                     "unit": "1/deg", "computed": slope, "reference": ps, "rel_error": abs(slope - ps) / ps,
                     "tol": 0.12, "pass": bool(abs(slope - ps) / ps <= 0.12), "kind": "independent_solver",
                     "source": "Hess-Smith (Moran 1984), invíscido"})
        return {"preset": self.name, "references": rows, "convergence": self._conv(), "all_pass": all(
            r["pass"] for r in rows)}

    def _conv(self):
        ns = (50, 100, 200, 400)
        cls = [panel_solve(NACA2412, 4.0, n_per_side=n, close_te=True).cl_kutta_joukowski for n in ns]
        rel = abs(cls[-1] - cls[-2]) / abs(cls[-1])
        return {"parameter": "painéis por face (Hess-Smith)", "values": list(ns), "Cl": cls, "rel_change_last": rel,
                "pass": rel < 0.005, "note": "bordo de fuga fechado (close_te); aberto, o Cl não converge"}

    def optimize(self, **kw):
        """Maximum L/D at the given angle with thickness >= 12 %, Cm >= -0.12 and a valid shape, inside the trained
        box (differential evolution on the ensemble mean), then the ensemble spread and the panel method checked at
        the optimum."""
        from scipy.optimize import differential_evolution
        p = self.resolve(**kw)
        keys = ("T0", "T1", "T2", "M0", "M1", "M2")
        b = [(self.param(k).lo, self.param(k).hi) for k in keys]
        m = self.model()

        def f(x):
            sh = np.asarray(x)
            pr = properties(sh)
            ok, _ = valid(sh)
            o = m.predict(**dict(zip(keys, map(float, x), strict=False)), alpha_deg=[p["alpha_deg"]])
            pen = 1e3 * (max(0, 0.12 - pr["thickness"]) + max(0, -0.12 - float(o["Cm"][0]))) + (0 if ok else 1e3)
            return -float(o["LD"][0]) + pen

        res = differential_evolution(f, b, seed=0, maxiter=60, popsize=15, tol=1e-8, polish=False)
        best = dict(zip(keys, map(float, res.x), strict=False))
        ver = self.run(**{**kw, **best})
        base = self.run(**kw)
        return {"objective": "L/D máximo no alpha dado com t/c >= 12 %, Cm >= -0,12, forma válida",
                "method": "evolução diferencial (scipy) no ensemble do zoo", "evaluations": int(res.nfev),
                **{k: round(v, 5) for k, v in best.items()}, "LD_opt": ver.outputs["LD"], "LD_base": base.outputs["LD"],
                "Cl_opt": ver.outputs["Cl"], "Cl_std_opt": ver.outputs["Cl_std"], "Cm_opt": ver.outputs["Cm"],
                "thickness_opt": ver.outputs["thickness"], "trust_at_opt": ver.trust.decisao,
                "panel_Cl_opt": ver.outputs["Cl_inviscid"]}

    def invert(self, true_offset: float = 1.3, noise: float = 0.01, seed: int = 0, **kw):
        """Rigging (incidence) error of a wing section from a wind-tunnel balance: Cl measured at five nominal
        angles, balance noise 0.01 in Cl. Synthetic truth: the model sits 1.3 deg nose-up of its nominal angle."""
        p = self.resolve(**kw)
        m = self.model()
        keys = ("T0", "T1", "T2", "M0", "M1", "M2")
        nom = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        sh = {k: p[k] for k in keys}
        cl = lambda d: m.predict(**sh, alpha_deg=nom + d)["Cl"]  # noqa: E731
        rng = np.random.default_rng(seed)
        y = cl(true_offset) + rng.normal(0, noise, len(nom))
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda d: float(((cl(d) - y) ** 2).sum()), bounds=(-3, 3), method="bounded",
                              options={"xatol": 1e-6})
        d = float(res.x)
        J = (cl(d + 1e-3) - cl(d - 1e-3)) / 2e-3
        sd = float(noise / math.sqrt(J @ J))
        return {"description": "Erro de calagem (incidência) de um perfil no túnel de vento a partir do Cl medido "
                               "pela balança em 5 ângulos nominais (ruído 0,01 em Cl). Verdade sintética: +1,3°.",
                "parameter": "offset de ângulo", "unit": "deg", "true": true_offset, "recovered": d,
                "ci95": [d - 1.96 * sd, d + 1.96 * sd], "rel_error": abs(d - true_offset) / true_offset,
                "true_inside_ci95": bool(abs(d - true_offset) <= 1.96 * sd), "noise": f"{noise} em Cl",
                "how": "mínimos quadrados (Brent) sobre o ensemble do zoo, IC pela sensibilidade dCl/dalpha"}

    def variants(self):
        out = {}
        for code in ("0012", "2412", "4412", "4415"):
            s = naca4(code)
            out[f"NACA {code}"] = dict(zip(("T0", "T1", "T2", "M0", "M1", "M2"), map(float, s), strict=False))
        return out

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p = result.params
        m = self.model()
        keys = ("T0", "T1", "T2", "M0", "M1", "M2")
        al = np.linspace(-2, 14, 33)
        pr = m.predict(**{k: p[k] for k in keys}, alpha_deg=al)
        pan = [panel_solve(self.shape(p), a, n_per_side=200, close_te=True).cl_kutta_joukowski for a in al]
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        ax[0].fill_between(al, pr["Cl"] - 2 * pr["Cl_std"], pr["Cl"] + 2 * pr["Cl_std"], alpha=0.3)
        ax[0].plot(al, pr["Cl"], label="RANS (zoo)")
        ax[0].plot(al, pan, "--", label="painéis (invíscido)")
        ax[0].set(xlabel="alpha [deg]", ylabel="Cl", title="Curva de sustentação")
        ax[0].legend(fontsize=8)
        ax[1].plot(pr["Cd"], pr["Cl"])
        ax[1].set(xlabel="Cd", ylabel="Cl", title="Polar de arrasto")
        x = np.linspace(0, 1, 200)
        yu, yl = surfaces(self.shape(p), x)
        ax[2].plot(x, yu, "k", x, yl, "k")
        ax[2].set_aspect("equal")
        ax[2].set_title(f"Forma (t/c {100 * result.outputs['thickness']:.1f} %)")
        fig.tight_layout()
        path = os.path.join(out_dir, "airfoil.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from ..viz3d import Scene, Surface, extrude
        pan = result.fields["panel"]
        sh = result.fields["shape"]
        n = 100
        b = np.linspace(0, np.pi, n + 1)
        x = 0.5 * (1 - np.cos(b))
        yu, yl = surfaces(sh, x)
        # closed section (y = chord, z = thickness) clockwise -> counter-clockwise loop for extrude
        sec = np.r_[np.c_[x[::-1], yl[::-1]], np.c_[x[1:-1], yu[1:-1]]]
        V, F, st = extrude(sec, np.linspace(0, 3.0, 12))
        # Cp at the panel mid-points: the first half of the panels is the lower surface (clockwise order)
        cp, xm = np.asarray(pan.cp), np.asarray(pan.x_mid)
        h = len(cp) // 2
        lo_x, lo_cp = xm[:h][::-1], cp[:h][::-1]
        up_x, up_cp = xm[h:], cp[h:]
        vx, vz = V[:, 1], V[:, 2]
        up = vz >= np.interp(vx, x, 0.5 * (yu + yl))
        cpv = np.where(up, np.interp(vx, up_x, up_cp), np.interp(vx, lo_x, lo_cp))
        V = np.c_[V[:, 1], V[:, 0], V[:, 2]]                       # chord along x, span along y
        return Scene([Surface("wing", V, F[:, ::-1], "white", {"cp": cpv})], title=self.title,
                     labels={"cp": f"Cp invíscido (painéis), alpha {result.params['alpha_deg']}°"})


class PresetCache:
    model = None
