"""Steel footbridge: crowd strength, deflection, lateral-torsional buckling and pedestrian vibration (EN 1990 A2)."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_reason import LIBRARY
from pinneapple_reason.modes.structural import STEEL, girder_loads, i_section, pedestrian_response
from pinneapple_simulation.numerical_solvers.beam_fe import beam_modes, beam_static

from ..core import Output, Param, Preset, SurrogateSpec, check

EC3 = "EN 1993-1-1:2005 6.2.5 / 6.3.2.2"
A2 = "EN 1990:2002 Anexo A2, A2.4.3.2: aceleração vertical <= 0,7 m/s2 (recomendado)"
BEAMFE = "pinneapple_simulation.numerical_solvers.beam_fe (Hermite, massa consistente, 40 elementos)"
MODES = ("bending_strength", "deflection", "lateral_torsional_buckling", "pedestrian_resonance")


def section_class_ok(h, b, tf, tw, fy) -> tuple:
    """EN 1993-1-1 Table 5.2, class 3 limits: web in bending c/t <= 124 eps, outstand flange c/t <= 14 eps
    (welded: c ignores the weld throat, conservative). Returns (ok, web c/t, web limit, flange c/t, flange limit)."""
    eps = math.sqrt(235e6 / fy)
    cw, cf = (h - 2 * tf) / tw, (b - tw) / 2 / tf
    return cw <= 124 * eps and cf <= 14 * eps, cw, 124 * eps, cf, 14 * eps


class SteelFootbridge(Preset):
    name = "steel_footbridge_pedestrian"
    title = "Passarela de aço de 30 m: resistência, flecha, flambagem lateral e vibração de pedestres"
    sector = "Estrutural / construção"
    problem = ("Passarela biapoiada de 30 m com duas vigas I soldadas e tabuleiro de 2,5 m (placas pré-moldadas). "
               "Verifica ELU de flexão sob multidão (EN 1991-2), flecha L/360, flambagem lateral com torção "
               "(EN 1993-1-1) e conforto à vibração por pedestres (EN 1990 A2, Bachmann & Ammann).")
    question = ("A seção das vigas atende os quatro modos de falha? Qual a seção soldada mais leve que atende? "
                "O primeiro modo vertical cai na faixa de passo de caminhada (1,6-2,4 Hz)?")
    params = (
        Param("span", "m", 30.0, "premissa: travessia típica de 30 m (via ou rio pequeno)", 8, 60),
        Param("deck_width", "m", 2.5, "largura útil de 2,5 m (passarela de pedestres e bicicletas)", 1.5, 6),
        Param("deck_dead_load", "Pa", 3000.0, "placas de concreto de 100 mm (25 kN/m3, EN 1991-1-1 Tab. A.1) + "
              "0,5 kN/m2 de acabamentos", 500, 8000),
        Param("girder_h", "m", 0.80, "premissa: L/37 (viga dimensionada só pela resistência)", 0.3, 2.0),
        Param("girder_b", "m", 0.25, "mesa de 250 mm", 0.1, 0.8),
        Param("girder_tf", "m", 0.020, "mesa de 20 mm (S355, t <= 40 mm)", 0.008, 0.06),
        Param("girder_tw", "m", 0.010, "alma de 10 mm", 0.006, 0.03),
        Param("fy", "Pa", 355e6, "EN 10025-2 S355, t <= 16 mm", 235e6, 460e6),
        Param("lateral_braces", "-", 0, "sem travamento lateral intermediário", 0, 10),
        Param("damping_ratio", "-", 0.005, "aço soldado: 0,5 % (Sétra 2006 / HiVoSS 2008, valor mínimo)", 0.002,
              0.05),
        Param("pedestrian_weight", "N", 700.0, "Bachmann & Ammann (1987)", 500, 1000),
        Param("acc_max", "m/s2", 0.7, A2, 0.2, 2.5),
        Param("deflection_ratio", "-", 360.0, "AASHTO LRFD Pedestrian Bridges (2009) sec. 5: L/360 sob carga de "
              "pedestres", 200, 1000),
    )
    outputs = (
        Output("f1", "Hz", "1ª frequência vertical"),
        Output("acceleration", "m/s2", "aceleração de pico, um pedestre em ressonância"),
        Output("deflection", "m", "flecha sob a multidão"),
        Output("util_bending", "-", "M_Ed / M_Rd"),
        Output("util_ltb", "-", "M_Ed / M_b,Rd (flambagem lateral)"),
        Output("M_cr", "N m", "momento crítico elástico"),
        Output("steel_mass", "kg", "massa de aço das vigas"),
    )
    validity = ("Vigas biapoiadas de seção I duplamente simétrica, Euler-Bernoulli, carga no centro de cisalhamento, "
                "C1 = 1 (conservador para carga distribuída), tabuleiro sem ação composta e sem travar a mesa "
                "comprimida. Vibração: um pedestre em regime permanente (limite superior); multidões e modos "
                "laterais (lock-in) não modelados.")
    pipeline = ("geometria: i_section(h, b, tf, tw) das vigas + tabuleiro",
                "modelo: cargas EN 1991-2 (multidão), combinações EN 1990 (1,35 G + 1,5 Q)",
                "solver: modos de falha do pinneapple_reason (bending_strength, deflection, "
                "lateral_torsional_buckling, pedestrian_resonance) + FE de viga Hermite (modos e flecha)",
                "pós: utilizações, f1, aceleração, M_cr",
                "decisão: todas as utilizações <= 1 e a <= 0,7 m/s2; seção mais leve por varredura")
    library = ("pinneapple_reason.LIBRARY.mode(...).evaluate", "pinneapple_reason.modes.structural",
               "pinneapple_simulation.numerical_solvers.beam_fe.beam_modes")
    surrogate_spec = SurrogateSpec(inputs=("span", "girder_h", "girder_b"),
                                   outputs=("f1", "util_ltb", "util_bending"), n_train=70, n_test=25,
                                   bounds={"span": (20.0, 45.0), "girder_h": (0.6, 1.6), "girder_b": (0.2, 0.5)},
                                   log_outputs=("util_ltb", "util_bending"))
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("util_ltb", "acceleration", "steel_mass")
    reason_case = "footbridge"
    missing = {}
    limitations = ("M_cr pela fórmula fechada de Timoshenko & Gere com C1 = 1; sem verificação independente "
                   "(FE de flambagem lateral não implementado).",)

    def _concept(self, p):
        return {**{k: p[k] for k in ("span", "deck_width", "deck_dead_load", "girder_h", "girder_b", "girder_tf",
                                     "girder_tw", "fy", "lateral_braces", "damping_ratio", "pedestrian_weight",
                                     "acc_max", "deflection_ratio")}, "n_girders": 2, "section_type": "welded"}

    def solve(self, p):
        c = self._concept(p)
        lim = {"acc_max": p["acc_max"], "deflection_ratio": p["deflection_ratio"]}
        v = {m: LIBRARY.mode(m).evaluate(c, limits=lim) for m in MODES}
        def q(m, k):
            return v[m].quantities[k][0]
        ld = girder_loads(c)
        r = pedestrian_response(c)
        out = {"f1": r["f1"], "acceleration": r["acc"], "deflection": v["deflection"].demand / 1e3
               if v["deflection"].unit == "mm" else v["deflection"].demand,
               "deflection_limit": p["span"] / p["deflection_ratio"],
               "util_bending": v["bending_strength"].demand / v["bending_strength"].capacity,
               "util_ltb": v["lateral_torsional_buckling"].demand / v["lateral_torsional_buckling"].capacity,
               "M_cr": q("lateral_torsional_buckling", "M_cr"), "M_Ed": ld["MEd"],
               "steel_mass": 2 * ld["sec"]["A"] * STEEL["rho"] * p["span"], "mass_per_m": ld["mass_per_m"],
               "modal_mass": r["M"], "EI_total": r["EI"], "qk": ld["qk"],
               "passed": {m: v[m].passed for m in MODES},
               "section_class3": section_class_ok(p["girder_h"], p["girder_b"], p["girder_tf"], p["girder_tw"],
                                                  p["fy"]),
               "self_checks": [c for m in MODES for c in v[m].self_checks]}
        fe = beam_modes(p["span"], r["EI"], ld["mass_per_m"], 40, 3)
        st = beam_static(p["span"], STEEL["E"] * ld["sec"]["Iy"], ld["qk"], 40)
        out.update({"f1_fe": float(fe["f"][0]), "f2_fe": float(fe["f"][1]), "deflection_fe": st["w_max"],
                    "fe_equilibrium": abs(abs(st["reaction_sum"]) - st["load"]) / st["load"],
                    "_fields": {"modes": fe, "static": st}})
        return out

    def checks(self, p, out):
        c = list(out.pop("self_checks", []))
        c += [check("FE beam: 1st frequency = closed form", abs(out["f1_fe"] - out["f1"]) / out["f1"] < 1e-3,
                    value=abs(out["f1_fe"] - out["f1"]) / out["f1"], limit=1e-3, source=BEAMFE),
              check("FE beam: crowd deflection = closed form", abs(out["deflection_fe"] - out["deflection"]) /
                    out["deflection"] < 1e-3, value=abs(out["deflection_fe"] - out["deflection"]) / out["deflection"],
                    limit=1e-3, source=BEAMFE),
              check("FE equilibrium: reactions = applied load", out["fe_equilibrium"] < 1e-9,
                    value=out["fe_equilibrium"], limit=1e-9, source=BEAMFE),
              check("cross-section class <= 3 (the elastic checks above hold)", out["section_class3"][0],
                    value=out["section_class3"][1], limit=out["section_class3"][2], source="EN 1993-1-1 Table 5.2",
                    reason=f"web c/t {out['section_class3'][1]:.0f} (<= {out['section_class3'][2]:.0f}), flange c/t "
                           f"{out['section_class3'][3]:.1f} (<= {out['section_class3'][4]:.1f})"),
              check("EN 1990 A2 needs the comfort check (f1 < 5 Hz)", True if out["f1"] < 5 else None,
                    value=out["f1"], limit=5.0, source=A2)]
        return c

    def complete(self, p, out):
        """The pedestrian acceleration jumps across the walking-pace bands (resonance), which a smooth surrogate
        cannot learn (hold-out R2 0.18 when tried); it is computed by the closed form, which costs microseconds."""
        return {**out, "acceleration": pedestrian_response(self._concept(p))["acc"]}

    def decide(self, p, out):
        names = {"bending_strength": "flexão (ELU)", "deflection": "flecha (ELS)",
                 "lateral_torsional_buckling": "flambagem lateral", "pedestrian_resonance": "vibração"}
        if "passed" in out:
            fails = [names[m] for m, ok in out["passed"].items() if not ok]
            if not out["section_class3"][0]:
                fails.append("classe da seção > 3 (alma ou mesa esbelta, EN 1993-1-1 Tab. 5.2)")
        else:                                              # surrogate: from the predicted utilisations
            fails = [n for n, bad in (("flambagem lateral", out["util_ltb"] > 1),
                                      ("vibração", out["acceleration"] > p["acc_max"])) if bad]
        return {"verdict": "OK" if not fails else "NOT OK",
                "reasons": [f"f1 = {out['f1']:.2f} Hz, aceleração {out['acceleration']:.2f} m/s2 (limite "
                            f"{p['acc_max']})", f"utilização flambagem lateral {out['util_ltb']:.2f}"]
                + ([f"utilização flexão {out['util_bending']:.2f}; flecha {1e3 * out['deflection']:.0f} mm "
                    f"(limite {1e3 * out['deflection_limit']:.0f} mm)"] if "util_bending" in out else [])
                + ([f"não atende: {', '.join(fails)}"] if fails else [])}

    # ------------------------------------------------------------------------------------------- validation
    def validate(self):
        p = self.defaults()
        r = self.run()
        o = r.outputs
        rows = [self._row("1ª frequência: fórmula fechada vs FE de viga", "f1", "Hz", o["f1"], o["f1_fe"], 1e-3,
                          "independent_solver", BEAMFE),
                self._row("flecha da multidão: 5qL^4/384EI vs FE de viga", "deflection", "m", o["deflection"],
                          o["deflection_fe"], 1e-3, "independent_solver", BEAMFE),
                self._row("2ª frequência = 4 x f1 (Euler-Bernoulli biapoiada)", "f2", "Hz", o["f2_fe"], 4 * o["f1"],
                          2e-3, "analytic", "Clough & Penzien, Dynamics of Structures, eq. 18-21: f_n = n^2 f_1")]
        # pedestrian response: steady resonant amplitude a = DLF G / M * 1/(2 xi) at f_pace = f1
        q = self._concept({**p, "span": 24.0})
        pr = pedestrian_response(q)
        if 1.6 <= pr["f1"] <= 2.4:
            ref = 0.4 * p["pedestrian_weight"] / pr["M"] / (2 * p["damping_ratio"])
            rows.append(self._row("aceleração em ressonância exata (vão 24 m, f1 na faixa de passo)", "acceleration",
                                  "m/s2", pr["acc"], ref, 1e-6, "analytic",
                                  "oscilador de 1 GL em ressonância: a = alpha G / (M_modal 2 xi), Bachmann & "
                                  "Ammann (1987)"))
        # FE convergence of f1
        ns = (4, 8, 16, 32)
        fs = [float(beam_modes(p["span"], o["EI_total"], o["mass_per_m"], n, 1)["f"][0]) for n in ns]
        rel = abs(fs[-1] - fs[-2]) / fs[-1]
        conv = {"parameter": "elementos de viga", "values": list(ns), "f1": fs, "rel_change_last": rel,
                "pass": rel < 1e-4}
        return {"preset": self.name, "references": rows, "convergence": conv,
                "all_pass": all(x["pass"] for x in rows) and conv["pass"]}

    @staticmethod
    def _row(case, output, unit, comp, ref, tol, kind, source):
        err = abs(comp - ref) / abs(ref)
        return {"case": case, "output": output, "unit": unit, "computed": comp, "reference": ref, "rel_error": err,
                "tol": tol, "pass": bool(err <= tol), "kind": kind, "source": source, "params": {}}

    # ------------------------------------------------------------------------------------------- extras
    @staticmethod
    def _brace_mass(p, n):
        """Cross-frames at n points: one cross-beam + two diagonals of L50x50x5 (3.77 kg/m, EN 10056-1)."""
        W, a = p["deck_width"], p["span"] / (n + 1)
        return n * 3.77 * (W + 2 * math.hypot(W, min(a, 3 * W) / 3))

    def optimize(self, **kw):
        """Lightest steel (girders + cross-frames) passing all four modes: welded sections (h, b, tf, tw on a
        fabrication grid) x number of lateral bracing points; the surrogate's pick over (h, b) is re-run on the
        solver."""
        p = self.resolve(**kw)
        best, n = None, 0
        lim = {"acc_max": p["acc_max"], "deflection_ratio": p["deflection_ratio"]}
        for nb in (0, 1, 2, 3, 5, 7):
            for h in np.arange(0.6, 1.61, 0.05):
                for b in np.arange(0.20, 0.51, 0.025):
                    for tf in (0.012, 0.015, 0.020, 0.025, 0.030):
                        for tw in (0.008, 0.010, 0.012):
                            n += 1
                            mass = 2 * i_section(h, b, tf, tw)["A"] * STEEL["rho"] * p["span"] + self._brace_mass(p, nb)
                            if (best is not None and mass >= best[1]) or not section_class_ok(h, b, tf, tw,
                                                                                               p["fy"])[0]:
                                continue
                            q = {**p, "girder_h": float(h), "girder_b": float(b), "girder_tf": tf, "girder_tw": tw,
                                 "lateral_braces": nb}
                            c = self._concept(q)
                            if all(LIBRARY.mode(m).evaluate(c, limits=lim).passed for m in MODES):
                                best = (q, mass)
        out = {"objective": "massa mínima de aço (vigas + contraventamentos) com os 4 modos atendidos e seção classe <= 3",
               "designs_scanned": n}
        if best is None:
            return {**out, "feasible": False}
        q, mass = best
        keys = ("girder_h", "girder_b", "girder_tf", "girder_tw", "lateral_braces")
        ver = self.run(**{**kw, **{k: q[k] for k in keys}})
        out.update({k: q[k] for k in keys})
        out.update({"steel_mass_kg": mass, "girder_mass_kg": ver.outputs["steel_mass"],
                    "base_girder_mass_kg": self.run(**kw).outputs["steel_mass"],
                    "verified_ok": ver.decision["verdict"] == "OK", "util_ltb": ver.outputs["util_ltb"],
                    "util_bending": ver.outputs["util_bending"], "acceleration": ver.outputs["acceleration"]})
        from ..surrogate import load_surrogate
        sur = load_surrogate(self.name)
        if sur is not None:
            cand = None
            for h in np.arange(0.6, 1.61, 0.02):
                for b in np.arange(0.2, 0.501, 0.01):
                    pr, ood = sur.predict_one({**p, "girder_h": h, "girder_b": b})
                    acc = pedestrian_response(self._concept({**p, "girder_h": h, "girder_b": b}))["acc"]
                    if ood or pr["util_ltb"] > 1 or acc > p["acc_max"] or not section_class_ok(
                            h, b, p["girder_tf"], p["girder_tw"], p["fy"])[0]:
                        continue
                    A = i_section(h, b, p["girder_tf"], p["girder_tw"])["A"]
                    if cand is None or A < cand[2]:
                        cand = (float(h), float(b), A, pr["util_ltb"])
            out["surrogate_question"] = ("seção (h, b) mais leve com util_ltb <= 1 e a <= limite, sem travamentos, "
                                         "tf e tw padrão")
            if cand:
                vs = self.run(girder_h=cand[0], girder_b=cand[1])
                out.update({"surrogate_choice_hb": [cand[0], cand[1]], "surrogate_util_ltb": cand[3],
                            "solver_util_ltb_at_choice": vs.outputs["util_ltb"],
                            "surrogate_vs_solver_diff_util": cand[3] - vs.outputs["util_ltb"],
                            "surrogate_choice_ok_on_solver": vs.decision["verdict"] == "OK"})
            else:
                out["surrogate_choice"] = "nenhuma seção do domínio do surrogate atende sem travamentos"
        return out

    def invert(self, noise: float = 0.005, seed: int = 0, added_mass_kg: float = 6000.0, **kw):
        """Effective bending stiffness and mass per metre from an ambient-vibration test: f1 and f2 measured on
        the bare bridge and again with known sandbags (6 t spread over the span). Synthetic truth: EI x 1.25
        (composite action of the deck) and mass x 1.10 (heavier surfacing); 0.5 % frequency noise."""
        p = self.resolve(**kw)
        o = self.run(**kw).outputs
        EI0, m0, L = o["EI_total"], o["mass_per_m"], p["span"]
        true = np.array([1.25, 1.10])

        def freqs(s):
            out = []
            for dm in (0.0, added_mass_kg / L):
                out += list(beam_modes(L, s[0] * EI0, s[1] * m0 + dm, 24, 2)["f"])
            return np.array(out)

        rng = np.random.default_rng(seed)
        y = freqs(true) * (1 + rng.normal(0, noise, 4))
        s = np.array([1.0, 1.0])
        for _ in range(30):
            r0 = (freqs(s) - y) / y
            J = np.column_stack([((freqs(s + e) - freqs(s - e)) / (2e-4)) / y for e in np.eye(2) * 1e-4])
            step = np.linalg.lstsq(J, -r0, rcond=None)[0]
            s = s + step
            if np.abs(step).max() < 1e-10:
                break
        cov = noise ** 2 * np.linalg.inv(J.T @ J)
        sd = np.sqrt(np.diag(cov))
        return {"description": "Rigidez efetiva EI e massa por metro a partir de ensaio de vibração ambiente: f1 e f2 "
                               f"medidas com a passarela vazia e com {added_mass_kg / 1e3:.0f} t de sacos de areia "
                               "distribuídos. Verdade sintética: EI x1,25 (ação composta do tabuleiro), massa x1,10; "
                               f"ruído de {100 * noise:.1f} % nas frequências.",
                "parameter": "fator de EI, fator de massa", "true": true.tolist(), "recovered": s.tolist(),
                "ci95": [[float(s[i] - 1.96 * sd[i]), float(s[i] + 1.96 * sd[i])] for i in range(2)],
                "rel_error": float(np.max(np.abs(s - true) / true)),
                "true_inside_ci95": bool(np.all(np.abs(s - true) <= 1.96 * sd)),
                "noise": f"{100 * noise:.1f} % (1 sigma) nas frequências",
                "how": "Gauss-Newton sobre o FE de viga (beam_modes), covariância sigma^2 (J^T J)^-1"}

    def variants(self):
        return {"base (h 0,80, b 0,25)": {}, "viga alta h 1,10": {"girder_h": 1.10},
                "mesa larga b 0,40": {"girder_b": 0.40}, "2 travamentos": {"lateral_braces": 2},
                "alta + larga": {"girder_h": 1.10, "girder_b": 0.40}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        F = result.fields
        p = result.params
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        md = F["modes"]
        for k in range(3):
            ax[0].plot(md["x"], md["shapes"][k] / np.abs(md["shapes"][k]).max(), label=f"{md['f'][k]:.2f} Hz")
        ax[0].set(xlabel="x [m]", ylabel="forma modal", title="Modos verticais (FE de viga)")
        ax[0].legend(fontsize=8)
        f1 = result.outputs["f1"]
        fp = np.linspace(1.0, 3.0, 400)
        xi = p["damping_ratio"]
        M = result.outputs["modal_mass"]
        acc = [0.4 * p["pedestrian_weight"] / M * (f / f1) ** 2 / math.sqrt((1 - (f / f1) ** 2) ** 2 +
                                                                          (2 * xi * f / f1) ** 2) for f in fp]
        ax[1].semilogy(fp, acc)
        ax[1].axvspan(1.6, 2.4, color="orange", alpha=0.2, label="passo de caminhada")
        ax[1].axhline(p["acc_max"], color="r", ls="--", lw=1, label="limite EN 1990 A2")
        ax[1].set(xlabel="frequência do passo [Hz]", ylabel="aceleração [m/s2]", title="Resposta a um pedestre")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(out_dir, "footbridge.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from pinneapple_tools.visualization.studio.animation import Animation

        from ..viz3d import Scene, Surface, box, extrude, merge
        p = result.params
        F = result.fields
        L, W = p["span"], p["deck_width"]
        h, b, tf, tw = p["girder_h"], p["girder_b"], p["girder_tf"], p["girder_tw"]
        x = np.linspace(0, L, 61)
        sec = np.array([[-b / 2, 0], [b / 2, 0], [b / 2, tf], [tw / 2, tf], [tw / 2, h - tf], [b / 2, h - tf],
                        [b / 2, h], [-b / 2, h], [-b / 2, h - tf], [-tw / 2, h - tf], [-tw / 2, tf], [-b / 2, tf]])
        parts = []
        ld = girder_loads(self._concept(p))
        M = ld["wed"] * x * (L - x) / 2                    # design moment, one girder
        sig = M / ld["sec"]["Wy"] / 1e6                    # extreme-fibre bending stress, MPa
        for yc in (-W / 2 + b / 2, W / 2 - b / 2):
            V, Fc, st = extrude(sec + [yc, 0], x)
            parts.append((V, Fc, sig[st], st))
        V, Fc, sg, st = merge(parts)
        dv, df = box([0, -W / 2, h], [L, W / 2, h + 0.10], (60, 4, 1))
        md = F["modes"]
        shape = np.interp(x, md["x"], md["shapes"][0])
        shape /= np.abs(shape).max()
        g = Surface("girders", V, Fc, "steel", {"sigma": sg, "mode1": np.abs(shape)[st]})
        dstat = np.interp(dv[:, 0], md["x"], np.abs(np.interp(md["x"], x, shape)))
        deck = Surface("deck", dv, df, "concrete", {"mode1": dstat})
        sc = Scene([g, deck], title=self.title,
                   labels={"sigma": "tensão de flexão de cálculo [MPa]", "mode1": f"1º modo, {result.outputs['f1']:.2f} Hz"})
        T = 24
        ph = np.linspace(0, 2 * np.pi, T, endpoint=False)
        amp = 0.06 * L
        an = Animation(np.linspace(0, 1 / result.outputs["f1"], T, endpoint=False), time_label="t", time_unit="s")
        wg = np.interp(V[:, 0], x, shape)
        wd = np.interp(dv[:, 0], x, shape)
        an.add_surface_shape("girders", np.stack([V + np.c_[0 * wg, 0 * wg, -amp * np.sin(a) * wg] for a in ph]))
        an.add_surface_shape("deck", np.stack([dv + np.c_[0 * wd, 0 * wd, -amp * np.sin(a) * wd] for a in ph]))
        sc.animation = an
        return sc
