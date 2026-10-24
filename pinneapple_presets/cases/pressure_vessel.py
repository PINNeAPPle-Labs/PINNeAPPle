"""Cylindrical pressure vessel shell and heads to ASME VIII Div. 1 (UG-27, UG-32), checked by Lamé and 3-D FEM."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_simulation.numerical_solvers.solid_fem import HexMesh, SolidFEM

from ..core import Output, Param, Preset, check

ASME = "ASME BPVC Section VIII Div. 1 (2023)"
LAME = "Lamé (1852); Timoshenko & Goodier, Theory of Elasticity, 3a ed., sec. 28"
PLATES = tuple(x / 1000 for x in (6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30, 32, 36, 40, 45, 50))


def ring_mesh(ri: float, ro: float, h: float, nr: int, nt: int, nz: int) -> HexMesh:
    """Quarter ring r in [ri, ro], theta in [0, 90 deg], z in [0, h] as an 8-node hexahedral mesh."""
    r = np.linspace(ri, ro, nr + 1)
    t = np.linspace(0, math.pi / 2, nt + 1)
    z = np.linspace(0, h, nz + 1)
    R, T, Z = np.meshgrid(r, t, z, indexing="ij")
    nodes = np.stack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel(), Z.ravel()], 1)
    nid = lambda i, j, k: (i * (nt + 1) + j) * (nz + 1) + k          # noqa: E731
    ii, jj, kk = (g.ravel() for g in np.meshgrid(np.arange(nr), np.arange(nt), np.arange(nz), indexing="ij"))
    el = np.stack([nid(ii, jj, kk), nid(ii + 1, jj, kk), nid(ii + 1, jj + 1, kk), nid(ii, jj + 1, kk),
                   nid(ii, jj, kk + 1), nid(ii + 1, jj, kk + 1), nid(ii + 1, jj + 1, kk + 1), nid(ii, jj + 1, kk + 1)], 1)
    return HexMesh(nodes, el, (nr, nt, nz), (2 * ro, 2 * ro, h))


def lame(p: float, ri: float, ro: float, r: float, E: float, nu: float) -> dict:
    """Lamé stresses at radius r and the plane-strain radial displacement (closed ends ignored)."""
    a = p * ri ** 2 / (ro ** 2 - ri ** 2)
    srr = a * (1 - ro ** 2 / r ** 2)
    stt = a * (1 + ro ** 2 / r ** 2)
    ur = (1 + nu) / E * a * ((1 - 2 * nu) * r + ro ** 2 / r)
    return {"s_rr": srr, "s_tt": stt, "u_r": ur}


class PressureVessel(Preset):
    name = "pressure_vessel_asme"
    title = "Vaso de pressão cilíndrico: espessura de casco e tampos pela ASME VIII Div. 1"
    sector = "Estrutural / caldeiraria e processo"
    problem = ("Vaso horizontal de aço carbono SA-516 Gr. 70 (reservatório de ar comprimido / vaso de processo), "
               "casco cilíndrico com tampos elípticos 2:1. Espessuras mínimas pela UG-27 e UG-32, chapa comercial, "
               "MAWP, pressão de teste hidrostático, conferidas pela solução de Lamé e por elementos finitos 3-D.")
    question = ("Qual a espessura de chapa do casco e dos tampos para esta pressão de projeto, qual a MAWP resultante e "
                "a pressão de teste? A tensão real (Lamé/FEM) confirma a fórmula do código?")
    params = (
        Param("p_design", "Pa", 1.0e6, "premissa: 10 bar(g), compressor de ar industrial / vaso de processo", 0.1e6,
              20e6),
        Param("d_inner", "m", 1.5, "premissa: diâmetro interno de 1500 mm", 0.2, 6),
        Param("length", "m", 4.5, "premissa: comprimento do casco entre linhas de tangência", 0.5, 30),
        Param("S", "Pa", 138e6, ASME + " / BPVC II-D Tab. 1A: SA-516 Gr. 70, 20,0 ksi até 650 °F", 50e6, 300e6),
        Param("E_joint", "-", 0.85, ASME + " UW-12: solda de topo dupla com radiografia por amostragem (spot)",
              0.45, 1.0),
        Param("ca", "m", 0.003, "premissa: sobre-espessura de corrosão de 3 mm (serviço de ar úmido)", 0.0, 0.01),
        Param("mill_tol", "m", 0.0003, ASME + " UG-16(c): tolerância de laminação até 0,3 mm (ou 6 %) dispensada",
              0.0, 0.002),
        Param("E_steel", "Pa", 200e9, "aço carbono, BPVC II-D Tab. TM-1 (~ 29,2e6 psi a 70 °F = 201 GPa)", 150e9,
              220e9),
        Param("nu_steel", "-", 0.3, "aço carbono", 0.25, 0.35),
        Param("rho_steel", "kg/m3", 7850.0, "aço carbono", 7700, 8100),
        Param("head", "-", "ellipsoidal_2_1", "tampo elíptico 2:1 (UG-32(d))",
              choices=("ellipsoidal_2_1", "hemispherical", "torispherical")),
        Param("fem_nr", "-", 4, "elementos na espessura do anel FEM (C3D8I); convergência verificada 2-8", 1, 16),
    )
    outputs = (
        Output("t_shell_req", "m", "espessura mínima do casco (UG-27(c)(1)) + corrosão"),
        Output("t_head_req", "m", "espessura mínima do tampo + corrosão"),
        Output("t_shell", "m", "chapa comercial do casco"), Output("t_head", "m", "chapa comercial do tampo"),
        Output("mawp", "Pa", "pressão máxima de trabalho admissível (corroída)"),
        Output("p_test", "Pa", "teste hidrostático UG-99(b): 1,3 x MAWP"),
        Output("hoop_lame", "Pa", "tensão circunferencial máxima de Lamé (corroída, P de projeto)"),
        Output("hoop_fem", "Pa", "a mesma pelo FEM 3-D"),
        Output("mass", "kg", "massa de aço do casco + tampos"),
    )
    validity = ("Casco cilíndrico sob pressão interna com t <= R/2 e P <= 0,385 S E (UG-27(c)(1)); tampos sem "
                "aberturas; temperatura <= 343 °C (650 °F) para o S dado. Bocais, suportes (selas), vento, sismo e "
                "fadiga não verificados.")
    pipeline = ("geometria: casco D_i x L, tampos (2:1, hemisférico ou torisférico)",
                "modelo: fórmulas ASME VIII UG-27 / UG-32 com eficiência de junta e corrosão",
                "solver: espessura -> chapa comercial -> MAWP; Lamé (cilindro espesso); FEM 3-D C3D8I de um quarto "
                "de anel em deformação plana",
                "pós: tensões, MAWP, pressão de teste, massa",
                "decisão: chapas >= requisito; tensão de Lamé <= S E na pressão de projeto")
    library = ("pinneapple_simulation.numerical_solvers.solid_fem.SolidFEM",)
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("t_shell", "t_head", "mass")
    missing = {"surrogate": "as fórmulas do código são instantâneas; o FEM serve só para verificar",
               "reason": "não há caso de vaso de pressão no pinneapple_reason"}

    @staticmethod
    def _plate(t):
        return next((x for x in PLATES if x >= t - 1e-12), float("nan"))

    def _head_t(self, p, P, S, E):
        D, R = p["d_inner"] + 2 * p["ca"], p["d_inner"] / 2 + p["ca"]
        if p["head"] == "ellipsoidal_2_1":
            return P * D / (2 * S * E - 0.2 * P)                       # UG-32(d)
        if p["head"] == "hemispherical":
            return P * R / (2 * S * E - 0.2 * P)                       # UG-32(f)
        return 0.885 * P * D / (S * E - 0.1 * P)                       # UG-32(e): L = D, r = 0.06 D

    def _head_mawp(self, p, t, S, E):
        D, R = p["d_inner"] + 2 * p["ca"], p["d_inner"] / 2 + p["ca"]
        if p["head"] == "ellipsoidal_2_1":
            return 2 * S * E * t / (D + 0.2 * t)
        if p["head"] == "hemispherical":
            return 2 * S * E * t / (R + 0.2 * t)
        return S * E * t / (0.885 * D + 0.1 * t)

    def fem_ring(self, p, P, t_corroded, nr=None, nt=12, nz=1):
        ri = p["d_inner"] / 2 + p["ca"]
        ro = ri + t_corroded
        m = ring_mesh(ri, ro, 0.05 * ri, int(nr or p["fem_nr"]), nt, nz)
        f = SolidFEM(m, p["E_steel"], p["nu_steel"])
        X = m.nodes
        r = np.hypot(X[:, 0], X[:, 1])
        f.fix(np.nonzero(np.abs(X[:, 1]) < 1e-9 * ro)[0], dofs=(1,))
        f.fix(np.nonzero(np.abs(X[:, 0]) < 1e-9 * ro)[0], dofs=(0,))
        f.fix(np.arange(len(X)), dofs=(2,))                            # plane strain
        f.pressure(np.nonzero(np.abs(r - ri) < 1e-9 * ro)[0], P)
        res = f.solve()
        inner = np.nonzero(np.abs(r - ri) < 1e-9 * ro)[0]
        th = np.arctan2(X[inner, 1], X[inner, 0])
        s = res.stress[inner]
        stt = s[:, 0] * np.sin(th) ** 2 + s[:, 1] * np.cos(th) ** 2 - 2 * s[:, 3] * np.sin(th) * np.cos(th)
        ur = (res.u[inner, 0] * np.cos(th) + res.u[inner, 1] * np.sin(th))
        return {"hoop_inner": float(stt.mean()), "ur_inner": float(ur.mean()), "res": res, "ri": ri, "ro": ro,
                "reaction_check": self._reaction(res, P, ri, 0.05 * ri)}

    @staticmethod
    def _reaction(res, P, ri, h):
        X = res.mesh.nodes
        on_y0 = np.abs(X[:, 1]) < 1e-9
        fy = res.reactions[on_y0, 1].sum()                             # symmetry-plane reaction = P ri h
        return abs(abs(fy) - P * ri * h) / (P * ri * h)

    def solve(self, p):
        P, S, E = p["p_design"], p["S"], p["E_joint"]
        R = p["d_inner"] / 2 + p["ca"]                                 # corroded radius
        t_c = P * R / (S * E - 0.6 * P)                                # UG-27(c)(1)
        t_l = P * R / (2 * S * E + 0.4 * P)                            # UG-27(c)(2)
        t_req = max(t_c, t_l) + p["ca"]
        t_shell = self._plate(t_req + p["mill_tol"])
        th_req = self._head_t(p, P, S, E) + p["ca"]
        t_head = self._plate(th_req + p["mill_tol"])
        tc_shell, tc_head = t_shell - p["ca"] - p["mill_tol"], t_head - p["ca"] - p["mill_tol"]
        mawp_shell = min(S * E * tc_shell / (R + 0.6 * tc_shell), 2 * S * E * tc_shell / (R - 0.4 * tc_shell))
        mawp = min(mawp_shell, self._head_mawp(p, tc_head, S, E))
        lm = lame(P, R, R + tc_shell, R, p["E_steel"], p["nu_steel"])
        fem = self.fem_ring(p, P, tc_shell)
        Di, L = p["d_inner"], p["length"]
        shell_m = math.pi * ((Di / 2 + t_shell) ** 2 - (Di / 2) ** 2) * L * p["rho_steel"]
        # head mass: 2:1 head ~ 1.09 D^2 t (blank area ~ 1.084 D^2 for 2:1 incl. short straight flange; a thin-shell
        # area of the semi-ellipsoid is 1.084 (D/2)^2 pi), hemisphere 2 pi R^2 t, torispherical ~ 0.93 D^2 t
        area = {"ellipsoidal_2_1": 1.084 * math.pi * (Di / 2) ** 2, "hemispherical": 2 * math.pi * (Di / 2) ** 2,
                "torispherical": 0.9286 * Di ** 2}[p["head"]]
        head_m = 2 * area * t_head * p["rho_steel"]
        return {"t_shell_req": t_req, "t_head_req": th_req, "t_shell": t_shell, "t_head": t_head, "mawp": mawp,
                "p_test": 1.3 * mawp, "hoop_lame": lm["s_tt"], "hoop_fem": fem["hoop_inner"],
                "ur_lame": lm["u_r"], "ur_fem": fem["ur_inner"], "hoop_asme": P * (R + 0.6 * tc_shell) / tc_shell,
                "mass": shell_m + head_m, "fem_reaction_err": fem["reaction_check"],
                "t_over_r": tc_shell / R, "p_over_se": P / (S * E), "_fields": {"fem": fem}}

    def checks(self, p, out):
        return [check("UG-27 limits: t <= R/2 and P <= 0.385 S E", out["t_over_r"] <= 0.5 and out["p_over_se"] <=
                      0.385, value=out["p_over_se"], limit=0.385, source=ASME + " UG-27(c)(1)"),
                check("FEM equilibrium: symmetry reaction = P r h", out["fem_reaction_err"] < 1e-6,
                      value=out["fem_reaction_err"], limit=1e-6, source="SolidFEM reactions"),
                check("FEM bore displacement = Lamé (plane strain)", abs(out["ur_fem"] - out["ur_lame"]) /
                      out["ur_lame"] < 0.01, value=abs(out["ur_fem"] - out["ur_lame"]) / out["ur_lame"], limit=0.01,
                      source=LAME),
                check("MAWP >= design pressure", out["mawp"] >= p["p_design"], value=out["mawp"], limit=p["p_design"])]

    def decide(self, p, out):
        ok = out["mawp"] >= p["p_design"] and out["hoop_lame"] <= p["S"] * p["E_joint"] * 1.0001
        return {"verdict": "OK" if ok else "NOT OK",
                "reasons": [f"casco {1e3 * out['t_shell']:.0f} mm (mín. {1e3 * out['t_shell_req']:.2f} mm), tampo "
                            f"{1e3 * out['t_head']:.0f} mm (mín. {1e3 * out['t_head_req']:.2f} mm)",
                            f"MAWP {out['mawp'] / 1e6:.3f} MPa, teste hidrostático {out['p_test'] / 1e6:.3f} MPa",
                            f"tensão circunferencial (Lamé) {out['hoop_lame'] / 1e6:.1f} MPa vs S E = "
                            f"{p['S'] * p['E_joint'] / 1e6:.1f} MPa"]}

    def validate(self):
        p = self.defaults()
        o = self.run().outputs
        rows = [self._row("tensão circunferencial no furo: FEM 3-D C3D8I vs Lamé", "hoop", "Pa", o["hoop_fem"],
                          o["hoop_lame"], 0.02, "analytic", LAME),
                self._row("deslocamento radial no furo: FEM vs Lamé (deformação plana)", "u_r", "m", o["ur_fem"],
                          o["ur_lame"], 0.005, "analytic", LAME),
                self._row("fórmula UG-27(c)(1) vs tensão máxima de Lamé", "hoop", "Pa", o["hoop_asme"], o["hoop_lame"],
                          0.02, "analytic", ASME + " App. 1-1: a fórmula aproxima Lamé para t <= R/2")]
        thick = {**p, "d_inner": 0.3, "p_design": 15e6}                # a thick case (t/R ~ 0.2)
        R = 0.15 + p["ca"]
        tt = 0.04
        rows.append(self._row("caso espesso (t/R 0,26): UG-27 vs Lamé", "hoop", "Pa",
                              thick["p_design"] * (R + 0.6 * tt) / tt, lame(thick["p_design"], R, R + tt, R, 2e11,
                                                                            0.3)["s_tt"], 0.03, "analytic",
                              ASME + " UG-27 / Lamé"))
        ns = (1, 2, 4, 8)
        tc = o["t_shell"] - p["ca"] - p["mill_tol"]
        hs = [self.fem_ring(p, p["p_design"], tc, nr=n)["hoop_inner"] for n in ns]
        rel = abs(hs[-1] - hs[-2]) / abs(hs[-1])
        conv = {"parameter": "elementos na espessura", "values": list(ns), "hoop": hs, "rel_change_last": rel,
                "pass": rel < 0.01}
        return {"preset": self.name, "references": rows, "convergence": conv,
                "all_pass": all(r["pass"] for r in rows) and conv["pass"]}

    @staticmethod
    def _row(case, output, unit, comp, ref, tol, kind, source):
        err = abs(comp - ref) / abs(ref)
        return {"case": case, "output": output, "unit": unit, "computed": comp, "reference": ref, "rel_error": err,
                "tol": tol, "pass": bool(err <= tol), "kind": kind, "source": source, "params": {}}

    def optimize(self, **kw):
        """Lightest vessel for the duty: head type x joint efficiency (radiography costs money, so the mass gain
        of E = 1.0 is reported next to E = 0.85 and 0.70)."""
        self.resolve(**kw)
        rows = []
        for head in ("ellipsoidal_2_1", "hemispherical", "torispherical"):
            for E in (0.70, 0.85, 1.0):
                r = self.run(**{**kw, "head": head, "E_joint": E})
                rows.append({"head": head, "E_joint": E, "t_shell": r.outputs["t_shell"], "t_head": r.outputs["t_head"],
                             "mass": r.outputs["mass"], "mawp": r.outputs["mawp"], "ok": r.decision["verdict"] == "OK"})
        best = min((r for r in rows if r["ok"]), key=lambda r: r["mass"])
        base = self.run(**kw).outputs["mass"]
        return {"objective": "massa mínima de aço (casco + tampos) com MAWP >= P de projeto",
                "best_head": best["head"], "best_E_joint": best["E_joint"], "best_mass_kg": best["mass"],
                "base_mass_kg": base, "table": rows}

    def invert(self, true_loss: float = 0.0025, noise_ue: float = 5.0, seed: int = 0, **kw):
        """Remaining wall from hoop strain gauges read during a hydrostatic test (four pressure steps up to the
        test pressure, gauge noise 5 microstrain): the corrosion loss of an old vessel. Synthetic truth: 2.5 mm
        lost from the nominal plate. Model: Lamé outer-surface hoop strain, closed ends (axial stress P ri^2 /
        (ro^2 - ri^2))."""
        p = self.resolve(**kw)
        o = self.run(**kw).outputs
        ri0 = p["d_inner"] / 2
        P = np.array([0.25, 0.5, 0.75, 1.0]) * o["p_test"]

        def strain(t):
            ri, ro = ri0, ri0 + t
            a = P * ri ** 2 / (ro ** 2 - ri ** 2)
            s_tt = 2 * a                                                  # at r = ro
            s_zz = a
            return (s_tt - p["nu_steel"] * s_zz) / p["E_steel"] * 1e6

        t_true = o["t_shell"] - true_loss
        rng = np.random.default_rng(seed)
        y = strain(t_true) + rng.normal(0, noise_ue, len(P))
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda t: ((strain(t) - y) ** 2).sum(), bounds=(0.2 * t_true, 3 * t_true),
                              method="bounded", options={"xatol": 1e-9})
        t = float(res.x)
        J = (strain(t + 1e-7) - strain(t - 1e-7)) / 2e-7
        sd = float(noise_ue / math.sqrt(J @ J))
        return {"description": "Espessura remanescente do casco a partir de extensômetros circunferenciais lidos no "
                               "teste hidrostático (4 patamares até 1,3 MAWP). Verdade sintética: chapa nominal menos "
                               f"{1e3 * true_loss:.1f} mm de corrosão; ruído de {noise_ue} microdeformações.",
                "parameter": "espessura remanescente", "unit": "m", "true": t_true, "recovered": t,
                "ci95": [t - 1.96 * sd, t + 1.96 * sd], "rel_error": abs(t - t_true) / t_true,
                "true_inside_ci95": bool(abs(t - t_true) <= 1.96 * sd), "noise": f"{noise_ue} microstrain",
                "how": "mínimos quadrados (Brent) sobre Lamé com tampos fechados; IC por sensibilidade",
                "remaining_mawp": self.run(**{**kw, "ca": 0.0}).outputs["mawp"] * 0 +
                p["S"] * p["E_joint"] * (t - p["mill_tol"]) / (ri0 + 0.6 * (t - p["mill_tol"]))}

    def variants(self):
        return {"2:1 E=0,85 (base)": {}, "hemisférico": {"head": "hemispherical"},
                "torisférico": {"head": "torispherical"}, "radiografia total E=1,0": {"E_joint": 1.0},
                "sem radiografia E=0,70": {"E_joint": 0.70}, "16 bar": {"p_design": 1.6e6}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fem = result.fields["fem"]
        res = fem["res"]
        X = res.mesh.nodes
        r = np.hypot(X[:, 0], X[:, 1])
        th = np.arctan2(X[:, 1], X[:, 0])
        s = res.stress
        stt = s[:, 0] * np.sin(th) ** 2 + s[:, 1] * np.cos(th) ** 2 - 2 * s[:, 3] * np.sin(th) * np.cos(th)
        srr = s[:, 0] * np.cos(th) ** 2 + s[:, 1] * np.sin(th) ** 2 + 2 * s[:, 3] * np.sin(th) * np.cos(th)
        p = result.params
        rr = np.linspace(fem["ri"], fem["ro"], 50)
        lm = lame(p["p_design"], fem["ri"], fem["ro"], rr, p["E_steel"], p["nu_steel"])
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot(1e3 * (rr - fem["ri"]), lm["s_tt"] / 1e6, "C0", label="Lamé, circunferencial")
        ax.plot(1e3 * (rr - fem["ri"]), lm["s_rr"] / 1e6, "C1", label="Lamé, radial")
        ax.plot(1e3 * (r - fem["ri"]), stt / 1e6, "C0o", ms=3, label="FEM C3D8I")
        ax.plot(1e3 * (r - fem["ri"]), srr / 1e6, "C1o", ms=3)
        ax.axhline(p["S"] * p["E_joint"] / 1e6, color="r", ls="--", lw=1, label="S E")
        ax.set(xlabel="profundidade a partir do furo [mm]", ylabel="tensão [MPa]",
               title="Casco corroído na pressão de projeto")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(out_dir, "vessel_stress.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from ..viz3d import Scene, Surface
        p, o = result.params, result.outputs
        Ri = p["d_inner"] / 2
        L = p["length"]
        th = np.linspace(0, 2 * np.pi, 64)
        P = p["p_design"]
        hoop = o["hoop_lame"] / 1e6
        # body of revolution about x: shell + two heads (outer surface), membrane stress for colour
        if p["head"] == "hemispherical":
            phi = np.linspace(0, np.pi / 2, 20)
            hx, hr = Ri * np.sin(phi), Ri * np.cos(phi)
            sig_h = np.full_like(phi, P * Ri / (2 * (o["t_head"] - p["ca"])) / 1e6)
        else:
            phi = np.linspace(0, np.pi / 2, 20)
            a, b = Ri, Ri / 2                                          # 2:1 (torispherical drawn the same)
            hx, hr = b * np.sin(phi), a * np.cos(phi)
            # meridional membrane stress of an ellipsoid, P R2 / (2 t) with R2 the second principal radius
            R2 = a ** 2 / np.sqrt(a ** 2 * np.cos(phi) ** 2 + b ** 2 * np.sin(phi) ** 2) * 1.0
            sig_h = P * R2 / (2 * (o["t_head"] - p["ca"])) / 1e6
        xs = np.r_[-hx[::-1], np.linspace(0, L, 40), L + hx]
        rs = np.r_[hr[::-1], np.full(40, Ri), hr]
        sg = np.r_[sig_h[::-1], np.full(40, hoop), sig_h]
        X, TH = np.meshgrid(xs, th, indexing="ij")
        Rr = np.repeat(rs[:, None], len(th), 1)
        V = np.c_[X.ravel(), (Rr * np.cos(TH)).ravel(), (Rr * np.sin(TH)).ravel()]
        idx = np.arange(V.shape[0]).reshape(X.shape)
        q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        F = np.r_[q[:, [0, 2, 1]], q[:, [0, 3, 2]]]
        body = Surface("vessel", V, F, "steel", {"sigma": np.repeat(sg, len(th))})
        # the FEM quarter ring next to it (deformed x 300), von Mises
        fem = result.fields["fem"]["res"]
        Vf, Ff, fl = fem.surface(scale=300.0)
        ring = Surface("fem_ring", Vf + [L / 2, 0, 1.6 * Ri], Ff, "steel",
                       {"sigma": fl["von_mises"] / 1e6})
        return Scene([body, ring], title=self.title,
                     labels={"sigma": "tensão de membrana / von Mises FEM [MPa]"})
