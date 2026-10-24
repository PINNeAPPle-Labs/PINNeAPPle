"""Insulation of a steam line: heat loss, surface temperature and the economic thickness (ISO 12241, VDI 2055)."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_physics.closed_form.pipe_insulation import (
    economic_thickness,
    heat_loss,
    radial_fd,
    surface_coefficient,
)

from ..core import Output, Param, Preset, check

ISO = "ISO 12241:2008 sec. 4 (método de cálculo da perda de calor)"
STD_T = tuple(x / 1000 for x in (0, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200))


class SteamLineInsulation(Preset):
    name = "steam_pipe_insulation"
    title = "Isolamento térmico de linha de vapor: perda, temperatura de superfície e espessura econômica"
    sector = "Processo e energia / utilidades"
    problem = ("Linha de vapor saturado de 10 bar(a) em tubo DN150 ao ar livre, isolada com lã de rocha e revestida "
               "de alumínio. Perda de calor por metro, temperatura da superfície e a espessura que minimiza o custo "
               "anual (energia perdida + isolamento instalado anualizado).")
    question = ("Qual espessura de isolamento minimiza o custo anual por metro, e qual a menor que mantém a superfície "
                "abaixo de 60 °C (proteção pessoal)? Quanto se perde sem isolamento?")
    params = (
        Param("d_pipe", "m", 0.1683, "DN150 (6\"), diâmetro externo 168,3 mm (ASME B36.10M / EN 10220)", 0.02, 1.2),
        Param("t_fluid", "degC", 179.9, "vapor saturado a 10 bar(a) (IAPWS-IF97: 179,88 °C)", 40, 600),
        Param("t_amb", "degC", 10.0, "premissa: média anual do ar externo de clima temperado", -30, 45),
        Param("wind", "m/s", 3.0, "premissa: vento médio de 3 m/s ao ar livre", 0, 20),
        Param("thickness", "m", 0.06, "premissa: 60 mm, espessura comum em linhas de vapor de média pressão", 0,
              0.3),
        Param("k0", "W/m K", 0.036, "lã de rocha: lambda(T) = k0 + k1 T; ordem dos valores declarados de seções "
              "de lã de rocha (EN 14303): ~0,041 a 50 °C, ~0,046 a 100 °C, ~0,056 a 200 °C", 0.015, 0.1),
        Param("k1", "W/m K2", 1.0e-4, "inclinação da mesma curva lambda(T)", 0, 5e-4),
        Param("eps", "-", 0.05, "alumínio brilhante: emissividade 0,05 (ISO 12241 Tab. 4)", 0.03, 0.95),
        Param("hours", "h/yr", 8000.0, "premissa: operação contínua com paradas de manutenção", 500, 8760),
        Param("energy_price", "EUR/kWh", 0.05, "premissa: gás natural industrial, ordem de grandeza Eurostat "
              "nrg_pc_203 (consumidores não-residenciais) 2023-2024", 0.005, 0.5),
        Param("efficiency", "-", 0.88, "premissa: caldeira + distribuição, 88 %", 0.5, 1.0),
        Param("cost_per_m3", "EUR/m3", 900.0, "premissa: lã de rocha instalada (material + mão de obra) por m3", 100,
              5000),
        Param("cost_per_m2", "EUR/m2", 40.0, "premissa: revestimento de alumínio instalado por m2", 0, 300),
        Param("rate", "-", 0.08, "premissa: custo de capital 8 % ao ano (VDI 2055-1 sec. 6)", 0.0, 0.3),
        Param("years", "yr", 10.0, "premissa: vida econômica do isolamento 10 anos", 1, 40),
        Param("t_surface_max", "degC", 60.0, "premissa usual de especificação para proteção pessoal (ISO 13732-1: "
              "limiar de queimadura de metal ~60 °C para contato de ~5 s)", 40, 80),
    )
    outputs = (
        Output("q", "W/m", "perda de calor por metro com a espessura dada"),
        Output("t_surface", "degC", "temperatura da superfície do revestimento"),
        Output("q_bare", "W/m", "perda do tubo nu (pintado, eps 0,9)"),
        Output("saving", "-", "fração da perda evitada"),
        Output("annual_cost", "EUR/m/yr", "custo anual por metro com a espessura dada"),
        Output("t_economic", "m", "espessura econômica (entre as espessuras comerciais)"),
        Output("t_safe", "m", "menor espessura com superfície <= limite"),
    )
    validity = ("Regime permanente, tubo horizontal, isolamento homogêneo e seco com lambda linear em T, sem pontes "
                "térmicas (suportes, flanges e válvulas acrescentam tipicamente 10-25 %, VDI 2055), resistência da "
                "parede e do filme interno desprezadas (lado seguro).")
    pipeline = ("geometria: tubo DN, camada de isolamento, revestimento",
                "modelo: condução radial com lambda(T) (transformada de Kirchhoff), radiação + convecção mista (Churchill-Chu e Churchill-Bernstein)",
                "solver: heat_loss (bisseção da temperatura de superfície); radial_fd (volumes finitos) para checar",
                "pós: perda W/m, temperatura de superfície, custo anual por espessura comercial",
                "decisão: espessura >= max(econômica, segura)")
    library = ("pinneapple_physics.closed_form.pipe_insulation.heat_loss",
               "pinneapple_physics.closed_form.pipe_insulation.economic_thickness",
               "pinneapple_physics.closed_form.pipe_insulation.radial_fd")
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("q", "t_surface", "annual_cost")
    missing = {"surrogate": "forma fechada (< 1 ms por avaliação): um surrogate não economiza tempo",
               "reason": "o caso do telhado (roof) cobre isolamento de forro, não de tubulação"}

    def _econ(self, p):
        return economic_thickness(p["d_pipe"], p["t_fluid"], p["t_amb"], STD_T, k0=p["k0"], k1=p["k1"],
                                  eps=p["eps"], wind=p["wind"], hours=p["hours"], energy_price=p["energy_price"],
                                  efficiency=p["efficiency"], cost_per_m3=p["cost_per_m3"],
                                  cost_per_m2=p["cost_per_m2"], rate=p["rate"], years=p["years"],
                                  t_surface_max=p["t_surface_max"])

    def solve(self, p):
        h = heat_loss(p["d_pipe"], p["thickness"], p["t_fluid"], p["t_amb"], k0=p["k0"], k1=p["k1"], eps=p["eps"],
                      wind=p["wind"])
        bare = heat_loss(p["d_pipe"], 0.0, p["t_fluid"], p["t_amb"], k0=p["k0"], eps=0.9, wind=p["wind"])
        e = self._econ({**p})
        one = economic_thickness(p["d_pipe"], p["t_fluid"], p["t_amb"], (p["thickness"],), k0=p["k0"], k1=p["k1"],
                                 eps=p["eps"], wind=p["wind"], hours=p["hours"], energy_price=p["energy_price"],
                                 efficiency=p["efficiency"], cost_per_m3=p["cost_per_m3"],
                                 cost_per_m2=p["cost_per_m2"], rate=p["rate"], years=p["years"])["rows"][0]
        fd = radial_fd(p["d_pipe"], p["thickness"], p["t_fluid"], h["t_surface"], k0=p["k0"], k1=p["k1"]) \
            if p["thickness"] > 0 else None
        return {"q": h["q"], "t_surface": h["t_surface"], "h_surface": h["h"], "regime": h["regime"], "re": h["re"],
                "q_bare": bare["q"], "saving": 1 - h["q"] / bare["q"], "annual_cost": one["total"],
                "t_economic": e["economic"]["thickness"],
                "t_safe": e["thinnest_safe"]["thickness"] if e["thinnest_safe"] else float("nan"),
                "q_fd": fd["q_inner"] if fd else h["q"], "fd_balance": abs(fd["q_inner"] - fd["q_outer"]) / fd["q_inner"]
                if fd else 0.0, "_fields": {"econ": e, "fd": fd}}

    def checks(self, p, out):
        rel = abs(out["q_fd"] - out["q"]) / out["q"]
        p["d_pipe"] + 2 * p["thickness"]
        return [check("Kirchhoff closed form = finite volumes with k(T)", rel < 1e-4, value=rel, limit=1e-4,
                      source="radial_fd (Picard, 400 volumes)"),
                check("FV energy balance: inner flow = outer flow", out["fd_balance"] < 1e-8,
                      value=out["fd_balance"], limit=1e-8),
                check("surface between ambient and fluid", p["t_amb"] <= out["t_surface"] <= p["t_fluid"]),
                check("Churchill-Bernstein range Re Pr > 0.2", out["re"] * 0.7 > 0.2 if out["regime"] == "forced"
                      else None, value=out["re"], source="Incropera eq. 7.54"),
                check("insulation thicker than the critical radius (adds resistance)",
                      p["thickness"] == 0 or out["q"] < out["q_bare"], value=out["q"], limit=out["q_bare"])]

    def decide(self, p, out):
        need = max(out["t_economic"], out["t_safe"] if not math.isnan(out["t_safe"]) else 0)
        ok = p["thickness"] >= need - 1e-9 and out["t_surface"] <= p["t_surface_max"]
        return {"verdict": "OK" if ok else "NOT OK",
                "reasons": [f"perda {out['q']:.1f} W/m (nu: {out['q_bare']:.0f} W/m, {100 * out['saving']:.1f} % evitado),"
                            f" superfície {out['t_surface']:.1f} °C",
                            f"espessura econômica {1e3 * out['t_economic']:.0f} mm; menor segura (<= "
                            f"{p['t_surface_max']:.0f} °C) {1e3 * out['t_safe']:.0f} mm; dada {1e3 * p['thickness']:.0f} mm"]}

    def validate(self):
        p = self.defaults()
        rows = []
        for t in (0.03, 0.06, 0.12):
            h = heat_loss(p["d_pipe"], t, p["t_fluid"], p["t_amb"], k0=p["k0"], k1=p["k1"], eps=p["eps"], wind=p["wind"])
            fd = radial_fd(p["d_pipe"], t, p["t_fluid"], h["t_surface"], k0=p["k0"], k1=p["k1"])
            rows.append(self._row(f"perda com {1e3 * t:.0f} mm: Kirchhoff vs volumes finitos com k(T)", "q", "W/m",
                                  h["q"], fd["q_inner"], 1e-4, "independent_solver",
                                  "radial_fd: equação de calor radial com k(T) resolvida numericamente"))
        # constant k: the textbook log formula (Incropera eq. 3.27 with the outer film)
        h = heat_loss(p["d_pipe"], 0.06, p["t_fluid"], p["t_amb"], k0=0.045, k1=0.0, eps=p["eps"], wind=p["wind"])
        de = p["d_pipe"] + 0.12
        r_tot = math.log(de / p["d_pipe"]) / (2 * math.pi * 0.045) + 1 / (h["h"] * math.pi * de)
        rows.append(self._row("k constante: resistências em série (Incropera eq. 3.27)", "q", "W/m", h["q"],
                              (p["t_fluid"] - p["t_amb"]) / r_tot, 1e-6, "analytic",
                              "Incropera et al., Fundamentals of Heat and Mass Transfer, eq. 3.27"))
        # convection coefficients against independent correlations (Incropera Table A.4 air at the film)
        from pinneapple_physics.closed_form.pipe_insulation import air
        ts = 40.0
        a = air(0.5 * (ts + p["t_amb"]))
        cb = surface_coefficient(ts, p["t_amb"], de, 0.0, p["wind"])
        re = cb["re"]
        C, m = (0.193, 0.618) if re < 40000 else (0.027, 0.805)
        hil = C * re ** m * a["pr"] ** (1 / 3) * a["k"] / de
        rows.append(self._row(f"h forçado (Re {re:.0f}): Churchill-Bernstein misto vs Hilpert", "h_cv", "W/m2 K",
                              cb["h_cv"], hil, 0.15, "independent_solver",
                              "Hilpert (1933) com as constantes da Tabela 7.2 de Incropera (incerteza ~±20 %)"))
        nat = surface_coefficient(ts, p["t_amb"], de, 0.0, 0.0)
        mca = 1.32 * ((ts - p["t_amb"]) / de) ** 0.25
        rows.append(self._row("h natural (sem vento): Churchill-Chu vs McAdams 1,32 (dT/D)^0,25", "h_cv", "W/m2 K",
                              nat["h_cv"], mca, 0.15, "independent_solver",
                              "McAdams, Heat Transmission (1954): cilindro horizontal laminar em ar"))
        return {"preset": self.name, "references": rows, "convergence": self._conv(p), "all_pass": all(
            r["pass"] for r in rows)}

    def _conv(self, p):
        h = heat_loss(p["d_pipe"], p["thickness"], p["t_fluid"], p["t_amb"], k0=p["k0"], k1=p["k1"], eps=p["eps"],
                      wind=p["wind"])
        ns = (25, 50, 100, 200)
        qs = [radial_fd(p["d_pipe"], p["thickness"], p["t_fluid"], h["t_surface"], k0=p["k0"], k1=p["k1"], n=n)
              ["q_inner"] for n in ns]
        rel = abs(qs[-1] - qs[-2]) / qs[-1]
        return {"parameter": "volumes radiais", "values": list(ns), "q": qs, "rel_change_last": rel, "pass": rel < 1e-4}

    @staticmethod
    def _row(case, output, unit, comp, ref, tol, kind, source):
        err = abs(comp - ref) / abs(ref)
        return {"case": case, "output": output, "unit": unit, "computed": comp, "reference": ref, "rel_error": err,
                "tol": tol, "pass": bool(err <= tol), "kind": kind, "source": source, "params": {}}

    def optimize(self, **kw):
        p = self.resolve(**kw)
        e = self._econ(p)
        best = e["economic"]
        need = max(best["thickness"], e["thinnest_safe"]["thickness"] if e["thinnest_safe"] else 0)
        ver = self.run(**{**kw, "thickness": need})
        base = self.run(**kw)
        return {"objective": "custo anual mínimo por metro (energia + isolamento anualizado), superfície <= limite",
                "economic_thickness_m": best["thickness"], "economic_annual_cost": best["total"],
                "recommended_thickness_m": need, "verified_t_surface": ver.outputs["t_surface"],
                "verified_annual_cost": ver.outputs["annual_cost"], "base_annual_cost": base.outputs["annual_cost"],
                "bare_annual_energy_cost": e["rows"][0]["energy_cost"],
                "payback_years_vs_bare": (ver.outputs["annual_cost"] and
                                          e["rows"][[r["thickness"] for r in e["rows"]].index(need)]["capital_cost"]
                                          / (e["rows"][0]["energy_cost"] -
                                             e["rows"][[r["thickness"] for r in e["rows"]].index(need)]["energy_cost"]))}

    def invert(self, true_factor: float = 1.6, noise_c: float = 0.5, seed: int = 0, **kw):
        """Effective conductivity of aged or wet insulation from infrared surface temperatures, 5 readings along the line at different wind speeds, IR noise 0.5 K. Synthetic truth: k x1.6
        (moisture); cladding emissivity known."""
        p = self.resolve(**kw)
        winds = np.array([0.5, 1.5, 3.0, 5.0, 8.0])

        def ts(f):
            return np.array([heat_loss(p["d_pipe"], p["thickness"], p["t_fluid"], p["t_amb"], k0=f * p["k0"],
                                       k1=f * p["k1"], eps=p["eps"], wind=w)["t_surface"] for w in winds])

        rng = np.random.default_rng(seed)
        y = ts(true_factor) + rng.normal(0, noise_c, len(winds))
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda f: ((ts(f) - y) ** 2).sum(), bounds=(0.3, 5.0), method="bounded",
                              options={"xatol": 1e-8})
        f = float(res.x)
        J = (ts(f + 1e-4) - ts(f - 1e-4)) / 2e-4
        sd = float(noise_c / math.sqrt(J @ J))
        return {"description": "Fator de condutividade efetiva do isolamento envelhecido/úmido a partir de termografia "
                               "da superfície (5 leituras, ventos de 0,5 a 8 m/s). Medição sintética com fator real "
                               f"{true_factor} e ruído de {noise_c} K.", "parameter": "k_eff / k_novo",
                "true": true_factor, "recovered": f, "ci95": [f - 1.96 * sd, f + 1.96 * sd],
                "rel_error": abs(f - true_factor) / true_factor,
                "true_inside_ci95": bool(abs(f - true_factor) <= 1.96 * sd), "noise": f"{noise_c} K",
                "how": "mínimos quadrados (Brent) sobre o modelo de perda, IC por sensibilidade"}

    def variants(self):
        return {"40 mm": {"thickness": 0.04}, "60 mm (base)": {}, "100 mm": {"thickness": 0.10},
                "60 mm, chapa pintada eps 0,9": {"eps": 0.9}, "60 mm, abrigado (sem vento)": {"wind": 0.0},
                "60 mm, vapor 20 bar (212 °C)": {"t_fluid": 212.4}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = result.fields["econ"]
        rows = e["rows"][1:]
        t = [1e3 * r["thickness"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(t, [r["energy_cost"] for r in rows], label="energia perdida")
        ax[0].plot(t, [r["annual_capital"] for r in rows], label="isolamento anualizado")
        ax[0].plot(t, [r["total"] for r in rows], "k", lw=2, label="total")
        ax[0].axvline(1e3 * e["economic"]["thickness"], color="g", ls="--", lw=1)
        ax[0].set(xlabel="espessura [mm]", ylabel="EUR / m / ano", title="Espessura econômica")
        ax[0].legend(fontsize=8)
        fd = result.fields["fd"]
        if fd is not None:
            ax[1].plot(1e3 * (fd["r"] - fd["r"][0]), fd["T"])
            ax[1].set(xlabel="distância da parede do tubo [mm]", ylabel="T [°C]", title="Perfil radial (volumes finitos)")
        fig.tight_layout()
        path = os.path.join(out_dir, "insulation.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from ..viz3d import Scene, Surface, tube
        p = result.params
        fd = result.fields["fd"]
        L = 2.0
        s = np.linspace(0, L, 30)
        ri, ro = p["d_pipe"] / 2, p["d_pipe"] / 2 + p["thickness"]
        surf = []
        V, F, _ = tube(np.c_[s, 0 * s, 0 * s], ri, 40)
        surf.append(Surface("pipe", V, F, "steel", {"temperature": np.full(len(V), p["t_fluid"])}))
        sc = s[s <= 0.7 * L]                                     # cut-away: cladding over the last 30 % only
        if p["thickness"] > 0 and fd is not None:
            V, F, _ = tube(np.c_[s[s >= 0.6 * L], 0 * s[s >= 0.6 * L], 0 * s[s >= 0.6 * L]], ro, 48)
            surf.append(Surface("cladding", V, F, "aluminium", {"temperature": np.full(len(V), result.outputs["t_surface"])}))
            # insulation end face: annulus at x = 0.6 L coloured by the radial temperature
            th = np.linspace(0, 2 * np.pi, 49)
            r = fd["r"][::10]
            T = fd["T"][::10]
            R, TH = np.meshgrid(r, th, indexing="ij")
            Va = np.c_[np.full(R.size, 0.6 * L), (R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel()]
            idx = np.arange(R.size).reshape(R.shape)
            q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
            Fa = np.r_[q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
            surf.append(Surface("insulation_face", Va, np.r_[Fa, Fa[:, ::-1]], "grey",
                                {"temperature": np.repeat(T, len(th))}))
        del sc
        return Scene(surf, title=self.title, labels={"temperature": "temperatura [°C]"})
