"""Water distribution network: pressures, flows and least-cost pipe sizing (global gradient algorithm)."""
from __future__ import annotations

import numpy as np

from pinneapple_simulation.numerical_solvers.pipe_network import (
    TWO_LOOP_COSTS,
    optimize_diameters,
    solve_network,
    two_loop_network,
)

from ..core import Output, Param, Preset, Reference, check

AS77 = "Alperovits & Shamir (1977), Water Resources Research 13(6):885-900 (rede de duas malhas)"
SW97 = "Savic & Walters (1997), J. Water Resour. Plann. Manage. 123(2):67-77: custo mínimo US$ 419.000"
EPA = "EPANET 2.2 (Rossman et al. 2020) via WNTR 1.5.0, accuracy 1e-6, rodado nesta máquina"
INCH = 0.0254
OPT_IN = (18, 10, 16, 4, 16, 10, 10, 1)

# EPANET 2.2 pressures (m) of the two-loop network with the default diameters, run here (scripts in the preset
# report; re-run live by validate() when wntr is installed)
EPANET_HW = {"2": 53.2467, "3": 30.4625, "4": 43.4492, "5": 33.8033, "6": 30.445, "7": 30.5523}
EPANET_DW = {"2": 52.8643, "3": 28.797, "4": 42.9119, "5": 31.7435, "6": 29.8895, "7": 29.815}


class WaterNetwork(Preset):
    name = "water_distribution_network"
    title = "Rede de distribuição de água: pressões, vazões e dimensionamento de menor custo"
    sector = "Fluidos / saneamento"
    problem = ("Rede malhada de distribuição (duas malhas, um reservatório, seis nós de consumo, oito tubos de "
               "1 km; rede de referência de Alperovits & Shamir 1977). Resolve pressões e vazões em regime "
               "permanente (Hazen-Williams ou Darcy-Weisbach/Colebrook) e dimensiona os diâmetros de menor custo.")
    question = ("Todos os nós têm pressão mínima de serviço (30 m) na demanda de projeto? Qual o conjunto de "
                "diâmetros comerciais de menor custo que garante isso?")
    params = (
        Param("reservoir_head", "m", 210.0, AS77 + ": nível do reservatório 210 m", 160, 300),
        Param("demand_factor", "-", 1.0, "1,0 = demandas de projeto de " + AS77 + " (100-330 m3/h por nó)", 0.1, 2.5),
        Param("headloss", "-", "HW", "Hazen-Williams como em " + AS77 + "; 'DW' = Darcy-Weisbach/Colebrook",
              choices=("HW", "DW")),
        Param("hw_c", "-", 130.0, AS77 + ": C = 130 (tubo novo)", 60, 150),
        Param("roughness_dw", "m", 2.6e-4, "ferro fundido, e = 0,26 mm (White, Fluid Mechanics, Tabela 6.1)", 1e-6,
              5e-3),
        Param("p_min", "m", 30.0, AS77 + ": pressão mínima de serviço 30 m", 5, 60),
        *(Param(f"d{i + 1}", "m", d * INCH, f"tubo {i + 1}: {d} pol, solução de menor custo de " + SW97, 0.0254,
                0.6096) for i, d in enumerate(OPT_IN)),
    )
    outputs = (
        Output("min_pressure", "m", "menor pressão nos nós"),
        Output("critical_node", "-", "nó com a menor pressão"),
        Output("supply", "m3/s", "vazão fornecida pelo reservatório"),
        Output("max_velocity", "m/s", "maior velocidade nos tubos"),
        Output("max_gradient", "m/km", "maior perda de carga unitária"),
        Output("cost", "US$", "custo dos tubos (tabela de " + AS77 + ")"),
        Output("p_node3", "m", "pressão no nó 3"),
        Output("p_node6", "m", "pressão no nó 6"),
    )
    validity = ("Regime permanente, escoamento incompressível, sem bombas nem válvulas, perdas localizadas "
                "desprezadas (tubos de 1 km). Hazen-Williams: água à temperatura ambiente, regime turbulento.")
    pipeline = ("topologia e cotas: two_loop_network(diâmetros, headloss)",
                "modelo: perda de carga Hazen-Williams (constante EPANET 10,667) ou Darcy-Weisbach com Colebrook",
                "solver: solve_network (algoritmo do gradiente global, Todini & Pilati 1988)",
                "pós: pressões nos nós, velocidades, gradientes, custo",
                "decisão: pressão mínima >= p_min; velocidade <= 3 m/s (faixa usual de projeto)",
                "otimização: optimize_diameters (algoritmo genético, 14 diâmetros comerciais)")
    library = ("pinneapple_simulation.numerical_solvers.pipe_network.solve_network",
               "pinneapple_simulation.numerical_solvers.pipe_network.optimize_diameters")
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("min_pressure", "cost", "max_velocity")
    missing = {"surrogate": "o solver leva ~5 ms por rede; um surrogate não economiza tempo e só acrescentaria erro"}

    def _net(self, p):
        d = [p[f"d{i + 1}"] for i in range(8)]
        r = p["hw_c"] if p["headloss"] == "HW" else p["roughness_dw"]
        net = two_loop_network(headloss=p["headloss"], roughness=r).with_diameters(d)
        net.reservoirs["1"] = p["reservoir_head"]
        net.junctions = {k: (z, q * p["demand_factor"]) for k, (z, q) in net.junctions.items()}
        return net

    @staticmethod
    def _cost(d_m):
        inch = [round(x / INCH) for x in d_m]
        if not all(i in TWO_LOOP_COSTS for i in inch):
            return float("nan")
        return float(sum(TWO_LOOP_COSTS[i] * 1000.0 for i in inch))

    def solve(self, p):
        net = self._net(p)
        r = solve_network(net)
        ph = r["pressure_head"]
        crit = min(ph, key=ph.get)
        grad = {k: 1000 * abs(v) / L for k, v, L in zip(r["pipes"], r["headloss"].values(), (x[3] for x in net.pipes), strict=False)}
        return {"min_pressure": ph[crit], "critical_node": crit, "supply": r["supply"],
                "max_velocity": max(abs(v) for v in r["velocity"].values()), "max_gradient": max(grad.values()),
                "cost": self._cost([x[4] for x in net.pipes]), "p_node3": ph["3"], "p_node6": ph["6"],
                "iterations": r["iterations"], "continuity_residual": r["continuity_residual"],
                "energy_residual": r["energy_residual"], "total_demand": sum(q for _, q in net.junctions.values()),
                "_fields": {"net": net, "sol": r}}

    def checks(self, p, out):
        return [check("continuity at every junction (m3/s)", out["continuity_residual"] < 1e-9,
                      value=out["continuity_residual"], limit=1e-9, source="solve_network"),
                check("energy along every pipe (m)", out["energy_residual"] < 1e-6, value=out["energy_residual"],
                      limit=1e-6, source="solve_network"),
                check("supply = total demand (mass balance)", abs(out["supply"] - out["total_demand"]) <
                      1e-9 * max(1.0, out["total_demand"]), value=out["supply"], limit=out["total_demand"],
                      source="global mass balance"),
                check("Newton converged", out["iterations"] < 100, value=out["iterations"], limit=100)]

    def decide(self, p, out):
        ok_p = out["min_pressure"] >= p["p_min"] - 1e-9
        ok_v = out["max_velocity"] <= 3.0
        return {"verdict": "OK" if ok_p and ok_v else "NOT OK",
                "reasons": [f"pressão mínima {out['min_pressure']:.2f} m no nó {out['critical_node']} "
                            f"{'>=' if ok_p else '<'} {p['p_min']:.0f} m",
                            f"velocidade máxima {out['max_velocity']:.2f} m/s {'<=' if ok_v else '>'} 3 m/s",
                            f"custo dos tubos US$ {out['cost']:,.0f}"]}

    def references(self):
        refs = [Reference(f"pressão no nó {k}, Hazen-Williams", {}, f"p_node{k}", EPANET_HW[k], "m", EPA,
                          "independent_solver", 1e-4) for k in ("3", "6")]
        refs += [Reference(f"pressão no nó {k}, Darcy-Weisbach", {"headloss": "DW"}, f"p_node{k}", EPANET_DW[k], "m",
                           EPA + " (EPANET usa Swamee-Jain e interpolação de Dunlop na transição; aqui Colebrook)",
                           "independent_solver", 0.01) for k in ("3", "6")]
        refs.append(Reference("custo da solução publicada", {}, "cost", 419000.0, "US$", SW97, "benchmark", 1e-9))
        return refs

    def validate(self):
        v = super().validate()
        try:                                          # live EPANET re-run when wntr is installed
            v["epanet_live"] = self._epanet_live()
        except ImportError:
            v["epanet_live"] = "wntr not installed: stored EPANET values used"
        opt = self.optimize()
        err = abs(opt["cost_ga"] - 419000.0) / 419000.0
        v["references"].append({"case": "otimização por AG vs ótimo publicado", "output": "cost", "unit": "US$",
                                "computed": opt["cost_ga"], "reference": 419000.0, "rel_error": err, "tol": 0.0,
                                "pass": bool(err <= 1e-9 and opt["feasible"]), "kind": "benchmark", "source": SW97,
                                "params": {}})
        v["all_pass"] = all(r["pass"] for r in v["references"])
        return v

    def _epanet_live(self):
        import wntr
        out = {}
        for hl, key, stored in (("HW", "H-W", EPANET_HW), ("DW", "D-W", EPANET_DW)):
            net = self._net(self.resolve(headloss=hl))
            wn = wntr.network.WaterNetworkModel()
            wn.options.hydraulic.headloss = key
            wn.options.hydraulic.accuracy = 1e-6
            wn.add_reservoir("1", base_head=net.reservoirs["1"])
            for k, (z, d) in net.junctions.items():
                wn.add_junction(k, base_demand=d, elevation=z)
            for (n, a, b, L, D, R) in net.pipes:
                wn.add_pipe("P" + n, a, b, length=L, diameter=D, roughness=R)
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=td + "/net")
            pr = {k: float(v) for k, v in res.node["pressure"].iloc[0].items() if k != "1"}
            out[hl] = {"pressures": pr, "max_diff_vs_stored": max(abs(pr[k] - stored[k]) for k in stored),
                       "wntr": wntr.__version__}
        return out

    def optimize(self, generations: int = 300, seed: int = 0, **kw):
        """Least-cost commercial diameters (GA, 14 sizes) with the minimum pressure; the GA result is re-solved
        with the network solver and compared with the published optimum."""
        p = self.resolve(**kw)
        sizes = [k * INCH for k in TWO_LOOP_COSTS]
        costs = list(TWO_LOOP_COSTS.values())
        r = optimize_diameters(self._net(p), sizes, costs, p["p_min"], generations=generations, seed=seed)
        chk = self.run(**{**kw, **{f"d{i + 1}": d for i, d in enumerate(r["diameters_m"])}})
        return {"objective": "custo mínimo dos tubos com pressão >= p_min em todos os nós",
                "method": "algoritmo genético (pop 100, torneio, cruzamento uniforme, mutação ±1/±2 classes)",
                "cost_ga": r["cost"], "diameters_in": [round(d / INCH) for d in r["diameters_m"]],
                "min_pressure_ga": r["min_pressure"], "feasible": r["feasible"], "hydraulic_solves": r["evaluations"],
                "published_cost": 419000.0, "published_diameters_in": list(OPT_IN),
                "verified_min_pressure": chk.outputs["min_pressure"], "verified_cost": chk.outputs["cost"]}

    def invert(self, noise_m: float = 0.05, seed: int = 0, **kw):
        """Hazen-Williams C of two pipe groups (trunk 1-3-5 aged, loop pipes) from pressures logged at nodes 3, 5
        and 7 at design demand and at 1.5x demand (a fire-flow test). Synthetic truth C = (95, 120)."""
        p = self.resolve(**kw)
        true = np.array([95.0, 120.0])
        trunk = {"1", "3", "5"}
        nodes = ("3", "5", "7")

        def pressures(c):
            out = []
            for fac in (1.0, 1.5):
                net = self._net({**p, "demand_factor": fac})
                net = net.with_roughness({x[0]: (c[0] if x[0] in trunk else c[1]) for x in net.pipes})
                ph = solve_network(net)["pressure_head"]
                out += [ph[n] for n in nodes]
            return np.array(out)

        rng = np.random.default_rng(seed)
        y = pressures(true) + rng.normal(0, noise_m, 6)
        c = np.array([130.0, 130.0])
        for _ in range(30):                                   # Gauss-Newton
            r0 = pressures(c) - y
            J = np.column_stack([(pressures(c + e) - pressures(c - e)) / 0.2 for e in np.eye(2) * 0.1])
            step = np.linalg.lstsq(J, -r0, rcond=None)[0]
            c = c + step
            if np.abs(step).max() < 1e-6:
                break
        J = np.column_stack([(pressures(c + e) - pressures(c - e)) / 0.2 for e in np.eye(2) * 0.1])
        cov = noise_m ** 2 * np.linalg.inv(J.T @ J)
        sd = np.sqrt(np.diag(cov))
        return {"description": "Calibração do C de Hazen-Williams de dois grupos de tubos (tronco 1-3-5 envelhecido "
                               "e tubos das malhas) a partir de pressões registradas nos nós 3, 5 e 7 na demanda de "
                               "projeto e a 1,5x (teste de vazão de incêndio). Medições sintéticas da própria rede "
                               f"com C real = (95, 120) e ruído de {noise_m} m.",
                "parameter": "C (tronco, malhas)", "true": true.tolist(), "recovered": c.tolist(),
                "ci95": [[float(c[i] - 1.96 * sd[i]), float(c[i] + 1.96 * sd[i])] for i in range(2)],
                "rel_error": float(np.max(np.abs(c - true) / true)),
                "true_inside_ci95": bool(np.all(np.abs(c - true) <= 1.96 * sd)), "noise": f"{noise_m} m (1 sigma)",
                "how": "Gauss-Newton sobre solve_network, covariância sigma^2 (J^T J)^-1", "n_measurements": 6}

    def variants(self):
        def uni(d):
            return {f"d{i + 1}": d * INCH for i in range(8)}
        return {"ótimo publicado": {}, "uniforme 12 pol": uni(12), "uniforme 16 pol": uni(16),
                "ótimo, tubos velhos C=100": {"hw_c": 100.0}, "ótimo, pico 1,3x": {"demand_factor": 1.3}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        net, sol = result.fields["net"], result.fields["sol"]
        fig, ax = plt.subplots(figsize=(5.4, 5.4))
        vmax = max(abs(v) for v in sol["velocity"].values())
        for (n, a, b, _L, D, _R) in net.pipes:
            (x0, y0), (x1, y1) = net.coords[a], net.coords[b]
            v = abs(sol["velocity"][n])
            ax.plot([x0, x1], [y0, y1], lw=1 + 20 * D, color=plt.cm.viridis(v / vmax))
            ax.text((x0 + x1) / 2 + 40, (y0 + y1) / 2 + 40, f"{n}: {round(D / INCH)}\"\n{sol['flow'][n] * 3600:.0f} m3/h",
                    fontsize=7)
        for k, (x, y) in net.coords.items():
            lab = f"R {net.reservoirs[k]:.0f} m" if k in net.reservoirs else f"{sol['pressure_head'][k]:.1f} m"
            ax.plot(x, y, "s" if k in net.reservoirs else "o", color="k", ms=8)
            ax.text(x - 330, y - 160, f"nó {k}\n{lab}", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xlim(500, 2600)
        ax.set_title("Pressão nos nós e vazão nos tubos (cor: velocidade)", fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        path = os.path.join(out_dir, "network.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from ..viz3d import Scene, Surface, box, merge, tube
        net, sol = result.fields["net"], result.fields["sol"]
        k = 0.01                                          # 1 km -> 10 display units
        zf = 0.05                                         # elevations x0.05 (relief readable, not to scale)
        z0 = min(z for z, _ in net.junctions.values())
        head = {**sol["head"], **net.reservoirs}

        def pos(nd):
            x, y = net.coords[nd]
            z = net.junctions[nd][0] if nd in net.junctions else net.reservoirs[nd] - 20
            return np.array([k * x, k * y, zf * (z - z0)])
        parts = []
        for (n, a, b, _L, D, _R) in net.pipes:
            s = np.linspace(0, 1, 30)
            P = pos(a)[None] * (1 - s[:, None]) + pos(b)[None] * s[:, None]
            V, F, i = tube(P, 0.15 + 0.35 * D / 0.6, 14)
            ha, hb = head[a], head[b]
            za = net.junctions[a][0] if a in net.junctions else net.reservoirs[a] - 20
            zb = net.junctions[b][0] if b in net.junctions else net.reservoirs[b] - 20
            pr = (ha + (hb - ha) * s) - (za + (zb - za) * s)
            parts.append((V, F, pr[i], np.full(len(V), abs(sol["velocity"][n]))))
        V, F, pr, vel = merge(parts)
        surf = [Surface("pipes", V, F, "steel", {"pressure": pr, "velocity": vel})]
        for nd in net.junctions:
            c = pos(nd)
            bv, bf = box(c - 0.45, c + 0.45)
            surf.append(Surface(f"node{nd}", bv, bf, "dark"))
        c = pos("1")
        bv, bf = box(c - [1.2, 1.2, 0], c + [1.2, 1.2, 1.5])
        surf.append(Surface("reservoir", bv, bf, "concrete"))
        return Scene(surf, title=self.title, labels={"pressure": "pressão [m c.a.]", "velocity": "velocidade [m/s]"})
