"""Junction temperatures on a power PCB (DC-DC / motor-drive stage) by 3-D finite volumes of the stack-up."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_physics.closed_form import pcb_thermal as pt
from pinneapple_physics.pcb.engine import calibrate, evaluate
from pinneapple_physics.pcb.solver import Board, Component, solve

from ..core import Output, Param, Preset, SurrogateSpec, check

JEDEC = ("JEDEC JESD51-7 (placa 2s2p) / JESD51-2 (ar parado); theta_JA mediano de datasheets publicados, tabela "
         "pinneapple_physics.closed_form.pcb_thermal.JEDEC_2S2P_THETA_JA")


class PowerPCB(Preset):
    name = "power_pcb_junction"
    title = "Temperatura de junção numa placa de potência (conversor DC-DC 4 camadas)"
    sector = "Térmica / eletrônica"
    problem = ("Placa de 4 camadas 100 x 80 mm, montada na vertical em gabinete sem ventilação, com dois MOSFETs "
               "D2PAK (meia-ponte), o indutor de saída, o controlador QFN e um regulador SOIC-8. Volumes finitos "
               "3-D por camada de cobre/dielétrico, modelos de 2 resistências JEDEC, convecção natural + radiação.")
    question = ("Todas as junções ficam abaixo do Tj máximo derrateado com 20 % de incerteza na convecção? "
                "Quantas vias térmicas sob os MOSFETs (ou que cobre) bastam, sem ventilador?")
    params = (
        Param("p_mosfet", "W", 2.0, "premissa: perdas de condução + chaveamento por MOSFET de um buck de ~150 W",
              0.1, 6.0),
        Param("p_inductor", "W", 1.2, "premissa: perdas no cobre + núcleo do indutor (~0,8 % de 150 W)", 0.0, 4.0),
        Param("p_controller", "W", 0.3, "premissa: controlador + drivers de gate", 0.0, 1.5),
        Param("p_ldo", "W", 0.25, "premissa: LDO 5 V -> 3,3 V a 75 mA", 0.0, 1.0),
        Param("t_ambient", "degC", 50.0, "premissa: ar interno de gabinete fechado (IEC 60068 classe industrial)",
              -20, 85),
        Param("velocity", "m/s", 0.0, "0 = convecção natural (sem ventilador)", 0.0, 5.0),
        Param("n_copper", "-", 4, "placa de 4 camadas (FR-4 1,6 mm)", choices=(2, 4, 6, 8)),
        Param("outer_oz", "oz", 1.0, "cobre externo 1 oz = 35 um (IPC-2221)", 0.5, 3.0),
        Param("vias", "-", 9, "premissa: 3 x 3 vias de 0,3 mm sob cada D2PAK", 0, 49),
        Param("tj_max", "degC", 125.0, "Tj 175 °C do datasheet derratado para 125 °C (prática de projeto)",
              80, 175),
        Param("board_w", "m", 0.100, "placa 100 mm", 0.03, 0.3),
        Param("board_d", "m", 0.080, "placa 80 mm (altura, montada na vertical)", 0.03, 0.3),
    )
    outputs = (
        Output("tj_max_c", "degC", "maior temperatura de junção"),
        Output("tj_max_hot_c", "degC", "a mesma com convecção 20 % menor"),
        Output("critical", "-", "componente crítico"),
        Output("tj_mosfet_c", "degC", "junção do MOSFET mais quente"),
        Output("board_max_c", "degC", "temperatura máxima da placa"),
        Output("energy_balance", "-", "erro relativo do balanço de energia"),
    )
    validity = ("Placa plana isolada no ar (sem gabinete radiante próximo), componentes como modelos de 2 "
                "resistências JEDEC (theta_JB, theta_JC), cobre como frações por camada, convecção natural "
                "(Churchill-Chu) ou forçada laminar de placa plana. theta dos encapsulamentos: valores típicos; para "
                "aprovação final use os do datasheet.")
    pipeline = ("geometria: Board(stack-up) + Component(x, y, encapsulamento, vias)",
                "modelo: condutividade anisotrópica do stack-up, vias como cobre paralelo, JEDEC 2R",
                "solver: pinneapple_physics.pcb.engine.evaluate (FVM 3-D + ponto fixo convecção/radiação)",
                "pós: Tj por componente com banda de ±20 % no h, mapas das faces",
                "decisão: Tj(h -20 %) <= Tj máx; menor alteração de projeto que atende")
    library = ("pinneapple_physics.pcb.engine.evaluate", "pinneapple_physics.pcb.engine.calibrate",
               "pinneapple_physics.pcb.solver.solve")
    surrogate_spec = SurrogateSpec(inputs=("p_mosfet", "t_ambient", "vias"), outputs=("tj_max_c",), n_train=60,
                                   n_test=20, bounds={"p_mosfet": (0.5, 4.0), "t_ambient": (20.0, 70.0),
                                                      "vias": (0, 36)})
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("tj_max_c", "tj_max_hot_c", "board_max_c")
    reason_case = "power_board"
    missing = {}

    def _board(self, p):
        b = Board(p["board_w"] * 1e3, p["board_d"] * 1e3, n_copper=int(p["n_copper"]), outer_oz=p["outer_oz"])
        v = int(round(p["vias"]))
        W, D = b.width_mm, b.depth_mm
        comps = [Component("Q1", 0.30 * W, 0.62 * D, p["p_mosfet"], "TO-263 D2PAK", tj_max_c=p["tj_max"], vias=v),
                 Component("Q2", 0.30 * W, 0.28 * D, p["p_mosfet"], "TO-263 D2PAK", tj_max_c=p["tj_max"], vias=v),
                 Component("L1", 0.62 * W, 0.45 * D, p["p_inductor"], "Inductor 10x10", tj_max_c=125),
                 Component("U1", 0.86 * W, 0.72 * D, p["p_controller"], "QFN-32 5x5", tj_max_c=p["tj_max"], vias=4),
                 Component("U2", 0.86 * W, 0.22 * D, p["p_ldo"], "SOIC-8", tj_max_c=p["tj_max"])]
        env = pt.Environment(p["t_ambient"], air_velocity_m_s=p["velocity"])
        return b, [c for c in comps if c.power_w > 0], env

    def solve(self, p):
        b, comps, env = self._board(p)
        r = evaluate(b, comps, env)
        cs = r["components"]
        tj = {c["name"]: c["t_junction_c"] for c in cs}
        hot = max(c["t_junction_band_c"][1] for c in cs)
        crit = max(cs, key=lambda c: c["t_junction_band_c"][1] - c["tj_max_c"])
        eb = next(c["value"] for c in r["checks"] if c["name"] == "energy_balance")
        return {"tj_max_c": max(tj.values()), "tj_max_hot_c": hot, "critical": crit["name"],
                "tj_mosfet_c": max(tj["Q1"], tj["Q2"]), "board_max_c": r["board_max_c"],
                "energy_balance": eb, "total_power": r["power_w"],
                "tj": tj, "checks_engine": r["checks"], "_fields": {"field3d": r["field3d"], "maps": r["maps"]}}

    def checks(self, p, out):
        c = []
        for ch in out["checks_engine"]:
            ok = {"pass": True, "fail": False}.get(ch["status"])
            c.append(check(ch["name"], ok, value=ch["value"] if isinstance(ch["value"], (int, float)) else None,
                           source=ch.get("law", ""), reason=ch.get("detail", "") if ok is None else "",
                           critical=ch["name"] == "energy_balance"))
        return c

    def decide(self, p, out):
        ok = out["tj_max_hot_c"] <= p["tj_max"]
        return {"verdict": "OK" if ok else "NOT OK",
                "reasons": [f"Tj máx {out['tj_max_c']:.1f} °C ({out['critical']}); com h -20 %: "
                            f"{out['tj_max_hot_c']:.1f} °C {'<=' if ok else '>'} {p['tj_max']:.0f} °C",
                            f"placa até {out['board_max_c']:.1f} °C"]}

    # ------------------------------------------------------------------------------------------- validation
    def validate(self):
        rows = []
        for pkg in ("TO-263 D2PAK", "QFN-32 5x5", "SOIC-8"):
            big = pt.PACKAGES[pkg]["w_mm"] > 27
            board = Board(114.3, 101.6 if big else 76.2, n_copper=4, thickness_mm=1.6, outer_oz=2, inner_oz=1,
                          outer_coverage=0.1, inner_coverage=0.95)
            vias = {"QFN-32 5x5": 9}.get(pkg, 0)
            r = solve(board, [Component("U", 57.15, board.depth_mm / 2, 1.0, pkg, vias=vias)],
                      pt.Environment(25.0, orientation="horizontal"), n_cells=60)
            th = r["components"][0]["t_junction_c"] - 25.0
            ref = pt.JEDEC_2S2P_THETA_JA[pkg]
            err = abs(th - ref) / ref
            rows.append({"case": f"theta_JA {pkg} na placa JEDEC 2s2p", "output": "theta_ja", "unit": "K/W",
                         "computed": th, "reference": ref, "rel_error": err, "tol": 0.08, "pass": bool(err <= 0.08),
                         "kind": "standard (consistência)", "source": JEDEC + ". Atenção: os theta típicos dos "
                         "encapsulamentos da lib foram calibrados para reproduzir esses valores; a linha confirma o "
                         "modelo placa+convecção+encapsulamento, não é uma validação independente", "params": {}})
        # grid convergence of the junction rise of the default board
        p = self.defaults()
        b, comps, env = self._board(p)
        ns = (24, 36, 48, 60, 80)
        tjs = [max(c["t_junction_c"] for c in solve(b, comps, env, n_cells=n)["components"]) for n in ns]
        rise = [t - p["t_ambient"] for t in tjs]
        rel = abs(rise[-1] - rise[-2]) / rise[-1]
        conv = {"parameter": "células por lado", "values": list(ns), "tj_max": tjs, "rel_change_last": rel,
                "pass": rel < 0.02}
        return {"preset": self.name, "references": rows, "convergence": conv,
                "all_pass": all(r["pass"] for r in rows) and conv["pass"]}

    # ------------------------------------------------------------------------------------------- extras
    def optimize(self, **kw):
        """Smallest change that meets Tj(h -20 %): no fan first, then the fewest thermal vias, then 1 oz copper,
        then the fewest layers (each step costs more in production than the one before)."""
        p = self.resolve(**kw)
        tried = 0
        for vel in (0.0, 0.5, 1.0, 2.0):
            for oz in (1.0, 2.0):
                for nl in (4, 6):
                    for v in (0, 4, 9, 16, 25, 36):
                        q = {**p, "velocity": vel, "outer_oz": oz, "n_copper": nl, "vias": v}
                        tried += 1
                        r = self.run(**q)
                        if r.decision["verdict"] == "OK":
                            out = {"objective": "menor mudança que atende Tj(h -20 %) <= Tj máx (sem ventilador > "
                                                "menos vias > cobre 1 oz > menos camadas)",
                                   "velocity": vel, "outer_oz": oz, "n_copper": nl, "vias": v,
                                   "tj_max_hot_c": r.outputs["tj_max_hot_c"], "designs_tried": tried}
                            out.update(self._surrogate_check(p, v, r))
                            return out
        return {"feasible": False, "designs_tried": tried}

    def _surrogate_check(self, p, v, r):
        from ..surrogate import load_surrogate
        sur = load_surrogate(self.name)
        if sur is None or sur.out_of_box(p):
            return {}
        best = None
        for vv in range(0, 37):
            pr, _ = sur.predict_one({**p, "vias": vv})
            if pr["tj_max_c"] <= p["tj_max"]:
                best = (vv, pr["tj_max_c"])
                break
        if best is None:
            return {}
        rs = self.run(**{**p, "vias": best[0]})
        return {"surrogate_question": "menos vias com Tj nominal <= Tj máx (sem a banda de h)",
                "surrogate_vias_nominal": best[0], "surrogate_tj_pred_c": best[1],
                "solver_tj_at_surrogate_choice_c": rs.outputs["tj_max_c"],
                "surrogate_vs_solver_diff_c": best[1] - rs.outputs["tj_max_c"]}

    def invert(self, noise_c: float = 0.3, seed: int = 0, **kw):
        """Board cooling and copper spreading factors from six thermocouples (Ensemble Kalman Inversion of the
        library, ``pinneapple_physics.pcb.engine.calibrate``). Synthetic truth: h_scale 0.75 (enclosure walls
        close by), in-plane copper 0.6 (less copper pour than drawn)."""
        p = self.resolve(**kw)
        b, comps, env = self._board(p)
        from pinneapple_physics.pcb.engine import _interp_map
        nom = solve(b, comps, env, n_cells=60)
        h = {k: v * 0.75 for k, v in nom["h"].items() if k in ("top", "bottom")}
        truth = solve(b, comps, env, n_cells=60, h_override=h, k_xy_scale=0.6)
        rng = np.random.default_rng(seed)
        W, D = b.width_mm, b.depth_mm
        pts = [(0.15, 0.45), (0.45, 0.45), (0.62, 0.85), (0.80, 0.50), (0.45, 0.10), (0.95, 0.95)]
        meas = [{"x_mm": fx * W, "y_mm": fy * D, "side": "top",
                 "t_c": float(_interp_map(truth["t_top_c"], W, D, fx * W, fy * D) + rng.normal(0, noise_c))}
                for fx, fy in pts]
        meas.append({"component": "Q1", "t_c": float(truth["components"][0]["t_junction_c"] + rng.normal(0, noise_c))})
        c = calibrate(b, comps, env, meas, noise_std_c=noise_c)
        par = c["parameters"]
        hs, ks = par["h_scale"], par["k_inplane_scale"]
        def ci(q):
            return [q["value"] * math.exp(-1.96 * q["std_log"]), q["value"] * math.exp(1.96 * q["std_log"])]
        return {"description": "Fatores de convecção (h_scale) e de espalhamento no cobre (k_inplane_scale) a partir de "
                               "6 termopares na face superior e o sensor de junção de Q1, por Ensemble Kalman Inversion da lib. Medição "
                               f"sintética do solver com (0,75; 0,6) e ruído de {noise_c} K.",
                "parameter": "h_scale, k_inplane_scale", "true": [0.75, 0.6], "recovered": [hs["value"], ks["value"]],
                "ci95": [ci(hs), ci(ks)],
                "rel_error": max(abs(hs["value"] - 0.75) / 0.75, abs(ks["value"] - 0.6) / 0.6),
                "noise": f"{noise_c} K (1 sigma)", "how": "pinneapple_physics.pcb.engine.calibrate (EKI, 16 membros)",
                "fit_rms_after_c": c["fit"]["rms_after_c"]}

    def variants(self):
        return {"base (4 cam., 9 vias)": {}, "sem vias": {"vias": 0}, "25 vias": {"vias": 25},
                "2 camadas": {"n_copper": 2}, "6 camadas, 2 oz": {"n_copper": 6, "outer_oz": 2.0},
                "ventilador 1 m/s": {"velocity": 1.0}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = result.fields["maps"]
        f = result.fields["field3d"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for a, k, t in ((ax[0], "top_c", "face superior"), (ax[1], "bottom_c", "face inferior")):
            im = a.imshow(np.asarray(m[k]), origin="lower", cmap="jet", extent=[0, m["width_mm"], 0, m["depth_mm"]])
            fig.colorbar(im, ax=a, label="T [°C]")
            a.set(xlabel="x [mm]", ylabel="y [mm]", title=t)
        for c in f["components"]:
            ax[0].add_patch(plt.Rectangle((c["x_mm"] - c["w_mm"] / 2, c["y_mm"] - c["d_mm"] / 2), c["w_mm"],
                                          c["d_mm"], fill=False, ec="w"))
            ax[0].text(c["x_mm"], c["y_mm"], f"{c['name']}\n{c['t_junction_c']:.0f}°C", color="w", fontsize=7,
                       ha="center", va="center")
        fig.tight_layout()
        path = os.path.join(out_dir, "pcb_maps.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from scipy.interpolate import RegularGridInterpolator

        from ..viz3d import Scene, Surface, box
        f = result.fields["field3d"]
        W, D, t = f["width_mm"], f["depth_mm"], f["thickness_mm"]
        top, bot = np.asarray(f["top_c"]), np.asarray(f["bottom_c"])
        ys = np.linspace(0, D, top.shape[0])
        xs = np.linspace(0, W, top.shape[1])
        V, F = box([0, 0, -t], [W, D, 0], (60, 48, 1))
        it = RegularGridInterpolator((ys, xs), top, bounds_error=False, fill_value=None)
        ib = RegularGridInterpolator((ys, xs), bot, bounds_error=False, fill_value=None)
        q = np.c_[np.clip(V[:, 1], 0, D), np.clip(V[:, 0], 0, W)]
        T = np.where(V[:, 2] > -t / 2, it(q), ib(q))
        surf = [Surface("board", V / 10, F, "pcb", {"temperature": T})]
        for c in f["components"]:
            cv, cf = box([c["x_mm"] - c["w_mm"] / 2, c["y_mm"] - c["d_mm"] / 2, 0],
                         [c["x_mm"] + c["w_mm"] / 2, c["y_mm"] + c["d_mm"] / 2, c["h_mm"]])
            surf.append(Surface(c["name"], cv / 10, cf, "dark", {"temperature": np.full(len(cv), c["t_junction_c"])}))
        return Scene(surf, title=self.title, labels={"temperature": "temperatura [°C] (componentes: junção)"})
