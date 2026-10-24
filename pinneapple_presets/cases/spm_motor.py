"""Surface-PM servo motor: torque, ripple, cogging and back-EMF by 2-D nonlinear magnetostatic FEM."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_simulation.electromagnetics.machine import SPMMachine
from pinneapple_simulation.electromagnetics.postprocess import back_emf

from ..core import Output, Param, Preset, SurrogateSpec, check

PYR = "Pyrhönen, Jokinen & Hrabovcová, Design of Rotating Electrical Machines, 2a ed. (2014)"


class SPMMotor(Preset):
    name = "spm_servo_motor"
    title = "Motor síncrono de ímãs superficiais (servo 4 polos, 24 ranhuras): torque, ondulação e cogging"
    sector = "Eletromagnetismo / máquinas elétricas"
    problem = ("Servo-motor de ímãs de NdFeB na superfície do rotor, 4 polos, 24 ranhuras, enrolamento distribuído "
               "trifásico, ferro não linear (curva BH). FEM magnetostático 2-D com banda deslizante, torque por tensor "
               "de Maxwell e por co-energia, fluxo concatenado e força contra-eletromotriz.")
    question = ("Que torque o motor entrega na corrente nominal (limite térmico de densidade de corrente), com quanta "
                "ondulação e cogging? Qual a menor espessura de ímã que mantém o torque?")
    params = (
        Param("I_peak", "A", 13.4, "corrente nominal: J_rms = 6 A/mm2 na área de cobre da ranhura (fator de "
              "enchimento 0,45, 20 espiras); J típico de motor TEFC 3-6,5 A/mm2 (" + PYR + ", Tab. 6.2)", 1, 60),
        Param("magnet_thickness", "m", 0.004, "ímã de 4 mm (servo-motores pequenos 3-6 mm)", 0.0015, 0.008),
        Param("magnet_arc", "-", 0.8, "arco do ímã = 80 % do passo polar", 0.5, 0.98),
        Param("Br", "T", 1.2, "NdFeB N38-N42 a 20 °C, Br 1,2-1,3 T (catálogos de ímãs sinterizados)", 0.9, 1.45),
        Param("magnetization", "-", "parallel", "magnetização paralela (bloco)", choices=("parallel", "radial")),
        Param("slot_opening", "-", 2 / 15, "abertura da ranhura = 2/15 do passo de ranhura", 0.05, 0.5),
        Param("rpm", "rpm", 3000.0, "rotação nominal de servo-motor", 100, 12000),
        Param("n_positions", "-", 6, "posições do rotor por passo de ranhura (ondulação, cogging)", 3, 24),
        Param("n_theta", "-", 360, "divisões angulares da malha: 15 por passo de ranhura, para que a abertura de 2/15 "
              "caia em nós (com 240 ou 480 a geometria da abertura muda com a malha); convergência 360 -> 720",
              choices=(360, 720)),
    )
    outputs = (
        Output("torque", "N m", "torque médio na corrente nominal (eixo q)"),
        Output("ripple", "-", "ondulação pico-a-pico / média"),
        Output("cogging_pp", "N m", "cogging pico-a-pico (sem corrente)"),
        Output("kt", "N m/A", "constante de torque (por A de pico)"),
        Output("ke", "V s/rad", "constante de FEM (fase, pico, por rad/s mecânico)"),
        Output("emf_peak", "V", "FEM de fase de pico na rotação dada"),
        Output("p_copper", "W", "perdas no cobre (3 fases, 20 °C)"),
    )
    validity = ("2-D (sem efeitos de extremidade nem skew), magnetostática (sem correntes induzidas nos ímãs), "
                "ferro com a curva BH ilustrativa da lib, enrolamento de camada única distribuído. Desmagnetização "
                "não verificada.")
    pipeline = ("geometria: SPMMachine (rotor, ímãs, entreferro, 24 ranhuras, banda deslizante)",
                "modelo: magnetostática não linear em A_z, ímãs por Br e mu_r",
                "solver: SPMMachine.solve por posição (Newton); torque por tensor de Maxwell e por co-energia",
                "pós: torque médio e ondulação num passo de ranhura, cogging, fluxo concatenado e FEM",
                "decisão: torque >= alvo e ondulação <= 5 %")
    library = ("pinneapple_simulation.electromagnetics.machine.SPMMachine",)
    surrogate_spec = SurrogateSpec(inputs=("I_peak", "magnet_thickness", "magnet_arc"),
                                   outputs=("torque",), n_train=40, n_test=15,
                                   bounds={"I_peak": (5.0, 30.0), "magnet_thickness": (0.0025, 0.006),
                                           "magnet_arc": (0.65, 0.92)})
    # cogging was trained too and dropped: hold-out max error 146 %, R2 -0.33 (it jumps with the magnet arc); the
    # FEM computes it
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("torque", "ripple", "cogging_pp")
    missing = {"reason": "não há caso de máquina elétrica no pinneapple_reason"}
    limitations = ("Sem referência experimental pública desta geometria ilustrativa: a validação é interna (dois "
                   "métodos de torque, balanço de potência, periodicidade, convergência de malha); a verificação "
                   "contra a solução exata de subdomínio da máquina sem ranhuras está em tests/test_motor_magnetics.py.",)

    def machine(self, p):
        return SPMMachine(n_theta=int(p["n_theta"]), magnet_thickness=p["magnet_thickness"],
                          magnet_arc=p["magnet_arc"], Br=p["Br"], magnetization=p["magnetization"],
                          slot_opening=p["slot_opening"], slot_width=0.4)

    def solve(self, p):
        m = self.machine(p)
        n = int(p["n_positions"])
        pitch = m.cogging_period()
        th = np.arange(n) * pitch / n
        T, last = [], None
        for t in th:
            last = m.solve(t, m.phase_currents(t, p["I_peak"], 0.0))
            T.append(m.torque(last))
        T = np.array(T)
        cog = m.cogging(th)
        steps = 36                                           # one electrical period, 10 el. deg steps
        the = (2 * math.pi / m.pole_pairs) * np.arange(steps) / steps
        psi = m.no_load_flux(the)
        w = p["rpm"] * 2 * math.pi / 60
        e = back_emf(psi, the, w)
        ke = float(np.abs(e[:, 0]).max() / w)
        ke1 = float(2 * np.abs(np.fft.rfft(e[:, 0])[1]) / steps / w)       # fundamental of the phase EMF
        # thermal rating: copper area of a slot, phase resistance (20 C)
        slot_pitch = 2 * math.pi * (m.r_bore) / m.slots
        a_slot = 0.4 * slot_pitch * m.slot_depth
        a_cu = 0.45 * a_slot / m.turns_per_slot
        n_series = m.turns_per_slot * m.slots / 3 / 2          # turns per phase (two coil sides per turn)
        l_turn = 2 * m.length + 2 * 0.6 * math.pi * m.r_bore * 2 / (2 * m.pole_pairs) * 2
        R = 1.72e-8 * n_series * l_turn / a_cu
        p_cu = 3 * R * (p["I_peak"] / math.sqrt(2)) ** 2
        # torque from the Maxwell stress and from the co-energy at one position (independent integrations)
        t0 = th[min(1, n - 1)]
        cur = m.phase_currents(t0, p["I_peak"], 0.0)
        t_mst = m.torque(m.solve(t0, cur))
        t_coe = m.torque_coenergy(t0, cur)
        # power balance: T - T_cog = i . e / w
        k = 3
        th_k = the[k]
        ck = np.array(m.phase_currents(th_k, p["I_peak"], 0.0))
        t_k = m.torque(m.solve(th_k, tuple(ck))) - m.torque(m.solve(th_k))
        return {"torque": float(T.mean()), "ripple": float((T.max() - T.min()) / abs(T.mean())),
                "cogging_pp": float(cog.max() - cog.min()), "kt": float(T.mean() / p["I_peak"]), "ke": ke,
                "emf_peak": ke * w, "p_copper": p_cu, "j_rms": p["I_peak"] / math.sqrt(2) / a_cu / 1e6,
                "t_mst": t_mst, "t_coe": t_coe, "t_power": float(ck @ e[k]) / w, "t_minus_cog": t_k,
                "kt_from_ke": 1.5 * ke1, "ke1": ke1,
                "cogging_mean_over_pp": float(abs(cog.mean()) / max(cog.max() - cog.min(), 1e-12)),
                "_fields": {"theta": th, "T": T, "cog": cog, "psi": psi, "the": the, "sol": last, "machine": m}}

    def checks(self, p, out):
        d1 = abs(out["t_mst"] - out["t_coe"]) / abs(out["t_coe"])
        d2 = abs(out["t_minus_cog"] - out["t_power"]) / abs(out["t_power"])
        return [check("Maxwell stress = co-energy torque", d1 < 0.01, value=d1, limit=0.01,
                      source="two independent torque integrations"),
                check("power balance T - T_cog = i.e / w", d2 < 0.03, value=d2, limit=0.03,
                      source="flux linkage derivative vs field torque"),
                check("cogging has zero mean over a slot pitch", out["cogging_mean_over_pp"] < 0.05,
                      value=out["cogging_mean_over_pp"], limit=0.05),
                check("current density within the TEFC range (<= 6.5 A/mm2)", out["j_rms"] <= 6.5,
                      value=out["j_rms"], limit=6.5, source=PYR)]

    def complete(self, p, out):
        return {**out, "ripple": float("nan"), "kt": out["torque"] / p["I_peak"]}

    def decide(self, p, out):
        ok_r = math.isnan(out["ripple"]) or out["ripple"] <= 0.05
        return {"verdict": "OK" if ok_r else "NOT OK",
                "reasons": [f"torque médio {out['torque']:.2f} N m a {p['I_peak']:.1f} A de pico "
                            f"(kt {out['kt']:.3f} N m/A)",
                            f"ondulação {100 * out['ripple']:.1f} % {'<=' if ok_r else '>'} 5 %, cogging "
                            f"{out['cogging_pp']:.3f} N m p-p"]}

    def validate(self):
        o = self.run().outputs
        rows = [self._row("torque: tensor de Maxwell vs co-energia", "torque", "N m", o["t_mst"], o["t_coe"], 0.01,
                          "independent_solver", "duas integrações independentes do mesmo campo (lib, tests/"
                          "test_motor_magnetics.py mede 0,09 %)"),
                self._row("torque - cogging vs i.e/w (balanço de potência)", "torque", "N m", o["t_minus_cog"],
                          o["t_power"], 0.03, "analytic", "conservação de energia: P_mec = sum i_k e_k"),
                self._row("kt (FEM) vs 3/2 ke1 (fundamental da FEM, máquina senoidal)", "kt", "N m/A", o["kt"], o["kt_from_ke"], 0.05,
                          "analytic", PYR + ", eq. de torque da máquina síncrona de ímãs")]
        p = self.defaults()
        ts = []
        for n in (360, 720):                                 # same slot-opening geometry on both meshes
            m = self.machine({**p, "n_theta": n})
            th = np.arange(12) * m.cogging_period() / 12     # 12 positions: the 16 % ripple averages out
            ts.append(float(np.mean([m.torque(m.solve(t, m.phase_currents(t, p["I_peak"], 0.0))) for t in th])))
        rel = abs(ts[-1] - ts[-2]) / abs(ts[-1])
        conv = {"parameter": "n_theta (12 posições por passo de ranhura)", "values": [360, 720], "torque": ts,
                "rel_change_last": rel, "pass": rel < 0.01}
        return {"preset": self.name, "references": rows, "convergence": conv,
                "all_pass": all(r["pass"] for r in rows) and conv["pass"]}

    @staticmethod
    def _row(case, output, unit, comp, ref, tol, kind, source):
        err = abs(comp - ref) / abs(ref)
        return {"case": case, "output": output, "unit": unit, "computed": comp, "reference": ref, "rel_error": err,
                "tol": tol, "pass": bool(err <= tol), "kind": kind, "source": source, "params": {}}

    def optimize(self, **kw):
        """Least magnet (thickness x arc) keeping the rated torque of the 4 mm / 0.8 design and cogging below
        2 % of it: searched on the surrogate, the pick re-run on the FEM."""
        p = self.resolve(**kw)
        base = self.run(**kw).outputs
        from ..surrogate import load_surrogate
        sur = load_surrogate(self.name)
        if sur is None:
            return None
        cands = []
        for h in np.linspace(0.0025, 0.006, 15):
            for a in np.linspace(0.65, 0.92, 12):
                pr, ood = sur.predict_one({**p, "magnet_thickness": h, "magnet_arc": a})
                if not ood and pr["torque"] >= base["torque"]:
                    cands.append((h * a, float(h), float(a), pr["torque"]))
        out = {"objective": "menor volume de ímã (espessura x arco) com torque >= base (surrogate, conferido no FEM) e "
                            "cogging <= 2 % do torque (FEM: o surrogate de cogging foi descartado)"}
        tried = []
        for vol, h, a, tq in sorted(cands)[:8]:
            ver = self.run(**{**kw, "magnet_thickness": h, "magnet_arc": a}).outputs
            ok = ver["torque"] >= 0.99 * base["torque"] and ver["cogging_pp"] <= 0.02 * base["torque"]
            tried.append({"magnet_thickness_m": h, "magnet_arc": a, "surrogate_torque": tq, "fem_torque": ver["torque"],
                          "fem_cogging": ver["cogging_pp"], "ok": ok})
            if ok:
                out.update({"magnet_thickness_m": h, "magnet_arc": a,
                            "magnet_volume_vs_base": vol / (p["magnet_thickness"] * p["magnet_arc"]),
                            "surrogate_torque": tq, "fem_torque": ver["torque"],
                            "surrogate_vs_fem_torque_rel": abs(tq - ver["torque"]) / ver["torque"],
                            "fem_cogging": ver["cogging_pp"], "verified_ok": True, "fem_checks": len(tried)})
                return out
        return {**out, "feasible": False, "fem_checks": len(tried), "tried": tried}

    def invert(self, true_br: float = 1.12, noise: float = 0.005, seed: int = 0, **kw):
        """Magnet remanence of a motor on the bench (hot or partly demagnetised magnets) from the open-circuit
        back-EMF measured at the rated speed, 0.5 % meter noise, 4 readings. Synthetic truth Br = 1.12 T."""
        p = self.resolve(**kw)

        def emf(br):
            m = self.machine({**p, "Br": br})
            th = (2 * math.pi / m.pole_pairs) * np.arange(12) / 12
            w = p["rpm"] * 2 * math.pi / 60
            return float(np.abs(back_emf(m.no_load_flux(th), th, w)[:, 0]).max())

        rng = np.random.default_rng(seed)
        y = emf(true_br) * (1 + rng.normal(0, noise, 4))
        ym = float(y.mean())
        lo, hi = 0.9, 1.45
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if emf(mid) < ym else (lo, mid)
        br = 0.5 * (lo + hi)
        dedb = (emf(br + 0.01) - emf(br - 0.01)) / 0.02
        sd = float(ym * noise / math.sqrt(len(y)) / dedb)
        return {"description": "Remanência dos ímãs (aquecidos ou parcialmente desmagnetizados) a partir da FEM a "
                               "vazio medida na rotação nominal (4 leituras, ruído de 0,5 %). Verdade sintética "
                               f"Br = {true_br} T.", "parameter": "Br", "unit": "T", "true": true_br,
                "recovered": br, "ci95": [br - 1.96 * sd, br + 1.96 * sd], "rel_error": abs(br - true_br) / true_br,
                "true_inside_ci95": bool(abs(br - true_br) <= 1.96 * sd), "noise": f"{100 * noise} %",
                "how": "bisseção sobre o FEM (FEM a vazio é monótona em Br), IC pela sensibilidade dE/dBr"}

    def variants(self):
        return {"base (4 mm, arco 0,80)": {}, "arco 0,67": {"magnet_arc": 0.67}, "arco 0,90": {"magnet_arc": 0.90},
                "ímã 3 mm": {"magnet_thickness": 0.003}, "magnetização radial": {"magnetization": "radial"}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        F = result.fields
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        deg = np.degrees(F["theta"])
        ax[0].plot(deg, F["T"], "o-", label="com corrente")
        ax[0].plot(deg, F["cog"], "s-", label="cogging")
        ax[0].set(xlabel="posição do rotor [graus mec.]", ylabel="torque [N m]", title="Um passo de ranhura")
        ax[0].legend(fontsize=8)
        ax[1].plot(np.degrees(F["the"]), F["psi"])
        ax[1].set(xlabel="graus mec.", ylabel="fluxo concatenado [Wb]", title="Fluxo a vazio, 3 fases")
        ms = F["sol"]
        mesh = ms.mesh
        Bm = np.hypot(ms.B[:, 0], ms.B[:, 1])
        tpc = ax[2].tripcolor(mesh.points[:, 0], mesh.points[:, 1], mesh.tris, facecolors=Bm, cmap="jet",
                              vmax=np.percentile(Bm, 99.5))
        fig.colorbar(tpc, ax=ax[2], label="|B| [T]")
        ax[2].set_aspect("equal")
        ax[2].axis("off")
        ax[2].set_title("|B| com corrente nominal")
        fig.tight_layout()
        path = os.path.join(out_dir, "motor.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from ..viz3d import Scene, Surface
        ms = result.fields["sol"]
        m = result.fields["machine"]
        mesh = ms.mesh
        P, T = mesh.points, mesh.tris
        Bm = np.hypot(ms.B[:, 0], ms.B[:, 1])
        acc, cnt = np.zeros(len(P)), np.zeros(len(P))
        for j in range(3):
            np.add.at(acc, T[:, j], Bm)
            np.add.at(cnt, T[:, j], 1)
        Bn = acc / np.maximum(cnt, 1)
        L = m.length
        surf = []
        for name, z in (("front", L), ("back", 0.0)):
            V = np.c_[P, np.full(len(P), z)]
            surf.append(Surface(name, V * 100, T if name == "front" else T[:, ::-1], "steel", {"B": Bn}))
        # outer stator skin, so the stack reads as a solid
        th = np.linspace(0, 2 * np.pi, 121)
        R = m.r_out
        V = np.r_[np.c_[R * np.cos(th), R * np.sin(th), np.zeros_like(th)],
                  np.c_[R * np.cos(th), R * np.sin(th), np.full_like(th, L)]]
        n = len(th)
        F = np.array([[i, i + 1, n + i + 1] for i in range(n - 1)] + [[i, n + i + 1, n + i] for i in range(n - 1)])
        surf.append(Surface("stator_skin", V * 100, F, "steel", {"B": np.full(len(V), float(np.median(Bn)))}))
        return Scene(surf, title=self.title, labels={"B": "|B| [T] (corrente nominal)"})
