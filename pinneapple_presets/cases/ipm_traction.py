"""Interior-PM traction motor (2004 Toyota Prius geometry): torque, ripple, MTPA and voltage at speed by FEA."""
from __future__ import annotations

import math

import numpy as np

from pinneapple_simulation.electromagnetics.machines.geometry import prius_2004
from pinneapple_simulation.electromagnetics.machines.pmsm import IPMMachine
from pinneapple_simulation.electromagnetics.machines.verification import (
    YANG_2013_PHI0_DEG,
    YANG_2013_TORQUE_NM,
)

from ..core import Output, Param, Preset, SurrogateSpec, check

YANG = ("Yang, Krishnamurthy & Brown, 'Electromagnetic and vibrational characteristic of IPM over full torque-speed "
        "range', IEEE IEMDC 2013, pp. 295-302 (Prius 2004, 250 A); valores tabulados na suíte de testes do pyleecan")
ORNL = "Staunton et al., ORNL/TM-2004/247, 'Evaluation of the 2004 Toyota Prius hybrid electric drive system' (2005)"
_M: dict = {}


def machine(n_gap: int) -> IPMMachine:
    if n_gap not in _M:
        _M[n_gap] = IPMMachine(prius_2004(), n_gap=n_gap)
    return _M[n_gap]


class IPMTraction(Preset):
    name = "ipm_traction_motor"
    title = "Motor de tração de ímãs internos (geometria do Toyota Prius 2004): torque, MTPA e tensão"
    sector = "Eletromagnetismo / mobilidade elétrica"
    problem = ("Motor síncrono de ímãs internos em V (8 polos, 48 ranhuras) com a geometria publicada do Prius "
               "2004. FEM magnetostático não linear por posição de rotor com correntes senoidais fixas no "
               "referencial dq: torque médio (Arkkio), ondulação, fluxos dq, ângulo de máximo torque por ampère "
               "(MTPA) e a tensão de fase na rotação pedida contra o limite do inversor.")
    question = ("Que torque o motor dá nesta corrente e neste ângulo, quanto falta para o MTPA, e a tensão nesta "
                "rotação cabe no barramento CC (ou é preciso enfraquecimento de campo)?")
    params = (
        Param("I_peak", "A", 250.0, "250 A de pico, o ponto publicado de Yang et al. (2013)", 10, 300),
        Param("gamma_deg", "deg", 40.0, "ângulo da corrente a partir do eixo q (0 = só i_q)", 0, 85),
        Param("rpm", "rpm", 1500.0, "premissa: rotação de base em torno de 1200-1500 rpm (" + ORNL + ")", 0, 6000),
        Param("v_dc", "V", 500.0, "barramento CC do Prius 2004 com o conversor elevador: até 500 V (" + ORNL + ")",
              100, 800),
        Param("n_positions", "-", 6, "posições do rotor por média de torque", 3, 24),
        Param("n_gap", "-", 480, "nós no entreferro (malha); 960 na verificação de convergência",
              choices=(480, 960)),
    )
    outputs = (
        Output("torque", "N m", "torque médio (tensor de Maxwell, Arkkio)"),
        Output("torque_dq", "N m", "torque pelos fluxos dq: 1,5 p (psi_d i_q - psi_q i_d)"),
        Output("ripple", "-", "ondulação pico-a-pico / média"),
        Output("psi_d", "Wb", "fluxo de eixo d"), Output("psi_q", "Wb", "fluxo de eixo q"),
        Output("v_phase_peak", "V", "tensão de fase de pico (sem resistência) na rotação dada"),
        Output("v_limit", "V", "limite de tensão de fase com modulação senoidal: V_dc / sqrt(3)"),
    )
    validity = ("2-D, magnetostático (sem correntes nos ímãs e laminações), aço do rotor/estator conforme a "
                "geometria da lib, 20 °C nos ímãs; perdas e temperatura não acopladas; a tensão desconsidera a queda "
                "resistiva e o efeito de extremidade.")
    pipeline = ("geometria: prius_2004() (estator de 48 ranhuras, ímãs em V)",
                "modelo: magnetostática não linear, correntes dq senoidais",
                "solver: IPMMachine.operating_point (média em posições de rotor), mtpa (seção áurea)",
                "pós: torque, ondulação, psi_d, psi_q, tensão de fase omega_e |psi|",
                "decisão: tensão <= V_dc / sqrt(3)")
    library = ("pinneapple_simulation.electromagnetics.machines.pmsm.IPMMachine",
               "pinneapple_simulation.electromagnetics.machines.verification")
    surrogate_spec = SurrogateSpec(inputs=("I_peak", "gamma_deg"), outputs=("torque", "psi_d", "psi_q"), n_train=40,
                                   n_test=15, bounds={"I_peak": (40.0, 250.0), "gamma_deg": (0.0, 80.0)})
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("torque", "ripple", "v_phase_peak")
    missing = {"reason": "não há caso de máquina elétrica no pinneapple_reason"}

    def solve(self, p):
        m = machine(int(p["n_gap"]))
        op = m.operating_point(I_peak=p["I_peak"], gamma=math.radians(p["gamma_deg"]),
                               n_positions=int(p["n_positions"]), keep=True)
        pp_ = m.pole_pairs
        we = p["rpm"] * 2 * math.pi / 60 * pp_
        v = we * math.hypot(op["psi_d"], op["psi_q"])
        sols = op.get("solutions") or []
        th = m.positions(1, period=m.load_period)[0]
        cur = m.phase_currents(th, p["I_peak"], math.radians(p["gamma_deg"]))
        return {"torque": op["torque_mean"], "torque_dq": op["torque_dq"], "ripple": op["ripple"],
                "psi_d": op["psi_d"], "psi_q": op["psi_q"], "v_phase_peak": v, "v_limit": p["v_dc"] / math.sqrt(3),
                "i_d": op["i_d"], "i_q": op["i_q"], "pole_pairs": pp_,
                "_fields": {"sol": sols[0] if sols else m.solve(th, cur), "machine": m, "theta": th}}

    def checks(self, p, out):
        d = abs(out["torque"] - out["torque_dq"]) / abs(out["torque"])
        return [check("Maxwell-stress torque = dq-flux torque (within 5 %)", d < 0.05, value=d, limit=0.05,
                      source="two post-processings of the same fields"),
                check("current within the 300 A inverter peak rating", p["I_peak"] <= 300, value=p["I_peak"],
                      limit=300, source=ORNL)]

    def complete(self, p, out):
        we = p["rpm"] * 2 * math.pi / 60 * 4
        i_d = -p["I_peak"] * math.sin(math.radians(p["gamma_deg"]))
        i_q = p["I_peak"] * math.cos(math.radians(p["gamma_deg"]))
        return {**out, "torque_dq": 1.5 * 4 * (out["psi_d"] * i_q - out["psi_q"] * i_d),
                "ripple": float("nan"), "v_phase_peak": we * math.hypot(out["psi_d"], out["psi_q"]),
                "v_limit": p["v_dc"] / math.sqrt(3)}

    def decide(self, p, out):
        ok = out["v_phase_peak"] <= out["v_limit"]
        return {"verdict": "OK" if ok else "NOT OK",
                "reasons": [f"torque {out['torque']:.1f} N m a {p['I_peak']:.0f} A, gamma {p['gamma_deg']:.0f}°",
                            f"tensão de fase {out['v_phase_peak']:.0f} V {'<=' if ok else '>'} limite "
                            f"{out['v_limit']:.0f} V a {p['rpm']:.0f} rpm"
                            + ("" if ok else " (precisa de enfraquecimento de campo: mais i_d negativo)")]}

    def validate(self):
        rows = []
        T = []
        m = machine(480)
        for phi0 in YANG_2013_PHI0_DEG:
            T.append(m.operating_point(I_peak=250.0, gamma=math.radians(phi0 - 90.0), n_positions=6)["torque_mean"])
        T = np.array(T)
        ref = YANG_2013_TORQUE_NM
        k = int(np.argmax(ref))
        rows.append({"case": "torque máximo da curva T(ângulo) a 250 A", "output": "torque", "unit": "N m",
                     "computed": float(T.max()), "reference": float(ref.max()),
                     "rel_error": abs(T.max() - ref.max()) / ref.max(), "tol": 0.05,
                     "pass": bool(abs(T.max() - ref.max()) / ref.max() <= 0.05), "kind": "benchmark", "source": YANG,
                     "params": {"I_peak": 250}})
        mae = float(np.mean(np.abs(T - ref)) / ref.max())
        rows.append({"case": "curva T(ângulo) a 250 A, 13 pontos: erro médio / torque máximo", "output": "torque",
                     "unit": "-", "computed": mae, "reference": 0.0, "rel_error": mae, "tol": 0.05,
                     "pass": bool(mae <= 0.05), "kind": "benchmark", "source": YANG, "params": {}})
        rows.append({"case": "ângulo de torque máximo (Phi0)", "output": "phi0", "unit": "deg",
                     "computed": float(YANG_2013_PHI0_DEG[int(np.argmax(T))]), "reference": float(YANG_2013_PHI0_DEG[k]),
                     "rel_error": abs(YANG_2013_PHI0_DEG[int(np.argmax(T))] - YANG_2013_PHI0_DEG[k]) / YANG_2013_PHI0_DEG[k],
                     "tol": 0.1, "pass": bool(abs(int(np.argmax(T)) - k) <= 1), "kind": "benchmark", "source": YANG,
                     "params": {}})
        th = m.positions(3, period=m.load_period)[1]
        cur = m.phase_currents(th, 250.0, math.radians(50.0))
        ta, tc = m.torque(m.solve(th, cur)), m.torque_coenergy(th, cur)
        rows.append({"case": "torque: tensor de Maxwell vs co-energia (250 A, 50°)", "output": "torque", "unit": "N m",
                     "computed": ta, "reference": tc, "rel_error": abs(ta - tc) / abs(tc), "tol": 0.02,
                     "pass": bool(abs(ta - tc) / abs(tc) <= 0.02), "kind": "independent_solver",
                     "source": "trabalho virtual dW'/dtheta (diferenças de 4a ordem)", "params": {}})
        t1 = machine(480).operating_point(I_peak=250.0, gamma=math.radians(40.0), n_positions=6)["torque_mean"]
        t2 = machine(960).operating_point(I_peak=250.0, gamma=math.radians(40.0), n_positions=6)["torque_mean"]
        conv = {"parameter": "nós no entreferro", "values": [480, 960], "torque": [t1, t2],
                "rel_change_last": abs(t2 - t1) / abs(t2), "pass": abs(t2 - t1) / abs(t2) < 0.02}
        self._yang = {"phi0": YANG_2013_PHI0_DEG.tolist(), "fem": T.tolist(), "ref": ref.tolist()}
        return {"preset": self.name, "references": rows, "convergence": conv, "yang_curve": self._yang,
                "all_pass": all(r["pass"] for r in rows) and conv["pass"]}

    def optimize(self, **kw):
        """MTPA angle at the given current by golden section on the FEA, against the surrogate's argmax re-run on
        the FEA."""
        p = self.resolve(**kw)
        m = machine(int(p["n_gap"]))
        g = m.mtpa(p["I_peak"], n_positions=4)
        gdeg = math.degrees(g)
        t_fea = self.run(**{**kw, "gamma_deg": gdeg}).outputs["torque"]
        out = {"objective": "ângulo de máximo torque por ampère (MTPA) na corrente dada", "gamma_mtpa_fea_deg": gdeg,
               "torque_mtpa_fea": t_fea, "torque_at_given_gamma": self.run(**kw).outputs["torque"]}
        from ..surrogate import load_surrogate
        sur = load_surrogate(self.name)
        if sur is not None and not sur.out_of_box({**p, "gamma_deg": 0.0}):
            gs = np.linspace(0, 80, 161)
            tq = sur.predict(np.c_[np.full_like(gs, p["I_peak"]), gs])["torque"][0]
            gsur = float(gs[int(np.argmax(tq))])
            ver = self.run(**{**kw, "gamma_deg": gsur}).outputs["torque"]
            out.update({"gamma_mtpa_surrogate_deg": gsur, "torque_surrogate_pred": float(tq.max()),
                        "torque_fea_at_surrogate_gamma": ver,
                        "surrogate_vs_fea_torque_rel": abs(float(tq.max()) - ver) / ver,
                        "torque_loss_vs_fea_mtpa": (t_fea - ver) / t_fea})
        return out

    def invert(self, true_offset_deg: float = 5.0, noise: float = 0.01, seed: int = 0, **kw):
        """Resolver offset (current-angle error of the drive) from a dynamometer torque sweep at 150 A over five
        commanded angles, 1 % torque-meter noise. Synthetic truth: the drive injects 5 deg more than it commands."""
        p = self.resolve(**kw)
        m = machine(int(p["n_gap"]))
        cmd = np.array([10.0, 25.0, 40.0, 55.0, 70.0])

        def torques(off):
            return np.array([m.operating_point(I_peak=150.0, gamma=math.radians(g + off), n_positions=4)["torque_mean"]
                             for g in cmd])

        rng = np.random.default_rng(seed)
        y0 = torques(true_offset_deg)
        y = y0 * (1 + rng.normal(0, noise, len(cmd)))
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda o: float((((torques(o) - y) / (noise * y0)) ** 2).sum()), bounds=(-15, 15),
                              method="bounded", options={"xatol": 1e-3})
        o = float(res.x)
        J = (torques(o + 0.5) - torques(o - 0.5)) / 1.0 / (noise * y0)
        sd = float(1 / math.sqrt(J @ J))
        return {"description": "Erro de offset do resolver (ângulo de corrente injetado diferente do comandado) a "
                               "partir de uma varredura de torque no dinamômetro a 150 A em 5 ângulos, ruído de 1 % "
                               "no torquímetro. Verdade sintética: +5°.", "parameter": "offset do ângulo",
                "unit": "deg", "true": true_offset_deg, "recovered": o, "ci95": [o - 1.96 * sd, o + 1.96 * sd],
                "rel_error": abs(o - true_offset_deg) / true_offset_deg,
                "true_inside_ci95": bool(abs(o - true_offset_deg) <= 1.96 * sd), "noise": f"{100 * noise} %",
                "how": "mínimos quadrados (Brent) sobre o FEA, IC pela sensibilidade"}

    def variants(self):
        return {"250 A, 40°": {}, "250 A, 20°": {"gamma_deg": 20.0}, "250 A, 60°": {"gamma_deg": 60.0},
                "150 A, 40°": {"I_peak": 150.0}, "250 A, 40°, 3000 rpm": {"rpm": 3000.0}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        y = getattr(self, "_yang", None)
        if y is None:
            self.validate()
            y = self._yang
        ax[0].plot(y["phi0"], y["ref"], "ko", label="Yang et al. 2013")
        ax[0].plot(y["phi0"], y["fem"], "C0-", label="FEA (lib)")
        ax[0].set(xlabel="ângulo da corrente Phi0 [graus]", ylabel="torque [N m]", title="Prius 2004 a 250 A")
        ax[0].legend(fontsize=8)
        ms = result.fields["sol"]
        mesh = ms.mesh
        Bm = np.hypot(ms.B[:, 0], ms.B[:, 1])
        tpc = ax[1].tripcolor(mesh.points[:, 0], mesh.points[:, 1], mesh.tris, facecolors=Bm, cmap="jet",
                              vmax=np.percentile(Bm, 99.5))
        fig.colorbar(tpc, ax=ax[1], label="|B| [T]")
        ax[1].set_aspect("equal")
        ax[1].axis("off")
        ax[1].set_title("|B| no ponto de operação")
        fig.tight_layout()
        path = os.path.join(out_dir, "ipm.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from ..viz3d import Scene, Surface
        ms = result.fields["sol"]
        mesh = ms.mesh
        P, T = mesh.points, mesh.tris
        Bm = np.hypot(ms.B[:, 0], ms.B[:, 1])
        acc, cnt = np.zeros(len(P)), np.zeros(len(P))
        for j in range(3):
            np.add.at(acc, T[:, j], Bm)
            np.add.at(cnt, T[:, j], 1)
        Bn = acc / np.maximum(cnt, 1)
        L = 0.0835                                               # Prius 2004 stack length 83.5 mm (ORNL 2005)
        return Scene([Surface("front", np.c_[P, np.full(len(P), L)] * 100, T, "steel", {"B": Bn}),
                      Surface("back", np.c_[P, np.zeros(len(P))] * 100, T[:, ::-1], "steel", {"B": Bn})],
                     title=self.title, labels={"B": "|B| [T] no ponto de operação"})
