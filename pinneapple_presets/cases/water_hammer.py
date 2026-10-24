"""Water hammer from a valve closure at the end of a steel water main (method of characteristics)."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from pinneapple_physics.closed_form.pipe_flow import colebrook
from pinneapple_simulation.numerical_solvers.water_hammer_moc import (
    G,
    joukowsky_head,
    michaud_head,
    valve_closure_moc,
    wave_speed,
)

from ..core import Output, Param, Preset, Reference, SurrogateSpec, check

WS = "Wylie & Streeter, Fluid Transients in Systems (1993)"


def rigid_column_max_head(*, length, diameter, friction_factor, q0, h_reservoir, h_downstream, t_close,
                          closure_exponent=1.0) -> float:
    """Peak valve head of the same closure by rigid-column theory (incompressible liquid, rigid pipe; Wylie &
    Streeter 1993 sec. 1.3): ``(L / g A) dQ/dt = H_res - h_f(Q) - H_valve``, valve ``Q = tau Q0 sqrt(dH/dH0)``.
    Valid when the closure is much slower than 2L/a; an independent check of the MOC there."""
    A = math.pi * diameter ** 2 / 4
    r = friction_factor * length / (2 * G * diameter * A ** 2)
    dh0 = h_reservoir - r * q0 ** 2 - h_downstream

    def hv(t, q):
        tau = max(1 - t / t_close, 0.0) ** closure_exponent
        return h_downstream + (dh0 * (q / (tau * q0)) ** 2 if tau > 1e-9 else np.inf)

    def rhs(t, y):
        q = y[0]
        tau = max(1 - t / t_close, 0.0) ** closure_exponent
        if tau < 1e-6:
            return [0.0]
        return [G * A / length * (h_reservoir - r * q * abs(q) - hv(t, q))]

    ts = np.linspace(0, t_close * (1 - 1e-4), 4000)
    sol = solve_ivp(rhs, (0, ts[-1]), [q0], t_eval=ts, rtol=1e-9, atol=1e-12, max_step=t_close / 2000)
    h = [hv(t, q) for t, q in zip(sol.t, sol.y[0], strict=False)]
    return float(np.max(h))


class WaterHammer(Preset):
    name = "water_hammer_valve_closure"
    title = "Golpe de aríete no fechamento de válvula de adutora de aço"
    sector = "Fluidos / saneamento e óleo e gás"
    problem = ("Adutora de aço DN500 (reservatório a montante, válvula na ponta) fecha a válvula em t_c segundos. "
               "Calcula a sobrepressão e a subpressão ao longo da linha pelo método das características.")
    question = ("Com este tempo de fechamento, a pressão máxima fica abaixo da classe PN da linha e a mínima acima da "
                "pressão de vapor (sem separação de coluna)? Qual o tempo mínimo de fechamento seguro?")
    params = (
        Param("length", "m", 1200.0, "premissa: adutora típica de 1,2 km entre reservatório e válvula", 100, 20000),
        Param("od", "m", 0.508, "ASME B36.10M, tubo NPS 20 (508 mm de diâmetro externo)", 0.05, 2.0),
        Param("wall", "m", 0.00953, "ASME B36.10M, NPS 20 parede STD (9,53 mm)", 0.002, 0.05),
        Param("velocity", "m/s", 1.5, "AWWA M11 (4a ed.) cap. 5: velocidades de projeto típicas 0,9-1,8 m/s",
              0.2, 4.0),
        Param("h_reservoir", "m", 60.0, "premissa: nível do reservatório 60 m acima do eixo da válvula", 5, 500),
        Param("t_close", "s", 10.0, "premissa: atuador de válvula borboleta de 10 s", 0.0, 120.0),
        Param("closure_exponent", "-", 1.0, "lei de fechamento tau=(1-t/tc)^Em; Em=1 linear (Wylie & Streeter 1993)",
              0.5, 4.0),
        Param("roughness", "m", 4.5e-5, "aço comercial, e = 0,045 mm (White, Fluid Mechanics, Tabela 6.1)",
              0.0, 3e-3),
        Param("friction", "-", "colebrook", "atrito de Darcy por Colebrook; 'none' só para o caso de referência",
              choices=("colebrook", "none")),
        Param("E_pipe", "Pa", 207e9, "aço carbono, E = 207 GPa (ASME B31.3 Tabela C-6, temperatura ambiente)",
              1e9, 250e9),
        Param("nu_pipe", "-", 0.3, "aço, coeficiente de Poisson 0,3", 0.2, 0.5),
        Param("K_water", "Pa", 2.19e9, "água a 20 °C, módulo volumétrico 2,19 GPa (Wylie & Streeter 1993, Tab. 2.1)",
              1.5e9, 2.5e9),
        Param("rho", "kg/m3", 998.2, "água a 20 °C (Incropera, Tabela A.6)", 950, 1050),
        Param("pn_rating", "Pa", 1.6e6, "classe PN16 (EN 1092-1): 1,6 MPa manométrica", 0.4e6, 10e6),
        Param("p_vapour_abs", "Pa", 2339.0, "pressão de vapor da água a 20 °C (Incropera, Tabela A.6)", 600, 1e5),
        Param("n_reaches", "-", 40, "malha do MOC (Courant 1); convergência verificada em 10-160", 10, 400),
    )
    outputs = (
        Output("wave_speed", "m/s", "celeridade da onda (Korteweg, tubo ancorado)"),
        Output("period", "s", "2L/a"),
        Output("max_head", "m", "carga piezométrica máxima (envoltória)"),
        Output("surge_head", "m", "sobrepressão máxima acima da carga estática na válvula"),
        Output("min_head", "m", "carga mínima (envoltória)"),
        Output("drawdown", "m", "subpressão máxima abaixo da carga estática na válvula"),
        Output("p_max", "Pa", "pressão manométrica máxima"),
        Output("p_min_abs", "Pa", "pressão absoluta mínima"),
        Output("joukowsky_head", "m", "aV/g"),
        Output("michaud_head", "m", "estimativa de Michaud"),
        Output("t_close_safe", "s", "menor tempo de fechamento (lei atual) com p_max <= PN"),
    )
    validity = ("Escoamento 1-D, líquido levemente compressível, tubo elástico de parede fina (D/e > 25), atrito "
                "permanente. Sem modelo de cavitação: depois que a carga cai à pressão de vapor a história não é "
                "física (o preset avisa). Sem bolsões de ar nem ventosas.")
    pipeline = ("geometria: D = od - 2 wall, A",
                "modelo: wave_speed(K, rho, D, e, E, 1-nu^2) (Korteweg); f = colebrook(Re, e/D)",
                "solver: valve_closure_moc(...) (MOC, Courant 1)",
                "pós: envoltórias h_max/h_min, p = rho g h",
                "decisão: p_max <= PN e p_min_abs > p_vapor; t_c seguro por bisseção do MOC")
    library = ("pinneapple_simulation.numerical_solvers.water_hammer_moc.valve_closure_moc",
               "pinneapple_physics.closed_form.pipe_flow.colebrook")
    surrogate_spec = SurrogateSpec(inputs=("t_close", "velocity", "length"), outputs=("surge_head", "drawdown"),
                                   n_train=120, n_test=30, log=("t_close", "length"),
                                   log_outputs=("surge_head", "drawdown"),
                                   bounds={"t_close": (0.5, 30.0), "velocity": (0.5, 3.0), "length": (300.0, 4000.0)})
    has_3d = True
    has_optimization = True
    has_inversion = True
    has_variants = True
    variant_outputs = ("p_max", "p_min_abs", "t_close_safe")
    variant_gallery = False
    missing = {}
    reason_case = "pumped_main"
    limitations = ("Atrito permanente: picos posteriores ficam pouco amortecidos (o primeiro pico, que dimensiona, "
                   "não muda).",)

    # --------------------------------------------------------------------------------------------- solver
    def _setup(self, p):
        D = p["od"] - 2 * p["wall"]
        A = math.pi * D ** 2 / 4
        a = wave_speed(p["K_water"], p["rho"], D, p["wall"], p["E_pipe"], 1 - p["nu_pipe"] ** 2)
        mu = 1.002e-3                                    # water 20 C, Incropera Table A.6
        Re = p["rho"] * p["velocity"] * D / mu
        f = 0.0 if p["friction"] == "none" else colebrook(Re, p["roughness"] / D)
        return D, A, a, Re, f

    def _moc(self, p, t_close=None, n=None, field=False):
        D, A, a, Re, f = self._setup(p)
        return valve_closure_moc(length=p["length"], diameter=D, wave_speed=a, friction_factor=f,
                                 q0=p["velocity"] * A, h_reservoir=p["h_reservoir"], h_downstream=0.0,
                                 t_close=p["t_close"] if t_close is None else t_close,
                                 closure_exponent=p["closure_exponent"], n_reaches=int(n or p["n_reaches"]),
                                 t_end=(p["t_close"] if t_close is None else t_close) + 6 * 2 * p["length"] / a,
                                 record_field=field)

    def _safe_tc(self, p, h_allow):
        D, A, a, Re, f = self._setup(p)
        hi = 2 * p["length"] / a
        for _ in range(12):                              # bracket: grow until safe
            if self._moc(p, hi)["max_head"] <= h_allow:
                break
            hi *= 2
        else:
            return float("inf")
        lo = 0.0
        if self._moc(p, lo)["max_head"] <= h_allow:
            return 0.0
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if self._moc(p, mid)["max_head"] <= h_allow else (mid, hi)
        return hi

    def solve(self, p):
        D, A, a, Re, f = self._setup(p)
        r = self._moc(p, field=True)
        g_rho = p["rho"] * G
        h_static = r["h_steady"][-1]
        p_atm = 101325.0
        h_allow = p["pn_rating"] / g_rho
        out = {"inner_diameter": D, "wave_speed": a, "period": r["period"], "reynolds": Re, "friction_factor": f,
               "max_head": r["max_head"], "max_head_x": r["max_head_at"], "max_head_t": r["max_head_time"],
               "surge_head": r["max_head"] - h_static, "min_head": r["min_head"],
               "drawdown": h_static - r["min_head"],
               "p_max": r["max_head"] * g_rho, "p_min_abs": r["min_head"] * g_rho + p_atm,
               "joukowsky_head": joukowsky_head(a, p["velocity"]),
               "michaud_head": michaud_head(a, p["velocity"], p["length"], max(p["t_close"], 1e-9)),
               "h_static_valve": h_static, "h_allow": h_allow,
               "t_close_safe": self._safe_tc(p, h_allow),
               "_fields": {"t": r["t"], "h_valve": r["h_valve"], "q_valve": r["q_valve"], "x": r["x"],
                           "h_max": r["h_max"], "h_min": r["h_min"], "h_steady": r["h_steady"],
                           "history": r.get("history")}}
        return out

    def complete(self, p, out):
        D, A, a, Re, f = self._setup(p)
        g_rho = p["rho"] * G
        h_static = p["h_reservoir"] - f * p["length"] / D * p["velocity"] ** 2 / (2 * G)
        return {**out, "wave_speed": a, "period": 2 * p["length"] / a, "max_head": out["surge_head"] + h_static,
                "p_max": (out["surge_head"] + h_static) * g_rho, "min_head": h_static - out["drawdown"],
                "p_min_abs": (h_static - out["drawdown"]) * g_rho + 101325.0,
                "h_static_valve": h_static, "t_close_safe": float("nan")}

    def checks(self, p, out):
        D = out["inner_diameter"]
        c = [check("Courant = 1 (MOC grid dt = dx/a)", True, source="valve_closure_moc"),
             check("thin wall D/e > 25 (Korteweg)", D / p["wall"] > 25, value=D / p["wall"], limit=25,
                   source=WS + " eq. 2.29"),
             check("surge does not exceed Joukowsky", out["surge_head"] <= out["joukowsky_head"] * 1.02 +
                   0.05 * out["h_static_valve"], value=out["surge_head"], limit=out["joukowsky_head"],
                   source="Joukowsky (1898): the largest surge of a single velocity change"),
             check("no column separation (p_min > p_vapour)", out["p_min_abs"] > p["p_vapour_abs"],
                   value=out["p_min_abs"], limit=p["p_vapour_abs"],
                   reason="below vapour pressure the MOC history after that instant is not physical")]
        return c

    def decide(self, p, out):
        ok_p = out["p_max"] <= p["pn_rating"]
        ok_v = out["p_min_abs"] > p["p_vapour_abs"]
        reasons = [f"p_max = {out['p_max'] / 1e6:.2f} MPa {'<=' if ok_p else '>'} PN {p['pn_rating'] / 1e6:.2f} MPa",
                   f"p_min_abs = {out['p_min_abs'] / 1e3:.1f} kPa {'>' if ok_v else '<='} p_vapour "
                   f"{p['p_vapour_abs'] / 1e3:.2f} kPa",
                   f"tempo de fechamento seguro (mesma lei) >= {out['t_close_safe']:.1f} s"]
        if math.isnan(out["t_close_safe"]):
            reasons.pop()
        return {"verdict": "OK" if ok_p and ok_v else "NOT OK", "reasons": reasons}

    # --------------------------------------------------------------------------------------------- validation
    def references(self):
        p = self.defaults()
        D = p["od"] - 2 * p["wall"]
        a = wave_speed(p["K_water"], p["rho"], D, p["wall"], p["E_pipe"], 1 - p["nu_pipe"] ** 2)
        refs = [Reference("fechamento instantâneo sem atrito = Joukowsky", {"t_close": 0.0, "friction": "none"},
                          "surge_head", joukowsky_head(a, p["velocity"]), "m",
                          "Joukowsky (1898); " + WS + " eq. 1.11: dH = a dV / g", "analytic", 0.005)]
        tc = 60.0                                        # 30 x 2L/a: rigid-column regime
        A = math.pi * D ** 2 / 4
        Re = p["rho"] * p["velocity"] * D / 1.002e-3
        f = colebrook(Re, p["roughness"] / D)
        h_rc = rigid_column_max_head(length=p["length"], diameter=D, friction_factor=f, q0=p["velocity"] * A,
                                     h_reservoir=p["h_reservoir"], h_downstream=0.0, t_close=tc)
        r_f = f * p["length"] / D * p["velocity"] ** 2 / (2 * G)
        refs.append(Reference("fechamento lento (60 s = 30·2L/a) = teoria da coluna rígida", {"t_close": tc},
                              "surge_head", h_rc - (p["h_reservoir"] - r_f), "m",
                              WS + " sec. 1.3 (rigid-column theory), integrada com scipy solve_ivp",
                              "independent_solver", 0.05))
        return refs

    def convergence(self):
        p = self.defaults()
        ns = [10, 20, 40, 80, 160]
        h = [self._moc(p, n=n)["max_head"] for n in ns]
        rel = abs(h[-2] - h[-1]) / abs(h[-1])
        return {"parameter": "n_reaches", "values": ns, "max_head": h, "rel_change_last": rel, "pass": rel < 0.01}

    # --------------------------------------------------------------------------------------------- extras
    def optimize(self, **kw):
        """Shortest closure time with p_max <= PN: by bisection on the MOC, and on the surrogate (when trained)
        with that optimum re-run on the MOC (the measured surrogate-vs-solver difference)."""
        p = self.resolve(**kw)
        h_allow = p["pn_rating"] / (p["rho"] * G)
        tc = self._safe_tc(p, h_allow)
        out = {"objective": "menor tempo de fechamento (operação mais rápida) com p_max <= PN",
               "t_close_opt_solver": tc, "p_max_at_opt_solver": self._moc(p, tc)["max_head"] * p["rho"] * G,
               "pn_rating": p["pn_rating"]}
        from ..surrogate import load_surrogate
        sur = load_surrogate(self.name)
        if sur is not None and all(sur.bounds[k][0] <= p[k] <= sur.bounds[k][1] for k in ("velocity", "length")):
            h_static = self._moc(p, 1e3, n=10)["h_steady"][-1]
            lo, hi = sur.bounds["t_close"]
            def f(t):
                return sur.predict_one({**p, "t_close": t})[0]["surge_head"] + h_static - h_allow
            if f(lo) > 0 > f(hi):
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
                pm = self._moc(p, hi)["max_head"] * p["rho"] * G
                out.update({"t_close_opt_surrogate": hi, "p_max_at_surrogate_opt_solver": pm,
                            "surrogate_opt_rel_diff_t": abs(hi - tc) / tc,
                            "surrogate_opt_feasible_on_solver": bool(pm <= p["pn_rating"])})
        return out

    def invert(self, true_factor: float = 0.82, noise_kpa: float = 5.0, seed: int = 0, **kw):
        """Effective wave speed from a measured valve-pressure history (transient-based condition assessment:
        entrained air or a weakened wall lower ``a``). Synthetic measurement: the MOC at ``a_true = true_factor *
        a_design`` plus white noise; least squares on the MOC over ``a``; 95 % interval from the Gauss-Newton
        curvature with the residual variance."""
        from scipy.optimize import minimize_scalar
        p = self.resolve(**kw)
        D, A, a0, Re, f = self._setup(p)
        tc = 2 * p["length"] / a0                       # fast closure: the wave period is visible
        t_meas = np.linspace(0, tc + 8 * 2 * p["length"] / (true_factor * a0), 400)
        g_rho = p["rho"] * G

        def p_valve(a):
            r = valve_closure_moc(length=p["length"], diameter=D, wave_speed=a, friction_factor=f,
                                  q0=p["velocity"] * A, h_reservoir=p["h_reservoir"], t_close=tc,
                                  closure_exponent=p["closure_exponent"], n_reaches=40, t_end=t_meas[-1] + 1)
            return np.interp(t_meas, r["t"], r["h_valve"]) * g_rho

        a_true = true_factor * a0
        rng = np.random.default_rng(seed)
        y = p_valve(a_true) + rng.normal(0, noise_kpa * 1e3, len(t_meas))
        def sse(a):
            return float(((p_valve(a) - y) ** 2).sum())
        grid = np.linspace(0.4 * a0, 1.1 * a0, 36)          # coarse scan avoids the periodic local minima
        a_g = grid[int(np.argmin([sse(a) for a in grid]))]
        da = grid[1] - grid[0]
        res = minimize_scalar(sse, bounds=(a_g - da, a_g + da), method="bounded", options={"xatol": 1e-3})
        a_hat = float(res.x)
        h = 1e-3 * a_hat
        J = (p_valve(a_hat + h) - p_valve(a_hat - h)) / (2 * h)
        s2 = res.fun / (len(y) - 1)
        sd = float(np.sqrt(s2 / (J @ J)))
        return {"description": "Celeridade efetiva da onda a partir do histórico de pressão medido na válvula "
                               "(ar dissolvido ou parede degradada reduzem a). Medição sintética: MOC com "
                               f"a_real = {true_factor} x a_projeto + ruído branco de {noise_kpa} kPa.",
                "parameter": "wave_speed", "unit": "m/s", "true": a_true, "recovered": a_hat,
                "ci95": [a_hat - 1.96 * sd, a_hat + 1.96 * sd], "rel_error": abs(a_hat - a_true) / a_true,
                "true_inside_ci95": bool(abs(a_hat - a_true) <= 1.96 * sd), "noise": f"{noise_kpa} kPa (1 sigma)",
                "how": "mínimos quadrados sobre o MOC (varredura + Brent), IC 95 % por Gauss-Newton",
                "n_samples": len(y)}

    def variants(self):
        """Closure laws and times a plant engineer actually chooses between."""
        return {"linear 10 s": {"closure_exponent": 1.0, "t_close": 10.0},
                "linear 2 s": {"closure_exponent": 1.0, "t_close": 2.0},
                "Em=2 (rápido no início) 10 s": {"closure_exponent": 2.0, "t_close": 10.0},
                "Em=0.5 (rápido no fim) 10 s": {"closure_exponent": 0.5, "t_close": 10.0},
                "linear 30 s": {"closure_exponent": 1.0, "t_close": 30.0}}

    def figures(self, result, out_dir):
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        F = result.fields
        g_rho = result.params["rho"] * G / 1e6
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(F["t"], np.asarray(F["h_valve"]) * g_rho, lw=1.2)
        ax[0].axhline(result.params["pn_rating"] / 1e6, color="r", ls="--", lw=1, label="PN")
        ax[0].set(xlabel="t [s]", ylabel="p na válvula [MPa]", title="Pressão na válvula")
        ax[0].legend()
        x = F["x"]
        for k, lab in (("h_max", "máx"), ("h_min", "mín"), ("h_steady", "regime")):
            ax[1].plot(x, np.asarray(F[k]) * g_rho, label=lab)
        ax[1].axhline(result.params["pn_rating"] / 1e6, color="r", ls="--", lw=1)
        ax[1].set(xlabel="x [m]", ylabel="p [MPa]", title="Envoltórias de pressão")
        ax[1].legend()
        fig.tight_layout()
        path = os.path.join(out_dir, "pressure.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    def scene(self, result):
        from pinneapple_tools.visualization.studio.animation import Animation

        from ..viz3d import Scene, Surface, box, tube
        F = result.fields
        p = result.params
        x = np.asarray(F["x"])
        L = p["length"]
        # display: the route drawn 100 units long with the pipe radius 1.2 (not to scale; the 1-D model depends
        # on the length, not on the route)
        s = x / L
        xs = np.linspace(0, 1, 200)
        path = np.c_[100 * xs, 12 * np.sin(2 * np.pi * xs) * xs, 3 * np.sin(np.pi * xs)]
        V, Fc, idx_fine = tube(path, 1.2, 24)
        idx = np.clip(np.searchsorted(s, xs[idx_fine], side="left"), 0, len(x) - 1)
        g_rho = p["rho"] * G / 1e6
        pipe = Surface("pipe", V, Fc, "steel", {"p_max": np.asarray(F["h_max"])[idx] * g_rho,
                                                "p_min": np.asarray(F["h_min"])[idx] * g_rho})
        rv, rf = box([-14, -8, -6], [0, 8, 6])
        vv, vf = box([99, -2.5, -2.5], [104, 2.5, 2.5])
        sc = Scene([pipe, Surface("reservoir", rv, rf, "concrete"), Surface("valve", vv, vf, "dark")],
                   title=self.title, labels={"p_max": "pressão máxima [MPa]", "p_min": "pressão mínima [MPa]"})
        hist = F.get("history")
        if hist is not None:
            th, H = np.asarray(hist["t"]), np.asarray(hist["h"])
            k = np.linspace(0, len(th) - 1, min(120, len(th))).astype(int)
            an = Animation(th[k])
            an.add_surface_field("pipe", H[k][:, idx] * g_rho, label="pressão [MPa]")
            an.add_hud("p_valve", H[k, -1] * g_rho, "p na válvula", "MPa", "{:.2f}")
            sc.animation = an
        return sc
