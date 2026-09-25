"""Climate digital twin from Open-Meteo data: the full notebook pipeline, end to end.

Open-Meteo archive -> virtual sensor -> twin state -> PINN (T(t), a, b) -> data and physics
validation -> what-if (+30% solar). Lumped thermal model::

    dT/dt - a*S(t) + b*(T - Tamb(t)) = 0

Three runs:

A. The notebook as written, with the physics loss built by ``PINNFactory`` straight from the
   equation string. S(t) and Tamb(t) are exogenous signals (known inputs) that the factory
   detects on its own; T0 is a runtime constant.
B. The notebook as written, with its hand-written PyTorch loss (cross-check of A: same numbers).
C. The corrected model. A and B cannot learn the physics: Tamb is the observed temperature itself
   (so b is unidentifiable), S is z-scored (negative at night, i.e. the sun "cools"), a and b are
   unconstrained (a comes out negative) and a plain tanh MLP on t in [0, 1] does not fit 7 daily
   cycles (normalised data MSE stays ~1). C uses:
   - T = soil temperature 0-7 cm (the body the sun heats), Tamb = 2 m air temperature,
     S = shortwave radiation / max (>= 0), T and Tamb on one shared scale;
   - a = exp(la), b = exp(lb) written in the equation itself, so both stay positive;
   - daily Fourier features of t as network input;
   - Adam, then SSBroyden refinement (pinneapple_neural.trainer.SelfScaledQuasiNewton).
   In ``--offline`` mode the soil temperature is generated from the same ODE with known a and b,
   so C must recover them.

    python examples/pinn_solver/06_open_meteo_climate_twin.py            # downloads 30 days
    python examples/pinn_solver/06_open_meteo_climate_twin.py --offline  # synthetic weather, no network

Data: Open-Meteo Historical Weather API, https://archive-api.open-meteo.com/v1/archive (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch

from pinneapple_neural.trainer.self_scaled_qn import SelfScaledQuasiNewton
from pinneapple_physics.pinn_solver.factory import (
    PINN, NeuralNetwork, PINNFactory, PINNProblemSpec, TabulatedSignal,
)

HOURLY = ["temperature_2m", "relative_humidity_2m", "pressure_msl", "wind_speed_10m",
          "wind_direction_10m", "precipitation", "shortwave_radiation", "soil_temperature_0_to_7cm"]
RENAME = {"time": "timestamp", "temperature_2m": "temperature", "relative_humidity_2m": "humidity",
          "pressure_msl": "pressure", "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_direction",
          "shortwave_radiation": "solar_radiation", "soil_temperature_0_to_7cm": "soil_temperature"}


# ── cells 5-7: data ──────────────────────────────────────────────────────
def fetch_open_meteo(lat=-5.795, lon=-35.209, start="2026-08-01", end="2026-08-30") -> pd.DataFrame:
    import requests
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "hourly": ",".join(HOURLY),
        "temperature_unit": "celsius", "wind_speed_unit": "kmh", "precipitation_unit": "mm",
        "timezone": "America/Fortaleza"})
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns=RENAME)


SYNTH_A, SYNTH_B = 40.0, 60.0  # true a, b of the offline soil model (units of the 7-day window C trains on)


def synthetic_weather(days=30, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-08-01", periods=days * 24, freq="h")
    h = np.arange(len(ts))
    solar = np.clip(900 * np.sin(2 * np.pi * (h % 24 - 6) / 24), 0, None) * rng.uniform(0.7, 1.0, len(ts))
    temp = 26 + 3 * np.sin(2 * np.pi * (h % 24 - 9) / 24) + rng.normal(0, 0.3, len(ts))
    soil = _soil_from_ode(temp, solar, SYNTH_A, SYNTH_B, window_hours=7 * 24)
    return pd.DataFrame({"soil_temperature": soil,"timestamp": ts, "temperature": temp, "humidity": 75 + rng.normal(0, 5, len(ts)),
                         "pressure": 1012 + rng.normal(0, 1, len(ts)), "wind_speed": 15 + rng.normal(0, 3, len(ts)),
                         "wind_direction": rng.uniform(0, 360, len(ts)), "precipitation": np.zeros(len(ts)),
                         "solar_radiation": solar})


def _soil_from_ode(air, solar, a, b, window_hours, substeps=20):
    """dT/dt = a*S - b*(T - Tamb) in C's units (time scaled by the 7-day window, S/max,
    temperatures on the air scale), integrated with small explicit steps."""
    mu, sd = air[: window_hours].mean(), air[: window_hours].std()
    Sn, An = solar / solar[: window_hours].max(), (air - mu) / sd
    dt = 1.0 / (window_hours - 1) / substeps
    T = np.empty(len(air)); T[0] = An[0]
    for i in range(1, len(air)):
        x = T[i - 1]
        for k in range(substeps):
            w = k / substeps
            S_ = (1 - w) * Sn[i - 1] + w * Sn[i]
            A_ = (1 - w) * An[i - 1] + w * An[i]
            x = x + dt * (a * S_ - b * (x - A_))
        T[i] = x
    return T * sd + mu


# ── cells 8-9: virtual sensor + twin state ──────────────────────────────
class VirtualWeatherSensor:
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)
        self.index = 0

    def reset(self):
        self.index = 0

    def read(self):
        if self.index >= len(self.data):
            return None
        row = self.data.iloc[self.index]
        self.index += 1
        return {"timestamp": row["timestamp"], **{k: float(row[k]) for k in (
            "temperature", "humidity", "pressure", "wind_speed", "wind_direction", "precipitation", "solar_radiation")}}


class ClimateDigitalTwin:
    def __init__(self):
        self.state = {}

    def update(self, measurement):
        self.state = measurement


# ── cells 10-12: normalisation and tensors ──────────────────────────────
def prepare(df: pd.DataFrame, days=7):
    d = df.iloc[: days * 24].copy().reset_index(drop=True)
    d["t"] = np.arange(len(d), dtype=np.float32)
    d["t"] = d["t"] / d["t"].max()
    stats = {"T_mean": d["temperature"].mean(), "T_std": d["temperature"].std()}
    d["T_norm"] = (d["temperature"] - stats["T_mean"]) / stats["T_std"]
    d["Tamb_norm"] = d["T_norm"]  # the notebook uses the same 2 m temperature as Tamb (see report)
    d["S_norm"] = (d["solar_radiation"] - d["solar_radiation"].mean()) / (d["solar_radiation"].std() + 1e-8)
    return d, stats


def make_model(seed):
    torch.manual_seed(seed)
    net = NeuralNetwork(num_inputs=1, num_outputs=1, num_layers=4, num_neurons=64, activation=torch.nn.Tanh())
    return PINN(net, inverse_params_names=["a", "b"], initial_guesses={"a": 0.1, "b": 0.1}, dtype=torch.float32)


# ── A: PINNFactory with exogenous signals ───────────────────────────────
def train_factory(d, epochs, seed=0, n_col=1000):
    spec = PINNProblemSpec(
        pde_residuals=["Derivative(T(t), t) - a*S(t) + b*(T(t) - Tamb(t))"],
        conditions=[{"name": "initial_condition", "equation": "T(t) - T0", "weight": 1.0}],
        independent_vars=["t"], dependent_vars=["T"], inverse_params=["a", "b"],
        loss_weights={"pde": 1.0, "conditions": 10.0, "data": 10.0}, verbose=True)
    factory = PINNFactory(spec)
    loss_fn = factory.generate_loss_function()
    model = make_model(seed)
    g = torch.Generator().manual_seed(seed)
    t_obs = torch.tensor(d["t"].values, dtype=torch.float32).reshape(-1, 1)
    batch = {
        "collocation": (torch.rand(n_col, 1, generator=g).requires_grad_(True),),
        "conditions": [(torch.zeros(1, 1, requires_grad=True),)],
        "data": ((t_obs,), torch.tensor(d["T_norm"].values, dtype=torch.float32).reshape(-1, 1)),
        "exogenous": {"S": TabulatedSignal(d["t"].values, d["S_norm"].values),
                      "Tamb": TabulatedSignal(d["t"].values, d["Tamb_norm"].values)},
        "constants": {"T0": float(d["T_norm"].iloc[0])},
    }
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss, comps = loss_fn(model, batch)
        loss.backward()
        opt.step()
        if ep == 1 or ep % 500 == 0:
            print(f"[A factory] ep {ep:5d} loss={comps['total']:.3e} pde={comps.get('pde', 0):.3e} "
                  f"data={comps.get('data', 0):.3e} a={float(model.inverse_params['a'].detach()):.4f} "
                  f"b={float(model.inverse_params['b'].detach()):.4f}")
    return model


# ── B: the notebook's hand-written loss (cells 17-25) ───────────────────
def train_notebook(d, epochs, seed=0, n_col=1000):
    from scipy.interpolate import interp1d
    model = make_model(seed)
    S_i = interp1d(d["t"].values, d["S_norm"].values, kind="linear", fill_value="extrapolate")
    Ta_i = interp1d(d["t"].values, d["Tamb_norm"].values, kind="linear", fill_value="extrapolate")
    rng = np.random.default_rng(seed)
    t_col_np = rng.uniform(0, 1, size=(n_col, 1)).astype(np.float32)
    t_col = torch.tensor(t_col_np, requires_grad=True)
    S_col = torch.tensor(S_i(t_col_np.ravel()).reshape(-1, 1), dtype=torch.float32)
    Ta_col = torch.tensor(Ta_i(t_col_np.ravel()).reshape(-1, 1), dtype=torch.float32)
    t_ic = torch.zeros((1, 1), requires_grad=True)
    T0 = torch.tensor([[float(d["T_norm"].iloc[0])]])
    t_obs = torch.tensor(d["t"].values.reshape(-1, 1), dtype=torch.float32).requires_grad_(True)
    T_obs = torch.tensor(d["T_norm"].values.reshape(-1, 1), dtype=torch.float32)

    def physics_loss(t, S, Tamb):
        T = model(t)
        dT = torch.autograd.grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        r = dT - model.inverse_params["a"] * S + model.inverse_params["b"] * (T - Tamb)
        return torch.mean(r ** 2)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        lp = physics_loss(t_col, S_col, Ta_col)
        ld = torch.mean((model(t_obs) - T_obs) ** 2)
        li = torch.mean((model(t_ic) - T0) ** 2)
        loss = lp + 10.0 * ld + 10.0 * li
        loss.backward()
        opt.step()
        if ep == 1 or ep % 500 == 0:
            print(f"[B notebook] ep {ep:5d} loss={loss.item():.3e} physics={lp.item():.3e} data={ld.item():.3e} "
                  f"a={float(model.inverse_params['a'].detach()):.4f} b={float(model.inverse_params['b'].detach()):.4f}")
    return model


# ── C: corrected model ──────────────────────────────────────────────────
class DailyFourier(torch.nn.Module):
    """[t, sin(2 pi k D t), cos(2 pi k D t)] for k = 1..K, with D days in t in [0, 1]."""

    def __init__(self, days, harmonics=3):
        super().__init__()
        self.register_buffer("w", 2 * math.pi * days * torch.arange(1, harmonics + 1, dtype=torch.float32))

    def forward(self, t):
        z = t * self.w
        return torch.cat([t, torch.sin(z), torch.cos(z)], dim=1)


def prepare_corrected(df, days=7):
    d = df.iloc[: days * 24].copy().reset_index(drop=True)
    d["t"] = np.arange(len(d), dtype=np.float32) / (len(d) - 1)
    mu, sd = d["temperature"].mean(), d["temperature"].std()  # one scale for T and Tamb
    d["T_norm"] = (d["soil_temperature"] - mu) / sd
    d["Tamb_norm"] = (d["temperature"] - mu) / sd
    d["S_norm"] = d["solar_radiation"] / d["solar_radiation"].max()
    return d, {"T_mean": mu, "T_std": sd, "days": days}


def train_corrected(d, stats, adam_epochs, qn_iters, seed=0, n_col=2000):
    spec = PINNProblemSpec(
        pde_residuals=["Derivative(T(t), t) - exp(la)*S(t) + exp(lb)*(T(t) - Tamb(t))"],
        conditions=[{"name": "initial_condition", "equation": "T(t) - T0", "weight": 1.0}],
        independent_vars=["t"], dependent_vars=["T"], inverse_params=["la", "lb"],
        loss_weights={"pde": 1.0, "conditions": 10.0, "data": 10.0}, verbose=True)
    loss_fn = PINNFactory(spec).generate_loss_function()
    torch.manual_seed(seed)
    net = torch.nn.Sequential(DailyFourier(stats["days"]), NeuralNetwork(7, 1, 3, 32, torch.nn.Tanh()))
    model = PINN(net, inverse_params_names=["la", "lb"], initial_guesses={"la": 1.0, "lb": 1.0})
    t_obs = torch.tensor(d["t"].values, dtype=torch.float32).reshape(-1, 1)
    batch = {
        "collocation": (torch.linspace(0, 1, n_col).reshape(-1, 1).requires_grad_(True),),
        "conditions": [(torch.zeros(1, 1, requires_grad=True),)],
        "data": ((t_obs,), torch.tensor(d["T_norm"].values, dtype=torch.float32).reshape(-1, 1)),
        "exogenous": {"S": TabulatedSignal(d["t"].values, d["S_norm"].values),
                      "Tamb": TabulatedSignal(d["t"].values, d["Tamb_norm"].values)},
        "constants": {"T0": float(d["T_norm"].iloc[0])},
    }

    def report(tag, comps):
        a, b = (float(torch.exp(model.inverse_params[k].detach())) for k in ("la", "lb"))
        print(f"[C corrected] {tag} loss={comps['total']:.3e} pde={comps.get('pde', 0):.3e} "
              f"data={comps.get('data', 0):.3e} a={a:.3f} b={b:.3f}")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(1, adam_epochs + 1):
        opt.zero_grad()
        loss, comps = loss_fn(model, batch)
        loss.backward()
        opt.step()
        if ep == 1 or ep % 1000 == 0:
            report(f"adam {ep:5d}", comps)
    if qn_iters:
        qn = SelfScaledQuasiNewton(model.parameters(), variant="ssbroyden", max_iter=qn_iters)

        def closure():
            qn.zero_grad()
            loss, _ = loss_fn(model, batch)
            loss.backward()
            return loss

        qn.step(closure)
        report(f"ssbroyden +{qn.state['n_iter']} it", loss_fn(model, batch)[1])
    return model


def evaluate_corrected(model, d, stats, offline):
    t = torch.tensor(d["t"].values, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        T_pred = model(t).numpy().ravel() * stats["T_std"] + stats["T_mean"]
    err = T_pred - d["soil_temperature"].values
    tt = t.clone().requires_grad_(True)
    Tn = model(tt)
    dT = torch.autograd.grad(Tn, tt, grad_outputs=torch.ones_like(Tn))[0]
    a = torch.exp(model.inverse_params["la"]).detach()
    b = torch.exp(model.inverse_params["lb"]).detach()
    S = torch.tensor(d["S_norm"].values, dtype=torch.float32).reshape(-1, 1)
    Ta = torch.tensor(d["Tamb_norm"].values, dtype=torch.float32).reshape(-1, 1)
    phys = float(torch.sqrt(torch.mean((dT.detach() - a * S + b * (Tn.detach() - Ta)) ** 2)))
    a_, b_ = float(a), float(b)
    hours = (len(d) - 1)

    def simulate(Sv, substeps=20):
        T = np.empty(len(d)); T[0] = d["T_norm"].iloc[0]
        dt = 1.0 / hours / substeps
        for i in range(1, len(d)):
            x = T[i - 1]
            for k in range(substeps):
                w = k / substeps
                x = x + dt * (a_ * ((1 - w) * Sv[i - 1] + w * Sv[i])
                              - b_ * (x - ((1 - w) * d["Tamb_norm"].values[i - 1] + w * d["Tamb_norm"].values[i])))
            T[i] = x
        return T * stats["T_std"] + stats["T_mean"]

    base = simulate(d["S_norm"].values)
    whatif = simulate(d["S_norm"].values * 1.30)
    sim_err = base - d["soil_temperature"].values
    res = {"path": "C: corrected (soil vs air, S>=0, a,b>0, Fourier, SSBroyden)", "a": a_, "b": b_,
           "time_constant_h": hours / b_,
           "MAE_C": float(np.mean(np.abs(err))), "RMSE_C": float(np.sqrt(np.mean(err ** 2))),
           "physics_residual_rmse": phys,
           "ode_with_learned_a_b_RMSE_C": float(np.sqrt(np.mean(sim_err ** 2))),
           "whatif_+30pct_solar_mean_delta_C": float(np.mean(whatif - base)),
           "whatif_+30pct_solar_max_delta_C": float(np.max(whatif - base))}
    if offline:
        res.update(a_true=SYNTH_A, b_true=SYNTH_B)
    print(res)
    return res


# ── cells 28-36: validation and what-if ─────────────────────────────────
def evaluate(model, d, stats, label):
    t = torch.tensor(d["t"].values, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        T_pred = model(t).numpy().ravel() * stats["T_std"] + stats["T_mean"]
    err = T_pred - d["temperature"].values
    mae, rmse = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))
    tt = t.clone().requires_grad_(True)
    Tn = model(tt)
    dT = torch.autograd.grad(Tn, tt, grad_outputs=torch.ones_like(Tn))[0]
    S = torch.tensor(d["S_norm"].values, dtype=torch.float32).reshape(-1, 1)
    Ta = torch.tensor(d["Tamb_norm"].values, dtype=torch.float32).reshape(-1, 1)
    a, b = model.inverse_params["a"], model.inverse_params["b"]
    phys = float(torch.sqrt(torch.mean((dT - a * S + b * (Tn - Ta)) ** 2)))
    a_, b_ = float(a.detach()), float(b.detach())

    def euler(Sv):
        T = np.zeros(len(d)); T[0] = d["T_norm"].iloc[0]
        dt = d["t"].values[1] - d["t"].values[0]
        for i in range(1, len(d)):
            T[i] = T[i - 1] + dt * (a_ * Sv[i - 1] - b_ * (T[i - 1] - d["Tamb_norm"].values[i - 1]))
        return T * stats["T_std"] + stats["T_mean"]

    base, whatif = euler(d["S_norm"].values), euler(d["S_norm"].values * 1.30)
    res = {"path": label, "a": a_, "b": b_, "MAE_C": mae, "RMSE_C": rmse, "physics_residual_rmse": phys,
           "whatif_final_delta_C": float(whatif[-1] - T_pred[-1]),
           "whatif_mean_delta_vs_euler_baseline_C": float(np.mean(whatif - base))}
    print(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--corrected-epochs", type=int, default=6000)
    ap.add_argument("--qn-iters", type=int, default=300)
    ap.add_argument("--only-corrected", action="store_true")
    args = ap.parse_args()
    df = synthetic_weather() if args.offline else fetch_open_meteo()
    print("data:", df.shape, "| NaN:", int(df.isna().sum().sum()))
    sensor, twin = VirtualWeatherSensor(df), ClimateDigitalTwin()
    twin.update(sensor.read())
    print("twin state:", {k: twin.state[k] for k in ("timestamp", "temperature", "solar_radiation")})
    results = []
    if not args.only_corrected:
        d, stats = prepare(df)
        results += [evaluate(train_factory(d, args.epochs), d, stats, "A: PINNFactory + exogenous"),
                    evaluate(train_notebook(d, args.epochs), d, stats, "B: notebook custom loss")]
    dc, sc = prepare_corrected(df)
    model = train_corrected(dc, sc, args.corrected_epochs, args.qn_iters)
    results.append(evaluate_corrected(model, dc, sc, args.offline))
    return results


if __name__ == "__main__":
    main()
