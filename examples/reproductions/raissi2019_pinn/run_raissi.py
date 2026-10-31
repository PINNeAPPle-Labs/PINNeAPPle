"""Reproduce the continuous-time PINN results of Raissi, Perdikaris & Karniadakis (arXiv:1711.10561v1) with the
protocol frozen in TARGETS.md.

    python examples/reproductions/raissi2019_pinn/run_raissi.py --part b0          # Burgers headline, 10 seeds
    python examples/reproductions/raissi2019_pinn/run_raissi.py --part table2      # Table 2, 3 seeds per cell
    python examples/reproductions/raissi2019_pinn/run_raissi.py --part table1      # Table 1, 3 seeds per cell
    python examples/reproductions/raissi2019_pinn/run_raissi.py --part s0          # Schrodinger, 3 seeds
    python examples/reproductions/raissi2019_pinn/run_raissi.py --part extras      # labelled variants (TARGETS 2, 4)
    python examples/reproductions/raissi2019_pinn/run_raissi.py --report
    python examples/reproductions/raissi2019_pinn/run_raissi.py --smoke            # tiny budgets, nothing kept

Every run writes ``runs/<name>.json`` (git-ignored) and is skipped when that file exists; ``--report`` gathers them
into all_runs.json, RESULTS.md, a figure and experiment.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn

from pinneapple_analysis.reproduction import seed_stats, status
from pinneapple_neural.trainer.scipy_lbfgsb import minimize_lbfgsb
from pinneapple_physics.paper_problems_raissi import (
    BURGERS_NU,
    burgers_candidates,
    burgers_grid,
    nls_grid,
    nls_reference,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
NU_ROWS, NF_COLS = (20, 40, 60, 80, 100, 200), (2000, 4000, 6000, 7000, 8000, 10000)
LAYER_ROWS, WIDTH_COLS = (2, 4, 6, 8), (10, 20, 40)
TABLE1 = dict(zip(NU_ROWS, [(2.9e-01, 4.4e-01, 8.9e-01, 1.2e+00, 9.9e-02, 4.2e-02),
                            (6.5e-02, 1.1e-02, 5.0e-01, 9.6e-03, 4.6e-01, 7.5e-02),
                            (3.6e-01, 1.2e-02, 1.7e-01, 5.9e-03, 1.9e-03, 8.2e-03),
                            (5.5e-03, 1.0e-03, 3.2e-03, 7.8e-03, 4.9e-02, 4.5e-03),
                            (6.6e-02, 2.7e-01, 7.2e-03, 6.8e-04, 2.2e-03, 6.7e-04),
                            (1.5e-01, 2.3e-03, 8.2e-04, 8.9e-04, 6.1e-04, 4.9e-04)], strict=True))
TABLE2 = dict(zip(LAYER_ROWS, [(7.4e-02, 5.3e-02, 1.0e-01), (3.0e-03, 9.4e-04, 6.4e-04),
                               (9.6e-03, 1.3e-03, 6.1e-04), (2.5e-03, 9.6e-04, 5.6e-04)], strict=True))
B0_TARGET, S0_TARGET = 6.7e-4, 1.97e-3


class MLP(nn.Module):
    """The official network: inputs scaled to [-1, 1] by the bounds, tanh hidden layers, linear output; weights
    ~ truncated normal (2 std) with std sqrt(2 / (fan_in + fan_out)), biases 0."""

    def __init__(self, layers, lb, ub):
        super().__init__()
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))
        self.lin = nn.ModuleList(nn.Linear(a, b) for a, b in zip(layers[:-1], layers[1:], strict=True))
        for m in self.lin:
            std = math.sqrt(2.0 / (m.in_features + m.out_features))
            nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(m.bias)

    def forward(self, X):
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for m in self.lin[:-1]:
            h = torch.tanh(m(h))
        return self.lin[-1](h)


def _grad(y, X, col):
    return torch.autograd.grad(y, X, torch.ones_like(y), create_graph=True)[0][:, col:col + 1]


def _lhs(n, lb, ub, rng):
    from scipy.stats import qmc
    return np.asarray(lb) + (np.asarray(ub) - np.asarray(lb)) * qmc.LatinHypercube(d=2, seed=rng).random(n)


def _save(name, out):
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, name + ".json"), "w") as fh:
        json.dump(out, fh, indent=1)


def _done(name):
    path = os.path.join(RUNS, name + ".json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def run_burgers(n_u, n_f, depth, width, seed, *, maxiter=50000, name=None, keep=True, dtype=torch.float32):
    """One Burgers run (``Burgers.py`` / ``main_loop`` of ``Burgers_systematic.py``); ``n_f = 0`` drops the
    physics term (the paper's footnote 1 baseline)."""
    if keep and name and (r := _done(name)):
        return r
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    lb, ub = np.array([-1.0, 0.0]), np.array([1.0, 0.99])
    Xc, Uc = burgers_candidates()
    idx = rng.choice(Xc.shape[0], n_u, replace=False)
    Xu = torch.tensor(Xc[idx], dtype=dtype)
    uu = torch.tensor(Uc[idx, None], dtype=dtype)
    Xf = torch.tensor(np.vstack([_lhs(n_f, lb, ub, rng), Xc]) if n_f else Xc[:0], dtype=dtype,
                      requires_grad=True)                       # all candidates appended, as Burgers.py does
    net = MLP([2] + [width] * depth + [1], lb, ub).to(dtype)

    def loss_fn():
        loss = ((net(Xu) - uu) ** 2).mean()
        if n_f:
            u = net(Xf)
            u_x = _grad(u, Xf, 0)
            f = _grad(u, Xf, 1) + u * u_x - BURGERS_NU * _grad(u_x, Xf, 0)
            loss = loss + (f ** 2).mean()
        return loss

    res = minimize_lbfgsb(net.parameters(), loss_fn, maxiter=maxiter, maxfun=maxiter)
    x, t, u_star = burgers_grid()
    X, T = np.meshgrid(x, t)
    with torch.no_grad():
        u_pred = net(torch.tensor(np.stack([X.ravel(), T.ravel()], 1), dtype=dtype)).numpy().ravel()
    err = float(np.linalg.norm(u_star.ravel() - u_pred) / np.linalg.norm(u_star.ravel()))
    out = {"problem": "burgers", "dtype": str(dtype).replace("torch.", ""), "n_u": n_u, "n_f": n_f, "depth": depth, "width": width, "seed": seed,
           "error_u": err, "loss": res.loss, "iterations": res.iterations, "evaluations": res.evaluations,
           "message": res.message, "seconds": res.seconds, "n_params": sum(p.numel() for p in net.parameters()),
           "machine": platform.platform(), "threads": torch.get_num_threads()}
    if keep and name:
        _save(name, out)
        if seed == 0 and (n_u, n_f, depth, width) == (100, 10000, 8, 20):
            np.save(os.path.join(RUNS, name + "_pred.npy"), u_pred.reshape(T.shape).astype(np.float32))
    return out


_NLS = {}


def run_schrodinger(seed, *, adam_steps=50000, maxiter=50000, name=None, keep=True):
    """One Schrodinger run (``Schrodinger.py``): Adam (lr 1e-3) for ``adam_steps``, then L-BFGS-B."""
    if keep and name and (r := _done(name)):
        return r
    if "h" not in _NLS:
        _NLS["h"] = nls_reference(20_000)
    H = _NLS["h"]                                              # (201, 256)
    x, t = nls_grid()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    lb, ub = np.array([-5.0, 0.0]), np.array([5.0, np.pi / 2])
    ix = rng.choice(x.size, 50, replace=False)
    it = rng.choice(t.size, 50, replace=False)
    f32 = dict(dtype=torch.float32)
    X0 = torch.tensor(np.stack([x[ix], np.zeros(50)], 1), **f32)
    h0 = torch.tensor(np.stack([H[0, ix].real, H[0, ix].imag], 1), **f32)
    Xlb = torch.tensor(np.stack([np.full(50, -5.0), t[it]], 1), **f32).requires_grad_(True)
    Xub = torch.tensor(np.stack([np.full(50, 5.0), t[it]], 1), **f32).requires_grad_(True)
    Xf = torch.tensor(_lhs(20000, lb, ub, rng), **f32).requires_grad_(True)
    net = MLP([2, 100, 100, 100, 100, 2], lb, ub)

    def loss_fn():
        loss = ((net(X0) - h0) ** 2).mean(0).sum()
        a, b = net(Xlb), net(Xub)
        loss = loss + ((a - b) ** 2).mean(0).sum()
        for j in range(2):
            loss = loss + ((_grad(a[:, j:j + 1], Xlb, 0) - _grad(b[:, j:j + 1], Xub, 0)) ** 2).mean()
        h = net(Xf)
        u, v = h[:, :1], h[:, 1:]
        u_x, v_x = _grad(u, Xf, 0), _grad(v, Xf, 0)
        q = u ** 2 + v ** 2
        f_u = _grad(u, Xf, 1) + 0.5 * _grad(v_x, Xf, 0) + q * v
        f_v = _grad(v, Xf, 1) - 0.5 * _grad(u_x, Xf, 0) - q * u
        return loss + (f_u ** 2).mean() + (f_v ** 2).mean()

    t0 = time.perf_counter()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(adam_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    adam_seconds = time.perf_counter() - t0
    res = minimize_lbfgsb(net.parameters(), loss_fn, maxiter=maxiter, maxfun=maxiter)
    X, T = np.meshgrid(x, t)
    with torch.no_grad():
        p = net(torch.tensor(np.stack([X.ravel(), T.ravel()], 1), **f32)).numpy()
    h_pred = np.hypot(p[:, 0], p[:, 1])
    h_star = np.abs(H).ravel()
    out = {"problem": "schrodinger", "seed": seed, "adam_steps": adam_steps,
           "error_h": float(np.linalg.norm(h_star - h_pred) / np.linalg.norm(h_star)),
           "error_u": float(np.linalg.norm(H.real.ravel() - p[:, 0]) / np.linalg.norm(H.real.ravel())),
           "error_v": float(np.linalg.norm(H.imag.ravel() - p[:, 1]) / np.linalg.norm(H.imag.ravel())),
           "loss": res.loss, "iterations": res.iterations, "evaluations": res.evaluations, "message": res.message,
           "seconds": adam_seconds + res.seconds, "adam_seconds": adam_seconds,
           "machine": platform.platform(), "threads": torch.get_num_threads()}
    if keep and name:
        _save(name, out)
        if seed == 0 and adam_steps:
            np.save(os.path.join(RUNS, name + "_pred.npy"), h_pred.reshape(T.shape).astype(np.float32))
    return out


def jobs(part):
    if part == "b0":
        return [(f"b0_s{s}", run_burgers, (100, 10000, 8, 20, s)) for s in range(10)]
    if part == "table2":
        return [(f"t2_L{L}_N{N}_s{s}", run_burgers, (100, 10000, L, N, s))
                for s in range(3) for L in LAYER_ROWS for N in WIDTH_COLS]
    if part == "table1":
        return [(f"t1_u{nu}_f{nf}_s{s}", run_burgers, (nu, nf, 8, 20, s))
                for s in range(3) for nu in NU_ROWS for nf in NF_COLS]
    if part == "s0":
        return [(f"s0_s{s}", run_schrodinger, (s,)) for s in range(3)]
    if part == "t2_float64":                                   # post hoc (README): Table 2 in float64, seed 0
        return [(f"x_t2f64_L{L}_N{N}_s0", lambda L, N, name, keep: run_burgers(100, 10000, L, N, 0, name=name, keep=keep,
                                                                                dtype=torch.float64), (L, N))
                for L in LAYER_ROWS for N in WIDTH_COLS]
    if part == "extras":
        return ([("x_nophysics_s0", run_burgers, (100, 0, 8, 20, 0)),
                 ("x_s0_lbfgs_only_s0", lambda s, name, keep: run_schrodinger(s, adam_steps=0, name=name, keep=keep),
                  (0,))]
                + [(f"x_b0_float64_s{s}", lambda s, name, keep: run_burgers(100, 10000, 8, 20, s, name=name, keep=keep,
                                                                            dtype=torch.float64), (s,)) for s in range(3)]
                + [(f"x_t1code_u{nu}_f{nf}_s0", run_burgers, (nu, nf, 8, 40, 0)) for nu in NU_ROWS for nf in NF_COLS]
                + [(f"x_t2code_L{L}_N{N}_s0", run_burgers, (200, 10000, L, N, 0))
                   for L in LAYER_ROWS for N in WIDTH_COLS])
    raise ValueError(part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", choices=("b0", "table2", "table1", "s0", "extras", "t2_float64"))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.smoke:
        r = run_burgers(100, 2000, 8, 20, 0, maxiter=200, keep=False)
        print("burgers", {k: r[k] for k in ("error_u", "loss", "iterations", "evaluations", "seconds")})
        r = run_schrodinger(0, adam_steps=20, maxiter=20, keep=False)
        print("schrodinger", {k: r[k] for k in ("error_h", "loss", "iterations", "seconds", "adam_seconds")})
        return
    for part in a.part or []:
        for name, fn, args in jobs(part):
            r = fn(*args, name=name, keep=True)
            err = r.get("error_u") if r["problem"] == "burgers" else r["error_h"]
            print(f"DONE {name}: error {err:.3g}, {r['iterations']} it, {r['seconds'] / 60:.1f} min", flush=True)
    if a.report:
        report()


def _cell(name_fn, seeds, target):
    vals = [r["error_u"] for s in seeds if (r := _done(name_fn(s)))]
    if not vals:
        return None, "not attempted", "-"
    st = seed_stats(vals)
    v = status(st.median, target)
    return st, v, f"{st.median:.2g} [{st.min:.2g}, {st.max:.2g}] ({len(vals)})"


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pinneapple_lab

    statuses, lines = {}, []
    st, v, txt = _cell(lambda s: f"b0_s{s}", range(10), B0_TARGET)
    statuses["B0 Burgers headline"] = v
    lines.append(f"## Burgers headline (B0)\n\nPaper: 6.7e-4. Ours, median [min, max] (seeds): {txt}. **{v}**.\n")
    rows = ["| N_u \\ N_f | " + " | ".join(map(str, NF_COLS)) + " |", "|---" * (len(NF_COLS) + 1) + "|"]
    for nu in NU_ROWS:
        cells = []
        for nf, p in zip(NF_COLS, TABLE1[nu], strict=True):
            st, v, txt = _cell(lambda s, nu=nu, nf=nf: f"t1_u{nu}_f{nf}_s{s}", range(3), p)
            statuses[f"Table 1 N_u={nu} N_f={nf}"] = v
            cells.append(f"{p:.2g} / {txt} {'✓' if v == 'reproduced' else '✗' if v != 'not attempted' else ''}")
        rows.append(f"| {nu} | " + " | ".join(cells) + " |")
    lines.append("## Table 1 (8 hidden layers × 20)\n\nEach cell: paper / ours median [min, max] (seeds); ✓ = within "
                 "a factor of 2.\n\n" + "\n".join(rows) + "\n")
    rows = ["| Layers \\ Neurons | " + " | ".join(map(str, WIDTH_COLS)) + " |", "|---" * (len(WIDTH_COLS) + 1) + "|"]
    for L in LAYER_ROWS:
        cells = []
        for N, p in zip(WIDTH_COLS, TABLE2[L], strict=True):
            st, v, txt = _cell(lambda s, L=L, N=N: f"t2_L{L}_N{N}_s{s}", range(3), p)
            statuses[f"Table 2 layers={L} neurons={N}"] = v
            cells.append(f"{p:.2g} / {txt} {'✓' if v == 'reproduced' else '✗' if v != 'not attempted' else ''}")
        rows.append(f"| {L} | " + " | ".join(cells) + " |")
    lines.append("## Table 2 (N_u = 100, N_f = 10 000)\n\n" + "\n".join(rows) + "\n")
    # post hoc, not the frozen protocol: the same runs in float64 (TARGETS.md amendment 1)
    f64 = [r["error_u"] for s in range(3) if (r := _done(f"x_b0_float64_s{s}"))]
    post = []
    if f64:
        st = seed_stats(f64)
        post.append(f"- B0 in float64: {st.median:.2g} [{st.min:.2g}, {st.max:.2g}] ({len(f64)} seeds), "
                    f"**{status(st.median, B0_TARGET)} (post hoc)** against 6.7e-4.")
    rows = ["| Layers × neurons | Paper | float32, median of 3 | float64, seed 0 | float64 status (post hoc) |",
            "|---|---|---|---|---|"]
    n_post = 0
    for L in LAYER_ROWS:
        for N, p in zip(WIDTH_COLS, TABLE2[L], strict=True):
            r64 = _done(f"x_t2f64_L{L}_N{N}_s0")
            v32 = [r["error_u"] for s in range(3) if (r := _done(f"t2_L{L}_N{N}_s{s}"))]
            if r64 is None:
                continue
            v = status(r64["error_u"], p)
            n_post += v == "reproduced"
            rows.append(f"| {L} × {N} | {p:.2g} | {np.median(v32):.2g} | {r64['error_u']:.2g} | {v} |")
    if len(rows) > 2:
        post.append(f"- Table 2 in float64 at seed 0: {n_post} of {len(rows) - 2} cells within a factor of 2.\n\n"
                    + "\n".join(rows))
    if post:
        lines.append("## float64 (post hoc, not scored)\n\n" + "\n".join(post) + "\n")
    s_runs = [r for s in range(3) if (r := _done(f"s0_s{s}"))]
    if s_runs:
        st = seed_stats([r["error_h"] for r in s_runs])
        v = status(st.median, S0_TARGET)
        lines.append(f"## Schrodinger headline (S0)\n\nPaper: 1.97e-3 (|h|). Ours: {st.median:.3g} [{st.min:.3g}, "
                     f"{st.max:.3g}] ({len(s_runs)} seeds). **{v}**.\n")
    else:
        v = "not attempted"
    statuses["S0 Schrodinger headline"] = v
    extras = [(n, r) for n, _, _ in jobs("extras") if (r := _done(n)) and not n.startswith("x_b0_float64")]
    if extras:
        lines.append("## Labelled variants (not scored)\n\n| Run | Error | Iterations | Minutes |\n|---|---|---|---|\n"
                     + "\n".join(f"| {n} | {r.get('error_u', r.get('error_h')):.3g} | {r['iterations']} | "
                                 f"{r['seconds'] / 60:.1f} |" for n, r in extras) + "\n")
    n_rep = sum(v == "reproduced" for v in statuses.values())
    n_att = sum(v != "not attempted" for v in statuses.values())
    head = (f"# Raissi et al. continuous-time PINNs: measured results\n\nGenerated by `run_raissi.py --report`.\n\n"
            f"- **{n_rep} of {n_att} attempted targets reproduced** ({len(statuses)} in the frozen scope).\n\n")
    with open(os.path.join(HERE, "RESULTS.md"), "w") as fh:
        fh.write(head + "\n".join(lines) + "\n![Burgers and Schrodinger, seed 0](predictions_seed0.png)\n")
    runs = {f[:-5]: _done(f[:-5]) for f in sorted(os.listdir(RUNS)) if f.endswith(".json")} if os.path.isdir(RUNS) \
        else {}
    with open(os.path.join(HERE, "all_runs.json"), "w") as fh:
        json.dump(runs, fh, indent=1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x, t, u = burgers_grid()
    if os.path.exists(p := os.path.join(RUNS, "b0_s0_pred.npy")):
        up = np.load(p)
        im = axes[0].imshow(up.T, extent=[0, 0.99, -1, 1], aspect="auto", origin="lower", cmap="RdBu_r")
        fig.colorbar(im, ax=axes[0])
        axes[0].set_title("Burgers, PINN seed 0")
        for tt, c in zip((25, 50, 75), ("C0", "C1", "C2"), strict=True):
            axes[1].plot(x, u[tt], c, lw=2, alpha=0.4)
            axes[1].plot(x, up[tt], c + "--", label=f"t = {t[tt]:.2f}")
        axes[1].set_title("Burgers: exact (solid), PINN (dashed)")
        axes[1].legend(fontsize=7)
    if os.path.exists(p := os.path.join(RUNS, "s0_s0_pred.npy")):
        xs, ts = nls_grid()
        im = axes[2].imshow(np.load(p).T, extent=[0, ts[-1], -5, 5], aspect="auto", origin="lower")
        fig.colorbar(im, ax=axes[2])
        axes[2].set_title("Schrodinger |h|, PINN seed 0")
    for ax in axes[[0, 2]]:
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "predictions_seed0.png"), dpi=100)
    plt.close(fig)
    with pinneapple_lab.experiment(
            "examples/reproductions/raissi2019_pinn", title="Continuous-time PINNs reproduction (Raissi et al.)",
            description="Burgers (headline, Tables 1-2) and Schrodinger PINNs of arXiv:1711.10561 at the paper's "
                        "settings, several seeds, scored against the frozen targets of TARGETS.md.",
            domain="physics-informed ML", methods=["PINN", "L-BFGS-B"],
            modules=["pinneapple_neural.trainer.scipy_lbfgsb", "pinneapple_physics.paper_problems_raissi",
                     "pinneapple_analysis.reproduction"], out_dir=HERE, register=False,
            links={"targets": "examples/reproductions/raissi2019_pinn/TARGETS.md"}) as ex:
        for k, v in statuses.items():
            if v != "not attempted":
                ex.check(f"{k}: {v}", v == "reproduced", kind="reference", label="reproduction")
    print(head + "\n".join(lines))


if __name__ == "__main__":
    main()
