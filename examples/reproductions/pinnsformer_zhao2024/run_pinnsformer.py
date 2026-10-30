"""Reproduce PINNsFormer (Zhao, Ding & Prakash, ICLR 2024) Tables 1-2 with the protocol frozen in TARGETS.md.

    python examples/reproductions/pinnsformer_zhao2024/run_pinnsformer.py --queue all    # every run, resumable
    python examples/reproductions/pinnsformer_zhao2024/run_pinnsformer.py --case reaction --model pinnsformer --seed 0
    python examples/reproductions/pinnsformer_zhao2024/run_pinnsformer.py --report   # tables, figure, experiment.json
    python examples/reproductions/pinnsformer_zhao2024/run_pinnsformer.py --smoke    # 3 outer steps, nothing kept

The losses, grids, optimizer and schedule replicate the official notebooks (TARGETS.md, ambiguities 1-9). Every run
writes ``runs/<case>_<model>_s<seed>.json`` and checkpoints every 10 outer steps, so a stopped run resumes.
Set ``PINNSFORMER_CONVECTION_MAT`` to the official ``demo/convection/convection.mat`` to also score convection
against the authors' numerical reference (ambiguity 8); the file is not redistributed here.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import torch
import torch.nn as nn

from pinneapple_analysis.reproduction import seed_stats, status
from pinneapple_neural.architectures.pinns.pinnsformer_paper import PINNsFormerPaper, init_official
from pinneapple_physics.paper_problems import pinnsformer_case, tensor_grid, time_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
# Tables 1-2 (TARGETS.md): (loss, rMAE, rRMSE)
TARGETS = {
    ("convection", "pinn"): (0.016, 0.778, 0.840), ("convection", "pinnsformer"): (3.7e-5, 0.023, 0.027),
    ("reaction", "pinn"): (0.199, 0.982, 0.981), ("reaction", "pinnsformer"): (3.0e-6, 0.015, 0.030),
    ("wave", "pinn"): (1.93e-2, 0.326, 0.335), ("wave", "pinnsformer"): (1.38e-2, 0.270, 0.283),
}
SEEDS = {"convection": (0, 1, 2), "reaction": (0, 1, 2), "wave": (0, 1)}   # wave seed 1 only if time allows
# seed-major order (TARGETS.md amendment 2): every cell gets seed 0 before any cell gets seed 1
_T1 = [(c, m, s) for s in SEEDS["reaction"] for c in ("reaction", "convection") for m in ("pinn", "pinnsformer")]
_WAVE = [("wave", m, s) for s in SEEDS["wave"] for m in ("pinn", "pinnsformer")]
QUEUES = {"table1": _T1, "wave": _WAVE}
QUEUE = _T1[:4] + _WAVE[:2] + _T1[4:] + _WAVE[2:]
K, DT = 5, 1e-4


def pinn_mlp() -> nn.Module:
    """The official ``PINNs(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4)``: 3 hidden tanh layers of 512."""
    net = nn.Sequential(nn.Linear(2, 512), nn.Tanh(), nn.Linear(512, 512), nn.Tanh(), nn.Linear(512, 512), nn.Tanh(),
                        nn.Linear(512, 1))
    init_official(net)
    return net


class _Seq(nn.Module):
    """Uniform interface: ``(N, L, 2) -> (N, L, 1)``; the PINN is applied point by point (L = 1)."""

    def __init__(self, model: str):
        super().__init__()
        self.model = model
        self.net = PINNsFormerPaper(2, 1, k=K, dt=DT) if model == "pinnsformer" else pinn_mlp()

    def forward(self, seq):
        return self.net.forward_sequence(seq) if self.model == "pinnsformer" else self.net(seq)


def _points(case, model):
    """Training points as (N, L, 2) leaf tensors: PINNsFormer on the 51 x 51 grid with k = 5 copies, PINN on the
    101 x 101 grid with L = 1 (TARGETS.md, points)."""
    n = 51 if model == "pinnsformer" else 101
    g = tensor_grid(case.x_range, case.t_range, n, n)
    out = {}
    for key in ("res", "left", "upper", "lower"):
        seq = time_sequence(g[key], K, DT) if model == "pinnsformer" else g[key][:, None, :]
        out[key] = torch.tensor(seq, dtype=torch.float32, requires_grad=True)   # float64 shift, then float32
    return out


def _d(u, X, col, create=True):
    return torch.autograd.grad(u, X, torch.ones_like(u), create_graph=create, retain_graph=True)[0][..., col:col + 1]


def losses(case_name, net, P):
    """The notebooks' three terms (res, bc, ic); means over every element, IC on the first copy only."""
    u = net(P["res"])
    up, lo = net(P["upper"]), net(P["lower"])
    left = net(P["left"])
    xl = P["left"][:, 0, 0]
    if case_name == "convection":
        res = _d(u, P["res"], 1) + 50.0 * _d(u, P["res"], 0)
        bc = ((up - lo) ** 2).mean()
        ic = ((left[:, 0, 0] - torch.sin(xl)) ** 2).mean()
    elif case_name == "reaction":
        res = _d(u, P["res"], 1) - 5.0 * u * (1 - u)
        bc = ((up - lo) ** 2).mean()
        ic = ((left[:, 0, 0] - torch.exp(-(xl - torch.pi) ** 2 / (2 * (torch.pi / 4) ** 2))) ** 2).mean()
    else:
        u_xx = _d(_d(u, P["res"], 0), P["res"], 0)
        u_tt = _d(_d(u, P["res"], 1), P["res"], 1)
        res = u_tt - 4.0 * u_xx
        bc = (up ** 2).mean() + (lo ** 2).mean()
        ui_t = _d(left, P["left"], 1)                      # every copy, as the notebook's loss_ic_2
        ic = ((left[:, 0, 0] - torch.sin(torch.pi * xl) - 0.5 * torch.sin(3 * torch.pi * xl)) ** 2).mean() \
            + (ui_t ** 2).mean()
    return (res ** 2).mean(), bc, ic


def predict(case, net, model):
    X, ref = case.test()
    with torch.no_grad():
        seq = time_sequence(X, K, DT) if model == "pinnsformer" else X[:, None, :]
        u = net(torch.tensor(seq, dtype=torch.float32))[:, 0, 0].numpy().astype(np.float64)
    return X, u, ref


def rel_errors(u, ref):
    return float(np.abs(u - ref).sum() / np.abs(ref).sum()), float(np.sqrt(((u - ref) ** 2).sum() / (ref ** 2).sum()))


def run(case_name, model, seed, steps=None, keep=True):
    case = pinnsformer_case(case_name)
    steps = steps or case.lbfgs_steps
    tag = f"{case_name}_{model}_s{seed}"
    out_path, ckpt = os.path.join(RUNS, tag + ".json"), os.path.join(RUNS, "ckpt", tag + ".pt")
    if keep and os.path.exists(out_path):
        with open(out_path) as fh:
            return json.load(fh)
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _Seq(model)
    P = _points(case, model)
    opt = torch.optim.LBFGS(net.parameters(), line_search_fn="strong_wolfe")
    track, start, wall = [], 0, 0.0
    if keep and os.path.exists(ckpt):
        st = torch.load(ckpt, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        track, start, wall = st["track"], st["step"], st["wall"]
        torch.set_rng_state(st["rng"])

    def closure():
        r, b, i = losses(case_name, net, P)
        track.append([r.item(), b.item(), i.item()])
        loss = r + b + i
        opt.zero_grad()
        loss.backward()
        return loss

    for step in range(start, steps):
        t0 = time.perf_counter()
        opt.step(closure)
        wall += time.perf_counter() - t0
        if keep and (step + 1) % 10 == 0:
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "track": track, "step": step + 1,
                        "wall": wall, "rng": torch.get_rng_state()}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
            print(f"{tag} step {step + 1}/{steps} loss {sum(track[-1]):.3g} ({wall / 60:.1f} min)", flush=True)
    X, u, ref = predict(case, net, model)
    rmae, rrmse = rel_errors(u, ref)
    out = {"case": case_name, "model": model, "seed": seed, "outer_steps": steps, "closure_evals": len(track),
           "loss": float(sum(track[-1])), "loss_terms": dict(zip(("res", "bc", "ic"), track[-1], strict=True)),
           "rmae": rmae, "rrmse": rrmse, "seconds": wall, "n_params": sum(p.numel() for p in net.parameters()),
           "loss_every_10": [float(sum(x)) for x in track[::10]], "machine": platform.platform(),
           "torch": torch.__version__, "threads": torch.get_num_threads()}
    mat = os.environ.get("PINNSFORMER_CONVECTION_MAT")
    if case_name == "convection" and mat and os.path.exists(mat):
        from scipy.io import loadmat
        out["rmae_vs_official_mat"], out["rrmse_vs_official_mat"] = rel_errors(u, loadmat(mat)["u"].reshape(-1))
    if keep:
        os.makedirs(RUNS, exist_ok=True)
        np.save(os.path.join(RUNS, tag + "_pred.npy"), u.astype(np.float32))
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=1)
        if os.path.exists(ckpt):
            os.remove(ckpt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=("convection", "reaction", "wave"))
    ap.add_argument("--model", choices=("pinn", "pinnsformer"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--queue", choices=("all", *QUEUES))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.smoke:
        for c in ("convection", "reaction", "wave"):
            for m in ("pinn", "pinnsformer"):
                r = run(c, m, 0, steps=1, keep=False)
                print(c, m, f"loss {r['loss']:.3g} rMAE {r['rmae']:.3g} evals {r['closure_evals']} "
                      f"{r['seconds']:.1f}s params {r['n_params']}", flush=True)
        return
    if a.queue:
        for c, m, s in (QUEUE if a.queue == "all" else QUEUES[a.queue]):
            r = run(c, m, s)
            print(f"DONE {c} {m} s{s}: loss {r['loss']:.3g} rMAE {r['rmae']:.3g} rRMSE {r['rrmse']:.3g} "
                  f"({r['seconds'] / 60:.0f} min)", flush=True)
    elif a.case:
        print(json.dumps(run(a.case, a.model, a.seed), indent=1))
    if a.report:
        report()


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pinneapple_lab

    runs = {}
    for c, m, s in QUEUE:
        path = os.path.join(RUNS, f"{c}_{m}_s{s}.json")
        if os.path.exists(path):
            with open(path) as fh:
                runs.setdefault((c, m), []).append(json.load(fh))
    rows = ["| Problem | Model | Metric | Paper | Ours: median [min, max] | Seed 0 | Seeds | Status |",
            "|---|---|---|---|---|---|---|---|"]
    statuses = {}
    for c in ("convection", "reaction", "wave"):
        med = {m: {k: float(np.median([r[k] for r in runs.get((c, m), [])])) if (c, m) in runs else None
                   for k in ("rmae", "rrmse")} for m in ("pinn", "pinnsformer")}
        ordering = None
        if all(med[m]["rmae"] is not None for m in med):
            ordering = all(med["pinnsformer"][k] < med["pinn"][k] for k in ("rmae", "rrmse"))
        for m in ("pinn", "pinnsformer"):
            rs = runs.get((c, m))
            for k, p in zip(("loss", "rmae", "rrmse"), TARGETS[(c, m)], strict=True):
                if not rs:
                    rows.append(f"| {c} | {m} | {k} | {p:.3g} | - | - | 0 | not attempted |")
                    statuses[f"{c} {m} {k}"] = "not attempted"
                    continue
                st = seed_stats([r[k] for r in rs])
                s0 = next((r[k] for r in rs if r["seed"] == 0), float("nan"))
                v = status(st.median, p, ordering_holds=ordering)
                statuses[f"{c} {m} {k}"] = v
                rows.append(f"| {c} | {m} | {k} | {p:.3g} | {st.median:.3g} [{st.min:.3g}, {st.max:.3g}] | {s0:.3g} | "
                            f"{len(rs)} | **{v}** |")
    extra = []
    for m in ("pinn", "pinnsformer"):
        rs = [r for r in runs.get(("convection", m), []) if "rmae_vs_official_mat" in r]
        if rs:
            extra.append(f"| convection | {m} | " + " | ".join(
                f"{np.median([r[k] for r in rs]):.3g}" for k in ("rmae_vs_official_mat", "rrmse_vs_official_mat"))
                + f" | {len(rs)} |")
    timing = [f"| {c} | {m} | {np.median([r['seconds'] for r in rs]) / 60:.0f} | "
              f"{int(np.median([r['closure_evals'] for r in rs]))} | {rs[0]['n_params']} |"
              for (c, m), rs in sorted(runs.items())]
    n_rep = sum(v == "reproduced" for v in statuses.values())
    n_part = sum(v == "partially reproduced" for v in statuses.values())
    text = ("# PINNsFormer (Zhao, Ding & Prakash, ICLR 2024): measured results\n\nGenerated by `run_pinnsformer.py "
            f"--report`. Scored against the exact solutions.\n\n- **{n_rep} of {len(statuses)} reproduced, {n_part} "
            "partially reproduced.**\n\n" + "\n".join(rows) + "\n")
    if extra:
        text += ("\nConvection against the official numerical reference `convection.mat` (median over seeds):\n\n"
                 "| Problem | Model | rMAE | rRMSE | Seeds |\n|---|---|---|---|---|\n" + "\n".join(extra) + "\n")
    text += ("\nCost per run (CPU, float32):\n\n| Problem | Model | Minutes (median) | Loss evaluations | Parameters |\n"
             "|---|---|---|---|---|\n" + "\n".join(timing) + "\n\n![seed 0 predictions](predictions_seed0.png)\n")
    with open(os.path.join(HERE, "RESULTS.md"), "w") as fh:
        fh.write(text)
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for i, c in enumerate(("convection", "reaction", "wave")):
        case = pinnsformer_case(c)
        X, ref = case.test()
        ext = [case.x_range[0], case.x_range[1], case.t_range[1], case.t_range[0]]
        panels = [("exact", ref)] + [(m, np.load(os.path.join(RUNS, f"{c}_{m}_s0_pred.npy")))
                                     for m in ("pinn", "pinnsformer")
                                     if os.path.exists(os.path.join(RUNS, f"{c}_{m}_s0_pred.npy"))]
        for j in range(3):
            ax = axes[i, j]
            if j < len(panels):
                name, u = panels[j]
                im = ax.imshow(np.asarray(u).reshape(101, 101), extent=ext, aspect="auto", cmap="RdBu_r")
                fig.colorbar(im, ax=ax)
                ax.set_title(f"{c}: {name}" + ("" if name == "exact" else f" (rRMSE {rel_errors(u, ref)[1]:.3f})"))
                ax.set_xlabel("x")
                ax.set_ylabel("t")
            else:
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "predictions_seed0.png"), dpi=100)
    plt.close(fig)
    with open(os.path.join(HERE, "all_runs.json"), "w") as fh:
        json.dump({f"{c}_{m}": rs for (c, m), rs in sorted(runs.items())}, fh, indent=1)
    with pinneapple_lab.experiment(
            "examples/reproductions/pinnsformer_zhao2024", title="PINNsFormer reproduction (Zhao, Ding & Prakash)",
            description="PINNsFormer and the PINN baseline on the convection, 1D reaction and 1D wave problems of "
                        "ICLR 2024 / arXiv:2307.11833, at the official notebooks' settings, several seeds, scored "
                        "against the frozen targets of TARGETS.md.",
            domain="physics-informed ML", methods=["PINNsFormer", "PINN"],
            modules=["pinneapple_neural.architectures.pinns.pinnsformer_paper", "pinneapple_physics.paper_problems",
                     "pinneapple_analysis.reproduction"], out_dir=HERE, register=False,
            links={"targets": "examples/reproductions/pinnsformer_zhao2024/TARGETS.md"}) as ex:
        for k, v in statuses.items():
            ex.check(f"{k}: {v}", v == "reproduced", kind="reference", label="reproduction")
        for (c, m), rs in sorted(runs.items()):
            ex.log_metric(f"{c}_{m}_rrmse_median", float(np.median([r["rrmse"] for r in rs])), "relative",
                          f"median over {len(rs)} seeds of rRMSE against the exact solution, 101 x 101 test grid")
    print(text)


if __name__ == "__main__":
    main()
