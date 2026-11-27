"""Large Physics Model at design-space scale: ~1000 analytic designs, batched, two backbones.

The design space is the ellipse family of :func:`pinneapple_neural.lpm.ellipse_design_space`
(a, b in [0.6, 1.4], uniform stream along x) with the exact potential-flow Cp as the answer, so a
thousand training simulations cost nothing to produce and the measured error belongs to the model.

    python -m benchmarks.lpm_scale --backbone attention --train 1000 --epochs 200
    python -m benchmarks.lpm_scale --backbone mlp --train 1000 --epochs 200

Held-out designs come from the same family with a different seed. Three out-of-distribution
probes: a square (same size, not a smooth ellipse), a circle of radius 3 (outside the size range)
and a thin ellipse ``a=0.25, b=1.2`` (outside the aspect-ratio range).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from pinneapple_neural.lpm import Design, LargePhysicsModel, ellipse_design_space, fit


def _device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run(n_train: int = 1000, n_test: int = 50, n_points: int = 64, epochs: int = 200,
        backbone: str = "attention", batch_designs: int = 64, width: int = 96, heads: int = 4,
        attn_layers: int = 2, seed: int = 0, device: str = "auto", lr: float = 2e-3) -> dict:
    dev = _device(device)
    torch.manual_seed(0)
    train = ellipse_design_space(n_train, n_points=n_points, seed=seed, device=dev)
    test = ellipse_design_space(n_test, n_points=n_points, seed=seed + 1000, device=dev)
    m = LargePhysicsModel(dim=2, n_fields=1, radii=(0.15, 0.5, 1.5), width=width, n_heads=heads,
                          backbone=backbone, attn_layers=attn_layers, attn_heads=4).to(dev)

    t0 = time.time()
    hist = fit(m, train, epochs=epochs, lr=lr, batch_designs=batch_designs, device=dev)
    seconds = time.time() - t0

    def rel(d: Design) -> float:
        pred = m.predict(d)["mean"].to(d.target.device)
        return float((pred - d.target).norm() / d.target.norm())

    m.eval()
    held = [rel(d) for d in test]
    probes = {
        "square": m.ood(Design.square(n=n_points)),
        "circle_r3": m.ood(Design.ellipse(3.0, 3.0, n_points)),
        "thin_ellipse_a025": m.ood(Design.ellipse(0.25, 1.2, n_points)),
    }
    return {
        "note": (f"{backbone} backbone, {n_train} training designs, {n_points} surface points, {epochs} epochs, "
                 f"batch {batch_designs}, width {width}, {heads} heads, {attn_layers} attention layers, device {dev}"),
        "backbone": backbone,
        "n_train": n_train, "n_test": n_test, "n_points": n_points, "epochs": epochs,
        "batch_designs": batch_designs, "width": width, "heads": heads, "attn_layers": attn_layers,
        "device": dev,
        "train_loss_first": hist[0], "train_loss_last": hist[-1],
        "heldout_rel_l2_mean": sum(held) / len(held),
        "heldout_rel_l2_p90": sorted(held)[int(0.9 * (len(held) - 1))],
        "heldout_rel_l2": held,
        "heldout_ood_scores": [m.ood(d)["ood_score"] for d in test],
        "probes": probes,
        "seconds": round(seconds, 1),
        "seconds_per_epoch": round(seconds / max(epochs, 1), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--test", type=int, default=50)
    p.add_argument("--points", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--backbone", choices=("mlp", "attention"), default="attention")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    r = run(n_train=a.train, n_test=a.test, n_points=a.points, epochs=a.epochs, backbone=a.backbone,
            batch_designs=a.batch, width=a.width, heads=a.heads, device=a.device)
    print(json.dumps({k: v for k, v in r.items() if k != "heldout_rel_l2"}, indent=1))
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out",
                                f"lpm_scale_{a.backbone}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(r, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
