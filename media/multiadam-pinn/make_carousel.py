"""Builds the 1080x1350 carousel from multiadam_bench.json (matplotlib figures + Pillow bands)."""
import io, json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "multiadam_bench.json")))
F = "/usr/share/fonts/opentype/inter/Inter-{}.otf"
W, H, BG, INK, INK2 = 1080, 1350, "#fcfcfb", "#0b0b0b", "#52514e"
COL = {"adam": "#2a78d6", "adam_b99": "#eb6834", "multiadam": "#1baf7a"}
LAB = {"adam": "Adam (0.9, 0.999)", "adam_b99": "Adam (0.99, 0.99)", "multiadam": "MultiAdam (0.99, 0.99)"}
SEEDS = D["config"]["seeds"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "axes.edgecolor": "#b0afa8",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "figure.facecolor": BG, "axes.facecolor": BG})


def font(w, s): return ImageFont.truetype(F.format(w), s)


def wrap(d, text, f, maxw):
    lines, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if d.textlength(t, font=f) <= maxw: cur = t
        else: lines.append(cur); cur = word
    lines.append(cur); return lines


def slide(name, title, caption, fig=None, body=None):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    y = 60
    for ln in wrap(d, title, font("Bold", 56), W - 120):
        d.text((60, y), ln, font=font("Bold", 56), fill=INK); y += 68
    y += 14
    for ln in wrap(d, caption, font("Regular", 34), W - 120):
        d.text((60, y), ln, font=font("Regular", 34), fill=INK2); y += 46
    y += 20
    d.line([(60, y), (W - 60, y)], fill="#e0dfda", width=2); y += 20
    foot = 1350 - 70
    if fig is not None:
        fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
        f = Image.open(buf).convert("RGB")
        bw, bh = W - 60, foot - y - 10
        s = min(bw / f.width, bh / f.height); f = f.resize((int(f.width * s), int(f.height * s)), Image.LANCZOS)
        im.paste(f, ((W - f.width) // 2, y + (bh - f.height) // 2))
    if body: body(d, y)
    d.text((60, foot), "github.com/PINNeAPPle-Labs/PINNeAPPle", font=font("Medium", 30), fill=INK2)
    num = name.split("_")[0]
    d.text((W - 60 - d.textlength(num + "/06", font=font("Medium", 30)), foot), num + "/06", font=font("Medium", 30), fill=INK2)
    im.save(os.path.join(HERE, name + ".png"), optimize=True)


def runs(prob, arm): return [D["problems"][prob]["runs"][f"{arm}/s{s}"] for s in SEEDS]


def band(ax, prob, arm, key="rel_l2", xkey="eval_step", label=True):
    rs = runs(prob, arm); x = np.array(rs[0][xkey])
    Y = np.array([r[key] for r in rs])
    ax.fill_between(x, Y.min(0), Y.max(0), color=COL[arm], alpha=0.15, lw=0)
    ax.plot(x, np.median(Y, 0), color=COL[arm], lw=2.5, label=LAB[arm] if label else None)


def med(prob, arm): return D["problems"][prob]["summary"][arm]["rel_l2_median"]

# 01 title / method
def body01(d, y):
    f, fm = font("Regular", 34), ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 30)
    lines = ["Plain Adam:  one (m, v) for g = grad(L_pde + w * L_bc)",
             "MultiAdam:   one (m_i, v_i) per term i,",
             "             step = mean_i( m_i / sqrt(v_i) )"]
    yy = y + 30
    for ln in lines: d.text((60, yy), ln, font=fm, fill=INK); yy += 44
    yy += 30
    code = ["from pinneapple_neural.trainer.multiadam \\", "    import MultiAdam", "",
            "opt = MultiAdam(net.parameters(), lr=1e-3)", "opt.step([loss_pde, loss_bc])  # no backward()"]
    d.rounded_rectangle([50, yy - 16, W - 50, yy + 44 * len(code) + 6], 16, fill="#f0efea")
    for ln in code: d.text((80, yy), ln, font=fm, fill=INK); yy += 44
    yy += 50
    for ln in wrap(d, "Rescale one loss term by any constant and its MultiAdam update stays the same (up to eps). "
                   "Algorithm: Yao et al., ICML 2023. Contributed by @F123Dash in PR #848.", f, W - 120):
        d.text((60, yy), ln, font=f, fill=INK2); yy += 46
    yy += 30
    for ln in wrap(d, "We ran it ourselves: 3 optimisers x 3 seeds x 3 cases, same net, points and steps, on 4 CPU cores. "
                   "Total: %.0f s." % D["total_seconds"], f, W - 120):
        d.text((60, yy), ln, font=f, fill=INK); yy += 46

slide("01_method", "MultiAdam: one Adam per loss term",
      "PINN losses add terms with very different scales. Hand-tuned weights are the usual fix.", body=body01)

# 02 poisson w=1e-3
P = "poisson1d_wbc1e-3"
fig, ax = plt.subplots(figsize=(10, 8.6))
for a in ["adam", "adam_b99", "multiadam"]: band(ax, P, a)
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("relative L2 error vs exact sin(pi x)")
ax.legend(frameon=False, loc="lower left"); ax.set_title("1-D Poisson, BC term x 1e-3 (median, band = 3 seeds)", fontsize=17, color=INK)
slide("02_poisson_weighted", "Down-weighted boundary: MultiAdam recovers",
      "Boundary term multiplied by 1e-3. Median rel-L2 after 2000 steps: Adam %.2f, MultiAdam %.4f."
      % (med(P, "adam"), med(P, "multiadam")), fig)

# 03 loss-term balance
fig, ax = plt.subplots(figsize=(10, 8.6))
for a in ["adam", "multiadam"]:
    rs = runs(P, a); x = np.array(rs[0]["step"]); T = np.array([r["terms"] for r in rs])
    ax.plot(x, np.median(T[:, :, 0], 0), color=COL[a], lw=2.5, label=LAB[a] + "  PDE term")
    ax.plot(x, np.median(T[:, :, 1], 0) / 1e-3, color=COL[a], lw=2.5, ls="--", label=LAB[a] + "  BC error (unweighted)")
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("loss term (median of 3 seeds)")
ax.legend(frameon=False, loc="lower left", fontsize=16)
slide("03_term_balance", "Where the error hides: the boundary",
      "Adam drives the PDE term down and leaves the boundary mean-square at 3 to 7 (final, per seed). "
      "MultiAdam pushes both down.", fig)

# 04 scale invariance
fig, ax = plt.subplots(figsize=(10, 8.6))
rows = [("poisson1d_wbc1", "w_bc = 1"), ("poisson1d_wbc1e-3", "w_bc = 1e-3")]
for j, a in enumerate(["adam", "adam_b99", "multiadam"]):
    for i, (p, lab) in enumerate(rows):
        v = D["problems"][p]["summary"][a]["rel_l2_per_seed"]
        yy = i * 4 + j
        ax.scatter(v, [yy] * 3, s=110, color=COL[a], edgecolor=BG, linewidth=2, zorder=3,
                   label=LAB[a] if i == 0 else None)
ax.set_xscale("log"); ax.set_yticks([1, 5]); ax.set_yticklabels(["w_bc = 1", "w_bc = 1e-3"])
ax.set_ylim(-1, 7); ax.set_xlabel("final relative L2 error (one dot per seed)"); ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=16)
slide("04_weight_invariance", "Same problem, two weights",
      "MultiAdam lands in the same range (1.7e-4 to 9.6e-3) for both weights. Adam needs w_bc = 1 and then "
      "beats MultiAdam: median %.1e vs %.1e." % (med("poisson1d_wbc1", "adam"), med("poisson1d_wbc1", "multiadam")), fig)

# 05 helmholtz
Hh = "helmholtz2d"
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 9.4), sharex=True)
for a in ["adam", "adam_b99", "multiadam"]:
    band(a1, Hh, a)
    rs = runs(Hh, a); x = np.array(rs[0]["step"]); T = np.array([r["terms"] for r in rs])[:, :, 0]
    a2.fill_between(x, T.min(0), T.max(0), color=COL[a], alpha=0.15, lw=0)
    a2.plot(x, np.median(T, 0), color=COL[a], lw=2.5)
a1.set_yscale("log"); a1.set_ylabel("rel-L2 vs exact"); a1.legend(frameon=False, fontsize=16, loc="upper right")
a2.set_yscale("log"); a2.set_ylabel("PDE residual term"); a2.set_xlabel("step")
fig.tight_layout()
slide("05_helmholtz_honest", "2-D Helmholtz: MultiAdam did not help here",
      "sin(pi x) sin(4 pi y), 3000 steps. Median rel-L2: Adam %.2f, MultiAdam %.2f, Adam with betas 0.99 %.2f. "
      "MultiAdam's PDE term was still falling at the end." % (med(Hh, "adam"), med(Hh, "multiadam"), med(Hh, "adam_b99")), fig)

# 06 summary table
def body06(d, y):
    f, fb = font("Regular", 32), font("SemiBold", 32)
    cols = [60, 440, 640, 830]
    yy = y + 30
    for c, t in zip(cols, ["median rel-L2", "Adam", "Adam .99", "MultiAdam"]): d.text((c, yy), t, font=fb, fill=INK)
    yy += 60
    for p, lab in [("poisson1d_wbc1", "Poisson, w_bc=1"), ("poisson1d_wbc1e-3", "Poisson, w_bc=1e-3"), ("helmholtz2d", "Helmholtz 2-D")]:
        vals = [med(p, a) for a in ["adam", "adam_b99", "multiadam"]]
        best = int(np.argmin(vals))
        d.text((cols[0], yy), lab, font=f, fill=INK)
        for k, v in enumerate(vals):
            d.text((cols[k + 1], yy), f"{v:.3g}", font=fb if k == best else f, fill=INK if k == best else INK2)
        yy += 56
        d.line([(60, yy - 8), (W - 60, yy - 8)], fill="#e6e5e0", width=2)
    yy += 30
    for t in ["Best in each row in bold. 3 seeds per cell, same net, points and steps.",
              "MultiAdam removes the need to tune the boundary weight when that weight is badly off.",
              "With a sensible weight, plain Adam was better. On Helmholtz, Adam with betas 0.99 was best and MultiAdam had not converged in 3000 steps.",
              "Limits: small nets, lr 1e-3 only, no lr sweep, CPU budget of a few minutes."]:
        for ln in wrap(d, t, f, W - 120): d.text((60, yy), ln, font=f, fill=INK2); yy += 44
        yy += 14
    yy += 10
    for ln in wrap(d, "Script + JSON: run_multiadam_bench.py, multiadam_bench.json", font("Medium", 30), W - 120):
        d.text((60, yy), ln, font=font("Medium", 30), fill=INK); yy += 42

slide("06_summary", "What we measured", "A tool for one failure mode, not a default optimiser.", body=body06)
