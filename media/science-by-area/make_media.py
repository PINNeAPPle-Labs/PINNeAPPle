"""Carousel (1080 x 1350 PNG) and short video for the pp.science post, built only from files committed in
PINNeAPPle (branch feat/science-research, PRs #882 and #883): the sealed figures, the 3D+t movies and the
result.json values.

    python make_media.py /path/to/PINNeAPPle out_dir
"""
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO, OUT = sys.argv[1], sys.argv[2]
EM = os.path.join(REPO, "examples", "science", "electromagnetism")
W, H, DPI = 1080, 1350, 100
INK, INK2, BG, ACC = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
os.makedirs(OUT, exist_ok=True)


def res(name, dim):
    with open(os.path.join(EM, name, dim, "result.json")) as fh:
        return json.load(fh)


def cmp(r, q):
    return next(c for c in r["comparisons"] if c["quantity"] == q)


def slide(path, title, lines, images=(), footer="pp.science · github.com/PINNeAPPle-Labs/PINNeAPPle"):
    """Title, text lines, then the images stacked in the remaining height (each as wide as fits)."""
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    fig.text(0.06, 0.95, title, fontsize=27, weight="bold", color=INK, va="top", wrap=True)
    y = 0.885
    for ln in lines:
        fig.text(0.06, y, ln, fontsize=16, color=INK if ln.startswith("•") else INK2, va="top")
        y -= 0.036
    ims = [mpimg.imread(i) if isinstance(i, str) else i for i in images]
    top, bottom = y - 0.005, 0.065
    gap = 0.012
    avail = top - bottom - gap * max(len(ims) - 1, 0)
    hs = [0.96 * (im.shape[0] / im.shape[1]) * (W / H) for im in ims]
    scale = min(1.0, avail / sum(hs)) if ims else 1.0
    for im, h in zip(ims, hs):
        h *= scale
        w = 0.96 * scale
        ax = fig.add_axes([(1 - w) / 2, top - h, w, h])
        ax.imshow(im)
        ax.axis("off")
        top -= h + gap
    fig.text(0.06, 0.028, footer, fontsize=12, color=INK2)
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)


def mp4_frame(path, k):
    import imageio.v2 as iio
    r = iio.get_reader(path)
    for i, f in enumerate(r):
        if i == k:
            return f
    return f


def atlas_chart(atlas):
    """Subareas per area by status (from atlas.json), as an image."""
    names = [a["title"].replace(" and ", " & ") for a in atlas["areas"]]
    cols = {"available": "#008300", "partial": "#eda100", "planned": "#a9a8a4"}
    fig, ax = plt.subplots(figsize=(9.6, 7.4), dpi=100, facecolor="white")
    left = np.zeros(len(names))
    for st, c in cols.items():
        v = np.array([sum(s["status"] == st for s in a["subareas"]) for a in atlas["areas"]])
        ax.barh(range(len(names)), v, left=left, color=c, height=0.62, label=st)
        left += v
    ax.set_yticks(range(len(names)), names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("subareas", fontsize=10)
    ax.legend(fontsize=10, loc="lower right", frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()
    im = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return im


def main():
    rows = []
    for exp in sorted(os.listdir(EM)):
        for dim in sorted(os.listdir(os.path.join(EM, exp))):
            p = os.path.join(EM, exp, dim, "result.json")
            if os.path.isfile(p):
                with open(p) as fh:
                    d = json.load(fh)
                rows += [(exp, dim, c) for c in d["comparisons"]]
    n_pass = sum(c["passed"] for _, _, c in rows)
    with open(os.path.join(REPO, "pinneapple_science", "data", "atlas.json")) as fh:
        atlas = json.load(fh)
    st = [s["status"] for a in atlas["areas"] for s in a["subareas"]]
    n_exp = len({e for e, _, _ in rows})

    slide(os.path.join(OUT, "01_atlas.png"), "Science by area in PINNeAPPle",
          [f"An atlas of {len(atlas['areas'])} areas, {len(st)} subareas:",
           f"• {st.count('available')} available, {st.count('partial')} partial (library code), "
           f"{st.count('planned')} planned",
           "• every pointer checked against the repository",
           f"• electromagnetism first: {n_exp} experiments, {len(rows)} comparisons,",
           f"   {n_pass} within tolerance of a cited reference"],
          [atlas_chart(atlas)])
    f = res("faraday_induction", "3D_plus_t")
    pk, tz = cmp(f, "peak EMF"), cmp(f, "time of the EMF sign change")
    slide(os.path.join(OUT, "02_faraday_3d_t.png"), "Faraday's law in 3-D + time",
          ["A magnet falls through a loop. Field of its bound current in 3-D,",
           "flux by Stokes' theorem, EMF = -dPhi/dt at every instant.",
           f"• peak EMF {pk['computed'] * 1e3:.2f} mV vs {pk['reference'] * 1e3:.2f} mV (Maxwell 1873): "
           f"{100 * pk['rel_error']:.2f} %",
           f"• sign change at {tz['computed'] * 1e3:.2f} ms vs free fall {tz['reference'] * 1e3:.2f} ms"],
          [mp4_frame(os.path.join(EM, "faraday_induction", "3D_plus_t", "faraday_3d_t.mp4"), 40),
           os.path.join(EM, "faraday_induction", "3D_plus_t", "faraday_traces.sealed.png")])
    c2 = res("parallel_plate_capacitor", "2D")
    c3 = res("parallel_plate_capacitor", "3D")
    p2 = cmp(c2, "C per unit length (extrapolated)")
    k3 = cmp(c3, "C, axisymmetric (extrapolated)")
    dev = c3["data"]["scaling"]["deviation_from_kirchhoff"]
    slide(os.path.join(OUT, "03_capacitor_edge.png"), "A capacitor's edge, measured",
          ["Finite volumes, three grids, Richardson extrapolation.",
           f"• 2-D strips, w/d = 10: {100 * p2['rel_error']:.2f} % from Palmer (1937); "
           f"fringing +{100 * c2['values']['fringe_excess']['value']:.1f} %",
           f"• 3-D discs, a/d = 10: {100 * k3['rel_error']:.2f} % from Kirchhoff (1877)",
           f"• Kirchhoff is asymptotic: +{100 * dev[0]:.1f} %, +{100 * dev[1]:.1f} %, +{100 * dev[2]:.1f} % "
           "at a/d = 2.5, 5, 10"],
          [os.path.join(EM, "parallel_plate_capacitor", "2D", "capacitor_2d.sealed.png"),
           os.path.join(EM, "parallel_plate_capacitor", "3D", "capacitor_3d.sealed.png")])
    m = res("bench_mu0_solenoid", "3D")
    mu = m["values"]["mu0_measured"]["value"]
    U = m["values"]["expanded_uncertainty_k2"]["value"]
    cov = m["values"]["coverage_k2"]["value"]
    slide(os.path.join(OUT, "04_bench_mu0.png"), "A virtual bench: mu0 from a solenoid",
          ["Hall probe and ammeter with noise, calibration error and resolution;",
           "caliper for L and R; weighted fit of B(I); GUM budget.",
           f"• mu0 = ({mu * 1e6:.4f} +- {U * 1e6:.4f}) x 1e-6 N/A^2 (k = 2), CODATA 2022 1.25664",
           f"• is the stated U honest? {100 * cov:.1f} % of 300 fresh benches cover the truth",
           "• below: capacitance from an RC discharge on an 8-bit oscilloscope"],
          [os.path.join(EM, "bench_mu0_solenoid", "3D", "bench_mu0.sealed.png"),
           os.path.join(EM, "bench_rc_discharge", "0D", "bench_rc.sealed.png")])
    d = res("research_fdtd_3d", "3D_plus_t")
    v = d["values"]
    slide(os.path.join(OUT, "05_fdtd_3d_t.png"), "Research level: FDTD in 3-D + time",
          ["Sources and materials from a dict; radiated power by near-to-far field.",
           f"• dipole power error {100 * v['power_error_10cpw']['value']:.2f} -> "
           f"{100 * v['power_error_14cpw']['value']:.2f} -> {100 * v['power_error_20cpw']['value']:.2f} % "
           f"(10, 14, 20 cells/lambda), order {v['observed_order_vacuum']['value']:.2f}",
           f"• with a dielectric sphere: no closed form, so GCI {100 * v['gci_fine_configured']['value']:.1f} % "
           f"and order {v['observed_order_configured']['value']:.2f} are reported",
           "• Maxwell residuals checked on the field in x, y, z and t"],
          [os.path.join(EM, "research_fdtd_3d", "3D_plus_t", "fdtd_convergence.sealed.png"),
           mp4_frame(os.path.join(EM, "research_fdtd_3d", "3D_plus_t", "fdtd_3d_t.mp4"), 14)])
    s = res("research_surrogate_hall", "2D")
    iv = res("research_inverse_permittivity", "1D")
    u = res("research_uq_waveguide", "2D")
    eps = iv["values"]["eps_r_recovered"]
    gum = cmp(u, "TE10: Monte Carlo std vs first-order GUM")
    slide(os.path.join(OUT, "06_research.png"), "Inverse problem, uncertainty, surrogate",
          [f"• eps_r from a noisy FDTD reflection spectrum: {eps['value']:.4f} +- {eps['uncertainty']:.4f} "
           "(truth 3.7)",
           f"• waveguide tolerances, Monte Carlo vs GUM std: {100 * gum['rel_error']:.1f} % apart",
           f"• GP surrogate, worst hold-out error {100 * s['values']['holdout_max_rel_error']['value']:.3f} %; "
           f"+-2 sigma covers {100 * s['values']['coverage_2sigma']['value']:.0f} % (stated)",
           "• every figure carries its Lab seal; every number has its source"],
          [os.path.join(EM, "research_inverse_permittivity", "1D", "inverse_permittivity.sealed.png"),
           os.path.join(EM, "research_surrogate_hall", "2D", "surrogate_hall.sealed.png")])
    video()


def video():
    """Title card, the falling magnet (3D+t), the FDTD field on three planes (3D+t), end card: 1080 x 1350."""
    import imageio.v2 as iio

    def card(text, sub):
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
        fig.text(0.5, 0.55, text, fontsize=34, weight="bold", ha="center", color=INK, wrap=True)
        fig.text(0.5, 0.45, sub, fontsize=18, ha="center", color=INK2, wrap=True)
        fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return a

    def fit(frame, caption):
        fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
        fig.text(0.5, 0.95, caption, fontsize=22, weight="bold", ha="center", va="top", color=INK)
        h, w = frame.shape[:2]
        width = 0.96
        height = width * (h / w) * (W / H)
        ax = fig.add_axes([(1 - width) / 2, 0.5 - height / 2, width, height])
        ax.imshow(frame)
        ax.axis("off")
        fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return a

    frames = [card("Science by area", "pp.science in PINNeAPPle: electromagnetism in 3-D + time")] * 36
    for f in iio.mimread(os.path.join(EM, "faraday_induction", "3D_plus_t", "faraday_3d_t.mp4"), memtest=False):
        frames.append(fit(f, "A magnet falls through a loop: |B| and the EMF"))
    for f in iio.mimread(os.path.join(EM, "research_fdtd_3d", "3D_plus_t", "fdtd_3d_t.mp4"), memtest=False):
        frames.append(fit(f, "FDTD: a dipole beside a dielectric sphere"))
    frames += [card("Every number against its reference", "github.com/PINNeAPPle-Labs/PINNeAPPle")] * 36
    path = os.path.join(OUT, "science_em_3d_t.mp4")
    iio.mimsave(path, frames, fps=12, codec="libx264", macro_block_size=1, pixelformat="yuv420p")
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    print("video", path, out, "s")


if __name__ == "__main__":
    main()
