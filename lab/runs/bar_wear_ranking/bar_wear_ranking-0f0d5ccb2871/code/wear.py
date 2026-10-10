"""Sliding wear of bars of different materials (``pinneapple_physics.tribology``).

* ``bar_wear``         -- one material: a crowned bar end pressed on a flat counterface and slid until it has run in;
                         Archard wear with the contact pressure redistributed on an elastic layer. Checks the exact
                         facts (worn volume = K F s, the closed-form initial contact, the flat-punch steady rate, the
                         running-in distance); renders the worn end in 3-D coloured by wear depth and a movie of the
                         profile and the pressure.
* ``bar_wear_ranking`` -- every ``bar_wear`` run in the database side by side: wear rate, life to 1 mm of wear and
                         the ranking of the materials.

    python -m pinneapple_lab sweep bar_wear -g material="mild steel","60/40 brass",PTFE,stellite,...
    python -m pinneapple_lab run bar_wear_ranking
"""
from __future__ import annotations

import os

import numpy as np

from ..spec import Experiment, register


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _worn_bar_scene(x_mm, gap_mm, wear_um, width_mm=10.0, height_mm=24.0, exaggerate=40.0, nb=24):
    """The bar turned tip-up (x along sliding, y across, z along the bar): its end follows the crown left after wear
    (``gap_mm`` above the lowest point, exaggerated in z) and is coloured by the wear depth."""
    from pinneapple_tools.visualization.studio.scene import Scene, Surface
    n = len(x_mm)
    top = height_mm - exaggerate * np.asarray(gap_mm)
    ys = np.linspace(-width_mm / 2, width_mm / 2, nb)
    V, F, W = [], [], []
    for y in ys:
        for i in range(n):
            V.append((x_mm[i], y, top[i]))
            W.append(wear_um[i])
    for j in range(nb - 1):
        for i in range(n - 1):
            a, b, c, d = j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i
            F += [(a, b, c), (a, c, d)]
    end = Surface("worn_end", np.asarray(V, float), np.asarray(F), "steel", {"wear_um": np.asarray(W, float)})
    V, F = [], []                                                 # side walls (plain steel) follow the end profile
    for y in (-width_mm / 2, width_mm / 2):
        for i in range(n):
            V += [(x_mm[i], y, 0.0), (x_mm[i], y, top[i])]
    for side in (0, 1):
        o = side * 2 * n
        for i in range(n - 1):
            a, b, c, d = o + 2 * i, o + 2 * i + 2, o + 2 * i + 3, o + 2 * i + 1
            F += [(a, b, c), (a, c, d)] if side else [(a, c, b), (a, d, c)]
    for i in (0, n - 1):                                          # the two end walls
        o = len(V)
        V += [(x_mm[i], -width_mm / 2, 0.0), (x_mm[i], width_mm / 2, 0.0), (x_mm[i], width_mm / 2, top[i]),
              (x_mm[i], -width_mm / 2, top[i])]
        F += [(o, o + 1, o + 2), (o, o + 2, o + 3)]
    walls = Surface("bar", np.asarray(V, float), np.asarray(F), "steel")
    sc = Scene([end, walls], axes="z_up", title="worn bar end")
    sc.labels["wear_um"] = "Wear depth (µm)"
    return sc


@register
class BarWear(Experiment):
    name = "bar_wear"
    version = "1"
    description = ("A crowned bar end slid on a flat counterface: Archard wear with the contact pressure redistributed "
                   "on an elastic layer, from the first Hertz-like contact through running-in to steady flat-punch "
                   "wear; one run per material (classic Archard & Hirst data).")
    tags = ["tribology", "wear", "materials", "mechanics", "dataset"]
    references = ["J. F. Archard, J. Appl. Phys. 24 (1953) 981", "J. F. Archard and W. Hirst, Proc. R. Soc. A 236 (1956) 397",
                  "I. Hutchings and P. Shipway, Tribology, 2nd ed. (2017), table 5.2"]
    limitations = ["one wear coefficient per material pair (no transition between mild and severe wear, no "
                   "temperature or debris effects)", "Winkler elastic layer instead of a full elastic half-space",
                   "the material data are order-of-magnitude values for dry sliding at low load"]
    case_param = "material"
    params = {"material": "60/40 brass", "load_N": 50.0, "length_mm": 20.0, "width_mm": 10.0,
              "crown_radius_mm": 400.0, "distance_runins": 2.0, "render": True}
    space = {"material": ["mild steel", "60/40 brass", "hardened tool steel", "stellite", "PTFE",
                          "ferritic stainless steel", "polyethylene", "tungsten carbide"]}

    def run(self, ctx):
        from pinneapple_physics.tribology import WEAR_MATERIALS, winkler_parabolic_contact
        from pinneapple_physics.tribology import BarWear as Bar
        p = ctx.params
        mat = WEAR_MATERIALS[p["material"]]
        bar = Bar(mat, load_N=p["load_N"], length_mm=p["length_mm"], width_mm=p["width_mm"],
                  crown_radius_mm=p["crown_radius_mm"])
        ctx.input("material", {"name": mat.name, "k": mat.k, "hardness_MPa": mat.hardness_MPa, "E_GPa": mat.E_GPa,
                               "K_mm3_per_Nm": mat.K})
        p0, _ = bar.pressure()
        exact0 = winkler_parabolic_contact(p["load_N"], p["width_mm"], p["crown_radius_mm"], bar.k_w)
        half = 0.5 * bar.dx * (p0 > 1e-9 * p0.max()).sum()
        s_run = bar.running_in_distance_m()
        with ctx.stage("slide"):
            r = bar.run(p["distance_runins"] * s_run)
        A = bar.area_mm2
        KF = mat.K * p["load_N"]
        conformed = np.searchsorted(r["contact"], 0.999)
        s_conf = float(r["s"][conformed]) if conformed < len(r["s"]) else float("nan")
        ctx.metric("K_mm3_per_Nm", mat.K)
        ctx.metric("wear_rate_mm3_per_m", KF)
        ctx.metric("steady_wear_rate_um_per_km", r["rate"][-1] * 1e6)
        ctx.metric("initial_peak_pressure_MPa", float(p0.max()))
        ctx.metric("steady_pressure_MPa", float(r["p_max"][-1]))
        ctx.metric("running_in_distance_m", s_conf)
        ctx.metric("life_to_1mm_km", A * 1.0 / KF / 1e3)
        ctx.metric("distance_m", float(r["s"][-1]))
        ctx.check("worn_volume_equals_K_F_s", value=abs(r["volume"][-1] / (KF * r["s"][-1]) - 1), max=1e-6,
                  detail="global Archard law, any pressure distribution", kind="physics")
        ctx.check("initial_contact_vs_closed_form", value=float(p0.max()), reference=exact0["p_max_MPa"], rtol=0.02,
                  detail=f"Winkler parabolic contact: half-width {exact0['half_width_mm']:.3f} mm (numerical "
                         f"{half:.3f})", kind="reference")
        ctx.check("steady_rate_vs_flat_punch", value=r["rate"][-1], reference=KF / A, rtol=0.01,
                  detail="dh/ds = K F / A once the end has conformed", kind="reference")
        ctx.check("running_in_vs_estimate", value=s_conf, reference=s_run, rtol=0.05,
                  detail="crown volume / (K F)", kind="reference")
        ds = ctx.dataset("wear_history", description="Wear history: sliding distance, worn volume, peak pressure, "
                         "contact fraction; profiles and pressures along the bar end",
                         units={"s": "m", "volume": "mm3", "p_max": "MPa", "x": "mm", "profiles": "mm", "pressures": "MPa"})
        ds.add(s=r["s"].astype(np.float32), volume=r["volume"].astype(np.float32), p_max=r["p_max"].astype(np.float32),
               contact=r["contact"].astype(np.float32), x=r["x"].astype(np.float32),
               profiles=r["profiles"].astype(np.float32), pressures=r["pressures"].astype(np.float32),
               s_saved=r["s_saved"].astype(np.float32), material=mat.name, K=mat.K, load_N=p["load_N"])
        plt = _plt()
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        cm = plt.get_cmap("viridis")
        idx = np.linspace(0, len(r["s_saved"]) - 1, 7).astype(int)
        for k, i in enumerate(idx):
            c = cm(k / (len(idx) - 1))
            ax[0].plot(r["x"], (r["crown"] + r["profiles"][i] - (r["crown"] + r["profiles"][i]).min()) * 1e3, color=c,
                       label=f"{r['s_saved'][i]:.0f} m")
            ax[1].plot(r["x"], r["pressures"][i], color=c)
        ax[0].set_title("bar end above its lowest point (µm)", fontsize=9)
        ax[0].legend(frameon=False, fontsize=7)
        ax[1].axhline(p["load_N"] / A, color="k", lw=0.8, ls="--")
        ax[1].set_title("contact pressure (MPa); dashed: F / A", fontsize=9)
        ax[2].plot(r["s"], r["volume"], color="#d95f02", label="worn volume")
        ax[2].plot(r["s"], KF * r["s"], "k--", lw=0.8, label="K F s (Archard)")
        ax[2].axvline(s_conf, color="#7570b3", lw=0.8)
        ax[2].set_xlabel("sliding distance (m)")
        ax[2].set_title(f"{mat.name}: running-in ends at {s_conf:.3g} m", fontsize=9)
        ax[2].legend(frameon=False, fontsize=8)
        for a in ax[:2]:
            a.set_xlabel("x along sliding (mm)")
        fig.tight_layout()
        ctx.figure("wear_profiles", fig)
        frames = []
        for i in range(0, len(r["s_saved"]), 2):
            fig, ax = plt.subplots(2, 1, figsize=(5.2, 4.2), sharex=True)
            surf = (r["crown"] + r["profiles"][i] - (r["crown"] + r["profiles"][i]).min()) * 1e3
            ax[0].fill_between(r["x"], surf, surf.max() + 20, color="#9aa4ab")
            ax[0].plot(r["x"], np.zeros_like(r["x"]), color="#30363b", lw=3)
            ax[0].set_ylim(-5, r["crown"].max() * 1e3 + 25)
            ax[0].set_ylabel("µm")
            ax[0].set_title(f"{mat.name} · s = {r['s_saved'][i]:.0f} m", fontsize=9)
            ax[1].fill_between(r["x"], r["pressures"][i], color="#d95f02", alpha=0.7)
            ax[1].axhline(p["load_N"] / A, color="k", lw=0.8, ls="--")
            ax[1].set_ylim(0, r["pressures"][0].max() * 1.05)
            ax[1].set_ylabel("p (MPa)")
            ax[1].set_xlabel("x (mm)")
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0)
            plt.close(fig)
        ctx.gif("running_in", frames, duration_ms=120)
        if p["render"]:
            try:
                import pinneapple as pp
                i = int(np.argmin(np.abs(r["s_saved"] - 0.45 * s_conf)))   # mid running-in: worn land + crown
                surf = r["crown"] + r["profiles"][i]
                wear = r["profiles"][i] * 1e3
                sc = _worn_bar_scene(r["x"], surf - surf.min(), wear, p["width_mm"],
                                     exaggerate=6.0 / max(r["crown"].max(), 1e-9))
                out = ctx.path("figures", "worn_end_render.jpg")
                pp.viz.render(sc, out, field="wear_um", view=(-1.0, -1.4, 1.1), distance=1.15, samples=48,
                              size=(1200, 720), title=f"{mat.name} at s = {r['s_saved'][i]:.0f} m (crown x"
                              f"{6.0 / max(r['crown'].max(), 1e-9):.0f} in height)")
                ctx.files["figures"].append(os.path.relpath(out, ctx.dir))
            except Exception as exc:                       # noqa: BLE001 - the render is optional (bpy)
                ctx.log(f"render skipped: {exc}")


@register
class BarWearRanking(Experiment):
    name = "bar_wear_ranking"
    version = "1"
    description = ("Every bar_wear run side by side: specific wear rate, life to 1 mm of wear at the same load and "
                   "the ranking of the materials, checked against the ranking of their Archard coefficients K = k/H.")
    tags = ["tribology", "wear", "materials", "comparison"]
    references = BarWear.references
    params = {"seed": 0}

    def run(self, ctx):
        from ..store import LabStore
        lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ctx.dir))))
        rows = [r for r in LabStore(lab_root).runs("bar_wear", status="completed")]
        if len(rows) < 3:
            raise RuntimeError("bar_wear_ranking needs at least 3 completed bar_wear runs")
        rows.sort(key=lambda r: r["metrics"]["wear_rate_mm3_per_m"])
        names = [r["params"]["material"] for r in rows]
        life = [r["metrics"]["life_to_1mm_km"] for r in rows]
        K = [r["metrics"]["K_mm3_per_Nm"] for r in rows]
        meas = [r["metrics"]["steady_wear_rate_um_per_km"] for r in rows]
        ctx.output("ranking", [{"material": n, "K_mm3_per_Nm": k, "life_to_1mm_km": li}
                               for n, k, li in zip(names, K, life, strict=True)])
        ctx.metric("materials", len(rows))
        ctx.metric("best_material_life_km", life[0])
        ctx.metric("worst_material_life_km", life[-1])
        ctx.check("ranking_follows_archard_K", passed=all(a <= b for a, b in zip(K, K[1:], strict=False))
                  and all(a <= b for a, b in zip(meas, meas[1:], strict=False)),
                  detail="simulated steady wear rates order the materials as K = k / H", kind="physics")
        ds = ctx.dataset("ranking", description="Materials ranked by simulated wear rate")
        for n, k, li, m in zip(names, K, life, meas, strict=True):
            ds.add(material=n, K=k, life_to_1mm_km=li, steady_rate_um_per_km=m)
        plt = _plt()
        fig, ax = plt.subplots(figsize=(8, 3.8))
        y = np.arange(len(names))
        ax.barh(y, life, color=plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(names))))
        ax.set_xscale("log")
        ax.set_xlim(min(life) / 2, max(life) * 12)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("sliding distance to wear 1 mm at 50 N (km)")
        for i, li in enumerate(life):
            ax.text(li, i, f"  {li:.3g} km", va="center", fontsize=8)
        ax.set_title("Wear life of a 20 × 10 mm bar end, Archard-Hirst data", fontsize=10)
        fig.tight_layout()
        ctx.figure("wear_life_ranking", fig)
