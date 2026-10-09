"""Earth-system blocks in one script (about one minute on a laptop CPU):

1. Rossby-Haurwitz wave (Williamson case 6) on the sphere at T42 for 4 days, with the conserved quantities;
2. 4D-Var on Lorenz-96: background, analysis and truth;
3. Richards infiltration (Celia et al. 1990) after 6 h, 12 h and 24 h.

Writes sphere_and_assimilation.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinneapple_analysis.data_assimilation import twin_experiment
from pinneapple_simulation.geophysics import (
    CELIA_1990_SOIL,
    Boundary,
    SphericalHarmonics,
    solve_richards,
    williamson_case6,
)


def main(days: float = 4.0, out: str = "sphere_and_assimilation.png") -> dict:
    sh = SphericalHarmonics(42)
    sw = williamson_case6(sh)
    sw.run(days * 86400, 600.0, diagnostics_every=86400)
    d0, d1 = sw.history[0], sw.history[-1]
    drift = {k: (d1[k] - d0[k]) / d0[k] for k in ("mass", "energy", "potential_enstrophy")}
    print("shallow water, relative drift after", days, "days:", {k: f"{v:.1e}" for k, v in drift.items()})

    da = twin_experiment(n=40, window_steps=20, seed=0)
    print(f"4D-Var RMSE at window end: background {da['rmse_background_end']:.2f}, analysis {da['rmse_analysis_end']:.2f}")

    soil = solve_richards(CELIA_1990_SOIL, 100.0, 100, -1000.0, 86400.0, Boundary("head", -75.0),
                          Boundary("free_drainage"), t_out=np.array([0, 21600, 43200, 86400]))
    print("Richards mass-balance ratio:", round(soil.mass_balance_ratio, 8))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    lon, lat = np.rad2deg(sh.lon), np.rad2deg(sh.lat)
    c = ax[0].contourf(lon, lat, sw.fields()["h"], 20, cmap="jet")
    fig.colorbar(c, ax=ax[0], label="h [m]")
    ax[0].set(title=f"Rossby-Haurwitz wave, day {days:g} (T42)", xlabel="longitude", ylabel="latitude")
    k_end = 20
    ax[1].plot(da["truth"][k_end].numpy(), "k", lw=2, label="truth")
    ax[1].plot(da["background_trajectory"][k_end].numpy(), "C0--", label=f"background (RMSE {da['rmse_background_end']:.2f})")
    ax[1].plot(da["analysis_trajectory"][k_end].numpy(), "C1", label=f"4D-Var analysis (RMSE {da['rmse_analysis_end']:.2f})")
    ax[1].set(title="Lorenz-96: state at the end of the window", xlabel="variable")
    ax[1].legend(fontsize=8)
    for k, t in enumerate(soil.t[1:], start=1):
        ax[2].plot(soil.theta[k], soil.z, label=f"{t / 3600:g} h")
    ax[2].set(title="Richards infiltration (Celia 1990)", xlabel="water content", ylabel="z [cm]")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    return {"drift": drift, "da": {k: v for k, v in da.items() if k.startswith("rmse")}, "mbr": soil.mass_balance_ratio}


if __name__ == "__main__":
    main()
