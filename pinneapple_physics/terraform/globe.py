"""The globe: a Mars that shrinks its caps over centuries, with a confidence bar that runs out (#374).

The picture is built the way the weather examples build theirs -- the same orthographic projection, the same dark
frame, the same habit of drawing an interpolated field from a coarse grid and saying so -- because
``pinneapple_physics.weather.viz`` already has the projection and there is no reason to write a second one.

What the globe shows is the model's own state, not a picture of Mars:

* the **surface** is shaded by a **synthetic** topography -- a Tharsis-sized bulge, a volcano, a rift and a basin,
  built from four Gaussian bumps. It is not MOLA and the code says so in the corner of the frame: the repo has no
  topography of Mars and this package does not invent one from the literature. It exists so that the globe reads as a
  globe rather than as a latitudinal stripe;
* the **temperature** of each latitude band is the model's annual mean, drawn interpolated over the coarse grid;
* the **caps** are white where the frost fraction of the band puts the cap edge, and they retreat poleward as the CO2
  sublimates -- the radius of the polar cap is computed from the *area* the model says is frosted, not drawn to look
  right;
* the **confidence bar** under the globe is ``1 - spread / SIGMA_THRESHOLD_K``: it reaches zero exactly at the century
  :func:`pinneapple_physics.terraform.terraforming.trust_horizon` returns, and the frame after it says so.

WhatsApp-sized honesty: the whole thing is a one-dimensional model wrapped round a sphere. The globe is a way to read
the model, and the alternative -- a map with a coastline -- would be a way to misread it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["PALETTE", "synthetic_topography", "cap_edge_latitude", "surface_field", "trust", "mars_gif",
           "century_figure"]

#: Mars-ish colours: the dark basaltic lowlands, the bright dust, and the frost.
PALETTE = {"bg": "#05080d", "fg": "#e8eef5", "dim": "#7d8a99", "dust": "#c96f3a", "rock": "#7a3b21",
           "high": "#e6a273", "frost": "#eef4fb", "good": "#3ddc97", "warn": "#f5b942", "bad": "#ff5d5d"}


def synthetic_topography(latitude_deg, longitude_deg) -> np.ndarray:
    """A declared synthetic relief for Mars (km), built from four Gaussian features. Not MOLA, not a map of Mars.

    Olympus Mons, the Tharsis bulge, Valles Marineris and Hellas are placed roughly where they are on the real planet
    so the globe is recognisable, and their amplitudes are round numbers chosen to look like a planet rather than
    measured from anything. Every figure this module draws says "synthetic topography" on it.
    """
    lat = np.radians(np.asarray(latitude_deg, float))
    lon = np.radians(np.asarray(longitude_deg, float))
    features = ((18.0, -133.0, 8.0, 21.0),        # Olympus Mons
                (0.0, -110.0, 55.0, 7.0),         # the Tharsis bulge
                (-12.0, -65.0, 18.0, -6.0),       # Valles Marineris, a rift
                (-42.0, 70.0, 25.0, -7.0))        # Hellas, a basin
    height = np.zeros_like(lat, dtype=float)
    for clat, clon, sigma, amplitude in features:
        dlat = lat - np.radians(clat)
        dlon = np.angle(np.exp(1j * (lon - np.radians(clon))))
        height += amplitude * np.exp(-(dlat ** 2 + (dlon * np.cos(lat)) ** 2) / (2.0 * np.radians(sigma) ** 2))
    return height


def cap_edge_latitude(cover, polar_latitude_deg: float = 70.0) -> np.ndarray:
    """The latitude of the cap edge from the fraction of the polar area that is frosted.

    A spherical cap poleward of ``phi`` covers ``2 pi R^2 (1 - sin phi)``, so a frost fraction ``f`` of the region
    poleward of ``phi_0`` puts the edge at ``sin phi = sin phi_0 + (1 - f)(1 - sin phi_0)``. ``f = 1`` is ``phi_0``;
    ``f = 0`` is the pole itself, and there is no cap.
    """
    f = np.clip(np.asarray(cover, float), 0.0, 1.0)
    s0 = np.sin(np.radians(polar_latitude_deg))
    return np.degrees(np.arcsin(np.clip(s0 + (1.0 - f) * (1.0 - s0), -1.0, 1.0)))


def surface_field(latitude_deg, longitude_deg, temperature, frost_fraction) -> tuple[np.ndarray, np.ndarray]:
    """The two fields the globe draws: the surface temperature (K) and the frost mask, both ``(lat, lon)``."""
    lat = np.asarray(latitude_deg, float)
    lon = np.asarray(longitude_deg, float)
    temperature = np.asarray(temperature, float)
    if temperature.ndim == 1:
        temperature = temperature[:, None]
    field = np.broadcast_to(temperature, (len(lat), len(lon))).astype(float).copy()
    edge = cap_edge_latitude(frost_fraction)
    pole = np.abs(lat)[:, None] >= edge                       # the same edge for both poles
    return field, pole.astype(float)


def trust(spread, threshold: float) -> np.ndarray:
    """The length of the confidence bar: 1 while the ensemble is tight, 0 once its spread reaches the threshold."""
    return np.clip(1.0 - np.asarray(spread, float) / float(threshold), 0.0, 1.0)


def _shading(size: int, light=(-0.55, -0.45, 0.70)) -> np.ndarray:
    """A Lambert-ish light on the visible disk, so a flat latitudinal field still reads as a sphere."""
    axis = np.linspace(-1.0, 1.0, size)
    X, Y = np.meshgrid(axis, -axis)
    r2 = X ** 2 + Y ** 2
    Z = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    vector = np.array(light, float)
    vector /= np.linalg.norm(vector)
    return np.clip(0.32 + 0.78 * (X * vector[0] + Y * vector[1] + Z * vector[2]), 0.12, 1.25)


def mars_gif(path, years, temperature, cover, pressure_pa, spread, latitude_deg, *, threshold: float = 2.0,
             frames: int = 60, size: int = 340, fps: int = 6, dpi: int = 80,
             title: str = "Mars, and what it does with the CO2 we release") -> str:
    """An animated globe: the caps retreat, the pressure climbs, and the confidence bar shortens to nothing.

    ``temperature`` is ``(time, band)`` in kelvin, ``cover`` the frosted fraction of the polar area, ``pressure_pa``
    the CO2 surface pressure and ``spread`` the ensemble's standard deviation on the global mean. All are sampled on
    ``years``; the animation takes ``frames`` of those samples, evenly. The globe is centred north of the equator so
    that the northern cap fills the frame; the model is zonally symmetric, so the southern cap is its mirror image.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.animation import FuncAnimation, PillowWriter

    from ..weather.viz import orthographic

    years = np.asarray(years, float)
    pick = np.unique(np.linspace(0, len(years) - 1, frames).astype(int))
    lon = np.arange(-180.0, 180.0, 3.0)
    lat_fine = np.arange(-88.0, 88.1, 2.0)
    relief = synthetic_topography(*np.meshgrid(lat_fine, lon, indexing="ij"))
    relief_unit = (relief - relief.min()) / (relief.max() - relief.min() + 1e-9)
    rgb = np.stack([0.45 + 0.48 * relief_unit, 0.20 + 0.44 * relief_unit, 0.11 + 0.30 * relief_unit])
    shade = _shading(size)
    centre = (42.0, -70.0)

    fig = plt.figure(figsize=(9.4, 5.2), facecolor=PALETTE["bg"])
    ax = fig.add_axes([0.015, 0.02, 0.47, 0.86])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")
    axt = fig.add_axes([0.52, 0.28, 0.46, 0.58])
    axt.axis("off")
    axb = fig.add_axes([0.52, 0.12, 0.42, 0.07])
    axb.set_facecolor("#111823")

    def picture(i: int, center_lon: float):
        edge = float(cap_edge_latitude(cover[i]))
        frost = np.repeat((np.abs(lat_fine) >= edge)[:, None].astype(float), len(lon), axis=1)
        channels = [orthographic(channel, lat_fine, lon, (centre[0], center_lon), size=size) for channel in rgb]
        frost_image = orthographic(frost, lat_fine, lon, (centre[0], center_lon), size=size)
        image = np.stack(channels, axis=-1)
        inside = np.isfinite(image[..., 0])
        image = np.where(np.isfinite(image), image, 0.0) * shade[..., None]
        keep = np.clip(frost_image, 0.0, 1.0)
        keep = np.where(np.isfinite(keep), keep, 0.0)
        image = image * (1.0 - keep[..., None]) + keep[..., None] * np.array([0.93, 0.95, 0.99]) * shade[..., None]
        return np.clip(image, 0.0, 1.0), inside

    image, inside = picture(0, centre[1])
    art = ax.imshow(np.dstack([image, inside.astype(float)]), interpolation="bilinear")
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)

    fig.text(0.02, 0.955, title, color=PALETTE["fg"], fontsize=13, weight="bold")
    sub = fig.text(0.02, 0.917, "", color=PALETTE["dim"], fontsize=8.5)
    year_text = axt.text(0.0, 1.02, "", color=PALETTE["fg"], fontsize=27, weight="bold", va="top")
    axt.text(0.0, 0.64, "72 latitude bands · annual means\nno seasons, no dust, no water\nsynthetic topography",
             color=PALETTE["dim"], fontsize=7.5, va="top", transform=axt.transAxes)
    readings = axt.text(0.0, 0.44, "", color=PALETTE["fg"], fontsize=10.5, va="top", transform=axt.transAxes)
    axt.text(0.0, 0.06, "CONFIDENCE", color=PALETTE["dim"], fontsize=7.5, transform=axt.transAxes)
    for side in axb.spines.values():
        side.set_color("#253040")
    axb.set_xticks([])
    axb.set_yticks([])
    axb.set_xlim(0, 1)
    axb.set_ylim(-0.5, 0.5)
    bar = axb.barh([0], [1.0], height=0.64, color=PALETTE["good"])[0]
    bar_text = axb.text(0.5, 1.5, "", color=PALETTE["dim"], fontsize=7.8, ha="center", va="bottom")

    for text in (year_text, readings):
        text.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground=PALETTE["bg"])])

    def frame(k: int):
        i = int(pick[k])
        center_lon = ((centre[1] + 320.0 * k / len(pick) + 180.0) % 360.0) - 180.0
        image, inside = picture(i, center_lon)
        art.set_data(np.dstack([image, inside.astype(float)]))
        year_text.set_text(f"year {years[i]:.0f}")
        readings.set_text(f"global mean      {np.mean(temperature[i]):6.1f} K\n"
                          f"CO$_2$ pressure  {pressure_pa[i] / 100:6.2f} mbar\n"
                          f"polar cap        {100 * cover[i]:4.0f} % of the cap area\n"
                          f"cap edge         {cap_edge_latitude(cover[i]):4.1f}°\n"
                          f"spread on the mean  ± {spread[i]:.2f} K")
        length = float(trust(spread[i], threshold))
        bar.set_width(max(length, 1e-3))
        bar.set_color(PALETTE["good"] if length > 0.5 else PALETTE["warn"] if length > 0.0 else PALETTE["bad"])
        bar_text.set_text(f"{100 * length:.0f} % of the trust left · ±{spread[i]:.2f} K on the global mean"
                          if length > 0.0 else "no trust left: the spread is the size of the answer")
        sub.set_text("a one-dimensional energy-balance model with the CO2 cycle · the bar reaches zero at the "
                     "trust horizon")
        return art, year_text, readings, bar, bar_text, sub

    animation = FuncAnimation(fig, frame, frames=len(pick), blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": PALETTE["bg"]})
    plt.close(fig)
    return str(path)


def century_figure(path, years, temperature, pressure_pa, cover, spread, *, threshold: float = 2.0,
                   horizon: float = float("nan"), dpi: int = 140) -> str:
    """The four curves the GIF is a view of: warming with the ensemble band, pressure, cap, and the spread."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = np.asarray(years, float)
    temperature = np.asarray(temperature, float)
    mean = temperature.mean(axis=1) if temperature.ndim > 1 else temperature
    lo = np.percentile(temperature, 16, axis=1) if temperature.ndim > 1 else mean
    hi = np.percentile(temperature, 84, axis=1) if temperature.ndim > 1 else mean
    pressure = np.asarray(pressure_pa, float) / 100.0
    cover = np.asarray(cover, float) * 100.0
    spread = np.asarray(spread, float)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), facecolor=PALETTE["bg"], sharex=True)
    specs = (("global mean surface temperature", "K", [(mean, PALETTE["fg"]), (lo, "#5fb3ff"), (hi, "#5fb3ff")]),
             ("CO$_2$ surface pressure", "mbar", [(pressure, "#ff9f43")]),
             ("polar cap", "% of the cap area", [(cover, PALETTE["frost"])]),
             ("spread of the ensemble", "K", [(spread, "#c792ea")]))
    for ax, (label, unit, series) in zip(axes.ravel(), specs, strict=True):
        ax.set_facecolor(PALETTE["bg"])
        if len(series) == 3:
            ax.fill_between(years, series[1][0], series[2][0], color="#5fb3ff", alpha=0.18)
        for values, colour in series:
            ax.plot(years, values, color=colour, lw=1.8)
        ax.set_ylabel(f"{label} ({unit})", color=PALETTE["dim"], fontsize=8.5)
        ax.tick_params(colors=PALETTE["dim"], labelsize=8)
        for side in ax.spines.values():
            side.set_color("#253040")
    if np.isfinite(horizon):
        for ax in axes.ravel():
            ax.axvline(horizon, color=PALETTE["bad"], ls="--", lw=1.2)
        axes[0, 1].text(horizon + 8, float(np.nanmin(pressure)) + 0.12 * (np.nanmax(pressure) - np.nanmin(pressure)),
                        f"trust horizon\nyear {horizon:.0f}", color=PALETTE["bad"], fontsize=8.5)
    axes[1, 1].axhline(threshold, color=PALETTE["dim"], ls=":", lw=1.0)
    axes[1, 1].text(years[1], threshold * 1.05, f"the declared threshold, {threshold:.0f} K",
                    color=PALETTE["dim"], fontsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("year of the scenario", color=PALETTE["dim"], fontsize=8.5)
    fig.suptitle("Mars over centuries: the caps retreat, the pressure stops at what the CO2 allows",
                 color=PALETTE["fg"], fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=dpi, facecolor=PALETTE["bg"])
    plt.close(fig)
    return str(path)
