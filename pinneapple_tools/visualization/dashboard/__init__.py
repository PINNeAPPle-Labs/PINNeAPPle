"""Animation-centred dashboards for any simulation: an animation in the middle, charts synced to its time cursor,
KPI tiles, a parameter panel and a colour bar; exported as interactive HTML (scrubbing a chart moves the
animation), a static PNG and a dashboard GIF/MP4.

    import pinneapple as pp
    from pinneapple_tools.visualization.dashboard import Dashboard, Chart, Series, Marker, Tile, Param, ColorBar
    db = Dashboard("Pendulum", t, frames,                      # frames: RGB arrays or image paths, one per t
                   charts=[Chart("Angle", "angle [deg]", "time [s]", [Series("theta", theta_deg)], slot="right")],
                   tiles=[Tile("Energy", E, "J")], params=[Param("Length", 1.0, "m")])
    db.to_html("pendulum.html"); db.to_png("pendulum.png"); db.to_video(gif="pendulum.gif", mp4="pendulum.mp4")

Chart kinds: ``"series"`` (one value per frame, cursor line, scrub to seek) and ``"phase"`` (fixed curves with
markers that move with the frames). See ``docs/core_concepts/dashboard.md``.
"""
from .field_frames import Table, cor_por_valor, frames_from_field
from .spec import PALETTE_DARK, THEME, Chart, ColorBar, Dashboard, Marker, Param, Series, Tile
from .surface3d import Cena3D, animacao_de_campos, contorno_de_step, gerar as gerar_animacao3d, quadros_de_campos

__all__ = ["Dashboard", "Chart", "Series", "Marker", "Tile", "Param", "ColorBar", "PALETTE_DARK", "THEME",
           "Table", "frames_from_field", "cor_por_valor",
           "Cena3D", "contorno_de_step", "quadros_de_campos", "gerar_animacao3d", "animacao_de_campos"]
