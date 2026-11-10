"""The passages that ship with the pipeline: a globe valve, an orifice plate, a converging nozzle.

These are *specs*, not code paths: each function returns the same :class:`~.passage.PassageSpec` a
caller could build by hand, so the solver, the checks, the tables and the pictures know nothing
about valves. Adding a diffuser, a venturi or a pipe reducer is a new function here and nothing
else.

The globe valve is the case the reference document is about. Its dimensions were read off the
customer's ``geometry.stp`` (30 solids, mm, one per metal part) with an inside/outside test per
solid on the meridian plane, because that file holds the metal and not the flow:

============================  =========  ==========================================================
quantity                      measured   how it was read
============================  =========  ==========================================================
bore and seat-port radius      10.0 mm    radial sweep at x = -105 and x = +50: metal only between
                                         r = 10 and 14 (4 mm wall); the seat ring's bore matches
inlet / outlet face            -118.4 /    bounding planes of the two flange solids
                               +61.5 mm
chamber radius                 30 mm      inner wall of the body, r = 27..30 along the chamber
seat plane                     x = -46 mm the disc section spans x = -48..-44 at every radius out
                                         to 29 mm; the seat ring upstream ends at x = -51
disc radius / thickness        22 / 6 mm  the plug is solid from the axis out to r = 22 mm, 6 mm
                                         thick
as-drawn lift                  4 mm       disc face at x_seat - 6 mm against the seat at x_seat - 2
============================  =========  ==========================================================

Two findings from that reading are reported to the customer, because they decide the case:

1. **The CAD has the valve almost shut.** On the drawing the disc face sits 4 mm from the seat and
   nearly touches the body wall (r = 22 mm in a 30 mm chamber with a 27-30 mm inner wall), so the
   throttling area is a narrow annular curtain. A run made on the file as drawn therefore solves a
   nearly closed valve -- and with 70 bar upstream, 60 bar downstream and no path, it can only show
   the acoustic transient of its own initial field, never a valve result. The study parametrises the
   lift and reports a curve instead of one point.
2. **The metal is thin-walled and the parts overlap by ~0.1 mm**, so a boolean of the solids against
   an envelope returns slivers rather than the flow volume (measured: 0.046-0.113 cm3). Rebuilding
   the domain from the dimensions above is the only route, and it is what this file records.
"""
from __future__ import annotations

from .passage import PassageSpec, Ring, Span

__all__ = ["globe_valve", "orifice_plate", "converging_nozzle", "STUDIES", "CAD_MEASUREMENTS"]

CAD_MEASUREMENTS = {
    "r_bore": (10.0, "radial sweep at x = -105: metal only in r = 10..14"),
    "x_inlet_face": (-118.45, "bounding plane of the inlet flange solid"),
    "x_outlet_face": (61.55, "bounding plane of the outlet flange solid"),
    "r_chamber": (30.0, "inner wall of the body, r = 27..30 along the chamber"),
    "x_seat": (-46.0, "the disc section spans x = -48..-44 at every radius out to 29 mm"),
    "r_disc": (22.0, "the disc section is solid from the axis out to r = 22 mm"),
    "disc_thickness": (6.0, "the disc section spans 6 mm along the axis"),
    "as_drawn_lift": (4.0, "disc face at x_seat - 6 mm; the seat ring ends at x_seat - 2 mm"),
}


def globe_valve(lift: float = 4.0, *, r_bore: float = 10.0, r_chamber: float = 30.0,
                r_disc: float = 22.0, x_seat: float = -46.0, disc_thickness: float = 6.0,
                x_bowl_in: float = -68.0, x_bowl_out: float = 8.0, drag: float = 7.0,
                h_pipe: float = 1.8, h_bowl: float = 1.4, h_gap: float = 0.6,
                n_theta: int = 20, radial_cells: tuple[int, int, int] = (10, 9, 6),
                r_core: float = 0.5, sector_deg: float = 90.0, xT: float = 0.72) -> PassageSpec:
    """The globe valve of the reference document, as a passage in the meridian plane.

    ``lift`` is the distance from the disc face to the seat plane along the axis: the parameter the
    study sweeps. Everything else defaults to the dimensions measured on the customer's CAD.
    """
    x_disc_up = x_seat - lift - disc_thickness
    x_disc_down = x_seat - lift
    x_in = x_bowl_in - drag * 2 * r_bore
    x_out = x_bowl_out + drag * 2 * r_bore
    pipe = (Ring(r_core, r_bore, radial_cells[0]),)
    bowl = (Ring(r_core, r_bore, radial_cells[0]), Ring(r_bore, r_disc, radial_cells[1]),
            Ring(r_disc, r_chamber, radial_cells[2]))

    spans = (
        Span("inlet", x_in, x_bowl_in, pipe, h_pipe),
        Span("bowl_in", x_bowl_in, x_disc_up, bowl, h_bowl),
        Span("disc", x_disc_up, x_disc_down, bowl[2:], h_bowl),
        Span("curtain", x_disc_down, x_seat, bowl, h_gap),
        Span("bowl_out", x_seat, x_bowl_out, bowl, h_bowl),
        Span("outlet", x_bowl_out, x_out, pipe, h_pipe),
    )
    curtain = 2.0 * 3.141592653589793 * 0.5 * (r_bore + r_disc) * lift if lift > 0 else 0.0
    port = 3.141592653589793 * r_bore ** 2
    throat = min(curtain, port)
    return PassageSpec(
        spans=spans, sector_deg=sector_deg, n_theta=n_theta, r_core=r_core,
        title=f"Válvula globo, abertura {lift:g} mm",
        meta={"kind": "globe_valve", "parameter_name": "lift_mm", "parameter_value": lift,
              "lift_mm": lift, "open_fraction": throat / port, "throttling_area_mm2": throat,
              "curtain_area_mm2": curtain, "port_area_mm2": port, "xT": xT,
              "x_seat_mm": x_seat, "r_bore_mm": r_bore, "r_chamber_mm": r_chamber,
              "r_disc_mm": r_disc, "disc_thickness_mm": disc_thickness},
    )


def orifice_plate(hole_ratio: float = 0.5, *, r_pipe: float = 25.0, thickness: float = 4.0,
                  upstream: float = 6.0, downstream: float = 10.0, h: float = 2.0,
                  n_theta: int = 24, n_radial: int = 12, r_core: float = 0.5,
                  sector_deg: float = 90.0, xT: float = 0.2) -> PassageSpec:
    """A thin orifice in a straight pipe — the other classic throttling passage.

    Same mesher, same solver, same checks: only the span list differs, which is the point of the
    pipeline. ``hole_ratio`` is the hole diameter over the pipe's (measured on the bore).
    """
    r_hole = hole_ratio * r_pipe
    n_hole = max(2, int(round(n_radial * hole_ratio)))
    duct = (Ring(r_core, r_hole, n_hole), Ring(r_hole, r_pipe, n_radial - n_hole))
    spans = (
        Span("inlet", -upstream * 2 * r_pipe, -thickness / 2, duct, h),
        Span("plate", -thickness / 2, thickness / 2, duct[:1], thickness),
        Span("outlet", thickness / 2, downstream * 2 * r_pipe, duct, h),
    )
    area = 3.141592653589793 * r_hole ** 2
    return PassageSpec(spans=spans, sector_deg=sector_deg, n_theta=n_theta, r_core=r_core,
                       title=f"Placa de orifício, razão de área {hole_ratio:g}",
                       meta={"kind": "orifice", "parameter_name": "hole_ratio",
                             "parameter_value": hole_ratio, "throttling_area_mm2": area,
                             "open_fraction": hole_ratio ** 2, "xT": xT,
                             "r_hole_mm": r_hole, "r_pipe_mm": r_pipe})


def converging_nozzle(area_ratio: float = 4.0, *, r_exit: float = 8.0, length: float = 30.0,
                      upstream: float = 6.0, h: float = 1.5, n_theta: int = 24, n_radial: int = 16,
                      r_core: float = 0.5, sector_deg: float = 90.0, xT: float = 0.2) -> PassageSpec:
    """A converging nozzle, meshed as a stack of short spans so the radius can taper.

    The band tapers linearly in x (:class:`~.passage.Ring` carries the radii at both ends), so the
    cone is one span and the wall is the cone, not a staircase.
    """
    r_inlet = (area_ratio ** 0.5) * r_exit
    cone = (Ring(r_core, r_inlet, n_radial, r1_end=r_exit),)
    spans = (Span("inlet", -upstream * 2 * r_inlet, -length / 2, (Ring(r_core, r_inlet, n_radial),), h),
             Span("cone", -length / 2, length / 2, cone, h),
             Span("outlet", length / 2, length / 2 + 6 * r_exit, (Ring(r_core, r_exit, n_radial),), h))
    return PassageSpec(spans=spans, sector_deg=sector_deg, n_theta=n_theta, r_core=r_core,
                       title=f"Bocal convergente, razão de áreas {area_ratio:g}",
                       meta={"kind": "nozzle", "parameter_name": "area_ratio",
                             "parameter_value": area_ratio,
                             "throttling_area_mm2": 3.141592653589793 * r_exit ** 2,
                             "open_fraction": 1.0 / area_ratio, "xT": xT,
                             "r_exit_mm": r_exit, "r_inlet_mm": r_inlet})


STUDIES = {"globe_valve": globe_valve, "orifice": orifice_plate, "nozzle": converging_nozzle}
