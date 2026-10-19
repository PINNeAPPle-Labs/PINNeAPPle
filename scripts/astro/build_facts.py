"""Build the catalogue part of the facts dataset (pinneapple_physics/astro/data/facts) from d3-celestial data.

  * messier.json  -> facts/messier_catalog.json: the 110 Messier objects (J2000 position, type, V magnitude, size, NGC
    designation), d3-celestial data (c) 2015 Olaf Frohn, BSD-3-Clause.
  * constellations.bounds.json -> facts/constellation_areas.json: the area of each constellation (square degrees),
    computed from the IAU boundaries (Delporte 1930) as precessed to J2000 in d3-celestial.

    python scripts/astro/build_facts.py D3_CELESTIAL_DATA_DIR        (npm pack d3-celestial; package/data)
"""
import json
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "pinneapple_physics", "astro", "data", "facts")


def _unit(lon_deg, lat_deg):
    lo, la = np.radians(lon_deg), np.radians(lat_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def polygon_area_deg2(ring) -> float:
    """Area (deg^2) of a spherical polygon whose edges run along lines of constant RA or Dec (the IAU boundaries).

    The ring is densified (edges of constant Dec are small circles, not great circles), rotated so that its mean
    direction sits on the equator (no pole inside), and integrated with Green's theorem on the sphere:
    A = | sum (lon_{i+1} - lon_i) (sin lat_i + sin lat_{i+1}) / 2 |."""
    _, v = _frame(ring)
    lon, lat = _lonlat(v)
    lon2, lat2 = np.roll(lon, -1), np.roll(lat, -1)
    dl = (lon2 - lon + np.pi) % (2 * np.pi) - np.pi
    return float(abs(np.sum(dl * (np.sin(lat) + np.sin(lat2)) / 2.0)) * (180.0 / np.pi) ** 2)


def _frame(ring):
    """Rotation that puts the ring's mean direction at (lon, lat) = (0, 0), and the densified ring in that frame."""
    pts = []
    for a, b in zip(ring[:-1], ring[1:], strict=True):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        d[0] = (d[0] + 180.0) % 360.0 - 180.0
        n = max(2, int(np.abs(d).max() * 8))
        pts += [a + d * k / n for k in range(n)]
    u = _unit(*np.array(pts).T)
    e1 = u.mean(0) / np.linalg.norm(u.mean(0))
    z = np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e3 = np.cross(e1, z)
    e3 /= np.linalg.norm(e3)
    R = np.stack([e1, np.cross(e3, e1), e3])
    return R, u @ R.T


def _lonlat(v):
    return np.arctan2(v[..., 1], v[..., 0]), np.arcsin(np.clip(v[..., 2], -1, 1))


def contains(ring, ra_deg, dec_deg) -> bool:
    """Is the point inside the boundary ring? Ray casting in the ring's own rotated frame (no pole, no RA wrap)."""
    R, v = _frame(ring)
    lon, lat = _lonlat(v)
    x, y = _lonlat(R @ _unit(ra_deg, dec_deg))
    inside = False
    for i in range(len(lon)):
        x1, y1, x2, y2 = lon[i], lat[i], lon[i - 1], lat[i - 1]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def constellation_of(d3_dir, ra_deg, dec_deg):
    bounds = json.load(open(os.path.join(d3_dir, "constellations.bounds.json"), encoding="utf-8"))
    for f in bounds["features"]:
        g = f["geometry"]
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
        if any(contains(r, ra_deg, dec_deg) for r in rings):
            return f["id"]
    return None


def constellation_areas(d3_dir):
    bounds = json.load(open(os.path.join(d3_dir, "constellations.bounds.json"), encoding="utf-8"))
    out = {}
    for f in bounds["features"]:
        g = f["geometry"]
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
        key = f["id"]
        if key == "Ser":                               # one feature per half; tell them apart by right ascension
            ra = np.degrees(np.arctan2(*np.mean([[np.sin(np.radians(p[0])), np.cos(np.radians(p[0]))]
                                                  for p in rings[0]], 0))) % 360
            key = "Ser1" if ra < 255 else "Ser2"
        out[key] = round(out.get(key, 0.0) + sum(polygon_area_deg2(r) for r in rings), 3)
    return out


def messier(d3_dir):
    d = json.load(open(os.path.join(d3_dir, "messier.json"), encoding="utf-8"))
    rows = []
    for f in d["features"]:
        p = f["properties"]
        ra, dec = f["geometry"]["coordinates"]
        mag = p.get("mag")
        rows.append({"id": f["id"], "desig": p.get("desig", ""), "alt_name": p.get("alt", ""), "type": p.get("type", ""),
                     "ra_deg": round(ra % 360, 4), "dec_deg": round(dec, 4),
                     "v_mag": None if mag in (None, "", 99) else float(mag), "size_arcmin": p.get("dim", ""),
                     "morphology": p.get("cl", p.get("morph", "")),
                     "constellation": constellation_of(d3_dir, ra % 360, dec)})
    rows.sort(key=lambda r: int(r["id"][1:]))
    return rows


def main(d3_dir):
    os.makedirs(OUT, exist_ok=True)
    areas = constellation_areas(d3_dir)
    src = {"source": "IAU constellation boundaries (Delporte 1930), J2000 polygons from d3-celestial (BSD-3, O. Frohn); "
                     "areas computed by scripts/astro/build_facts.py"}
    json.dump({**src, "unit": "deg2", "areas": dict(sorted(areas.items()))},
              open(os.path.join(OUT, "constellation_areas.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rows = messier(d3_dir)
    json.dump({"source": "d3-celestial messier.json (BSD-3, O. Frohn): J2000 position, type, V magnitude, size",
               "objects": rows}, open(os.path.join(OUT, "messier_catalog.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    print(len(areas), "constellations, total", round(sum(areas.values()), 2), "deg2;", len(rows), "Messier objects")


if __name__ == "__main__":
    main(sys.argv[1])
