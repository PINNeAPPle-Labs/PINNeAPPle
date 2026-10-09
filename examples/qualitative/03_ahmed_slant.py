"""Qualitative preview of the Ahmed body's rear slant angle, compared with the wind-tunnel measurements.

Ahmed, Ramm & Faltin (1984, SAE 840300) measured the drag coefficient against the slant angle: it falls slightly
from 0 to 12.5 degrees, rises to a peak just below 30 degrees and drops abruptly when the flow over the slant fully
separates. The preview's flow model knows attached pressure recovery and separation beyond a threshold angle, but not
the pair of longitudinal vortices that make the 20-30 degree range so draggy, and it says so: its confidence drops
near the separation angle. ``quantify`` against the measurements shows where the qualitative reading holds and where
it does not, which is the point of doing it before trusting it.
"""
from pinneapple_design.geometry.bodies import ahmed_body
from pinneapple_design.qualitative import preview

# total drag coefficient, Ahmed et al. (1984), Re = 4.3e6
MEASURED_CD = {0.0: 0.250, 12.5: 0.230, 25.0: 0.285, 30.0: 0.378, 35.0: 0.260, 40.0: 0.250}


def main(lang="pt", out="ahmed_preview.png"):
    geoms = {f"slant {a:g}°": ahmed_body(a) for a in MEASURED_CD}
    p = preview(geoms, "minimizar coeficiente de arrasto" if lang == "pt" else "minimize drag coefficient",
                {"U": 60.0}, lang=lang)
    print(p.text())
    p.figure(out)
    check = p.quantify(lambda name, g: MEASURED_CD[float(name.split()[1].rstrip("°"))], variants=list(geoms))
    print()
    print(check.text())
    return p, check


if __name__ == "__main__":
    main()
