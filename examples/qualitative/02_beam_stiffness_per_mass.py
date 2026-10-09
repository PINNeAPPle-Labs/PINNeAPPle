"""Qualitative preview of beam / bracket variants for stiffness per mass, checked with a 3-D voxel FEM.

Same overall envelope, different ways of placing the material: a solid bar, an I-section, a stepped taper (deep at
the clamp, shallow at the tip), and a lightening cut-out either near the clamp or near the tip. The preview explains
each result through where the flexibility comes from (root, middle or tip third of the span) and maps the bending
stress on the surface; ``part_sensitivity`` shows that material added to the flanges near the clamp pays most. The
quantitative step is a 3-D linear-elastic FEM on the voxelized shape (incompatible-mode bricks), which also sees
shear and the stress concentrations that beam theory leaves out.
"""
from pinneapple_design.geometry.bodies import box
from pinneapple_design.qualitative import Assembly, part_sensitivity, preview
from pinneapple_design.qualitative.quantitative import voxel_fem_cantilever

L, B, H = 1.0, 0.05, 0.08                       # span (x), width (y), depth (z), m


def solid():
    return Assembly({"bar": box((L, B, H), (L / 2, 0, 0))})


def i_section(tf=0.012, tw=0.012):
    return Assembly({"top flange": box((L, B, tf), (L / 2, 0, H / 2 - tf / 2)),
                     "web": box((L, tw, H - 2 * tf), (L / 2, 0, 0)),
                     "bottom flange": box((L, B, tf), (L / 2, 0, -H / 2 + tf / 2))})


def stepped_taper(depths=(0.08, 0.065, 0.05, 0.035)):
    n = len(depths)
    return Assembly({f"segment {i + 1}": box((L / n, B, h), (L / n * (i + 0.5), 0, 0)) for i, h in enumerate(depths)})


def with_cutout(where="root", length=0.3, keep=0.02):
    """Solid bar with a through cut-out of ``length`` along x near the clamp or the tip, leaving top and bottom
    strips of depth ``keep``."""
    x0 = 0.1 if where == "root" else L - 0.1 - length
    parts = {"before": box((x0, B, H), (x0 / 2, 0, 0)),
             "top strip": box((length, B, keep), (x0 + length / 2, 0, H / 2 - keep / 2)),
             "bottom strip": box((length, B, keep), (x0 + length / 2, 0, -H / 2 + keep / 2)),
             "after": box((L - x0 - length, B, H), ((x0 + length + L) / 2, 0, 0))}
    return Assembly(parts)


def fem(name, geom):
    return voxel_fem_cantilever(geom, load=1000.0, n_slices=50, cells_across=10)["stiffness_to_mass"]


def main(lang="pt", out="beam_preview.png"):
    geoms = {"solid bar": solid(), "I-section": i_section(), "stepped taper": stepped_taper(),
             "cut-out near clamp": with_cutout("root"), "cut-out near tip": with_cutout("tip")}
    cond = {"E": 70e9, "rho": 2700, "load": 1000.0}
    p = preview(geoms, "aumentar rigidez por massa" if lang == "pt" else "maximize stiffness per mass", cond, lang=lang)
    print(p.text())
    p.figure(out, view=(18, -60))
    print()
    print(part_sensitivity(i_section(), "reduzir a deflexão", cond, step=0.2, lang=lang, axes="z").text())
    check = p.quantify(fem, variants=list(geoms))
    print()
    print(check.text())
    return p, check


if __name__ == "__main__":
    main()
