# HDRIs

Environment images for `render(background="hdri:<name>")`. `pp.viz.hdri_path("<name>")` finds `<name>.hdr` here (this
directory is the default `$PINNEAPPLE_HDRI_DIR`), so `background="hdri:machine_shop_02"` is enough.

The image is both the light and what the metals reflect, which is what makes a brushed or cast surface read as metal
at all: a procedural sky gives a smooth gradient, and a smooth gradient reflected in steel looks like grey plastic.

| File | What it is | Source |
|---|---|---|
| `studio_small_09.hdr` | small photographic studio, soft key and rim | [Poly Haven](https://polyhaven.com/a/studio_small_09) |
| `empty_warehouse_01.hdr` | empty industrial warehouse, skylights down one side | [Poly Haven](https://polyhaven.com/a/empty_warehouse_01) |
| `machine_shop_02.hdr` | machine shop: overhead luminaires, tools, concrete | [Poly Haven](https://polyhaven.com/a/machine_shop_02) |

All three are **CC0** (public domain, no attribution required) from [Poly Haven](https://polyhaven.com), taken at
1k resolution (1.5 MB each) — enough for the reflections of a figure, not for a 4K hero shot, where the 4k/8k
originals are the right download. Add your own by dropping a `.hdr`, `.exr` or `.png` here under any name, or by
pointing `$PINNEAPPLE_HDRI_DIR` at another directory.
