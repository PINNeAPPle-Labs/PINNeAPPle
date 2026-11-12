# Cinematic flow films

A real 3-D solve of the library, lit and rendered as smoke. Nothing in the picture is a texture or a stock
effect: the volume is the simulated vorticity, the surface is the simulated body, and every film ships with a
record of the mesh, the Reynolds number and the steps it came from.

```bash
python -m pinneapple_simulation.pipelines.cinematic_flow carro  --out filmes/carro
python -m pinneapple_simulation.pipelines.cinematic_flow asa    --out filmes/asa
python -m pinneapple_simulation.pipelines.cinematic_flow foguete --out filmes/foguete
python -m pinneapple_simulation.pipelines.cinematic_flow carro --out filmes/carro --still   # um quadro só
```

## The three steps, and why they are separate

| Step | Module | Writes | Re-run cost |
|---|---|---|---|
| Solve | `solve.py` | `snap_*.npz`, `meta.json` | minutes to an hour |
| Render | `scene.py` (inside Blender) | `frames/*.png` | minutes |
| Assemble | `film.py` | `<caso>.mp4`, `ficha.json` | seconds |

The look is tuned on the render side only: the stored fields are the two the picture needs (speed deficit and
vorticity, both dimensionless), and the scalar that becomes density is built at render time. Changing the
threshold that decides what reads as smoke does not cost another solve.

## What is on screen

- **The smoke** is `density = vorticity + speed deficit`, cut below a percentile. Vorticity is what draws the
  filaments and the shear layers; the deficit is what fills the wake. A coarse lattice leaves a low-level noise
  floor in the free stream, so the cut is not optional — without it the whole tunnel turns into fog.
- **The colour** walks from the tint of the thin wake to white where the field is dense, which is what reads as
  depth in a cloud. There is no colour map: the picture is not a plot, and `ficha.json` carries the numbers.
- **The glow** is an emission node multiplied by the density itself. `Principled Volume`'s own emission is
  emitted through the volume's whole bounding box whatever the density is, which turns a sparse field into a
  solid slab — the reason the shader here is built by hand.

## Recorded with every film

`ficha.json` holds the mesh (`malha`, `celulas`, `dx_m`), the lattice Reynolds number actually reached and the
physical Reynolds number the film stands for at `u_inf`, the step count, the render settings, and the field the
volume was built from. A film without that record is a picture of nothing.

## On this machine

The solver is the library's `WindTunnelLBM` (D3Q19, Smagorinsky LES) on PyTorch; `--device mps` is about twice
`cpu` on an Apple M5. The render is EEVEE with OpenVDB volumes: on the same machine Cycles takes tens of minutes
per 720p frame with a volume in the scene, and EEVEE takes seconds — the difference between a film per evening
and a film per week.
