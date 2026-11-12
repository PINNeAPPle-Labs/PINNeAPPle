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

The look is tuned on the render side only: the stored fields are the two the picture needs (speed and the
Q-criterion, both dimensionless), and the scalar that becomes density is built at render time. Changing the
threshold that decides what reads as smoke does not cost another solve.

## What is on screen

- **The smoke** is the Q-criterion, `Q = (|Ω|² − |S|²)/2`, cut above a high percentile of the fluid. Vorticity
  magnitude will not do here: the road under a car is a moving wall, and its boundary layer carries more
  vorticity than the wake does, so a picture built on `|ω|` paints the floor and leaves the wake empty. Q is
  zero for pure shear and positive only where rotation beats strain, which is the structure the picture wants.
- **The volume is tapered into the walls.** The field is a box and a box has edges: where the wake still has
  density at the boundary the render shows a straight cut, and that cut reads as a box rather than as air. A
  raised cosine over the outer 14 % of each axis takes the field to zero before it reaches the wall.
- **The grid is refined before it is written.** The stored field is subsampled to keep the files small; a
  trilinear pass to twice the resolution adds no detail, but it removes the stair-step of an 8 cm voxel along a
  curved surface, which is what makes a volume read as blocks instead of smoke.
- **The colour** walks from the tint of the thin wake to white where the field is dense, which is what reads as
  depth in a cloud. There is no colour map: the picture is not a plot, and `ficha.json` carries the numbers.
- **The glow** is an emission node multiplied by the density itself. `Principled Volume`'s own emission is
  emitted through the volume's whole bounding box whatever the density is, which turns a sparse field into a
  solid slab — the reason the shader here is built by hand. For the same reason the texture that breaks up the
  cloud multiplies the field down to zero instead of down to a fraction of itself.
- **The snapshots crossfade on overlapping windows.** Keyed end to end, the outgoing and the incoming grid are
  both at zero on the same frame and the smoke blinks out once per snapshot — a fault that exists only in the
  animation, never in a still.

The caption on the finished film is drawn as a transparent overlay at assembly time, not burned by the
renderer: text rendered inside the scene goes through the camera's focus and its motion blur, and a caption
that blurs with the subject reads as a mistake.

## Recorded with every film

`ficha.json` holds the mesh (`malha`, `celulas`, `dx_m`), the lattice Reynolds number actually reached and the
physical Reynolds number the film stands for at `u_inf`, the step count, the render settings, and the field the
volume was built from. A film without that record is a picture of nothing.

## On this machine

The solver is the library's `WindTunnelLBM` (D3Q19, Smagorinsky LES) on PyTorch; `--device mps` is about twice
`cpu` on an Apple M5. The render is EEVEE with OpenVDB volumes: on the same machine Cycles takes tens of minutes
per 720p frame with a volume in the scene, and EEVEE takes seconds — the difference between a film per evening
and a film per week.
