# Optics Lab

**The aerial image of a lithographic clip, in 3-D through focus, with the closed-form checks beside every number.**

Type an optical system — wavelength, numerical aperture, partial coherence, illumination shape — and the app images
the bundled clip (three 64 nm lines at 192 nm pitch plus an elbow) with the library's Abbe sum, over the pupil and the
source it declares. It reports the printed CD against the mask CD, the NILS, the resolution limit λ/(NA(1 + σ)), the
implied k₁, and the process window (depth of focus and exposure latitude) from an EPE map over focus × dose. The
same image is served at five defocus planes as a 3-D view: the same aerial image, seen at five depths.

![Optics Lab](../../docs/assets/apps/optics-lab.png)

## What it shows, and what it checks

| on screen | where it comes from |
|---|---|
| CD impressa, viés contra a máscara | `pinneapple_physics.lithography.metrics.cd_along` on the nominal-focus image |
| NILS | `…metrics.nils` (CD × log-slope at the edge) |
| meia-passo mínimo e k₁ | `…imaging.resolution_limit` = λ/(NA(1+σ)) |
| profundidade de foco, latitude de exposição, EPE | `…metrics.process_window` |
| campos em 3-D | the aerial image at each defocus, as a glTF field view (`pinneapple_tools.visualization.studio`) |

The three checks the page repeats at the numbers it shows are the ones the library's own tests make:

1. **Clear-mask gain**: a fully clear mask has to image to exactly 1.0 (measured: 2e-15).
2. **Modulation around the limit**: at 0.95× the resolution limit there is no modulation; at 1.10× there is.
3. **Abbe vs Hopkins**: the Abbe sum over source points and the SOCS kernels of the same system agree (measured:
   ~1e-14).

## Run

```bash
pip install -e . -r apps/optics_lab/requirements.txt
cd apps/optics_lab && PYTHONPATH=../..:../_shared uvicorn optics_lab.api:app --port 8166
# or, from the repository root:
docker build -f apps/optics_lab/Dockerfile -t optics-lab . && docker run -p 8166:8166 optics-lab
```

`POST /api/image` answers in about half a second at the 128² grid (121 source points, five defocus planes and the
25-point process window included).

## Limits, stated in the app

- **Scalar imaging, no polarisation**: the pupil is a scalar low-pass. Vector imaging (high-NA polarisation effects)
  is not modelled.
- **A perfect projection lens**: no aberration of its own beyond the defocus of the focus list; no flare, no
  mask-3-D (the mask is a thin transmission function).
- **The resist is a threshold**: `printed` uses a constant threshold, optionally blurred by `diffusion`; no chemically
  amplified resist model, no development.
- **The clip is a test clip**, not a product layer: three lines at 192 nm pitch and an elbow on a 1024 nm field.
- The defocus planes in the 3-D view are **spaced for the drawing** (0.35 µm apart); the physical range is the focus
  list, −60 to +60 nm.
