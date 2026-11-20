# Rendering without a display

PyVista and VTK open an OpenGL context even for an off-screen render. On a workstation there is a window server
to open one against; on a server, in a container or in CI there is not, and the render fails — sometimes with an
exception, sometimes by aborting the interpreter. This page is the recipe, and the command that tells you which
part of it your machine is missing.

## Ask the machine first

```console
$ python -m pinneapple_tools.visualization.headless
python     3.12.6 on Linux-6.8.0-45-generic-x86_64-with-glibc2.39
pyvista    0.49.2
vtk        9.7.2
display    no
xvfb-run   not installed
libOSMesa  not found
libEGL     /usr/lib/x86_64-linux-gnu/libEGL.so.1

recipe     egl: no usable display and no xvfb, but libEGL is installed: VTK can open a context without a window server
render     failed: RuntimeError: ...

to fix it:
  apt-get install -y libegl1 libgl1 libglvnd0
  PYVISTA_OFF_SCREEN=true python -m pinneapple_tools.visualization.headless
```

The same report as a dict is `pp.viz.headless()`: `can_render` (`True`, `False` or `None` when nothing was
attempted), `error`, `recipe`, `why`, `commands` and `environment`. The render is attempted in a subprocess, and a
frame of one flat colour counts as a failure — a context that opens on a window nothing was drawn into is the
usual symptom of a half-configured Mesa, and it would otherwise look like a success.

**macOS and Windows need none of this.** A render with no `DISPLAY` is normal there, because the window server is
always running; the report says `recipe: ready` and the frame comes back. Everything below is Linux.

If the report says `recipe: install`, rendering is not the problem — pyvista is not installed, and the command it
prints is the one that installs it:

```console
  pip install "pinneapple[pyvista]"
```

## The three recipes, shortest first

**1. A virtual display (`xvfb`).** The least invasive: the standard VTK wheel keeps working, `xvfb-run` gives it an
X display of its own, and Mesa's software GL (`libgl1-mesa-dri`) draws into it.

```console
  apt-get install -y xvfb
  xvfb-run -a python -m pinneapple_tools.visualization.headless
```

Prefix any script with `xvfb-run -a`, or set `DISPLAY` from a running `Xvfb :99` if you would rather start it once.

**2. OSMesa.** VTK renders into a memory buffer with no window at all. It needs the OSMesa build of VTK, which on
Linux is a separate wheel:

```console
  pip uninstall -y vtk
  pip install vtk-osmesa
  PYVISTA_OFF_SCREEN=true python -m pinneapple_tools.visualization.headless
```

Installing `vtk` again later (it is a dependency of `pyvista`) puts the standard wheel back and undoes this; the
report makes that visible, because `recipe` goes back to `xvfb` or `docker`.

**3. EGL.** VTK 9.2 and later can open a context through EGL, which needs no X server and no vtk-osmesa:

```console
  apt-get install -y libegl1 libgl1 libglvnd0
  PYVISTA_OFF_SCREEN=true python -m pinneapple_tools.visualization.headless
```

EGL is the least predictable of the three — whether the context opens depends on the driver, and a container
with no GPU falls back to Mesa's `libgl1-mesa-dri`. Run the check rather than assuming it worked.

## The container

[`docker/headless/Dockerfile`](../../docker/headless/Dockerfile) is the recipe with nothing left to decide: xvfb,
Mesa's software GL, ffmpeg (for `pp.viz.turntable`'s MP4) and the package with its rendering extras. Build it from
the repository root, because it installs the package from the build context:

```console
  docker build -t pinneapple-headless -f docker/headless/Dockerfile .
  docker run --rm -v "$PWD:/work" -w /work pinneapple-headless
```

The `ENTRYPOINT` is `xvfb-run -a`, so a command replaces the default one and still runs under the virtual display:
`docker run --rm -v "$PWD:/work" -w /work pinneapple-headless python your_render.py`. With no command it runs the
check above, which is how you confirm the image before blaming a case for failing in it.

The image pulls torch as part of the package, which is most of its size. An image that could only render would be
smaller and a different thing: it could not run the case it is given.

## In CI

The GitHub runner has no display and no GPU. Either use the container above, or add the two lines that make the
runner behave like a machine with one:

```yaml
- run: sudo apt-get update && sudo apt-get install -y xvfb libgl1-mesa-dri
- run: xvfb-run -a python -m pytest tests/test_headless_rendering.py
```

Whatever the runner does, gate the job on the check: a rendering test that silently skips everywhere is a test
that says nothing. `tests/test_headless_rendering.py` is written that way — it runs the probe where a context can
be opened and asserts, and where it cannot, it asserts the report names the right recipe instead of quietly
passing.
