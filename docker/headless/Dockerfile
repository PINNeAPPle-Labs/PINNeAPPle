# Rendering without a display (#66).
#
# Build from the repository root, because the image installs the package from the context:
#     docker build -t pinneapple-headless -f docker/headless/Dockerfile .
# Then, with your case in the current directory:
#     docker run --rm -v "$PWD:/work" -w /work pinneapple-headless python your_render.py
#
# Run with no arguments to check the image itself: `python -m pinneapple_tools.visualization.headless` renders a
# 320x240 sphere and prints the diagnosis, exiting 1 when it cannot.

FROM python:3.12-slim-bookworm

# xvfb gives the render a virtual X display, and Mesa's llvmpipe (libgl1-mesa-dri) is the software GL behind it:
# no GPU, no X server on the host, nothing to configure at run time. libosmesa6 is along for the case where a
# caller would rather have OSMesa than the virtual display; ffmpeg is what the MP4 export pipes into
# (pp.viz.turntable).
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        libgl1 \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libegl-mesa0 \
        libosmesa6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYVISTA_OFF_SCREEN=true \
    PYTHONUNBUFFERED=1

COPY . /src
# The package with its rendering extras. This pulls torch, which is most of the image: an image that can only
# render would be smaller and a different thing -- it could not run the case it is given.
RUN pip install --no-cache-dir "/src[pyvista]" && rm -rf /src

WORKDIR /work
# Entry point rather than a prefix in every command: `docker run image python your_render.py` runs the render
# under xvfb-run without the caller having to remember it.
ENTRYPOINT ["xvfb-run", "-a"]
CMD ["python", "-m", "pinneapple_tools.visualization.headless"]
