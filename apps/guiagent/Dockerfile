# Engineering GUI Agent -- a separate, optional heavy image (FreeCAD + CfdOF + OpenFOAM + ParaView + Xvfb).
# Build from the repository root:
#   docker build -f apps/guiagent/Dockerfile -t guiagent .
#   docker run -p 8115:8115 guiagent
ARG FREECAD_VERSION=1.0.2
ARG CFDOF_VERSION=v1.37.3

# ---------------------------------------------------------------- FreeCAD (AppImage, extracted) and CfdOF
FROM ubuntu:24.04 AS apps
ARG FREECAD_VERSION
ARG CFDOF_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN curl -fsSL -o FreeCAD.AppImage \
      "https://github.com/FreeCAD/FreeCAD/releases/download/${FREECAD_VERSION}/FreeCAD_${FREECAD_VERSION}-conda-Linux-x86_64-py311.AppImage" \
    && chmod +x FreeCAD.AppImage && ./FreeCAD.AppImage --appimage-extract >/dev/null \
    && mv squashfs-root freecad && rm FreeCAD.AppImage \
    # parts the agent never loads (AI inference, pythonocc, IFC/BIM): about 600 MB
    && rm -rf freecad/usr/lib/openvino-* freecad/usr/lib/python3.11/site-packages/OCC \
              freecad/usr/lib/python3.11/site-packages/ifcopenshell
RUN git clone -q --depth 1 --branch "${CFDOF_VERSION}" https://github.com/jaheyns/CfdOF /opt/freecad-mods/CfdOF \
    && rm -rf /opt/freecad-mods/CfdOF/.git /opt/freecad-mods/CfdOF/Demos

# ---------------------------------------------------------------- runtime
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    LANG=C.UTF-8 QT_QPA_PLATFORM=xcb XDG_RUNTIME_DIR=/tmp/runtime-agent
# OpenFOAM as the repository's other OpenFOAM work installs it (Ubuntu package, v1912); ParaView with pvpython;
# Xvfb + xdotool for the GUI steps; the X/GL libraries FreeCAD's Qt needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
      openfoam paraview python3-paraview xvfb xdotool \
      python3 python3-venv ca-certificates fontconfig fonts-dejavu-core \
      libgl1 libglx-mesa0 libegl1 libglu1-mesa libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
      libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libxcb-cursor0 libxrender1 \
      libxi6 libsm6 libice6 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=apps /opt/freecad /opt/freecad
COPY --from=apps /opt/freecad-mods /opt/freecad-mods

# The app's own Python, separate from the system Python that pvpython uses. PINNeAPPle depends on PyTorch (imported
# by pinneapple_tools); the CPU wheel keeps the image small.
RUN python3 -m venv /opt/app \
    && /opt/app/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /srv
COPY . .
RUN /opt/app/bin/pip install --no-cache-dir . -r apps/guiagent/requirements.txt \
    && useradd -m -u 10001 agent \
    && mkdir -p /home/agent/.local/share/FreeCAD/Mod /data "$XDG_RUNTIME_DIR" \
    && ln -s /opt/freecad-mods/CfdOF /home/agent/.local/share/FreeCAD/Mod/CfdOF \
    && chown -R agent:agent /home/agent /data "$XDG_RUNTIME_DIR"

# runs, job queue, API keys and usage live in /data/guiagent (apps/_shared runtime)
ENV PATH=/opt/app/bin:$PATH PYTHONPATH=/srv:/srv/apps/guiagent \
    PINNEAPPLE_FREECAD=/opt/freecad/usr/bin/freecad PINNEAPPLE_CFDOF=/opt/freecad-mods/CfdOF \
    FOAM_BASHRC=/usr/share/openfoam/etc/bashrc PINNEAPPLE_PVPYTHON=/usr/bin/pvpython \
    PINNEAPPLE_APP_DATA=/data
USER agent
WORKDIR /srv/apps/guiagent
VOLUME /data
EXPOSE 8115
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8115/health')"
# one worker: runs are tracked in this process (GUIAGENT_MAX_JOBS, default 1)
CMD ["uvicorn", "guiagent.api:app", "--host", "0.0.0.0", "--port", "8115", "--workers", "1"]
