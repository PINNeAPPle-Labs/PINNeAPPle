# PCB Hotspot -- build from the repository root:
#   docker build -f apps/pcb_hotspot/Dockerfile -t pcb-hotspot .
#   docker run -p 8081:8081 pcb-hotspot
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# PINNeAPPle itself (see Dockerfile.dockerignore for what is left out)
COPY . .
RUN pip install --no-cache-dir . -r apps/pcb_hotspot/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/pcb_hotspot
WORKDIR /srv/apps/pcb_hotspot
EXPOSE 8081
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8081/health')"
CMD ["uvicorn", "pcb_hotspot.api:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "2"]
