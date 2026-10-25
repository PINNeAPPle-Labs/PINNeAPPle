# Parameter Discovery -- build from the repository root:
#   docker build -f apps/parameter_discovery/Dockerfile -t parameter-discovery .
#   docker run -p 8163:8163 parameter-discovery
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/parameter_discovery/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/parameter_discovery:/srv/apps/_shared
WORKDIR /srv/apps/parameter_discovery
EXPOSE 8163
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8163/health')"
# the PINN trains in a background thread of this one worker
CMD ["uvicorn", "parameter_discovery.api:app", "--host", "0.0.0.0", "--port", "8163", "--workers", "1"]
