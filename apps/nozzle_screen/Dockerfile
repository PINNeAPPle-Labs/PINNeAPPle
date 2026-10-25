# NozzleScreen -- build from the repository root:
#   docker build -f apps/nozzle_screen/Dockerfile -t nozzle-screen .
#   docker run -p 8146:8146 -v nozzle_screen_data:/data nozzle-screen
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/nozzle_screen/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/nozzle_screen:/srv/apps/_shared
WORKDIR /srv/apps/nozzle_screen
EXPOSE 8146
VOLUME ["/data"]
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8146/health')"
CMD ["uvicorn", "nozzle_screen.api:app", "--host", "0.0.0.0", "--port", "8146", "--workers", "1"]
