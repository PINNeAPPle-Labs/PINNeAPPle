# PINNeAPPle Observatory -- build from the repository root:
#   docker build -f apps/observatory/Dockerfile -t observatory .
#   docker run -p 8094:8094 observatory
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/observatory/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data /srv/cache && chown -R 10001:0 /data /srv/cache && chmod -R g+rwX /data /srv/cache
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/observatory OBS_CACHE_DIR=/srv/cache/tle
WORKDIR /srv/apps/observatory
EXPOSE 8094
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8094/health')"
# one worker: the PINN / lensing caches and the validation table live in process memory
CMD ["uvicorn", "observatory.api:app", "--host", "0.0.0.0", "--port", "8094", "--workers", "1"]
