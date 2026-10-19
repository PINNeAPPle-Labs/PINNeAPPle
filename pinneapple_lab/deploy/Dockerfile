# PINNeAPPle Lab catalogue -- build from the repository root:
#   docker build -f pinneapple_lab/deploy/Dockerfile -t pinneapple-lab .
#   docker run -p 8093:8093 -v $PWD/lab:/data/lab pinneapple-lab
# The lab folder (runs, figures, dataset shards) is a volume: sweeps run elsewhere (or in this container with
# `docker exec ... python -m pinneapple_lab sweep ...`) and the page picks the new runs up on reload.
FROM python:3.11-slim

# PINNEAPPLE_REPO: the checkout copied below; /experiments reads the example manifests and their media from it
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PINNEAPPLE_LAB=/data/lab PINNEAPPLE_REPO=/srv
WORKDIR /srv

# The server needs only numpy and Pillow; PyTorch (CPU wheel) is installed so the built-in experiments also run here.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN pip install --no-cache-dir . pillow matplotlib scipy numba \
 && mkdir -p /data && cp -r lab /data/lab

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

EXPOSE 8093
VOLUME ["/data/lab"]
HEALTHCHECK --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8093/health')"
CMD ["python", "-m", "pinneapple_lab", "serve", "--host", "0.0.0.0", "--port", "8093"]
