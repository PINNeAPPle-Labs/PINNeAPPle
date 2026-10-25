# PINNeAPPle Hub -- build from the repository root:
#   docker build -f apps/hub/Dockerfile -t pinneapple-hub .
#   docker run -p 8116:8116 -v hub_data:/data pinneapple-hub
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# The Hub serves what its index points to: data/web, data/geometries, lab/runs, examples, docs/org/reference/figures,
# docs/assets (apps/hub/Dockerfile.dockerignore keeps them in the build context)
COPY . .
RUN pip install --no-cache-dir . -r apps/hub/requirements.txt && mkdir -p /data
# The index is not committed: build it from the sources and the recorded runs (data/web/hub/snippet_runs)
RUN python -m pinneapple_hub.library build

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib PINNEAPPLE_LAB=/srv/lab
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/_shared:/srv/apps/hub
WORKDIR /srv/apps/hub
EXPOSE 8116
VOLUME ["/data"]
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8116/health')"
# Validation runs are jobs (HUB_MAX_JOBS at once) in their own interpreters with time and memory limits.
CMD ["uvicorn", "hub.api:app", "--host", "0.0.0.0", "--port", "8116", "--workers", "1"]
