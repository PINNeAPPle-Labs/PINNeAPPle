# Meshless Method Lab -- build from the repository root:
#   docker build -f apps/meshless_lab/Dockerfile -t meshless-lab .
#   docker run -p 8109:8109 -v meshless_data:/data meshless-lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/meshless_lab/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/meshless_lab:/srv/apps/_shared
WORKDIR /srv/apps/meshless_lab
EXPOSE 8109
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8109/health')"
# Studies run in the shared job queue (MESHLESS_MAX_JOBS at once); state (jobs, uploads, usage) lives in /data.
CMD ["uvicorn", "meshless_lab.api:app", "--host", "0.0.0.0", "--port", "8109", "--workers", "1"]
