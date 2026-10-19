# Parameter Identifiability Lab -- build from the repository root:
#   docker build -f apps/identifiability_lab/Dockerfile -t identifiability-lab .
#   docker run -p 8108:8108 -v ident_data:/data identifiability-lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/identifiability_lab/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/identifiability_lab:/srv/apps/_shared
WORKDIR /srv/apps/identifiability_lab
EXPOSE 8108
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8108/health')"
# Jobs, usage and API keys live in SQLite under /data (mount a volume); the queue is safe with several workers.
CMD ["uvicorn", "identifiability_lab.api:app", "--host", "0.0.0.0", "--port", "8108", "--workers", "2"]
