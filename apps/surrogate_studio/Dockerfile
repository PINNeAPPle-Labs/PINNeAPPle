# Surrogate Studio -- build from the repository root:
#   docker build -f apps/surrogate_studio/Dockerfile -t surrogate-studio .
#   docker run -p 8097:8097 -v surrogate_data:/data surrogate-studio
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir ".[surrogate]" -r apps/surrogate_studio/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/surrogate_studio:/srv/apps/_shared
WORKDIR /srv/apps/surrogate_studio
EXPOSE 8097
VOLUME ["/data"]
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8097/health')"
# one worker: trainings run in the job queue (SUR_MAX_JOBS at once), and a 1 GB container holds one torch process
CMD ["uvicorn", "surrogate_studio.api:app", "--host", "0.0.0.0", "--port", "8097", "--workers", "1"]
