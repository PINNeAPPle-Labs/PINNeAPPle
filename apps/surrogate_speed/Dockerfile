# Surrogate Speed -- build from the repository root:
#   docker build -f apps/surrogate_speed/Dockerfile -t surrogate-speed .
#   docker run -p 8165:8165 surrogate-speed
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/surrogate_speed/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/surrogate_speed:/srv/apps/_shared
WORKDIR /srv/apps/surrogate_speed
EXPOSE 8165
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8165/health')"
# the surrogate and the sweep live in this one worker
CMD ["uvicorn", "surrogate_speed.api:app", "--host", "0.0.0.0", "--port", "8165", "--workers", "1"]
