# BrakeCool AI -- build from the repository root:
#   docker build -f apps/brakecool_ai/Dockerfile -t brakecool-ai .
#   docker run -p 8161:8161 brakecool-ai
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small (the surrogate is a small MLP).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/brakecool_ai/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/brakecool_ai:/srv/apps/_shared
WORKDIR /srv/apps/brakecool_ai
EXPOSE 8161
# /health answers at once; the surrogate trains in the background (a few seconds), /api/optimize answers 503 until then
HEALTHCHECK --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8161/health')"
# one worker: the surrogate lives in the worker's memory and trains once at start-up
CMD ["uvicorn", "brakecool_ai.api:app", "--host", "0.0.0.0", "--port", "8161", "--workers", "1"]
