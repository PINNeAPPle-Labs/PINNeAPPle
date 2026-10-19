# Engineering Design Optimizer -- build from the repository root:
#   docker build -f apps/engineering_optimizer/Dockerfile -t engineering-optimizer .
#   docker run -p 8100:8100 -v edo_data:/data engineering-optimizer
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/engineering_optimizer/requirements.txt && mkdir -p /data

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

# apps/_shared: appkit (login, job queue routes); the repository root: pinneapple_data.contracts
ENV PYTHONPATH=/srv:/srv/apps/_shared:/srv/apps/engineering_optimizer
WORKDIR /srv/apps/engineering_optimizer
EXPOSE 8100
VOLUME ["/data"]
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health')"
# One worker is enough: optimisations run as jobs (EDO_MAX_JOBS at once); the queue is safe with more workers.
CMD ["uvicorn", "engineering_optimizer.api:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
