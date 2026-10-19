# Physics Anomaly Detector -- build from the repository root:
#   docker build -f apps/anomaly_detector/Dockerfile -t anomaly-detector .
#   docker run -p 8111:8111 -v anomaly_data:/data anomaly-detector
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/anomaly_detector/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/anomaly_detector:/srv/apps/_shared
WORKDIR /srv/apps/anomaly_detector
EXPOSE 8111
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8111/health')"
# One worker within the 1 GB default (torch is imported); heavy work runs in the shared job queue.
CMD ["uvicorn", "anomaly_detector.api:app", "--host", "0.0.0.0", "--port", "8111", "--workers", "1"]
