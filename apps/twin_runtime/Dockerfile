# Digital Twin Runtime -- build from the repository root:
#   docker build -f apps/twin_runtime/Dockerfile -t twin-runtime .
#   docker run -p 8103:8103 -v twin_data:/data twin-runtime
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/twin_runtime/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/twin_runtime:/srv/apps/_shared
WORKDIR /srv/apps/twin_runtime
# 8103: HTTP. 1883: the embedded MQTT broker, only when TWIN_MQTT_HOST=0.0.0.0 and the port is published.
EXPOSE 8103 1883
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8103/health')"
# One worker: live twins (state, threads, the embedded broker) live in the worker's memory.
CMD ["uvicorn", "twin_runtime.api:app", "--host", "0.0.0.0", "--port", "8103", "--workers", "1"]
