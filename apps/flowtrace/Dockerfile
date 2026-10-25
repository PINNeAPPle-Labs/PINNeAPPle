# FlowTrace -- build from the repository root:
#   docker build -f apps/flowtrace/Dockerfile -t flowtrace .
#   docker run -p 8145:8145 -v flowtrace_data:/data flowtrace
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/flowtrace/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/flowtrace:/srv/apps/_shared
WORKDIR /srv/apps/flowtrace
EXPOSE 8145
VOLUME ["/data"]
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8145/health')"
CMD ["uvicorn", "flowtrace.api:app", "--host", "0.0.0.0", "--port", "8145", "--workers", "1"]
