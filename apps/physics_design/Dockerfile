# Physics Design Optimizer -- build from the repository root:
#   docker build -f apps/physics_design/Dockerfile -t physics-design .
#   docker run -p 8106:8106 -v pdo_data:/data physics-design
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/physics_design/requirements.txt && mkdir -p /data

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

# apps/_shared: appkit (login, job queue routes); the repository root: pinneapple_data.contracts
ENV PYTHONPATH=/srv:/srv/apps/_shared:/srv/apps/physics_design
WORKDIR /srv/apps/physics_design
EXPOSE 8106
VOLUME ["/data"]
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8106/health')"
# One worker is enough: studies run as jobs (PDO_MAX_JOBS at once); the queue is safe with more workers.
CMD ["uvicorn", "physics_design.api:app", "--host", "0.0.0.0", "--port", "8106", "--workers", "1"]
