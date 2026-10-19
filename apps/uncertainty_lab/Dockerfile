# Engineering Uncertainty Lab -- build from the repository root:
#   docker build -f apps/uncertainty_lab/Dockerfile -t uncertainty-lab .
#   docker run -p 8098:8098 uncertainty-lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/uncertainty_lab/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/uncertainty_lab
WORKDIR /srv/apps/uncertainty_lab
EXPOSE 8098
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8098/health')"
# One worker: studies run in request threads, at most UQL_MAX_HEAVY at a time.
CMD ["uvicorn", "uncertainty_lab.api:app", "--host", "0.0.0.0", "--port", "8098", "--workers", "1"]
