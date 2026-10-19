# Physics Validation Lab -- build from the repository root:
#   docker build -f apps/validation_lab/Dockerfile -t validation-lab .
#   docker run -p 8107:8107 validation-lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
# One worker: torch is loaded for zoo models; VAL_MAX_HEAVY validations run at once in that worker (1 GB default).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/validation_lab/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/validation_lab
WORKDIR /srv/apps/validation_lab
EXPOSE 8107
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8107/health')"
CMD ["uvicorn", "validation_lab.api:app", "--host", "0.0.0.0", "--port", "8107", "--workers", "1"]
