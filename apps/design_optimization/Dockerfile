# Design Optimization -- build from the repository root:
#   docker build -f apps/design_optimization/Dockerfile -t design-optimization .
#   docker run -p 8181:8181 design-optimization
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/design_optimization/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/design_optimization:/srv/apps/_shared
WORKDIR /srv/apps/design_optimization
EXPOSE 8181
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8181/health')"
# each request trains a small surrogate; one torch process
CMD ["uvicorn", "design_optimization.api:app", "--host", "0.0.0.0", "--port", "8181", "--workers", "1"]
