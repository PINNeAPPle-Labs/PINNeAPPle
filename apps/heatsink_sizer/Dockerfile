# HeatSink Sizer -- build from the repository root:
#   docker build -f apps/heatsink_sizer/Dockerfile -t heatsink-sizer .
#   docker run -p 8080:8080 heatsink-sizer
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# CPU-only PyTorch keeps the image small; the surrogates are tiny MLPs.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# PINNeAPPle itself (see Dockerfile.dockerignore for what is left out)
COPY . .
RUN pip install --no-cache-dir . -r apps/heatsink_sizer/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/heatsink_sizer
WORKDIR /srv/apps/heatsink_sizer
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "heatsink_sizer.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
