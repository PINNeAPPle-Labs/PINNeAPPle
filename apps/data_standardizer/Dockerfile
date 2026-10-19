# Engineering Data Standardizer -- build from the repository root:
#   docker build -f apps/data_standardizer/Dockerfile -t data-standardizer .
#   docker run -p 8083:8083 data-standardizer
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/data_standardizer/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/data_standardizer
WORKDIR /srv/apps/data_standardizer
EXPOSE 8083
HEALTHCHECK --start-period=90s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8083/health')"
CMD ["uvicorn", "data_standardizer.api:app", "--host", "0.0.0.0", "--port", "8083", "--workers", "2"]
