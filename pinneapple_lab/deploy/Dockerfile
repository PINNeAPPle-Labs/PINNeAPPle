# PINNeAPPle Lab catalogue -- build from the repository root:
#   docker build -f pinneapple_lab/deploy/Dockerfile -t pinneapple-lab .
#   docker run -p 8093:8093 -v $PWD/lab:/data/lab pinneapple-lab
# The lab folder (runs, figures, dataset shards) is a volume: sweeps run elsewhere (or in this container with
# `docker exec ... python -m pinneapple_lab sweep ...`) and the page picks the new runs up on reload.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PINNEAPPLE_LAB=/data/lab
WORKDIR /srv

# The server needs only numpy and Pillow; PyTorch (CPU wheel) is installed so the built-in experiments also run here.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN pip install --no-cache-dir . pillow matplotlib scipy numba \
 && mkdir -p /data && cp -r lab /data/lab

EXPOSE 8093
VOLUME ["/data/lab"]
HEALTHCHECK --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8093/health')"
CMD ["python", "-m", "pinneapple_lab", "serve", "--host", "0.0.0.0", "--port", "8093"]
