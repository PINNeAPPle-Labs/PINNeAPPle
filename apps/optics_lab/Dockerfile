# Optics Lab -- build from the repository root:
#   docker build -f apps/optics_lab/Dockerfile -t optics-lab .
#   docker run -p 8166:8166 optics-lab
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# the imaging is a few FFTs per source point; PyTorch CPU is all the library needs here
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/optics_lab/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/optics_lab:/srv/apps/_shared
WORKDIR /srv/apps/optics_lab
EXPOSE 8166
HEALTHCHECK --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8166/health')"
# one worker: the 3-D views live in the worker's memory (a small LRU) and the imaging is CPU-bound per request
CMD ["uvicorn", "optics_lab.api:app", "--host", "0.0.0.0", "--port", "8166", "--workers", "1"]
