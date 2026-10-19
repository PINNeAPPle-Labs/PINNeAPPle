# Simulation Campaign Planner -- build from the repository root:
#   docker build -f apps/simulation_campaign/Dockerfile -t simulation-campaign .
#   docker run -p 8096:8096 -v campaign_data:/data simulation-campaign
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/simulation_campaign/requirements.txt

# jobs, API keys, usage and uploads (apps/_shared runtime); mount a volume here
RUN mkdir -p /data

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0
VOLUME /data

ENV PYTHONPATH=/srv:/srv/apps/simulation_campaign
WORKDIR /srv/apps/simulation_campaign
EXPOSE 8096
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8096/health')"
CMD ["uvicorn", "simulation_campaign.api:app", "--host", "0.0.0.0", "--port", "8096", "--workers", "2"]
