# Simulation Preflight -- build from the repository root:
#   docker build -f apps/simulation_preflight/Dockerfile -t simulation-preflight .
#   docker run -p 8087:8087 simulation-preflight
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/simulation_preflight/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/simulation_preflight
WORKDIR /srv/apps/simulation_preflight
EXPOSE 8087
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8087/health')"
CMD ["uvicorn", "simulation_preflight.api:app", "--host", "0.0.0.0", "--port", "8087", "--workers", "2"]
