# Simulation Comparator -- build from the repository root:
#   docker build -f apps/simulation_comparator/Dockerfile -t simulation-comparator .
#   docker run -p 8089:8089 simulation-comparator
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/simulation_comparator/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/simulation_comparator
WORKDIR /srv/apps/simulation_comparator
EXPOSE 8089
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8089/health')"
CMD ["uvicorn", "simulation_comparator.api:app", "--host", "0.0.0.0", "--port", "8089", "--workers", "2"]
