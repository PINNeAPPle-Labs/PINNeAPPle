# Inverse Heat Lab -- build from the repository root:
#   docker build -f apps/inverse_heat/Dockerfile -t inverse-heat .
#   docker run -p 8086:8086 inverse-heat
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# PINNeAPPle itself plus the fin_convection_inverse example (its reference solver and full 2D/3D results)
COPY . .
RUN pip install --no-cache-dir . -r apps/inverse_heat/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/inverse_heat
WORKDIR /srv/apps/inverse_heat
EXPOSE 8086
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8086/health')"
# One worker: training jobs live in the worker's memory (IHL_MAX_JOBS run in parallel threads).
CMD ["uvicorn", "inverse_heat.api:app", "--host", "0.0.0.0", "--port", "8086", "--workers", "1"]
