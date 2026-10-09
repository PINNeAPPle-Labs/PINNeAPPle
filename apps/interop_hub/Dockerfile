# Interoperability Hub -- build from the repository root:
#   docker build -f apps/interop_hub/Dockerfile -t interop-hub .
#   docker run -p 8091:8091 interop-hub
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/interop_hub/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/interop_hub
WORKDIR /srv/apps/interop_hub
EXPOSE 8091
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8091/health')"
CMD ["uvicorn", "interop_hub.api:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "2"]
