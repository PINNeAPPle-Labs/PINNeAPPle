# Mesh Quality -- build from the repository root:
#   docker build -f apps/mesh_quality/Dockerfile -t mesh-quality .
#   docker run -p 8088:8088 mesh-quality
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/mesh_quality/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/mesh_quality
WORKDIR /srv/apps/mesh_quality
EXPOSE 8088
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8088/health')"
CMD ["uvicorn", "mesh_quality.api:app", "--host", "0.0.0.0", "--port", "8088", "--workers", "2"]
