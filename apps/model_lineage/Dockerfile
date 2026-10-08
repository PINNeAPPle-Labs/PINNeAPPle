# Engineering Model Lineage -- build from the repository root:
#   docker build -f apps/model_lineage/Dockerfile -t model-lineage .
#   docker run -p 8090:8090 model-lineage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/model_lineage/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/model_lineage
WORKDIR /srv/apps/model_lineage
EXPOSE 8090
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')"
CMD ["uvicorn", "model_lineage.api:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "2"]
