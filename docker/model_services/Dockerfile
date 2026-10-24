# PINNeAPPle Model Services: zoo surrogates, physics checks and the intent extractor behind NIM-compatible routes
# (GET /v1/health/ready, GET /v1/models, POST /v1/infer, POST /v1/chat/completions). CPU only.
#   docker build -f docker/model_services/Dockerfile -t pinneapple-model-services .
#   docker run -p 8000:8000 pinneapple-model-services
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . /app
RUN pip install --no-cache-dir /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if b'ready' in urllib.request.urlopen('http://127.0.0.1:8000/v1/health/ready').read() else 1)"
CMD ["python", "-m", "pinneapple_tools.model_services", "serve", "--host", "0.0.0.0", "--port", "8000"]
