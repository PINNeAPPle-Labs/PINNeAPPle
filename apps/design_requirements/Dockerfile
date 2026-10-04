# Form Compiler (ASME U-DR-1 built in) -- build from the repository root:
#   docker build -f apps/design_requirements/Dockerfile -t design-requirements .
#   docker run -p 8085:8085 design-requirements
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/design_requirements/requirements.txt

ENV PYTHONPATH=/srv:/srv/apps/design_requirements
WORKDIR /srv/apps/design_requirements
EXPOSE 8085
HEALTHCHECK --start-period=90s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8085/health')"
CMD ["uvicorn", "design_requirements.api:app", "--host", "0.0.0.0", "--port", "8085", "--workers", "2"]
