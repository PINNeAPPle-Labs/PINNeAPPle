# Simulation Workflow Runner -- build from the repository root:
#   docker build -f apps/simulation_runner/Dockerfile -t simulation-runner .
#   docker run -p 8110:8110 -v runner_data:/data simulation-runner
# The public image runs the built-in templates only (RUNNER_SELF_HOSTED unset), as an unprivileged user.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/simulation_runner/requirements.txt \
    && useradd --create-home --uid 10001 runner && mkdir -p /data && chown runner /data

ENV PYTHONPATH=/srv:/srv/apps/simulation_runner RUNNER_DATA_DIR=/data/runner
USER runner
WORKDIR /srv/apps/simulation_runner
EXPOSE 8110
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8110/health')"
CMD ["uvicorn", "simulation_runner.api:app", "--host", "0.0.0.0", "--port", "8110", "--workers", "2"]
