# Engineering Evidence Report -- build from the repository root:
#   docker build -f apps/evidence_report/Dockerfile -t evidence-report .
#   docker run -p 8113:8113 evidence-report
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
# One worker: torch (example study) + matplotlib + reportlab stay well inside the 1 GB default; EVR_MAX_HEAVY
# requests run at once in that worker.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/evidence_report/requirements.txt

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/evidence_report
WORKDIR /srv/apps/evidence_report
EXPOSE 8113
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8113/health')"
CMD ["uvicorn", "evidence_report.api:app", "--host", "0.0.0.0", "--port", "8113", "--workers", "1"]
