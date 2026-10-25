# PINNeAPPle Arena -- build from the repository root:
#   docker build -f apps/pinn_arena/Dockerfile -t pinn-arena .
#   docker run -p 8130:8130 -v pinnarena_data:/data pinn-arena
# Stage 1 builds the React frontend (apps/pinn_arena/frontend, npm ci from package-lock.json); stage 2 serves it
# from the FastAPI app (pinn_arena/static) next to the API.
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY apps/pinn_arena/frontend/package.json apps/pinn_arena/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY apps/pinn_arena/frontend/ ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
WORKDIR /srv

# PINNeAPPle depends on PyTorch; the CPU wheel keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir . -r apps/pinn_arena/requirements.txt
COPY --from=frontend /fe/dist /srv/apps/pinn_arena/pinn_arena/static

# Unprivileged runtime user: /data (the volume) and /tmp are the only places it writes (apps/deploy/README.md)
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib
USER 10001:0

ENV PYTHONPATH=/srv:/srv/apps/pinn_arena:/srv/apps/_shared
WORKDIR /srv/apps/pinn_arena
EXPOSE 8130
VOLUME ["/data"]
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8130/health')"
# one worker: experiments train in this process (PINN_ARENA_MAX_RUNNING at once) and the external-catalog job store
# is in memory
CMD ["uvicorn", "pinn_arena.api:app", "--host", "0.0.0.0", "--port", "8130", "--workers", "1"]
