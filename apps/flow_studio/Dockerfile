# PINNeAPPle Flow Studio -- build from the repository root:
#   docker build -f apps/flow_studio/Dockerfile -t flow-studio .
#   docker compose -f apps/flow_studio/docker-compose.yml up            (app; + ollama / postgres with profiles)

# 1. the frontend (React + Vite + Tailwind); the repository also carries a built copy in frontend/dist
FROM node:20-slim AS ui
WORKDIR /ui
COPY apps/flow_studio/frontend/package.json apps/flow_studio/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY apps/flow_studio/frontend/ ./
RUN npm run build

# 2. the library and the API
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /srv
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN pip install --no-cache-dir -e ".[flow,llm]" "psycopg[binary]>=3.1" scikit-learn statsmodels
COPY --from=ui /ui/dist /srv/apps/flow_studio/frontend/dist

# Unprivileged runtime user; /data (the volume) holds the SQLite store, run folders and the Lab
RUN useradd --system --uid 10001 --gid 0 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin app \
    && mkdir -p /data && chown -R 10001:0 /data && chmod -R g+rwX /data
ENV HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib PINNEAPPLE_FLOW_HOME=/data
USER 10001:0
EXPOSE 8765
HEALTHCHECK --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')"
# Inside the container the server listens on 0.0.0.0. Without FLOW_USER / FLOW_PASSWORD_HASH it refuses to start
# unless --insecure-no-auth is given; docker-compose.yml publishes the port on the host's 127.0.0.1 only.
CMD ["python", "-m", "pinneapple_flow", "studio", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
