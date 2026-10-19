# PINNeAPPle Playground: a static site (the models run in the browser). Build from the repository root:
#   docker build -f apps/playground/Dockerfile -t pinneapple-playground .
#   docker run -p 8095:8095 pinneapple-playground
# Stage 1 assembles the site with the standard library only; stage 2 serves it with Caddy.
FROM python:3.11-slim AS build
WORKDIR /src
COPY apps/playground apps/playground
COPY pinneapple_hub/zoo/web pinneapple_hub/zoo/web
COPY pinneapple_tools/visualization/studio/web/vendor/three pinneapple_tools/visualization/studio/web/vendor/three
COPY data/geometries data/geometries
COPY data/web data/web
RUN python apps/playground/build.py --out /site

FROM caddy:2-alpine@sha256:d8542f48d34a9cf4e4c11a478865229840e87e4c96ea3f439101f31a5d35f75f
COPY apps/playground/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /site /srv
USER 10001:0
EXPOSE 8095
HEALTHCHECK --start-period=10s CMD wget -q -O /dev/null http://localhost:8095/index.html || exit 1
