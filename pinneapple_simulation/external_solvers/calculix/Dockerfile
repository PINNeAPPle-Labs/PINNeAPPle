# CalculiX CrunchiX (ccx) from the official Debian package: reproducible, no third-party image.
# The official Debian image is pulled from the AWS mirror of the Docker library: anonymous pulls from Docker Hub are
# rate limited (429 Too Many Requests on shared CI runners). Override with --build-arg BASE=debian:bookworm-slim.
ARG BASE=public.ecr.aws/docker/library/debian:bookworm-slim
FROM ${BASE}
RUN apt-get -o Acquire::Retries=5 update \
 && apt-get -o Acquire::Retries=5 install -y --no-install-recommends calculix-ccx \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
ENTRYPOINT ["ccx"]
