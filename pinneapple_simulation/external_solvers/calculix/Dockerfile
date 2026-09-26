# CalculiX CrunchiX (ccx) from the official Debian package: reproducible, no third-party image.
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends calculix-ccx \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
ENTRYPOINT ["ccx"]
