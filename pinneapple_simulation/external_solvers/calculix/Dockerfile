# CalculiX CrunchiX (ccx) from the official Debian package: reproducible, no third-party image.
FROM debian:bookworm-slim
RUN apt-get -o Acquire::Retries=5 update \
 && apt-get -o Acquire::Retries=5 install -y --no-install-recommends calculix-ccx \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /work
ENTRYPOINT ["ccx"]
