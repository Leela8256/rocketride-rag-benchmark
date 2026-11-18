# The RocketRide engine exactly as released. Nothing inside the release is changed.
#
# The engine ships for linux-x64, so every image of this benchmark is linux/amd64.
FROM --platform=linux/amd64 python:3.12-slim-bookworm

# libc++ is a runtime dependency of the engine binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libc++1 libc++abi1 ca-certificates curl procps \
    && rm -rf /var/lib/apt/lists/*

# The release under test. The archive hash is checked, so a different archive stops the build.
# To test another release, pass all three values:
#   docker build --build-arg ENGINE_VERSION=... --build-arg ENGINE_URL=... --build-arg ENGINE_ARCHIVE_SHA256=...
ARG ENGINE_VERSION=3.6.0
ARG ENGINE_URL=https://github.com/rocketride-org/rocketride-server/releases/download/server-v3.6.0/rocketride-server-v3.6.0-linux-x64.tar.gz
ARG ENGINE_ARCHIVE_SHA256=621a52106bb91ab5500992ee59893919b4496f41b00cbe0f578b6f65b4639c1b

WORKDIR /opt/rocketride/engine
RUN curl -fsSL --retry 3 "$ENGINE_URL" -o /tmp/engine.tgz \
    && echo "${ENGINE_ARCHIVE_SHA256}  /tmp/engine.tgz" | sha256sum -c - \
    && tar -xzf /tmp/engine.tgz -C . \
    && rm /tmp/engine.tgz \
    && test -x ./engine && test -f ai/eaas.py \
    && sha256sum engine ai/common/models/transformers/sentence_transformers.py nodes/store_postgres/postgres.py \
       > /opt/rocketride/release-files.sha256

LABEL benchmark.rocketride.engine_version="${ENGINE_VERSION}" \
      benchmark.rocketride.engine_url="${ENGINE_URL}" \
      benchmark.rocketride.archive_sha256="${ENGINE_ARCHIVE_SHA256}"

# The key the benchmark client connects with. The engine listens on 127.0.0.1 only.
ENV ROCKETRIDE_APIKEY=local-dev \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["./engine"]
CMD ["ai/eaas.py", "--host=127.0.0.1", "--port=5565"]
