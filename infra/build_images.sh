#!/usr/bin/env bash
# Build every image of the benchmark. Run on the benchmark machine, from anywhere:
#   bash infra/build_images.sh
#
#   pgvector/pgvector:pg17        the database (pulled, not built)
#   ragbench-rocketride:base      the RocketRide engine release, unchanged
#   ragbench-client:latest        the load client with the RocketRide client SDK
#   ragbench-rocketride:warm      base after one warm-up run (the engine has installed the packages of the nodes
#                                 the benchmark uses and downloaded the embedding model)
#   ragbench-rocketride:latest    warm plus the two packages the engine's postgres node asks for
#   ragbench-parser:latest        RocketRide's parser jars behind a small HTTP server, for the LangGraph side
#   ragbench-langgraph:latest     the LangGraph service
#
# Needs Docker and network access. Nothing else may be listening on port 5565 while this runs.
#
# Another engine release: set all three of ENGINE_VERSION, ENGINE_URL and ENGINE_ARCHIVE_SHA256 in the environment.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
PORT=5565
mkdir -p build-records
trap 'docker rm -f ragbench-rr-warm >/dev/null 2>&1 || true' EXIT
ENGINE_ARGS=()
for name in ENGINE_VERSION ENGINE_URL ENGINE_ARCHIVE_SHA256; do
  if [ -n "${!name:-}" ]; then ENGINE_ARGS+=(--build-arg "$name=${!name}"); fi
done

echo "== database image"
docker pull -q pgvector/pgvector:pg17
docker image inspect pgvector/pgvector:pg17 --format '{{index .RepoDigests 0}}' | tee build-records/db-image-digest.txt

echo "== RocketRide engine, as released"
docker build ${ENGINE_ARGS[@]+"${ENGINE_ARGS[@]}"} -t ragbench-rocketride:base rocketride
docker run --rm --entrypoint cat ragbench-rocketride:base /opt/rocketride/release-files.sha256 | tee build-records/rr-release-files.sha256

echo "== load client"
docker build -t ragbench-client:latest -f infra/client.Dockerfile infra

echo "== warm the engine once and keep the result"
docker rm -f ragbench-rr-warm >/dev/null 2>&1 || true
docker run -d --name ragbench-rr-warm --network host ragbench-rocketride:base ai/eaas.py --host=127.0.0.1 --port=$PORT
for i in $(seq 1 180); do curl -fsS --max-time 2 "http://127.0.0.1:$PORT/version" >/dev/null 2>&1 && break; sleep 2; done
curl -fsS --max-time 5 "http://127.0.0.1:$PORT/version" | tee build-records/rr-version.json; echo
docker run --rm --network host \
  -v "$REPO/harness:/harness:ro" -v "$REPO/rocketride/pipelines:/pipelines:ro" \
  -v "$REPO/langgraph/app/pipelines/document_pdf/fixtures:/fixtures:ro" \
  ragbench-client:latest /harness/warm_rocketride.py "$PORT" /fixtures/warmup.pdf
docker stop ragbench-rr-warm >/dev/null
docker commit ragbench-rr-warm ragbench-rocketride:warm >/dev/null
docker rm ragbench-rr-warm >/dev/null

echo "== RocketRide engine with the packages of its postgres node"
docker build -t ragbench-rocketride:latest -f rocketride/Dockerfile.store --build-arg BASE=ragbench-rocketride:warm rocketride
docker run --rm --entrypoint cat ragbench-rocketride:latest /opt/rocketride/freeze.txt > build-records/rr-freeze.txt
docker run --rm --entrypoint sh ragbench-rocketride:latest -c 'diff /opt/rocketride/freeze-before-store.txt /opt/rocketride/freeze.txt || true' | tee build-records/rr-packages-added.txt

echo "== parser for the LangGraph side"
docker build -t ragbench-parser:latest --build-arg ENGINE_IMAGE=ragbench-rocketride:base langgraph/parser

echo "== LangGraph service"
docker build -t ragbench-langgraph:latest langgraph
docker run --rm --entrypoint cat ragbench-langgraph:latest /app/pip-freeze.txt > build-records/lg-freeze.txt

echo "== the two systems should run the same torch, transformers and sentence-transformers"
grep -i -E '^(torch|transformers|sentence-transformers|tokenizers|numpy)==' build-records/rr-freeze.txt | sed 's/^/rocketride  /' || true
grep -i -E '^(torch|transformers|sentence-transformers|tokenizers|numpy)==' build-records/lg-freeze.txt | sed 's/^/langgraph   /' || true
echo "If the versions differ, change langgraph/requirements.txt to the engine's versions and build again."

docker image inspect ragbench-rocketride:latest ragbench-langgraph:latest ragbench-parser:latest ragbench-client:latest \
  --format '{{index .RepoTags 0}} {{.Id}}' | tee build-records/image-ids.txt
echo BUILD_OK
