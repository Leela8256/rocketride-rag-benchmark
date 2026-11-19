#!/usr/bin/env bash
# Stage both systems as plain processes under one folder, for a machine where containers cannot be used (plans made
# with `make_plan.py --backend process`). It does with folders what infra/build_images.sh does with images:
#
#   <stage>/engine/dist                  the RocketRide engine release, unpacked; the archive's SHA-256 is checked
#   <stage>/engine/release-files.sha256  hashes of three release files, taken right after unpacking
#   <stage>/client/venv                  Python with the RocketRide client SDK from PyPI
#   (warm-up)                            one PDF through the pipeline: the engine installs the packages of its nodes
#                                        and downloads the embedding model, so no run installs or downloads anything
#   (store packages)                     the two packages the engine's postgres node asks for, everything already
#                                        installed passed as a constraint
#   <stage>/parser-classes               the helper class of the LangGraph side's parser, compiled against the
#                                        release's own jars
#   <stage>/lg/venv, <stage>/lg/hf       the LangGraph service's pinned packages and the embedding model at its
#                                        fixed revision
#   <stage>/process-config.json          the paths, for make_plan.py --process-config
#
# Everything is written under <stage> (and the usual package caches of the user). Nothing is installed on the machine.
#
# Needs: python3 with venv, curl, network access, a Java that has the compiler module (to compile one class; set
# JAVAC_JAVA, default `java`), and a Postgres 17 with pgvector unpacked somewhere:
#   PG_BIN   folder with postgres, initdb, psql, pg_ctl, pg_isready, createdb
#   PG_LIB   folder with the libraries those programs need (their LD_LIBRARY_PATH)
#
# Usage: PG_BIN=... PG_LIB=... bash infra/stage_processes.sh <stage folder>
# A second start continues: finished steps leave a marker in <stage>/done and are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
STAGE="$(mkdir -p "${1:?usage: stage_processes.sh <stage folder>}" && cd "$1" && pwd)"
: "${PG_BIN:?set PG_BIN}" "${PG_LIB:?set PG_LIB}"
PORT=5565
ENGINE_VERSION="${ENGINE_VERSION:-3.6.0}"
ENGINE_URL="${ENGINE_URL:-https://github.com/rocketride-org/rocketride-server/releases/download/server-v3.6.0/rocketride-server-v3.6.0-linux-x64.tar.gz}"
ENGINE_ARCHIVE_SHA256="${ENGINE_ARCHIVE_SHA256:-621a52106bb91ab5500992ee59893919b4496f41b00cbe0f578b6f65b4639c1b}"
CLIENT_VERSION="${ROCKETRIDE_CLIENT_VERSION:-1.6.0}"
MODEL_ID="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
MODEL_REVISION="b207367332321f8e44f96e224ef15bc607f4dbf0"
DIST="$STAGE/engine/dist"
mkdir -p "$STAGE/engine" "$STAGE/done" "$STAGE/records" "$STAGE/tmp"
step() { [ -e "$STAGE/done/$1" ] && { echo "== $1: already done"; return 1; }; echo "== $1"; return 0; }
ok() { touch "$STAGE/done/$1"; }
ENGINE_PID=""
stop_engine() { if [ -n "$ENGINE_PID" ]; then kill -- "-$ENGINE_PID" 2>/dev/null || true; sleep 3; kill -9 -- "-$ENGINE_PID" 2>/dev/null || true; ENGINE_PID=""; fi; }
trap stop_engine EXIT

if step engine; then
  if [ ! -x "$DIST/engine" ]; then
    curl -fsSL --retry 3 "$ENGINE_URL" -o "$STAGE/engine/engine.tgz"
    echo "${ENGINE_ARCHIVE_SHA256}  $STAGE/engine/engine.tgz" | sha256sum -c -
    mkdir -p "$DIST" && tar -xzf "$STAGE/engine/engine.tgz" -C "$DIST"
    (cd "$DIST" && sha256sum engine ai/common/models/transformers/sentence_transformers.py nodes/store_postgres/postgres.py) > "$STAGE/engine/release-files.sha256"
  fi
  test -x "$DIST/engine" && test -f "$DIST/ai/eaas.py" && test -f "$STAGE/engine/release-files.sha256"
  echo "{\"version\": \"$ENGINE_VERSION\", \"url\": \"$ENGINE_URL\", \"archive_sha256\": \"$ENGINE_ARCHIVE_SHA256\"}" > "$STAGE/records/engine-release.json"
  cp "$STAGE/engine/release-files.sha256" "$STAGE/records/rr-release-files.sha256"
  ok engine
fi

if step client; then
  python3 -m venv "$STAGE/client/venv"
  "$STAGE/client/venv/bin/pip" install --quiet --no-cache-dir "rocketride==${CLIENT_VERSION}"
  "$STAGE/client/venv/bin/pip" freeze > "$STAGE/records/client-freeze.txt"
  # The drivers need these parts of the SDK. The step stops here when a client version lacks one.
  "$STAGE/client/venv/bin/python" -c "import inspect; from rocketride import RocketRideClient as C; from rocketride.schema import DocFilter, Question; \
assert 'env' in inspect.signature(C.use).parameters and 'threads' in inspect.signature(C.use).parameters; \
assert all(hasattr(C, m) for m in ['validate', 'pipe', 'chat', 'terminate']); \
assert 'limit' in DocFilter.model_fields and 'objectIds' in DocFilter.model_fields; print('client SDK ok')"
  ok client
fi

if step warm; then
  ss -tln | grep -q ":$PORT " && { echo "port $PORT is in use"; exit 1; }
  # The environment a run gives the engine (campaign.py, clean_env), so the warm-up fills the caches a run reads.
  (cd "$DIST" && exec env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME="$HOME" LANG=C.UTF-8 \
      PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ROCKETRIDE_APIKEY=local-dev TMPDIR="$STAGE/tmp" \
      setsid ./engine ai/eaas.py --host=127.0.0.1 --port=$PORT) > "$STAGE/records/rr-warm.log" 2>&1 < /dev/null &
  ENGINE_PID=$!
  for i in $(seq 1 300); do curl -fsS --max-time 2 "http://127.0.0.1:$PORT/version" >/dev/null 2>&1 && break; sleep 2; done
  curl -fsS --max-time 5 "http://127.0.0.1:$PORT/version" | tee "$STAGE/records/rr-version.json"; echo
  RAGBENCH_PIPELINES="$REPO/rocketride/pipelines" "$STAGE/client/venv/bin/python" -u harness/warm_rocketride.py "$PORT" \
      langgraph/app/pipelines/document_pdf/fixtures/warmup.pdf
  stop_engine
  ok warm
fi

if step store-packages; then
  (cd "$DIST" \
    && ./engine -m pip freeze 2>/dev/null | grep -E '^[A-Za-z0-9_.-]+==' > "$STAGE/records/rr-freeze-before-store.txt" \
    && ./engine -m pip install --no-cache-dir -c "$STAGE/records/rr-freeze-before-store.txt" psycopg2-binary pgvector \
    && ./engine -m pip freeze 2>/dev/null | grep -E '^[A-Za-z0-9_.-]+==' > "$STAGE/records/rr-freeze.txt" \
    && (diff "$STAGE/records/rr-freeze-before-store.txt" "$STAGE/records/rr-freeze.txt" | tee "$STAGE/records/rr-packages-added.txt" || true) \
    && sha256sum -c "$STAGE/engine/release-files.sha256")   # the files of the release must still be the files of the release
  ok store-packages
fi

if step parser; then
  rm -rf "$STAGE/parser-classes" && mkdir -p "$STAGE/parser-classes"
  CP="$(ls "$DIST"/java/lib/*.jar | tr '\n' ':')"
  "${JAVAC_JAVA:-java}" -m jdk.compiler/com.sun.tools.javac.Main -proc:none -source 17 -target 17 -nowarn -cp "$CP" \
      -d "$STAGE/parser-classes" langgraph/parser/BenchTextServer.java
  test -n "$(find "$STAGE/parser-classes" -name 'BenchTextServer*.class')"
  ok parser
fi

if step langgraph; then
  python3 -m venv "$STAGE/lg/venv"
  "$STAGE/lg/venv/bin/pip" install --quiet -r langgraph/requirements.txt
  "$STAGE/lg/venv/bin/pip" freeze > "$STAGE/records/lg-freeze.txt"
  # The embedding model at a fixed revision, as in langgraph/Dockerfile. The service loads it by name, offline.
  export HF_HOME="$STAGE/lg/hf"
  "$STAGE/lg/venv/bin/python" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_ID}', revision='${MODEL_REVISION}', device='cpu')"
  MODEL_DIR="$HF_HOME/hub/models--$(echo "${MODEL_ID}" | sed 's#/#--#g')"
  test -d "$MODEL_DIR/snapshots/${MODEL_REVISION}"
  mkdir -p "$MODEL_DIR/refs" && printf '%s' "${MODEL_REVISION}" > "$MODEL_DIR/refs/main"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$STAGE/lg/venv/bin/python" -c "from sentence_transformers import SentenceTransformer; print('offline load ok, dimension', SentenceTransformer('${MODEL_ID}', device='cpu').get_sentence_embedding_dimension())"
  ok langgraph
fi

echo "== the two systems should run the same torch, transformers and sentence-transformers"
grep -i -E '^(torch|transformers|sentence-transformers|tokenizers|numpy)==' "$STAGE/records/rr-freeze.txt" | sed 's/^/rocketride  /' || true
grep -i -E '^(torch|transformers|sentence-transformers|tokenizers|numpy)==' "$STAGE/records/lg-freeze.txt" | sed 's/^/langgraph   /' || true
echo "If the versions differ, change langgraph/requirements.txt to the engine's versions and stage the LangGraph side again."

DB_DESCRIPTION="$(LD_LIBRARY_PATH="$PG_LIB" "$PG_BIN/postgres" --version) with pgvector, unpacked packages, no container"
STAGE="$STAGE" REPO="$REPO" DIST="$DIST" PG_BIN="$PG_BIN" PG_LIB="$PG_LIB" DB_DESCRIPTION="$DB_DESCRIPTION" python3 - <<'EOF'
import json, os
e = os.environ
json.dump({
    'rr_dist': e['DIST'], 'release_files': e['STAGE'] + '/engine/release-files.sha256',
    'lg_python': e['STAGE'] + '/lg/venv/bin/python', 'lg_app': e['REPO'] + '/langgraph/app',
    # What langgraph/Dockerfile sets: the model comes from the staged folder and a run never reaches the hub.
    'lg_env': {'HF_HOME': e['STAGE'] + '/lg/hf', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
    'parser_classes': e['STAGE'] + '/parser-classes', 'client_python': e['STAGE'] + '/client/venv/bin/python',
    'pg_bin': e['PG_BIN'], 'pg_lib': e['PG_LIB'], 'db_description': e['DB_DESCRIPTION'],
}, open(e['STAGE'] + '/process-config.json', 'w'), indent=2)
EOF
cat "$STAGE/process-config.json"; echo
echo STAGE_OK
