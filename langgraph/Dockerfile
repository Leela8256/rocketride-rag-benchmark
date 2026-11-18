# The LangGraph side: a FastAPI service that runs LangGraph graphs, with LangChain's Postgres vector store.
#
# linux/amd64 like the RocketRide engine image, so both systems run on the same architecture.
FROM --platform=linux/amd64 python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip freeze > /app/pip-freeze.txt

# The embedding model is put into the image at a fixed revision, so a run needs no network and every container
# embeds with the same weights. It is the model RocketRide's `miniLM` profile names.
ARG MODEL_ID=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
ARG MODEL_REVISION=b207367332321f8e44f96e224ef15bc607f4dbf0
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_ID}', revision='${MODEL_REVISION}', device='cpu')" \
    && MODEL_DIR="/opt/hf/hub/models--$(echo "${MODEL_ID}" | sed 's#/#--#g')" \
    && test -d "$MODEL_DIR/snapshots/${MODEL_REVISION}" \
    # The service loads the model by name. Offline, the name resolves through refs/main, so point it at the revision.
    && mkdir -p "$MODEL_DIR/refs" && printf '%s' "${MODEL_REVISION}" > "$MODEL_DIR/refs/main" \
    && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "from sentence_transformers import SentenceTransformer; print('offline load ok, dimension', SentenceTransformer('${MODEL_ID}', device='cpu').get_sentence_embedding_dimension())"

# A run must never reach the hub: a missing file should fail loudly, not download other weights.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app/ /app/

# Defaults; the campaign sets PIPELINES, EXTRACTOR, TIKA_URL and the database settings for every run.
ENV PIPELINES=document-pdf-v1 \
    TEMP_DIRECTORY=/tmp

# No thread setting of any math library is set here or by the campaign: the service runs with library defaults.
# One process is the deployment shape. It listens on 127.0.0.1 because every container uses the host network.
CMD ["uvicorn", "lg_rag_runtime:app", "--host", "127.0.0.1", "--port", "8100", "--workers", "1"]
