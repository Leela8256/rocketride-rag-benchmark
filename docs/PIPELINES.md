# The RAG pipeline in each framework

The same pipeline is built twice. What is deliberately the same on both sides is listed first, then how each
framework expresses the pipeline, then how the two stores behave, because that is where they differ most.

## What is the same on both sides

| Part | What both use |
|---|---|
| Parser | RocketRide's Java parser (its Tika jars and configuration). RocketRide calls it from its `parse` node. The LangGraph service calls the same jars through a small HTTP server ([langgraph/parser/](../langgraph/parser/)). |
| Splitter | LangChain `RecursiveCharacterTextSplitter` at library defaults, measuring length in characters |
| Embedding model | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`, 384 numbers per vector, normalized, on CPU |
| Vector database | One Postgres 17 with the pgvector extension, cosine distance |
| Input | The same PDFs in the same order, the same questions |

The screening step of every campaign checks this: a PDF is used only when both systems cut it into chunks with
identical text.

## RocketRide

A pipeline is a `.pipe` file: a list of nodes and the lanes that connect them. The engine runs it. The three
files are in [rocketride/pipelines/](../rocketride/pipelines/).

**Ingestion**, [ingest.pipe](../rocketride/pipelines/ingest.pipe):

```
webhook -> parse -> preprocessor_langchain -> embedding_transformer -> postgres
```

- The client connects once over a websocket, starts one task with `use(pipeline, threads=32)`, and sends every
  PDF as one `pipe()` object: open, write the bytes, close. `close()` returns when the document is stored.
- The store is the last node. There is no response node, so nothing is sent back but the document's id.
- The database password is not in the file. The file holds the placeholder `${ROCKETRIDE_PG_PASSWORD}` and the
  client sends the value with `use(env=...)`.

**Ingestion without the store**, [ingest_no_store.pipe](../rocketride/pipelines/ingest_no_store.pipe): the same
nodes with `response_documents` in place of `postgres`, so the chunks and vectors come back to the client. The
screening runs use it to build the reference.

**Search**, [search.pipe](../rocketride/pipelines/search.pipe):

```
chat -> embedding_transformer -> postgres -> response_answers
```

- The client calls `client.chat()` with the question and a limit of 10 results.
- The question is embedded by the same node type and profile as the documents.
- The store's `answers` lane is the response: the nearest chunks with their scores.

## LangGraph

A pipeline is a compiled `StateGraph` behind a FastAPI service ([langgraph/app/](../langgraph/app/)). One
process, plain synchronous graph steps, which LangGraph runs on its default thread pool.

**Ingestion**, graph `document-pdf-rag-v1`
([graph.py](../langgraph/app/pipelines/document_pdf_rag/graph.py)):

```
extract -> chunk -> embed -> store
```

- One HTTP request per PDF: `POST /v1/process/document-pdf-rag-v1`.
- `store` calls LangChain's `PGVectorStore.add_embeddings()` with the vectors the `embed` step computed
  ([store_pg.py](../langgraph/app/workload/document/store_pg.py)). The response carries the number of rows written.

**Ingestion without the store**, graph `document-pdf-v1`: the same first three steps, returning chunks and
vectors. Used by the screening runs.

**Search**, graph `search-pdf-rag-v1` ([graph.py](../langgraph/app/pipelines/search_pdf_rag/graph.py)):

```
embed -> retrieve
```

- One HTTP request per question, with the question and the number of results in the request's options.
- `retrieve` calls `PGVectorStore.similarity_search_with_score_by_vector()`.

Nothing in the service adds a lock, a limit, a queue, a cache or a batch. The vector store is created the way
LangChain's documentation shows, at library defaults.

## How the two differ

These are properties of the two products, read from their source code (RocketRide at release 3.6.0,
langchain-postgres 0.0.19 with SQLAlchemy 2.1). They are the reason the numbers differ, so they are worth knowing
before reading a result.

### Embedding

| | RocketRide 3.6.0 | LangGraph side |
|---|---|---|
| Calls to the model at once | One. The loader holds a lock per model around the forward pass (`ai/common/models/transformers/sentence_transformers.py`). | As many as there are worker threads. `SentenceTransformer.encode()` is called with no lock. |

### Writing to the store

| | RocketRide `postgres` node, 3.6.0 | LangChain `PGVectorStore`, langchain-postgres 0.0.19 |
|---|---|---|
| Connections | One for the pipeline, shared by all threads. One statement or transaction at a time. | A pool that keeps 5 connections and opens up to 10 more under load (SQLAlchemy's defaults) |
| Who runs a store call | The pipeline thread that holds the document, on the shared connection | One background thread with one event loop for the whole process (`langchain_postgres/v2/engine.py`). Every synchronous call of the store is handed to it. Several statements can wait on the database at the same time, each on its own connection, but the Python part of every call runs on that one thread. |
| Per document | One transaction: delete the rows of that document id, one `INSERT` per chunk, commit | One `INSERT ... ON CONFLICT` and one commit per chunk |
| Sending the same document again | Replaces its rows | Adds the rows again under new ids |
| Index on the document id | None, so the delete scans the table | Not needed for the write |
| Vector index | None created | None created |

**One store call per document in this pipeline.** `preprocessor_langchain` hands the chunks of a document's
text to `embedding_transformer` in one call, when the document ends. That node passes its whole buffer on as
soon as it holds 64 chunks or the document ends, so the text of a document reaches the store in one call and one
transaction, however many chunks it has. A table the parser recognises is handed on in a call of its own; in the
runs this benchmark was developed with, the parser returned no tables for PDFs, and the database counted about
one commit per document.

In 3.6.0 every store call first deletes the rows of the document ids it carries. A document that reached the
store in several calls would therefore keep only the rows of the last one. The read-back check would report the
others as missing chunks and fail the run.

### Searching the store

| | RocketRide `postgres` node, 3.6.0 | LangChain `PGVectorStore` |
|---|---|---|
| Query | `SELECT *, embedding <=> q AS distance ... ORDER BY distance LIMIT k` | The library's own nearest-neighbour query |
| Connections | The same single connection, one search at a time | The same pool, behind the same single background thread |
| Results it drops | Any under a similarity threshold: 0.5 by default, never under 0.20 | None |

Because of the last row the benchmark sets RocketRide's threshold to 0.20 and leaves out results under 0.20 on
both sides when it compares result lists.

Neither system creates a vector index, so a search compares the question with every stored vector. The harness
therefore measures search twice: on the table as each system leaves it, and after it has built a pgvector HNSW
index itself.

### How a request reaches the system

| | RocketRide | LangGraph side |
|---|---|---|
| Transport | One websocket for all requests | One HTTP request each |
| Load client of this benchmark | The RocketRide client SDK: one event loop, one connection | Python's `urllib` from a thread pool: a new connection for every request |
| Concurrency inside | The task's threads (32 in the default plan; a task accepts 1 to 64) | The default thread pool of the process |

The transport and the client belong to each stack as it is measured here. A request is timed by the client, from
sending it to its answer, so the time of the transport is inside every latency. The report shows how much CPU
the client used in every timed part, to show when a client and not a system sets a rate.
