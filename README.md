# RAG pipeline benchmark: RocketRide and LangGraph

A benchmark of a complete retrieval pipeline (RAG) on two frameworks: the RocketRide engine as it is released, and
LangGraph with LangChain. Both halves of the pipeline are timed:

- **Ingestion.** A PDF is parsed to text, the text is cut into chunks, every chunk is turned into a vector by an
  embedding model, and chunks and vectors are stored in a vector database.
- **Search.** A question is turned into a vector and the database returns the stored chunks nearest to it.

Both systems run **as shipped**. Nothing inside the RocketRide release is changed, and nothing is added inside
either system to lock, limit or batch its work. The two use the same parser, the same splitter, the same
embedding model and the same database.

What still differs is how each stack is built and reached: the framework, its vector store code (connections and
transactions), where the parser runs, and the transport (one websocket for RocketRide, one HTTP request per call
for LangGraph). So a difference in a rate is a difference between the two stacks as they are deployed here. The
benchmark does not show which part causes it. [docs/PIPELINES.md](docs/PIPELINES.md) lists the parts.

Two numbers are chosen by the harness, and chosen to match: the RocketRide task is started with 32 threads
(`use(threads=32)`, the client's own parameter), which is the size of the LangGraph process's default thread
pool on the benchmark machine, and the parser helper of the LangGraph side also serves with 32 threads.

This is retrieval only. No language model writes an answer from the retrieved chunks.

**Who wrote this.** The benchmark is written and maintained by RocketRide engineers. The pipelines of both sides,
the harness and every check are in this repository so that anyone can read them and run them again. A result
taken with it should be given with every metric of [docs/METRICS.md](docs/METRICS.md), whichever system leads,
together with the report's list of what was left out and what failed.

## What is in this repository

| Folder | What it holds |
|---|---|
| [rocketride/](rocketride/) | The engine image (the release archive, unchanged) and the three pipeline files: ingestion, ingestion without the store, search |
| [langgraph/](langgraph/) | The LangGraph side: a FastAPI service, the ingestion and search graphs, LangChain's Postgres store, and the parser helper |
| [infra/](infra/) | Image build script, load client image, database start-up script |
| [harness/](harness/) | Plan, campaign runner, load clients for ingestion and search, read-back check, CPU and memory sampler, analysis and report |
| [tests/](tests/) | Tests that run without Docker |
| [docs/](docs/) | [PIPELINES.md](docs/PIPELINES.md), [INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md), [METRICS.md](docs/METRICS.md) |

## The pipeline in each framework

| Step | RocketRide | LangGraph |
|---|---|---|
| Shape | A `.pipe` file of nodes, run by the engine | A `StateGraph` inside a FastAPI service |
| How a PDF arrives | Client SDK over one websocket, one `pipe()` per file | One HTTP request per file |
| Parse | `parse` node | The same RocketRide parser, behind a small HTTP server |
| Chunk | `preprocessor_langchain` (LangChain recursive splitter, defaults) | The same splitter |
| Embed | `embedding_transformer`, profile `miniLM` | The same model through sentence-transformers |
| Store | `postgres` node (pgvector) | LangChain `PGVectorStore.add_embeddings()` |
| Search | `chat -> embedding_transformer -> postgres -> response_answers` | Graph `embed -> retrieve` with `similarity_search_with_score_by_vector()` |

Details, and how the two stores behave, are in [docs/PIPELINES.md](docs/PIPELINES.md).

## Machine layout

Everything runs on one Linux machine, in containers pinned to separate CPUs:

| Container | CPUs (default plan, 32-CPU machine) | Memory |
|---|---|---|
| System under test, one at a time | 24 | 48 GiB |
| Postgres 17 with pgvector, empty at the start of every run | 4 | 8 GiB |
| Load client | 4 | |

See [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) for the images, the CPU lists and how to move the layout
to another machine.

## What is measured

- **Ingestion:** PDFs per second, CPU-seconds per PDF, peak memory, time per PDF, time the database spends per
  PDF, commits.
- **Search:** valid answers per second and latency at several numbers of questions in flight, with the table as
  each system leaves it and again with a vector index; time inside the database per search.
- **Correctness:** every run's stored chunks and vectors are read back and compared with a reference, every
  search answer is checked before it is counted, and the two systems' search results are compared question by
  question.
- **The load client itself:** its CPU use in every timed part, so a rate that the client limits is visible.

Every metric, with how it is taken, is in [docs/METRICS.md](docs/METRICS.md).

## Run it

On a Linux x86-64 machine with Docker, Python 3.9 or newer and a folder of PDFs:

```bash
# 1. Build the images (needs network access; it downloads the engine release and the packages of both systems)
bash infra/build_images.sh

# 2. Check the whole path on 40 PDFs and 30 questions first
python3 harness/make_plan.py --corpus /data/pdfs --out /data/rag-smoke --smoke
bash harness/run_all.sh /data/rag-smoke
cat /data/rag-smoke/REPORT.md

# 3. The benchmark: 3,000 PDFs, 1,000 questions, three timed runs per system (--repeats 5 for a result you publish)
python3 harness/make_plan.py --corpus /data/pdfs --out /data/rag-run
nohup bash harness/run_all.sh /data/rag-run > /data/rag-run/chain.out 2>&1 < /dev/null &
```

`run_all.sh` does five things in order:

1. **Screening.** Every candidate PDF goes through both systems without the store. The RocketRide run becomes the
   reference, and the questions are cut from the chunks it returns.
2. **Finalize.** The PDFs both systems turn into the same chunks and vectors are kept, and the questions are
   fixed.
3. **Timed runs.** One run per system and repeat: warm up, remove the warm-up rows, ingest, read the table back,
   search without an index, build a vector index, search again.
4. **Analysis.** Every number is recomputed from the run files into `analysis.json`.
5. **Report.** `REPORT.md` is written from `analysis.json`.

Other machines need their own CPU lists: see `python3 harness/make_plan.py --help`.

## Tests without Docker

```bash
python3 tests/test_plan_dry_run.py            # plan, docker commands, pipeline files
python3 tests/test_search_validation.py       # the check of every search answer, with a stand-in for the client SDK
python3 tests/test_finalize.py                # which PDFs the screening keeps, and why it leaves the others out
python3 tests/test_store_and_dbcheck.py       # needs a scratch Postgres with pgvector, see the file
```

## Status

- The harness was developed on RocketRide engine 3.5.0. The analysis and report scripts were checked against
  the run files of a 3,000-PDF campaign from that work, and the four tests above pass.
- The default engine release here is 3.6.0. `infra/build_images.sh` has been run against it on a 32-CPU Linux
  machine: every image built, the engine ran the ingestion pipeline in the warm-up step with client SDK 1.6.0,
  and the parser helper compiled against the release's jars. The images have not been built again since two
  later changes to the LangGraph side: one package pin (numpy, to the engine's version) and the removal of
  settings its service read and never applied.
- The campaign itself (screening, timed runs, search) has **not** been run on 3.6.0 in this form yet. Run the
  smoke plan first and expect to adjust small things.
- Three things were added after a review and are covered by the tests only, not yet by a run against the engine:
  the check of every search answer, the removal of the warm-up rows before the timed requests, and the comparison
  of every document's commit time with the moment the client had its answer. They depend on three facts about the
  engine that the smoke run shows:
  - a RocketRide answer without results has the form the search client expects. The client asks for one before
    the first timed question; a form it cannot read stops the run, and the report says whether the form was
    confirmed;
  - RocketRide's store writes a document's file name into the `parent` column. The warm-up rows are removed and
    the commit times are matched by it; if it does not hold, the run stops before the timed requests and names
    the rows;
  - the store's own row has the id `schema`. If it does not, the same stop applies.
- No measured results are published in this repository.

## Limits to keep in mind

- **One machine.** The database shares the machine with the system under test, on its own 4 CPUs. Search without
  an index is limited by those CPUs, so those numbers describe the database as much as the two systems.
- **One store, one model.** Postgres with pgvector, and one small embedding model on CPU.
- **The vector index is not part of either system.** Neither creates one. The harness builds it for the second
  search phase, at pgvector's default settings.
- **One RocketRide setting differs from its default.** Its store drops results under a similarity of 0.5 by
  default. It is set to 0.20, the lowest the node allows, so that it returns what LangChain returns.
- **The questions are cut from the documents.** They are not questions a person wrote, and answer quality is not
  judged.
- **Screened PDFs only.** The timed runs use the PDFs that both systems turn into the same chunks and vectors.
  A PDF one system fails on, or reads differently, is left out and listed in the report. The rates say nothing
  about such PDFs.
- **Two different clients.** RocketRide is driven through its client SDK on one websocket, LangGraph through
  plain HTTP requests. The clients are part of each stack as measured. The report shows the CPU use of the client
  in every timed part.
- **No cause is shown.** The benchmark compares two stacks. Saying which part of a stack makes it faster or
  slower (its scheduler, its store code, its transport) needs an experiment that changes that one part.
- **Sittings do not compare.** Rates from two sittings on the same machine can differ by more than the effect
  being measured. Compare the two systems inside one campaign.
