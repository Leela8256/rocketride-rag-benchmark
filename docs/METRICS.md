# Metrics

Every number in a report is recomputed by [harness/analyze.py](../harness/analyze.py) from files the runs wrote.
This page says what each number means and where it comes from.

Two words used below:

- **In flight:** sent and not yet answered. The client keeps a fixed number of requests in flight: as soon as
  one finishes it sends the next.
- **Service under test:** the RocketRide engine container, or the LangGraph service container together with its
  parser container.

## How a timed run goes

1. An empty database and the service are started. Four warm-up PDFs are sent. They are not timed.
2. The host settles the database (`ANALYZE`, `CHECKPOINT`) and clears its statement statistics.
3. **Ingestion, timed.** The PDFs are sent with 64 in flight until all are stored.
4. The table is read back and checked (see Correctness).
5. **Search without an index, timed.** After 16 warm-up questions, the questions are sent at each number in
   flight in turn (1, 8, 32 by default). Every level sends every question once.
6. The harness builds a pgvector HNSW index on the table, and step 5 is repeated.

Timing covers only the requests of steps 3, 5 and 6. Validation and writing of results happen after the clock
has stopped.

## Ingestion

| Metric | What it tells you | How it is taken |
|---|---|---|
| PDFs per second | Speed of the whole ingestion path | Number of PDFs divided by the wall time from the first request sent to the last one answered ([driver.py](../harness/driver.py)) |
| Share of LangGraph | RocketRide's rate as a percentage of LangGraph's in the same campaign | From the means of the two systems' runs |
| CPU-seconds per PDF | How much computing one PDF costs | CPU time of the service's containers during the timed requests, from the kernel's cgroup counters (`cpu.stat`), divided by the PDFs |
| Average CPUs busy | How much of its CPUs the service used | The same CPU time divided by the wall time |
| Peak own memory | Memory the service's processes hold themselves | Highest half-second sample of `anon` from the cgroup's `memory.stat`. File cache is not counted. ([sampler.py](../harness/sampler.py)) |
| Time per PDF, middle value | How long one PDF takes while 64 are in flight | Median of the time from sending a PDF to its answer, timed by the client |
| Database time per PDF | How hard the system works the database | Total execution time of the system's statements during the timed requests, as Postgres reports it (`pg_stat_statements`), divided by the PDFs |
| Commits, inserts, deletes | How the system writes | Calls per statement kind from `pg_stat_statements` |
| Write-ahead log volume | How much the database had to write | `pg_stat_wal`, difference over the timed requests |
| Database CPU and memory | Load on the database container | Its cgroup counters, sampled on the host |
| Time per graph step (LangGraph) | Where a LangGraph request spends its time | The `Server-Timing` header the service returns with every response |

## Search

Reported for each state of the table (without an index, with the index) and each number of questions in flight.

| Metric | What it tells you | How it is taken |
|---|---|---|
| Questions per second | Search speed | Number of questions divided by the wall time of the level ([search_driver.py](../harness/search_driver.py)) |
| Latency: middle value, 95th and 99th percentile | How long one question takes | Time from sending a question to its answer, timed by the client |
| Service CPU per question | Computing cost of one question | cgroup CPU time of the service during the level, divided by the questions |
| Database CPUs busy | Whether the database is the limit | cgroup CPU time of the database container during the level, divided by the wall time |
| Time inside the database per search | Database cost of one search, as the database sees it | Execution time of the search statement divided by its calls, from `pg_stat_statements` |
| Index build time and size | Cost of the vector index | Timed and measured by the harness when it builds the index |

## Correctness

A rate only counts when the run did the right work. Three checks:

**1. Screening, before any timed run.** Every candidate PDF goes through both systems without the store. A PDF
is used only when both return a valid result and the chunk texts are identical. The LangGraph run's vectors are
compared with the RocketRide run's.

**2. Read-back, after every ingestion** ([dbcheck.py](../harness/dbcheck.py)). The database computes the SHA-256
of every stored chunk text and returns every stored vector. They are compared with the reference, which is the
RocketRide screening run. A run passes when:

- every expected chunk is in the table,
- no expected chunk is in the table more often than expected,
- every stored vector equals the reference vector of its chunk within tolerance (absolute 1e-5 plus relative 1e-4).

Both systems are checked against the same reference, so two passing runs stored the same chunks and vectors.

**3. Search results.** Every answer must be complete, and the result lists are compared (see next section).

A run that fails a check has no valid result. It is listed as failed, not averaged in, and a second start of
the campaign runs it again.

## Do both systems find the same chunks

| Metric | What it tells you | How it is taken |
|---|---|---|
| Questions with the same result list | Whether the two systems return the same chunks in the same order | Each run's lists compared with the first LangGraph run's, by the SHA-256 of the chunk texts. Results under similarity 0.20 are left out on both sides, because RocketRide's store never returns them. |
| Share of results in common | The same, when lists differ only partly | Mean overlap of the two lists per question |
| Index against its own exact scan | How much of the exact answer the approximate index returns | A run's results with the index compared with the same run's results without it |
| Source chunk among the results | Whether the chunk a question was cut from comes back | Every question is 12 consecutive words from the middle of one stored chunk, so that chunk is known |
| Questions whose list changes with the load | Whether results stay the same under load | Each level's lists compared with the first level's |

## What is not measured

- **Answer quality.** No language model writes an answer, and no person judged the results. "Source chunk among
  the results" depends on the questions (12 words) and the chunks (up to 4,000 characters) far more than on the
  frameworks.
- **Anything inside the engine.** The engine runs as shipped, with no profiler or tracing.
- **GPU.** Both systems embed on CPU.
- **Other stores.** Only Postgres with pgvector.
