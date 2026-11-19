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
2. The host removes the rows the warm-up left in the table (the four PDFs, and the one document the LangGraph
   service stores when it starts), so that a search can only return chunks of the measured PDFs. A row it cannot
   explain stops the run here. It then settles the database (`VACUUM`, `ANALYZE`, `CHECKPOINT`) and clears its
   statement statistics.
3. **Ingestion, timed.** The PDFs are sent with 64 in flight until all are stored.
4. The table is read back and checked (see Correctness).
5. **Search without an index, timed.** After 16 warm-up questions, the questions are sent at each number in
   flight in turn (1, 8, 32 by default). Every level sends every question once. Every answer is checked before
   it is counted.
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
| Client CPUs busy | Whether the load client, and not the system, set the rate | CPU time of the client process during the timed requests divided by the wall time. Each client does its Python work on one CPU at a time. The report flags 0.8 and above as a rate the client may have set. A lower value makes that unlikely; it does not rule it out. |

## Search

Reported for each state of the table (without an index, with the index) and each number of questions in flight.

| Metric | What it tells you | How it is taken |
|---|---|---|
| Valid answers per second | Search speed | Number of answers that passed their check, divided by the wall time of the level ([search_driver.py](../harness/search_driver.py)) |
| Answers not valid | Whether a rate can be trusted | Responses that are not an answer: an error, or another shape than a list of result documents. One of them fails the run, so a reported rate never contains any. |
| Valid answers without a result | How often a system finds nothing | Answers that are well formed and hold no result. They count as answers. The first five are kept in the run's folder for inspection, and the report flags a level where more than 1 % of the answers are empty, because an empty answer is quick. |
| Latency: middle value, 95th and 99th percentile | How long one question takes | Time from sending a question to its answer, timed by the client, over the valid answers |
| Service CPU per question | Computing cost of one question | cgroup CPU time of the service during the level, divided by the questions |
| Database CPUs busy | Whether the database is the limit | cgroup CPU time of the database container during the level, divided by the wall time |
| Time inside the database per search | Database cost of one search, as the database sees it | Execution time of the search statement divided by its calls, from `pg_stat_statements` |
| Index build time and size | Cost of the vector index | Timed and measured by the harness when it builds the index |
| Results in common with the search without the index | What the index's speed costs in results | The same run's result lists with and without the index, compared question by question. The index is approximate, so the report prints this beside the rate with the index. |
| Client CPUs busy | Whether the load client set the rate | As for ingestion, for every level |

## Correctness

A rate only counts when the run did the right work. Four checks:

**1. Screening, before any timed run.** Every candidate PDF goes through both systems without the store. A PDF
is used only when both return a valid result, the chunk texts are identical and the vectors are equal within the
tolerance of the read-back ([finalize.py](../harness/finalize.py)). Every PDF left out is listed with its reason,
and the errors of each screening run are counted by message, so a failure in one system cannot pass unseen. A
file with the same bytes as an earlier one is not a candidate at all.

The timed runs therefore cover only PDFs that both systems handle the same way. The report says how many
candidates were left out and why.

**2. Read-back, after every ingestion** ([dbcheck.py](../harness/dbcheck.py)). The database computes the SHA-256
of every stored chunk text and returns every stored vector. They are compared with the reference, which is the
RocketRide screening run. A run passes when:

- every expected chunk is in the table,
- no expected chunk is in the table more often than expected,
- every stored vector equals the reference vector of its chunk within tolerance (absolute 1e-5 plus relative 1e-4),
- the table holds no other row. The one exception is a row RocketRide's store keeps for itself, which its own
  search never returns. Any other row fails the run and is listed by document, because a search could return it,
- no document has a row that was committed after the client had that document's answer, and no row was
  committed after the client's clock stopped (allowance 1 ms). The database keeps the commit time of every row
  (`track_commit_timestamp`), and the client records when it sent each document and how long the answer took.
  So an answer came only after its document was stored, and the ingestion time is the time to store.

Both systems are checked against the same reference, so two passing runs stored the same chunks and vectors, and
nothing else.

**3. Search answers.** Every response is checked before it counts
([search_driver.py](../harness/search_driver.py)):

- A RocketRide answer must be the object its client returns for the search pipeline: no error, and exactly one
  answer that is a list of result documents. A LangGraph answer must carry its request id, the hash of its own
  body and a list of result documents.
- A well formed answer with no result is valid and is counted separately.
- Anything else is not an answer. It is never counted in a rate, the first five such responses are kept, and
  the run fails. A request that fails outright (no response at all) stops the run at once.
- Before the first timed question the client asks RocketRide one question that cannot match (limited to a
  document id no table holds). When the engine answers it without results, the form of an empty answer is
  confirmed, and a form the client does not read as a valid answer stops the run there. When the engine refuses
  the question or ignores the limit, the form stays unconfirmed and the run goes on. The report says for every
  search phase which of the two happened. In both cases a response in another form is never counted.

**4. Search results.** The result lists of the two systems are compared (see next section).

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

## Reading a difference

- **Runs per system.** Three by default, in alternating order (`--repeats`). Use five for a result you publish.
- **Spread.** The report prints every run, the mean, and the lowest and highest run of each system.
- **Not conclusive.** When the runs of the two systems overlap, the report says so, and the difference should
  not be reported as one.
- **Everything, not a selection.** A report lists the candidates left out, the runs that failed and the attempts
  that were run again. Give all metrics of this page, whichever system leads.

## What is not measured

- **Answer quality.** No language model writes an answer, and no person judged the results. "Source chunk among
  the results" depends on the questions (12 words) and the chunks (up to 4,000 characters) far more than on the
  frameworks.
- **Anything inside the engine.** The engine runs as shipped, with no profiler or tracing.
- **The cause of a difference.** The two stacks differ in framework, store code, transport and client (see
  [PIPELINES.md](PIPELINES.md)). A rate compares the stacks. It does not say which part is responsible.
- **GPU.** Both systems embed on CPU.
- **Other stores.** Only Postgres with pgvector.
