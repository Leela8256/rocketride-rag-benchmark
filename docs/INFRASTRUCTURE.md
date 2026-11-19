# Infrastructure

Everything runs on one Linux machine. The system under test, the database and the load client each get their own
container and their own CPUs, so one cannot take CPU time from another. The harness scripts on the host pin
themselves to the client's CPUs.

## The machine

The layout was developed on one AWS EC2 `c7i.8xlarge`: 32 virtual CPUs (16 physical cores with two hardware
threads each), 64 GiB of memory, no GPU. Any x86-64 Linux machine with enough CPUs works; give the plan your own
CPU lists.

What the machine needs:

- **x86-64 Linux.** The RocketRide engine release used here is linux-x64, and both systems run on the same
  architecture.
- **Docker** with cgroup v2. The harness reads CPU time and memory of every container from `/sys/fs/cgroup`.
- **Python 3.9 or newer** on the host. The harness uses the standard library only.
- **Network access while the images are built.** The images are built so that a run downloads and installs
  nothing: the embedding model and the packages of the nodes are already inside. The smoke run shows whether that
  holds for the engine release you test (look for package installation in the engine's log of a run).
- **Disk.** Tens of gigabytes for the images (both systems install torch), plus the campaign folder (the two
  screening runs keep every vector).
- **Nothing else running.** A campaign refuses to start while another container is running.

## Containers and CPUs

Default plan, for a 32-CPU machine whose CPU `n` and CPU `n + 16` are the two threads of one core:

| Container | Image | CPUs | Memory | Lives for |
|---|---|---|---|---|
| RocketRide engine | `ragbench-rocketride:latest` | `0-11,16-27` (12 cores) | 48 GiB | one run |
| LangGraph service | `ragbench-langgraph:latest` | `0-11,16-27` | 48 GiB | one run |
| Parser for the LangGraph service | `ragbench-parser:latest` | `0-11,16-27` | | one run |
| Postgres with pgvector | `pgvector/pgvector:pg17` | `12-13,28-29` (2 cores) | 8 GiB | one run |
| Load client | `ragbench-client:latest` | `14-15,30-31` (2 cores) | | one run |

- **One system at a time.** A run starts either the engine, or the LangGraph service with its parser.
- **The parser counts as LangGraph.** Its CPU time is added to the service's, because RocketRide parses inside
  the engine. The parser container has no memory limit of its own, so the 48 GiB limit covers the LangGraph
  service but not its parser, while it covers all of the RocketRide engine.
- **Whole cores.** Each group gets both threads of its cores, so no core is shared between two groups. Check
  your machine with `lscpu -e` and pass `--service-cpus`, `--db-cpus` and `--client-cpus` to `make_plan.py`.
- **Host network, localhost only.** Every container uses the host network and listens on `127.0.0.1`: engine
  5565, LangGraph service 8100, parser 9998, Postgres 5432.

## The database

- **Fresh for every run.** Each run starts an empty Postgres container and removes it afterwards, with its data.
  No run sees another run's rows.
- **Settings.** `shared_buffers=2GB`, `max_connections=200`, `maintenance_work_mem=1GB` (so the vector index of
  the search phase is built in memory), `pg_stat_statements` loaded so the database reports the time it spends
  per statement, and `track_commit_timestamp=on` so the read-back can compare the commit time of every row with
  the instant the client's clock stopped.
- **Extensions.** `vector` and `pg_stat_statements` are created at start by [infra/db-init.sql](../infra/db-init.sql).
- **Tables.** Each system creates its own table (`rr_chunks`, `lc_chunks`) with its own layout. The harness does
  not create or alter them, with two exceptions: it deletes the rows the warm-up left before the timed requests,
  and it builds the vector index for the second search phase.
- **Password.** A new random password per run. It is written to a file outside the campaign folder (mode 0600),
  handed to the containers as an environment file, and deleted when the run ends. It is never printed and never
  stored with the results.

## Images

[infra/build_images.sh](../infra/build_images.sh) builds them in this order:

| Image | Built from | What it is |
|---|---|---|
| `ragbench-rocketride:base` | [rocketride/Dockerfile](../rocketride/Dockerfile) | The engine release archive, unpacked. The archive's SHA-256 is checked. |
| `ragbench-client:latest` | [infra/client.Dockerfile](../infra/client.Dockerfile) | Python with the RocketRide client SDK from PyPI |
| `ragbench-rocketride:warm` | the base image after one warm-up run | The engine installs the packages of a node the first time a pipeline uses it, and downloads the model. One PDF is sent through the pipeline and the container is saved, so no run installs or downloads anything. |
| `ragbench-rocketride:latest` | [rocketride/Dockerfile.store](../rocketride/Dockerfile.store) | The warm image plus the two packages the engine's `postgres` node asks for (`psycopg2-binary`, `pgvector`) |
| `ragbench-parser:latest` | [langgraph/parser/Dockerfile](../langgraph/parser/Dockerfile) | The engine base image with one helper class added. Only the engine's Java runtime and parser jars run in it, served over HTTP. |
| `ragbench-langgraph:latest` | [langgraph/Dockerfile](../langgraph/Dockerfile) | The LangGraph service, its pinned packages and the embedding model |

- **The release stays the release.** The hashes of the engine binary, the embedding loader and the store node
  are taken when the archive is unpacked. They are checked again when the last image is built, and in every
  RocketRide run after its warm-up, before the timed requests.
- **Another release.** Set `ENGINE_VERSION`, `ENGINE_URL` and `ENGINE_ARCHIVE_SHA256` in the environment of
  `build_images.sh`.
- **Same model on both sides.** The LangGraph image holds the embedding model at a fixed revision. The engine
  downloads the model its `miniLM` profile names during the warm-up build step. If the two were ever different
  weights, the screening comparison and every read-back would fail on the vectors.
- **Same model code on both sides.** The build prints the torch, transformers and sentence-transformers versions
  of both images. If they differ, change [langgraph/requirements.txt](../langgraph/requirements.txt) to the
  engine's versions and build again.

## The PDFs

Any folder of `.pdf` files. The campaign this layout was developed for used PDFs from the public Govdocs1
collection.

`make_plan.py` takes the first four files by name for warm-up and spreads the candidates evenly over the rest.
A file with the same bytes as an earlier one is left out and the next file takes its place, so no document is
stored twice. The default plan needs 3,304 different PDFs: 3,300 candidates, of which the first 3,000 that pass
the screening are used.

## How long it takes

On the machine above, the default plan (two screening runs, then three timed runs per system on 3,000 PDFs with
1,000 questions) takes several hours. The smoke plan is a matter of minutes once the images exist.

## The campaign folder

Everything one sitting produces is in the folder given to `make_plan.py`:

| Path | What it is |
|---|---|
| `plan.json` | Machine layout, images, database settings, one entry per run, and the commit of this repository |
| `source/corpus-manifest.json` | The PDFs with their SHA-256, and the files left out as copies |
| `source/queries.jsonl` | The questions, each with the hash of the chunk it was cut from |
| `finalize.json` | Which candidates the screening kept, and every one it left out with the reason |
| `host-screening.json`, `host-campaign.json`, `host-cells.json` | The machine as the harness found it at the start of a stage (or of a start with `--only`): CPU model, CPUs, memory, kernel, Docker version |
| `cells/<run>/` | Everything one run wrote: commands, container records with image ids, the commit of the harness, samples, results, logs |
| `cells/<run>.failed-<n>/` | An attempt that failed and was run again. It is kept, and the report lists it. |
| `status.json` | Which runs completed, failed or were skipped |
| `analysis.json`, `REPORT.md` | Written by `analyze.py` and `report.py` |

A campaign folder cannot be moved: the plan holds absolute paths. Make a new plan for a new place.
