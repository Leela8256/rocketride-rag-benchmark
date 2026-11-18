"""Create a campaign folder: the plan of every run and the list of candidate PDFs.

A campaign folder holds everything one benchmark sitting produces. This script writes:

  plan.json                     machine layout, images, database settings and one entry per run ("cell")
  source/corpus-manifest.json   the candidate PDFs (spread evenly over the corpus) and four warm-up PDFs,
                                with the SHA-256 of every file

Nothing is started here. Standard library only.

Usage:
  python3 harness/make_plan.py --corpus /data/pdfs --out /data/rag-run
  python3 harness/make_plan.py --corpus /data/pdfs --out /data/rag-smoke --smoke

The defaults describe one 32-CPU machine (16 physical cores, two hardware threads each): 24 CPUs for the system
under test, 4 for the database, 4 for the load client. Give your own CPU lists on another machine.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cell(label, arm, stage, documents, **more):
    """One run: a fresh database (when the run stores), a fresh service container, one system."""
    c = {'label': label, 'arm': arm, 'stage': stage, 'documents': documents, 'store': True, 'search': True,
         'timeout_s': 3 * 3600, 'request_timeout_s': 3600}
    c.update(more)
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', required=True, help='folder of PDF files')
    p.add_argument('--out', required=True, help='campaign folder to create')
    p.add_argument('--documents', type=int, default=3000, help='PDFs every timed run ingests')
    p.add_argument('--candidates', type=int, help='PDFs the screening runs try (default: documents + 10 %%)')
    p.add_argument('--questions', type=int, default=1000, help='search questions')
    p.add_argument('--repeats', type=int, default=2, help='timed runs per system')
    p.add_argument('--in-flight', type=int, default=64, help='PDFs in flight during ingestion')
    p.add_argument('--engine-threads', type=int, default=32, help='threads of the RocketRide task (1 to 64)')
    p.add_argument('--levels', default='1,8,32', help='questions in flight during search, one level after the other')
    p.add_argument('--k', type=int, default=10, help='results asked for per question')
    p.add_argument('--service-cpus', default='0-11,16-27')
    p.add_argument('--db-cpus', default='12-13,28-29')
    p.add_argument('--client-cpus', default='14-15,30-31')
    p.add_argument('--service-memory', default='48g')
    p.add_argument('--db-memory', default='8g')
    p.add_argument('--db-env-dir', default=str(Path.home() / '.ragbench-env'),
                   help='where a run keeps its database password while it runs; must be outside the campaign folder')
    p.add_argument('--rr-image', default='ragbench-rocketride:latest')
    p.add_argument('--lg-image', default='ragbench-langgraph:latest')
    p.add_argument('--parser-image', default='ragbench-parser:latest')
    p.add_argument('--client-image', default='ragbench-client:latest')
    p.add_argument('--db-image', default='pgvector/pgvector:pg17')
    p.add_argument('--smoke', action='store_true', help='a small plan to check the whole path: 40 PDFs, 30 questions')
    args = p.parse_args()

    if args.smoke:
        args.documents, args.candidates, args.questions, args.repeats, args.levels = 40, 44, 30, 1, '1,8'
    candidates = args.candidates or args.documents + args.documents // 10
    levels = [int(x) for x in args.levels.split(',')]
    assert 1 <= args.engine_threads <= 64, 'a RocketRide task takes 1 to 64 threads'
    assert candidates >= args.documents and args.repeats >= 1 and levels

    corpus = Path(args.corpus).resolve()
    out = Path(args.out).resolve()
    assert corpus.is_dir(), 'not a folder: %s' % corpus
    assert not (out / 'plan.json').exists(), 'this campaign folder already has a plan'
    env_dir = Path(args.db_env_dir).expanduser().resolve()
    assert out not in env_dir.parents and env_dir != out, 'the password folder must be outside the campaign folder'

    names = sorted(x.name for x in corpus.glob('*.pdf'))
    assert len(names) >= candidates + 4, 'the corpus has %d PDFs; %d are needed' % (len(names), candidates + 4)
    # Four PDFs warm every run up and are never measured. The candidates are spread evenly over the rest.
    warm, rest = names[:4], names[4:]
    step = len(rest) / candidates
    picked = [rest[int(i * step)] for i in range(candidates)]
    assert len(set(picked)) == candidates

    (out / 'source').mkdir(parents=True)
    (out / 'cells').mkdir()
    manifest = {'measured': picked, 'warm': warm, 'stage': 'candidates',
                'selection': '%d PDFs spread evenly over the %d files of the corpus by name; the first 4 files warm up' % (
                    candidates, len(names)),
                'sha256': {n: sha(corpus / n) for n in picked + warm}}
    (out / 'source/corpus-manifest.json').write_text(json.dumps(manifest) + '\n')

    # Screening: every candidate through both systems without the store. The RocketRide run becomes the reference
    # every later run is checked against, and the questions are cut from the chunks it returns.
    screening = {'documents': candidates, 'store': False, 'search': False, 'tolerant': True, 'request_timeout_s': 900}
    cells = [cell('screen-rr', 'rr', 'screening', make_queries=True, **screening),
             cell('screen-lg', 'lg', 'screening', **screening)]
    # Timed runs, alternating so that neither system always runs first: rr lg lg rr rr lg ...
    order = []
    for i in range(args.repeats):
        order += [('rr', i + 1), ('lg', i + 1)] if i % 2 == 0 else [('lg', i + 1), ('rr', i + 1)]
    cells += [cell('%s-%d' % (arm, n), arm, 'campaign', args.documents) for arm, n in order]

    plan = {
        'tag': out.name, 'root': str(out), 'repo': str(REPO), 'corpus': str(corpus),
        'documents': args.documents, 'candidates': candidates, 'questions': args.questions,
        'concurrency': args.in_flight, 'workers': args.engine_threads,
        'service_cpus': args.service_cpus, 'client_cpus': args.client_cpus, 'service_memory': args.service_memory,
        'client_image': args.client_image,
        'rr': {'image': args.rr_image, 'port': 5565, 'engine_dir': '/opt/rocketride/engine',
               'ingest_pipeline': 'ingest.pipe', 'screen_pipeline': 'ingest_no_store.pipe', 'search_pipeline': 'search.pipe',
               # Recorded for every run, so a run shows which engine files it ran.
               'files': ['engine', 'ai/common/models/transformers/sentence_transformers.py', 'nodes/store_postgres/postgres.py',
                         'nodes/parse/libparse.so', 'java/lib/rocketride-parse.jar']},
        'lg': {'image': args.lg_image, 'port': 8100, 'parser_image': args.parser_image, 'parser_port': 9998,
               'ingest_pipeline': 'document-pdf-rag-v1', 'screen_pipeline': 'document-pdf-v1',
               'search_pipeline': 'search-pdf-rag-v1',
               'files': ['/app/lg_rag_runtime.py', '/app/workload/document/embed.py', '/app/workload/document/store_pg.py',
                         '/app/pipelines/document_pdf_rag/graph.py', '/app/pipelines/search_pdf_rag/graph.py']},
        'db': {'image': args.db_image, 'host': '127.0.0.1', 'port': 5432, 'user': 'postgres', 'database': 'ragbench',
               'cpus': args.db_cpus, 'memory': args.db_memory, 'rr_table': 'rr_chunks', 'lg_table': 'lc_chunks',
               'env_dir': str(env_dir),
               # pg_stat_statements gives the time the database spends per statement. maintenance_work_mem lets the
               # vector index of the search phase be built in memory.
               'args': ['-c', 'listen_addresses=127.0.0.1', '-c', 'port=5432', '-c', 'max_connections=200',
                        '-c', 'shared_buffers=2GB', '-c', 'shared_preload_libraries=pg_stat_statements',
                        '-c', 'pg_stat_statements.max=5000', '-c', 'maintenance_work_mem=1GB']},
        'reference_cell': 'cells/screen-rr',
        'search': {'k': args.k, 'queries': args.questions, 'warm': min(16, args.questions), 'levels': levels,
                   # exact: the table as each system leaves it (neither creates a vector index).
                   # hnsw: the same table after the harness has built a pgvector HNSW index at its defaults.
                   'phases': ['exact', 'hnsw'],
                   # RocketRide's store drops results under a similarity threshold (0.5 by default, never under
                   # 0.20). It is set to 0.20 so that the store returns what LangChain's store returns.
                   'rr_score': 0.2},
        'cells': cells,
    }
    labels = [c['label'] for c in cells]
    assert len(set(labels)) == len(labels)
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print('campaign folder:', out)
    print('candidates: %d PDFs, %d bytes' % (candidates, sum((corpus / n).stat().st_size for n in picked)))
    print('screening:', ' '.join(c['label'] for c in cells if c['stage'] == 'screening'))
    print('campaign: ', ' '.join(c['label'] for c in cells if c['stage'] == 'campaign'))


if __name__ == '__main__':
    main()
