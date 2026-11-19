"""Create a campaign folder: the plan of every run and the list of candidate PDFs.

A campaign folder holds everything one benchmark sitting produces. This script writes:

  plan.json                     machine layout, images, database settings and one entry per run ("cell")
  source/corpus-manifest.json   the candidate PDFs (spread evenly over the corpus, a file with the same bytes as
                                an earlier one left out) and four warm-up PDFs, with the SHA-256 of every file

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
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def harness_revision():
    """The commit of this repository, so a campaign records which harness made it. None outside a git checkout."""
    def git(*args):
        return subprocess.run(['git', '-C', str(REPO)] + list(args), capture_output=True, text=True)
    try:
        head = git('rev-parse', 'HEAD')
        if head.returncode != 0:
            return {'commit': None, 'uncommitted_changes': None}
        changed = git('status', '--porcelain', '--untracked-files=no').stdout.strip()
    except OSError:                 # no git on this machine
        return {'commit': None, 'uncommitted_changes': None}
    return {'commit': head.stdout.strip(), 'uncommitted_changes': bool(changed)}


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
    p.add_argument('--repeats', type=int, default=3, help='timed runs per system')
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
    p.add_argument('--manifest', help='a fixed list of PDFs (JSON with "warm", "measured" and "sha256"): these files in '
                                      'this order and no other selection; every timed run then ingests all of them '
                                      'that pass the screening')
    p.add_argument('--questions-file', help='given questions (JSON lines with id, doc, text, chunk_sha256) to use in '
                                            'place of the ones cut from the screening run')
    p.add_argument('--rr-gates', default='', help='also run RocketRide with its embedding lock replaced by a gate of '
                                                  'these sizes, for example 4,8 (not as shipped; needs --gate-dir)')
    p.add_argument('--gate-dir', help='folder with bench_gate.py, the changed loader and gate-files.json')
    p.add_argument('--screen-request-timeout', type=int, default=900, help='seconds one PDF may take in a screening run')
    p.add_argument('--cell-hours', type=float, default=3.0, help='time limit of the ingestion of one run')
    args = p.parse_args()

    if args.smoke:
        args.documents, args.candidates, args.questions, args.repeats, args.levels = 40, 44, 30, 1, '1,8'
    candidates = args.candidates or args.documents + args.documents // 10
    levels = [int(x) for x in args.levels.split(',')]
    assert 1 <= args.engine_threads <= 64, 'a RocketRide task takes 1 to 64 threads'
    assert candidates >= args.documents and args.repeats >= 1 and levels
    gates = [int(x) for x in args.rr_gates.split(',') if x.strip()]
    assert all(g >= 1 for g in gates) and len(set(gates)) == len(gates), 'gate sizes are different positive numbers'
    assert not gates or args.gate_dir, '--rr-gates needs --gate-dir'
    assert not (args.manifest and args.smoke), 'a fixed list is used as it is; --smoke picks its own 44 PDFs'

    corpus = Path(args.corpus).resolve()
    out = Path(args.out).resolve()
    assert corpus.is_dir(), 'not a folder: %s' % corpus
    assert not (out / 'plan.json').exists(), 'this campaign folder already has a plan'
    env_dir = Path(args.db_env_dir).expanduser().resolve()
    assert out not in env_dir.parents and env_dir != out, 'the password folder must be outside the campaign folder'

    names = sorted(x.name for x in corpus.glob('*.pdf'))
    given = json.loads(Path(args.manifest).read_text()) if args.manifest else None
    if given:
        # A fixed list: these warm-up PDFs and these candidates, in this order. Every file must be the file the
        # list names (SHA-256). Nothing is added and nothing is left out here, not even a file with the same bytes
        # as another: the list is the definition of the run.
        warm, picked = list(given['warm']), list(given['measured'])
        assert len(warm) == 4, 'the client warms up with four PDFs'
        assert len(set(warm + picked)) == len(warm) + len(picked), 'a name is in the list twice'
        hashes = {}
        for n in warm + picked:
            assert (corpus / n).is_file(), 'not in the corpus folder: %s' % n
            hashes[n] = sha(corpus / n)
            assert hashes[n] == given['sha256'][n], 'not the file the list names (SHA-256 differs): %s' % n
        candidates = args.documents = len(picked)
        same_bytes = len(picked) + len(warm) - len(set(hashes.values()))
        selection = ('the %d PDFs of the given list %s in its order, every file verified by SHA-256; its 4 warm-up PDFs; '
                     '%d of the files have the same bytes as another file of the list and are kept, as the list has them'
                     % (len(picked), Path(args.manifest).name, same_bytes))
        duplicates = []
    assert given or len(names) >= candidates + 4, 'the corpus has %d PDFs; %d are needed' % (len(names), candidates + 4)
    # Four PDFs warm every run up and are never measured. The candidates are spread evenly over the rest. A file
    # with the same bytes as a warm-up file or an earlier candidate is left out and the next file takes its place:
    # the same document twice would put the same chunks into the table twice.
    if not given:
        warm, rest = names[:4], names[4:]
        hashes = {n: sha(corpus / n) for n in warm}
        seen, taken, picked, duplicates = set(hashes.values()), set(), [], []
        step = len(rest) / candidates
    for i in range(0 if given else candidates):
        j = int(i * step)
        while True:
            assert j < len(rest), 'the corpus has too few different PDFs: %d of the %d needed' % (len(picked), candidates)
            name, j = rest[j], j + 1
            if name in taken:
                continue
            taken.add(name)
            hashes[name] = sha(corpus / name)
            if hashes[name] not in seen:
                seen.add(hashes[name])
                picked.append(name)
                break
            duplicates.append(name)
    assert len(set(picked)) == candidates

    (out / 'source').mkdir(parents=True)
    (out / 'cells').mkdir()
    if not given:
        selection = ('%d PDFs spread evenly over the %d files of the corpus by name; the first 4 files warm up; '
                     '%d files with the same bytes as an earlier one left out' % (candidates, len(names), len(duplicates)))
    manifest = {'measured': picked, 'warm': warm, 'stage': 'candidates', 'selection': selection,
                'duplicates_left_out': duplicates,
                'sha256': {n: hashes[n] for n in picked + warm}}
    (out / 'source/corpus-manifest.json').write_text(json.dumps(manifest) + '\n')

    # Screening: every candidate through both systems without the store. The RocketRide run becomes the reference
    # every later run is checked against, and the questions are cut from the chunks it returns.
    limit = int(args.cell_hours * 3600)
    screening = {'documents': candidates, 'store': False, 'search': False, 'tolerant': True,
                 'request_timeout_s': args.screen_request_timeout, 'timeout_s': limit}
    cells = [cell('screen-rr', 'rr', 'screening', make_queries=True, **screening),
             cell('screen-lg', 'lg', 'screening', **screening)]
    # Timed runs, alternating so that neither system always runs first: rr lg lg rr rr lg ...
    order = []
    for i in range(args.repeats):
        order += [('rr', i + 1), ('lg', i + 1)] if i % 2 == 0 else [('lg', i + 1), ('rr', i + 1)]
    for i in range(args.repeats):
        for arm, n in order[2 * i:2 * i + 2]:
            cells.append(cell('%s-%d' % (arm, n), arm, 'campaign', args.documents, timeout_s=limit))
        # RocketRide with its embedding lock replaced by a gate of N calls at once. Not as shipped. The system is
        # still `rr` (same pipelines, same screening run, same checks); `system` names the variant in the report.
        for g in gates:
            cells.append(cell('rr-g%d-%d' % (g, i + 1), 'rr', 'campaign', args.documents, timeout_s=limit,
                              gate=g, system='rr-g%d' % g))
    gate = None
    if gates:
        gate_dir = Path(args.gate_dir).expanduser().resolve()
        record = json.loads((gate_dir / 'gate-files.json').read_text())
        for name, key in [('bench_gate.py', 'module_sha256'), ('sentence_transformers.py', 'changed_sha256')]:
            assert sha(gate_dir / name) == record[key], '%s is not the file gate-files.json describes' % name
        gate = dict(record, dir=str(gate_dir))
    if args.questions_file:
        given_questions = [json.loads(x) for x in Path(args.questions_file).read_text().splitlines() if x.strip()]
        assert given_questions and all({'id', 'doc', 'text', 'chunk_sha256'} <= set(q) for q in given_questions)
        assert all(isinstance(q['id'], int) and not isinstance(q['id'], bool) for q in given_questions), 'a question id is a whole number'
        assert len({q['id'] for q in given_questions}) == len(given_questions), 'two given questions have the same id'
        (out / 'source/questions-given.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in given_questions))
        args.questions = len(given_questions)

    plan = {
        'tag': out.name, 'root': str(out), 'repo': str(REPO), 'harness': harness_revision(), 'corpus': str(corpus),
        'documents': args.documents, 'candidates': candidates, 'questions': args.questions,
        # A fixed list: the timed runs ingest every PDF of it that passes the screening, however many those are.
        'measure_all_kept': bool(given), 'manifest_given': Path(args.manifest).name if given else None,
        'questions_given': 'source/questions-given.jsonl' if args.questions_file else None,
        'concurrency': args.in_flight, 'workers': args.engine_threads,
        'service_cpus': args.service_cpus, 'client_cpus': args.client_cpus, 'service_memory': args.service_memory,
        'client_image': args.client_image,
        'rr': {'image': args.rr_image, 'port': 5565, 'engine_dir': '/opt/rocketride/engine',
               # The one row RocketRide's store keeps in its table for itself. Its own search never returns it.
               'own_object_id': 'schema',
               # Only for the runs with `gate` in their entry: the two files mounted into the engine container.
               'gate': gate,
               'ingest_pipeline': 'ingest.pipe', 'screen_pipeline': 'ingest_no_store.pipe', 'search_pipeline': 'search.pipe',
               # Recorded for every run, so a run shows which engine files it ran.
               'files': ['engine', 'ai/common/models/transformers/sentence_transformers.py', 'nodes/store_postgres/postgres.py',
                         'nodes/parse/libparse.so', 'java/lib/rocketride-parse.jar']},
        'lg': {'image': args.lg_image, 'port': 8100, 'parser_image': args.parser_image, 'parser_port': 9998,
               'ingest_pipeline': 'document-pdf-rag-v1', 'screen_pipeline': 'document-pdf-v1',
               'search_pipeline': 'search-pdf-rag-v1',
               # The document the service stores when it starts, to prove its write path before it reports ready.
               'fixture_documents': ['warmup.pdf'],
               'files': ['/app/lg_rag_runtime.py', '/app/workload/document/embed.py', '/app/workload/document/store_pg.py',
                         '/app/pipelines/document_pdf_rag/graph.py', '/app/pipelines/search_pdf_rag/graph.py']},
        'db': {'image': args.db_image, 'host': '127.0.0.1', 'port': 5432, 'user': 'postgres', 'database': 'ragbench',
               'cpus': args.db_cpus, 'memory': args.db_memory, 'rr_table': 'rr_chunks', 'lg_table': 'lc_chunks',
               'env_dir': str(env_dir),
               # pg_stat_statements gives the time the database spends per statement. maintenance_work_mem lets the
               # vector index of the search phase be built in memory. track_commit_timestamp keeps the commit time
               # of every row, which the read-back compares with the instant the client's clock stopped.
               'args': ['-c', 'listen_addresses=127.0.0.1', '-c', 'port=5432', '-c', 'max_connections=200',
                        '-c', 'shared_buffers=2GB', '-c', 'shared_preload_libraries=pg_stat_statements',
                        '-c', 'pg_stat_statements.max=5000', '-c', 'maintenance_work_mem=1GB',
                        '-c', 'track_commit_timestamp=on']},
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
    print('candidates: %d PDFs, %d bytes; %d left out as copies of an earlier file' % (
        candidates, sum((corpus / n).stat().st_size for n in picked), len(duplicates)))
    print('screening:', ' '.join(c['label'] for c in cells if c['stage'] == 'screening'))
    print('campaign: ', ' '.join(c['label'] for c in cells if c['stage'] == 'campaign'))


if __name__ == '__main__':
    main()
