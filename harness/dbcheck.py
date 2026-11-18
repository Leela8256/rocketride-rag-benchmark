"""Read back what a run stored in Postgres, check it against the reference run, and record what the database
did during the timed requests.

Runs on the host after the timed requests of a cell, while the database container is still up. Standard library
only.

The check. The database itself computes the SHA-256 of every stored chunk text and returns every stored vector.
They are compared with the chunk texts and vectors of the same documents in the reference run, which is the
RocketRide screening run of the campaign (per_doc.jsonl and vectors.f32 of that cell):

  every expected chunk is in the table;
  no expected chunk is in the table more often than expected;
  every stored vector equals the reference vector of its chunk within the tolerance the embedding runs use.

Rows whose text is not an expected chunk are counted and reported, not failed: the warm-up documents, the
LangGraph start-up fixture and a row a store keeps for itself.

Both systems are checked against the same reference, so two passing cells stored the same chunks and vectors.

What the database did. The statement statistics (pg_stat_statements) are cleared by the host just before the timed
requests. They are read here first, before this script runs any query of its own on the table: how many inserts,
deletes and commits the system under test made, and how long the database spent on them.

Usage: dbcheck.py <cell directory> [psql command prefix ...]
  The default prefix runs psql inside the cell's database container.
"""
import array
import collections
import json
import subprocess
import sys
from pathlib import Path

ATOL, RTOL, DIM = 1e-5, 1e-4, 384


def read(path):
    return json.loads(Path(path).read_text())


def main():
    out = Path(sys.argv[1]).resolve()
    root = out.parents[1]
    plan, cfg = read(root / 'plan.json'), read(out / 'config.json')
    db = plan['db']
    arm = cfg['arm']
    table = db['rr_table'] if arm == 'rr' else db['lg_table']
    prefix = sys.argv[2:] or ['docker', 'exec', 'ragbench-%s-db' % cfg['label'], 'psql', '-U', db['user'], '-d', db['database']]

    def psql(sql):
        return subprocess.run(prefix + ['-At', '-F', '\t', '-v', 'ON_ERROR_STOP=1', '-c', sql], check=True,
                              capture_output=True, text=True).stdout

    # First, before any query of this script touches the table: what the database did since the host cleared the
    # statistics, just before the timed requests.
    statements, by_kind = [], collections.defaultdict(lambda: {'calls': 0, 'ms': 0.0, 'rows': 0})
    has_stats = psql("SELECT count(*) FROM pg_extension WHERE extname = 'pg_stat_statements'").strip() == '1'
    if has_stats:
        raw = psql("SELECT s.calls, round(s.total_exec_time::numeric, 3), s.rows, left(regexp_replace(s.query, '\\s+', ' ', 'g'), 220) "
                   "FROM pg_stat_statements s JOIN pg_database d ON d.oid = s.dbid WHERE d.datname = current_database() "
                   "AND s.query NOT LIKE '%pg_stat_statements%' ORDER BY s.total_exec_time DESC")
        for line in raw.splitlines():
            calls, ms, rows, query = line.split('\t', 3)
            kind = query.split(None, 1)[0].upper() if query.strip() else '?'
            by_kind[kind]['calls'] += int(calls)
            by_kind[kind]['ms'] += float(ms)
            by_kind[kind]['rows'] += int(rows)
            if len(statements) < 12:
                statements.append({'calls': int(calls), 'ms': float(ms), 'rows': int(rows), 'query': query})
    wal = psql('SELECT wal_records, wal_bytes FROM pg_stat_wal').strip().split('\t')
    counters = psql('SELECT xact_commit, xact_rollback, tup_inserted, tup_deleted FROM pg_stat_database '
                    'WHERE datname = current_database()').strip().split('\t')
    before = read(out / 'db-prepare.json') if (out / 'db-prepare.json').exists() else {}

    # The reference: chunk text hashes and vectors of this cell's documents, from the campaign's reference run.
    ref = root / plan['reference_cell']
    names = read(root / 'source/corpus-manifest.json')['measured'][:cfg['documents']]
    wanted = set(names)
    vectors = array.array('f')
    with (ref / 'vectors.f32').open('rb') as f:
        vectors.fromfile(f, (ref / 'vectors.f32').stat().st_size // vectors.itemsize)
    expected, ref_vector, seen = collections.Counter(), {}, set()
    for line in (ref / 'per_doc.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['doc'] not in wanted:
            continue
        seen.add(row['doc'])
        for i, digest in enumerate(row['text_sha256']):
            expected[digest] += 1
            start = (row['vector_offset'] + i) * DIM
            ref_vector.setdefault(digest, vectors[start:start + DIM])
    assert seen == wanted, 'the reference run does not cover every document of this cell'

    total = int(psql('SELECT count(*) FROM %s' % table))
    dump = psql("COPY (SELECT encode(sha256(convert_to(content, 'UTF8')), 'hex'), embedding::text FROM %s "
                "WHERE content IS NOT NULL AND embedding IS NOT NULL) TO STDOUT" % table)
    found = collections.Counter()
    worst, outside, compared, other = 0.0, 0, 0, 0
    for line in dump.splitlines():
        digest, _, text = line.partition('\t')
        if digest not in expected:
            other += 1
            continue
        found[digest] += 1
        values = [float(x) for x in text.strip('[]').split(',')]
        assert len(values) == DIM, 'a stored vector has %d values' % len(values)
        compared += 1
        bad = False
        for a, b in zip(values, ref_vector[digest]):
            d = abs(a - b)
            if d > worst:
                worst = d
            if d > ATOL + RTOL * abs(b):
                bad = True
        outside += bad
    missing = sum((expected - found).values())
    surplus = sum((found - expected).values())
    # How many times each expected chunk is stored, as a multiple of how often the documents contain it.
    copies = collections.Counter('%g' % (found[h] / expected[h]) for h in expected)

    if arm == 'rr':
        keys = psql('SELECT count(DISTINCT objectid), count(DISTINCT parent), count(*) FILTER (WHERE embedding IS NULL) FROM %s' % table)
        sample = psql('SELECT objectid, parent, chunkid, vectorsize, modelname, length(content) FROM %s ORDER BY id LIMIT 4' % table)
    else:
        keys = psql("SELECT count(DISTINCT langchain_metadata->>'source'), count(DISTINCT langchain_id), 0 FROM %s" % table)
        sample = psql("SELECT langchain_metadata->>'source', langchain_metadata->>'chunk', length(content) FROM %s LIMIT 4" % table)
    result = {
        'table': table, 'documents': len(names), 'expected_chunks': sum(expected.values()),
        'rows_total': total,
        'rows_matching_an_expected_chunk': sum(found.values()), 'rows_other': total - sum(found.values()),
        'rows_other_with_text_and_vector': other,
        'expected_chunks_missing': missing, 'expected_chunks_stored_too_often': surplus, 'copies_of_each_expected_chunk': dict(copies),
        'vectors_compared': compared, 'vectors_outside_tolerance': outside, 'max_abs_difference': worst,
        'tolerance': {'atol': ATOL, 'rtol': RTOL},
        'database': {'server_version': psql('SHOW server_version').strip(),
                     'vector_extension': psql("SELECT extversion FROM pg_extension WHERE extname = 'vector'").strip(),
                     'indexes': psql("SELECT indexname FROM pg_indexes WHERE tablename = '%s' ORDER BY 1" % table).split(),
                     'table_bytes': int(psql("SELECT pg_total_relation_size('%s')" % table)),
                     'heap_bytes': int(psql("SELECT pg_relation_size('%s')" % table)),
                     'columns': psql("SELECT column_name || ' ' || data_type FROM information_schema.columns "
                                     "WHERE table_name = '%s' ORDER BY ordinal_position" % table).splitlines()},
        # Since the host cleared the statistics just before the timed requests.
        'during_timed_requests': {
            'statement_statistics': has_stats,
            'by_kind': {k: {'calls': v['calls'], 'ms': round(v['ms'], 1), 'rows': v['rows']} for k, v in sorted(by_kind.items())},
            'database_ms_total': round(sum(v['ms'] for v in by_kind.values()), 1),
            'top_statements': statements,
            'wal_records': int(wal[0]) - before.get('wal_records', 0), 'wal_bytes': int(wal[1]) - before.get('wal_bytes', 0),
            # These two counters are reported by Postgres with a delay, so they are a lower bound.
            'commits_counter': int(counters[0]) - before.get('xact_commit', 0),
            'rollbacks_counter': int(counters[1]) - before.get('xact_rollback', 0)},
        'prepared': before,
        'distinct_documents_or_ids_and_rows_without_vector': keys.strip().split('\t'),
        'sample_rows_without_text': sample.splitlines(),
        'reference': str(ref),
        'pass': missing == 0 and outside == 0 and compared == sum(found.values()) and surplus == 0,
    }
    (out / 'db-check.json').write_text(json.dumps(result, indent=2) + '\n')
    print('DBCHECK', json.dumps({k: result[k] for k in ['table', 'expected_chunks', 'rows_total', 'rows_other',
                                                         'expected_chunks_missing', 'expected_chunks_stored_too_often',
                                                         'copies_of_each_expected_chunk', 'vectors_outside_tolerance', 'pass']}))


if __name__ == '__main__':
    main()
