"""Read back what a run stored in Postgres, check it against the reference run, and record what the database
did during the timed requests.

Runs on the host after the timed requests of a cell, while the database container is still up. Standard library
only.

The check. The database itself computes the SHA-256 of every stored chunk text and returns every stored vector.
They are compared with the chunk texts and vectors of the same documents in the reference run, which is the
RocketRide screening run of the campaign (per_doc.jsonl and vectors.f32 of that cell):

  every expected chunk is in the table under its own PDF and its own chunk number. A chunk is the three together:
  the PDF it belongs to, its number inside that PDF, and its text. A row with the right text under another PDF or
  another number is not that chunk: it is counted as misfiled and fails the check;
  no expected chunk is in the table more often than expected;
  every stored vector equals the reference vector of its chunk within the tolerance the embedding runs use;
  the table holds no other row, except the one row RocketRide's store keeps for itself (the plan names it). The
  campaign removes the warm-up documents and LangGraph's start-up fixture before the timed requests, so any other
  row is unexplained and fails the check: a search could return it;
  no document has a row that was committed after the client had that document's answer, and no row was
  committed after the client's clock stopped. Every PDF that has chunks must be covered by that comparison: a PDF
  whose rows or whose answer time cannot be found makes the check fail. The database keeps the commit time of every row
  (track_commit_timestamp) and the client records when it sent every document and how long the answer took. So
  an answer came only after its document was stored, and the timed window holds all of the storing.

Both systems are checked against the same reference, so two passing cells stored the same chunks and vectors, and
nothing else that a search could return.

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
LATE_S = 0.001      # allowance when a commit time of the database is compared with a time taken by the client


def read(path):
    return json.loads(Path(path).read_text())


def quoted(text):
    return "'" + str(text).replace("'", "''") + "'"


def unhex(text):
    """A document name the database sent as hex, so that no character of a file name can break a line."""
    return bytes.fromhex(text).decode('utf-8', 'replace')


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
    # An expected chunk is (PDF, chunk number, hash of the text). Texts and vectors come from the reference run.
    # The number is the one this system itself gave the chunk in its own screening run: the two systems need not
    # number their chunks the same way, and a system is checked against its own numbering. The screening has
    # already shown that both return the same texts in the same order, so position i is the same chunk in both.
    # Without that file (or in one written before numbers were recorded) the position in the PDF is the number.
    own_numbers, own_screening = {}, root / 'cells' / ('screen-%s' % arm) / 'per_doc.jsonl'
    if own_screening.exists():
        for line in own_screening.read_text().splitlines():
            row = json.loads(line)
            if row['doc'] in wanted and row.get('chunk_numbers') is not None:
                own_numbers[row['doc']] = row['chunk_numbers']
    expected, ref_vector, expected_texts, seen = collections.Counter(), {}, set(), set()
    for line in (ref / 'per_doc.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['doc'] not in wanted:
            continue
        seen.add(row['doc'])
        numbers = own_numbers.get(row['doc'], row.get('chunk_numbers')) or list(range(len(row['text_sha256'])))
        assert len(numbers) == len(row['text_sha256']), 'another number of chunk numbers than chunks: %s' % row['doc']
        for i, digest in enumerate(row['text_sha256']):
            key = (row['doc'], numbers[i], digest)
            expected[key] += 1
            expected_texts.add(digest)
            start = (row['vector_offset'] + i) * DIM
            ref_vector.setdefault(key, vectors[start:start + DIM])
    assert seen == wanted, 'the reference run does not cover every document of this cell'
    documents_with_chunks = {key[0] for key in expected}

    # Which rows are the store's own, and which document a row belongs to.
    own_id = plan['rr'].get('own_object_id') if arm == 'rr' else None
    own = 'objectid IS NOT DISTINCT FROM %s' % quoted(own_id) if own_id else 'false'
    document = "encode(convert_to(coalesce(%s, ''), 'UTF8'), 'hex')" % ('parent' if arm == 'rr' else "langchain_metadata->>'source'")
    number = "coalesce(%s, '')" % ('chunkid::text' if arm == 'rr' else "langchain_metadata->>'chunk'")
    total = int(psql('SELECT count(*) FROM %s' % table))
    own_total = int(psql('SELECT count(*) FROM %s WHERE %s' % (table, own)))
    dump = psql("COPY (SELECT encode(sha256(convert_to(content, 'UTF8')), 'hex'), (%s)::int, %s, %s, embedding::text FROM %s "
                "WHERE content IS NOT NULL AND embedding IS NOT NULL) TO STDOUT" % (own, document, number, table))
    found, unexplained, misfiled = collections.Counter(), collections.Counter(), collections.Counter()
    worst, outside, compared, own_dumped, dumped = 0.0, 0, 0, 0, 0
    for line in dump.split('\n'):
        if not line:
            continue
        digest, is_own, name, chunk_number, text = line.split('\t', 4)
        name = unhex(name)
        dumped += 1
        if is_own == '1':
            own_dumped += 1
            continue
        key = (name, int(chunk_number) if chunk_number.lstrip('-').isdigit() else None, digest)
        if key not in expected:
            # The text of an expected chunk under another PDF or another chunk number is not that chunk.
            (misfiled if digest in expected_texts else unexplained)[name or '(no document name)'] += 1
            continue
        found[key] += 1
        values = [float(x) for x in text.strip('[]').split(',')]
        assert len(values) == DIM, 'a stored vector has %d values' % len(values)
        compared += 1
        bad = False
        for a, b in zip(values, ref_vector[key]):
            d = abs(a - b)
            if d > worst:
                worst = d
            if d > ATOL + RTOL * abs(b):
                bad = True
        outside += bad
    missing = sum((expected - found).values())
    surplus = sum((found - expected).values())
    # Rows without a text or a vector are not in the dump. Those that are not the store's own are unexplained too.
    bare = (total - dumped) - (own_total - own_dumped)
    if bare:
        unexplained['(rows without a text or a vector)'] += bare
    rows_unexplained = sum(unexplained.values())
    rows_misfiled = sum(misfiled.values())

    # Was every document stored before the client had its answer, and every row before the client's clock stopped?
    # The store's own row is left out: a store may touch it when its task ends.
    stop = read(out / 'measurement-end.json').get('clock_stop_epoch') if (out / 'measurement-end.json').exists() else None
    answered = {}                # document -> the moment the client had its answer
    if (out / 'per_doc.jsonl').exists():
        for line in (out / 'per_doc.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row.get('start_epoch') is not None and row.get('latency_s') is not None:
                answered[row['doc']] = row['start_epoch'] + row['latency_s']
    required = 'track_commit_timestamp=on' in db.get('args', [])
    late = late_documents = last_commit = None
    compared_documents = 0
    if psql('SHOW track_commit_timestamp').strip() == 'on':
        raw = psql("COPY (SELECT %s, extract(epoch FROM max(pg_xact_commit_timestamp(xmin))) FROM %s WHERE NOT (%s) "
                   "GROUP BY 1) TO STDOUT" % (document, table, own))
        commits = {unhex(name): float(last) for name, last in (line.split('\t') for line in raw.split('\n') if line) if last != '\\N'}
        last_commit = max(commits.values()) if commits else None
        if stop is not None:
            late = int(psql('SELECT count(*) FROM %s WHERE NOT (%s) AND pg_xact_commit_timestamp(xmin) > to_timestamp(%.6f)' % (
                table, own, stop + LATE_S)))
        if answered:
            covered = [name for name in documents_with_chunks if name in commits and name in answered]
            compared_documents = len(covered)
            late_documents = sorted(name for name in covered if commits[name] > answered[name] + LATE_S)
    # Checked means: for every PDF that has chunks, its rows were found under its name and compared with its answer.
    commit_times_checked = (late is not None and late_documents is not None
                            and compared_documents == len(documents_with_chunks))
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
        'rows_of_the_store_itself': own_total,
        'rows_unexplained': rows_unexplained,
        'unexplained_rows_by_document': dict(unexplained.most_common(10)),
        # Rows that hold the text of an expected chunk under another PDF or another chunk number.
        'rows_with_an_expected_text_in_the_wrong_place': rows_misfiled,
        'misfiled_rows_by_document': dict(misfiled.most_common(10)),
        # None when it could not be checked: the client recorded no times, or the database does not keep commit
        # times. A plan that starts the database with track_commit_timestamp=on requires the check.
        'commit_times_checked': commit_times_checked, 'commit_times_required': required,
        'rows_committed_after_the_clock_stopped': late,
        'documents_with_chunks': len(documents_with_chunks),
        'documents_compared_with_their_answer': compared_documents,
        'documents_committed_after_their_answer': None if late_documents is None else len(late_documents),
        'first_documents_committed_after_their_answer': (late_documents or [])[:10],
        'clock_stop_epoch': stop, 'last_row_commit_epoch': last_commit, 'allowance_s': LATE_S,
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
        'pass': (missing == 0 and outside == 0 and compared == sum(found.values()) and surplus == 0
                 and rows_unexplained == 0 and rows_misfiled == 0 and own_total <= 1 and not late and not late_documents
                 and (commit_times_checked or not required)),
    }
    (out / 'db-check.json').write_text(json.dumps(result, indent=2) + '\n')
    print('DBCHECK', json.dumps({k: result[k] for k in ['table', 'expected_chunks', 'rows_total', 'rows_of_the_store_itself',
                                                         'rows_unexplained', 'rows_with_an_expected_text_in_the_wrong_place',
                                                         'expected_chunks_missing',
                                                         'expected_chunks_stored_too_often', 'copies_of_each_expected_chunk',
                                                         'vectors_outside_tolerance', 'commit_times_checked',
                                                         'rows_committed_after_the_clock_stopped',
                                                         'documents_committed_after_their_answer', 'pass']}))


if __name__ == '__main__':
    main()
