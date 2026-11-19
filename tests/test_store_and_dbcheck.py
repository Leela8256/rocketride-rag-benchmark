"""Test of the store code, of the step that empties the table of warm-up rows, and of the read-back check, against
a real Postgres with pgvector. No Docker, no model.

The test follows the order of a run: warm-up, the step between the warm-up and the timed requests, the timed
requests, the read-back.

  1. The LangGraph store code (workload/document/store_pg.py) writes through LangChain's PGVectorStore, from 32
     threads at once, the way the service's thread pool calls it. A table with RocketRide's own definition is
     filled the way its node writes (delete the document's rows, one INSERT per chunk, one commit per document),
     with the one row the store keeps for itself.
  2. After the warm-up, campaign.db_prepare() removes the warm-up rows and LangGraph's start-up fixture, keeps
     RocketRide's own row, and refuses to go on when a row it cannot explain is left.
  3. After the measured documents are stored, dbcheck.py passes on both tables, and fails when a chunk is
     missing, stored twice, its vector is changed, its text sits under another PDF or another chunk number, a row
     it cannot explain is in the table (with or without a vector), a document was committed after the client had
     its answer, a row was committed after the client's clock stopped, or the commit times could not be checked
     although the plan asks for them. The store's own row may change after the clock stopped. An empty table
     fails the check without breaking it.
  4. The two functions that show whether Postgres uses the vector index: the index scan counter, and the plan of
     a search statement as the database records it (with placeholders).

Needs: a scratch Postgres on 127.0.0.1:54329, started with `-c track_commit_timestamp=on`, with database ragbench
and the vector extension, user postgres without a password (trust authentication); a Python with
langchain-postgres and psycopg (see langgraph/requirements.txt). The embedding model is not loaded: vectors are
made up. The test drops and creates the tables lc_chunks, rr_chunks and keep in that database.

Usage: <python> tests/test_store_and_dbcheck.py [psql binary]
"""
import array
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
APP, PSQL = HERE / 'langgraph/app', (sys.argv[1] if len(sys.argv) > 1 else 'psql')
PREFIX = [PSQL, '-h', '127.0.0.1', '-p', '54329', '-U', 'postgres', '-d', 'ragbench']
DIM = 384
RR_DDL = ('CREATE TABLE IF NOT EXISTS rr_chunks (id bigserial PRIMARY KEY, content text, objectId text, nodeId text, '
          'parent text, permissionId int, isDeleted boolean, chunkId int, isTable boolean, tableId int, vectorSize int, '
          'modelName text, embedding vector(384));')
OWN, FIXTURE = 'schema', 'warmup.pdf'


def psql(sql):
    return subprocess.run(PREFIX + ['-At', '-F', '\t', '-v', 'ON_ERROR_STOP=1', '-c', sql], check=True, capture_output=True, text=True).stdout


def f32(values):
    return list(array.array('f', values))


def main():
    assert psql('SHOW track_commit_timestamp').strip() == 'on', 'start the scratch Postgres with -c track_commit_timestamp=on'
    rng = random.Random(7)
    docs = {}
    for d in range(12):
        chunks = []
        for i in range(rng.randint(1, 30)):
            text = 'doc %d chunk %d %s\twith a tab, a "quote" and a \\ backslash\nand a new line' % (d, i, 'x' * rng.randint(0, 3000))
            chunks.append({'index': i, 'text': text, 'vector': f32([rng.uniform(-1, 1) for _ in range(DIM)])})
        docs['doc-%02d.pdf' % d] = chunks
    docs['doc-11.pdf'].append(dict(docs['doc-11.pdf'][0], index=len(docs['doc-11.pdf'])))   # the same text twice in one document
    # The run covers 10 documents; the other 2 warm it up. One of the 10 holds the same text twice.
    measured, warm = sorted(docs)[2:], sorted(docs)[:2]
    fixture = [{'index': 0, 'text': 'the text of the start-up fixture', 'vector': f32([rng.uniform(-1, 1) for _ in range(DIM)])}]
    expected = sum(len(docs[n]) for n in measured)

    for table in ['lc_chunks', 'rr_chunks', 'keep']:
        psql('DROP TABLE IF EXISTS %s' % table)

    os.environ.update(RAG_PG_HOST='127.0.0.1', RAG_PG_PORT='54329', RAG_PG_USER='postgres', RAG_PG_PASSWORD='unused',
                      RAG_PG_DATABASE='ragbench')
    sys.path.insert(0, str(APP))
    sys.path.insert(0, str(HERE / 'harness'))
    import campaign
    import psycopg
    from workload.document import store_pg

    def lg_store(names):
        """LangGraph side: the real store code, 32 threads."""
        def one(name):
            chunks = fixture if name == FIXTURE else docs[name]
            return store_pg.store_chunks(name, [{'index': c['index'], 'text': c['text']} for c in chunks], [c['vector'] for c in chunks])
        with ThreadPoolExecutor(32) as pool:
            return list(pool.map(one, names))

    def rr_store(names):
        """RocketRide's table, written the way its node writes."""
        with psycopg.connect('host=127.0.0.1 port=54329 user=postgres dbname=ragbench') as conn:
            with conn.cursor() as cur:
                for name in names:
                    cur.execute('DELETE FROM rr_chunks WHERE objectId = ANY(%s)', (['id-of-' + name],))
                    for c in docs[name]:
                        cur.execute('INSERT INTO rr_chunks (content, objectId, parent, chunkId, vectorSize, modelName, embedding) '
                                    'VALUES (%s, %s, %s, %s, %s, %s, %s::vector)',
                                    (c['text'], 'id-of-' + name, name, c['index'], DIM, 'm', str(c['vector'])))
                    conn.commit()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'source').mkdir()
        ref = root / 'cells/screen-rr'
        ref.mkdir(parents=True)
        vectors, lines, offset = array.array('f'), [], 0
        for name in sorted(docs):
            chunks = docs[name]
            lines.append(json.dumps({'doc': name, 'chunks': len(chunks), 'vector_offset': offset,
                                     'text_sha256': [hashlib.sha256(c['text'].encode()).hexdigest() for c in chunks]}))
            for c in chunks:
                vectors.extend(c['vector'])
            offset += len(chunks)
        (ref / 'per_doc.jsonl').write_text('\n'.join(lines) + '\n')
        with (ref / 'vectors.f32').open('wb') as f:
            vectors.tofile(f)
        plan = {'db': {'rr_table': 'rr_chunks', 'lg_table': 'lc_chunks', 'user': 'postgres', 'database': 'ragbench',
                       'args': ['-c', 'track_commit_timestamp=on']},
                'rr': {'own_object_id': OWN}, 'lg': {'fixture_documents': [FIXTURE]}, 'reference_cell': 'cells/screen-rr'}
        (root / 'source/corpus-manifest.json').write_text(json.dumps({'measured': measured, 'warm': warm}))
        (root / 'plan.json').write_text(json.dumps(plan))
        campaign.PLAN, campaign.R = plan, root

        def cell_of(arm):
            cell = root / 'cells' / (arm + '-test')
            cell.mkdir(parents=True, exist_ok=True)
            (cell / 'config.json').write_text(json.dumps({'label': arm + '-test', 'arm': arm, 'documents': len(measured)}))
            return cell

        def check(arm, clock_stop=None, answered=None, answer_times=True):
            """dbcheck on the arm's table. The client's clock stops now, and it has every document's answer now,
            unless other instants are given. `answered` maps a document to the moment the client had its answer."""
            cell = cell_of(arm)
            now = time.time()
            (cell / 'measurement-end.json').write_text(json.dumps({'clock_stop_epoch': now if clock_stop is None else clock_stop}))
            (cell / 'per_doc.jsonl').unlink(missing_ok=True)
            if answer_times:
                (cell / 'per_doc.jsonl').write_text(''.join(json.dumps(
                    {'doc': n, 'start_epoch': (answered or {}).get(n, now) - 2.0, 'latency_s': 2.0}) + '\n' for n in measured))
            subprocess.run([sys.executable, str(HERE / 'harness/dbcheck.py'), str(cell)] + PREFIX, check=True, capture_output=True)
            return json.loads((cell / 'db-check.json').read_text())

        def sql(text):
            # The scratch server does not load pg_stat_statements; the step's own reset is not what is tested here.
            return '' if 'pg_stat_statements_reset' in text else psql(text).strip()

        # 1. The warm-up: the service stores its fixture when it starts, then the warm-up documents arrive.
        assert not store_pg.is_loaded()
        store_pg.load_store()
        assert lg_store([FIXTURE] + warm) == [1] + [len(docs[n]) for n in warm]
        info = store_pg.store_info()
        assert 'unused' not in json.dumps(info) and info['loaded'], info
        try:
            store_pg._engine.init_vectorstore_table(table_name=store_pg.TABLE, vector_size=DIM)
            raise SystemExit('a second table creation must fail')
        except SystemExit:
            raise
        except Exception as e:
            second = type(e).__name__
        psql(RR_DDL)
        psql("INSERT INTO rr_chunks (content, objectId, parent, vectorSize, modelName, embedding) "
             "VALUES ('', '%s', '', %d, 'm', '%s'::vector)" % (OWN, DIM, str([0.0] * DIM)))
        rr_store(warm)

        # 2. The campaign's step between the warm-up and the timed requests.
        stray = {'rr': "INSERT INTO rr_chunks (content, objectId, parent, embedding) SELECT 'a row nobody can explain', 'stray', 'stray.pdf', embedding FROM rr_chunks LIMIT 1",
                 'lg': "INSERT INTO lc_chunks (langchain_id, content, embedding, langchain_metadata) SELECT gen_random_uuid(), "
                       "'a row nobody can explain', embedding, '{\"source\": \"stray.pdf\"}' FROM lc_chunks LIMIT 1"}
        unstray = {'rr': "DELETE FROM rr_chunks WHERE parent = 'stray.pdf'", 'lg': "DELETE FROM lc_chunks WHERE langchain_metadata->>'source' = 'stray.pdf'"}
        warm_rows = sum(len(docs[n]) for n in warm)
        for arm, table, own in [('lg', 'lc_chunks', 0), ('rr', 'rr_chunks', 1)]:
            cfg = {'arm': arm, 'label': arm + '-test'}
            assert int(psql('SELECT count(*) FROM %s' % table)) == warm_rows + (own if arm == 'rr' else 1)
            psql(stray[arm])
            try:
                campaign.db_prepare(cfg, cell_of(arm), sql)
                raise SystemExit('a row that cannot be explained must stop the run before the timed requests')
            except AssertionError as e:
                assert 'neither warm-up rows' in str(e) and 'stray.pdf' in str(e), e
            psql(unstray[arm])
            lg_store([FIXTURE] + warm) if arm == 'lg' else rr_store(warm)     # the failed attempt had removed them
            campaign.db_prepare(cfg, cell_of(arm), sql)
            prepared = json.loads((cell_of(arm) / 'db-prepare.json').read_text())
            assert prepared['rows_before_timed_requests'] == own == int(psql('SELECT count(*) FROM %s' % table)), prepared
            assert prepared['rows_after_warm_up'] == warm_rows + (own if arm == 'rr' else 1), prepared
        assert psql("SELECT objectid FROM rr_chunks").strip() == OWN, 'the store\'s own row must stay'

        # 3. The timed requests, then the read-back: it passes, and fails for each thing that can be wrong.
        before_the_rows = time.time()
        counts = lg_store(measured)
        assert counts == [len(docs[n]) for n in measured], counts
        rr_store(measured)
        for arm in ['lg', 'rr']:
            r = check(arm)
            assert r['pass'] and r['expected_chunks'] == expected == r['rows_matching_an_expected_chunk'] == r['vectors_compared'], r
            assert r['rows_unexplained'] == 0 and r['rows_of_the_store_itself'] == (1 if arm == 'rr' else 0), r
            assert r['rows_committed_after_the_clock_stopped'] == 0 and r['last_row_commit_epoch'] <= r['clock_stop_epoch'], r
            assert r['vectors_outside_tolerance'] == 0 and r['max_abs_difference'] < 1e-6, r
            assert r['commit_times_checked'] and r['documents_compared_with_their_answer'] == len(measured) == r['documents_with_chunks'], r
            assert r['rows_with_an_expected_text_in_the_wrong_place'] == 0, r
            assert r['documents_committed_after_their_answer'] == 0, r
            # A clock that stopped before the rows were committed.
            r = check(arm, before_the_rows)
            assert not r['pass'] and r['rows_committed_after_the_clock_stopped'] == expected and r['rows_unexplained'] == 0, r
            assert r['documents_committed_after_their_answer'] == 0, r
            # One document whose answer the client had before its rows were committed, inside the timed window.
            r = check(arm, answered={measured[3]: before_the_rows})
            assert not r['pass'] and r['documents_committed_after_their_answer'] == 1 and r['rows_committed_after_the_clock_stopped'] == 0, r
            assert r['first_documents_committed_after_their_answer'] == [measured[3]], r
            # The plan asks for commit times; a run whose client recorded no answer times cannot be checked.
            r = check(arm, answer_times=False)
            assert not r['pass'] and not r['commit_times_checked'] and r['commit_times_required'], r

        # The store may touch its own row after the clock stopped (when its task ends); that is not a late row.
        stopped = time.time()
        time.sleep(0.05)
        psql("UPDATE rr_chunks SET modelName = 'touched' WHERE objectId = '%s'" % OWN)
        r = check('rr', stopped)
        assert r['pass'] and r['rows_committed_after_the_clock_stopped'] == 0 and r['last_row_commit_epoch'] < stopped, r
        # A second row under the store's own id is not explained by "the store keeps one row".
        psql("INSERT INTO rr_chunks (content, objectId, parent) VALUES ('', '%s', '')" % OWN)
        r = check('rr')
        assert not r['pass'] and r['rows_of_the_store_itself'] == 2, r
        psql("DELETE FROM rr_chunks WHERE objectId = '%s' AND embedding IS NULL" % OWN)
        # A row without a vector is not in the dump of texts and vectors; it is unexplained all the same.
        psql("INSERT INTO rr_chunks (content, objectId, parent) VALUES ('a text without a vector', 'bare', 'bare.pdf')")
        r = check('rr')
        assert not r['pass'] and r['unexplained_rows_by_document'] == {'(rows without a text or a vector)': 1}, r
        psql("DELETE FROM rr_chunks WHERE objectId = 'bare'")
        assert check('rr')['pass']

        # 4. Is the vector index used? The counter of index scans, and the plan of a statement with placeholders.
        psql('CREATE INDEX lc_chunks_hnsw ON lc_chunks USING hnsw (embedding vector_cosine_ops)')
        lg_cell = {'arm': 'lg', 'label': 'lg-test'}
        before = campaign.index_scans(lg_cell, sql)
        vector = str(docs[measured[0]][0]['vector'])
        psql("SET enable_seqscan = off; SELECT langchain_id FROM lc_chunks ORDER BY embedding <=> '%s' LIMIT 5" % vector)
        assert campaign.index_scans(lg_cell, sql, at_least=before + 1, wait_s=10) == before + 1, 'one search through the index is one index scan'
        psql("SET enable_indexscan = off; SELECT langchain_id FROM lc_chunks ORDER BY embedding <=> '%s' LIMIT 5" % vector)
        assert campaign.index_scans(lg_cell, sql, wait_s=0) == before + 1, 'a search that does not use the index adds no index scan'
        recorded = 'SELECT *, embedding <=> $1 AS distance FROM lc_chunks ORDER BY distance LIMIT $2;'
        plan = campaign.search_plan(lg_cell, lambda text: recorded if 'pg_stat_statements' in text else sql(text))
        assert plan['plan'] and plan['query'] == recorded and isinstance(plan['names_the_vector_index'], bool), plan
        assert campaign.search_plan(lg_cell, sql)['plan'] is None, 'without recorded statements there is no plan, and no failure'
        psql('DROP INDEX lc_chunks_hnsw')

        assert campaign.index_use(1000, 0) is True and campaign.index_use(0, 0) is False, 'scans decide'
        assert campaign.index_use(0, 2) is None, 'no scans counted while a session is inside a transaction: not determined'

        # Each system is checked against its own numbering of the chunks. Here the reference run numbers them from
        # 1 and LangGraph's own screening run from 0, as its table does: LangGraph passes, and the RocketRide table
        # (numbered from 0) no longer fits its own reference and fails.
        ref_lines = (ref / 'per_doc.jsonl').read_text()
        rows = [json.loads(x) for x in ref_lines.splitlines()]
        (ref / 'per_doc.jsonl').write_text(''.join(json.dumps(dict(r, chunk_numbers=[i + 1 for i in range(r['chunks'])])) + '\n' for r in rows))
        (root / 'cells/screen-lg').mkdir()
        (root / 'cells/screen-lg/per_doc.jsonl').write_text(''.join(json.dumps(dict(r, chunk_numbers=list(range(r['chunks'])))) + '\n' for r in rows))
        r = check('lg')
        assert r['pass'] and r['rows_with_an_expected_text_in_the_wrong_place'] == 0, ('own numbering', r)
        r = check('rr')
        assert not r['pass'] and r['rows_with_an_expected_text_in_the_wrong_place'] == expected, ('own numbering', r)
        (ref / 'per_doc.jsonl').write_text(ref_lines)
        (root / 'cells/screen-lg/per_doc.jsonl').unlink()

        victim = hashlib.sha256(docs[measured[0]][0]['text'].encode()).hexdigest()
        where = "encode(sha256(convert_to(content, 'UTF8')), 'hex') = '%s'" % victim
        for arm, table in [('lg', 'lc_chunks'), ('rr', 'rr_chunks')]:
            psql('CREATE TABLE keep AS SELECT * FROM %s' % table)
            psql("UPDATE %s SET embedding = (SELECT embedding FROM %s WHERE NOT (%s) AND content <> '' LIMIT 1) WHERE %s" % (table, table, where, where))
            r = check(arm)
            assert not r['pass'] and r['vectors_outside_tolerance'] == 1 and r['rows_unexplained'] == 0, ('changed vector', r)
            psql('DELETE FROM %s WHERE %s' % (table, where))
            r = check(arm)
            assert not r['pass'] and r['expected_chunks_missing'] == 1, ('missing chunk', r)
            psql('TRUNCATE %s' % table)
            psql('INSERT INTO %s SELECT * FROM keep' % table)
            assert check(arm)['pass'], 'the restored table must pass again'
            # The same chunk a second time: same PDF, same chunk number, same text.
            cols = 'content, embedding, ' + ('langchain_metadata, langchain_id' if arm == 'lg' else 'objectId, parent, chunkId')
            vals = 'content, embedding, ' + ('langchain_metadata, gen_random_uuid()' if arm == 'lg' else 'objectId, parent, chunkId')
            psql('INSERT INTO %s (%s) SELECT %s FROM keep WHERE %s' % (table, cols, vals, where))
            r = check(arm)
            assert not r['pass'] and r['expected_chunks_stored_too_often'] == 1 and r['rows_with_an_expected_text_in_the_wrong_place'] == 0, ('stored twice', r)

            def restore():
                psql('TRUNCATE %s' % table)
                psql('INSERT INTO %s SELECT * FROM keep' % table)
            # One more row with an expected text, under no PDF: it is not that chunk.
            restore()
            psql('INSERT INTO %s (content, embedding%s) SELECT content, embedding%s FROM keep WHERE %s' % (
                table, ', langchain_id' if arm == 'lg' else '', ', gen_random_uuid()' if arm == 'lg' else '', where))
            r = check(arm)
            assert not r['pass'] and r['rows_with_an_expected_text_in_the_wrong_place'] == 1 and r['expected_chunks_stored_too_often'] == 0, ('misfiled row', r)
            # Every row keeps its text and its vector but is filed under another PDF.
            restore()
            psql("UPDATE lc_chunks SET langchain_metadata = jsonb_set(langchain_metadata::jsonb, '{source}', '\"unrelated.pdf\"')::json" if arm == 'lg'
                 else "UPDATE rr_chunks SET parent = 'unrelated.pdf' WHERE objectId <> '%s'" % OWN)
            r = check(arm)
            assert not r['pass'] and r['rows_with_an_expected_text_in_the_wrong_place'] == expected == r['expected_chunks_missing'], ('wrong PDF', r)
            assert not r['commit_times_checked'] and r['documents_compared_with_their_answer'] == 0, ('wrong PDF', r)
            assert r['misfiled_rows_by_document'] == {'unrelated.pdf': expected}, r['misfiled_rows_by_document']
            # Every row under its own PDF, with another chunk number.
            restore()
            psql("UPDATE lc_chunks SET langchain_metadata = jsonb_set(langchain_metadata::jsonb, '{chunk}', "
                 "to_jsonb((langchain_metadata->>'chunk')::int + 1000))::json" if arm == 'lg'
                 else "UPDATE rr_chunks SET chunkId = chunkId + 1000 WHERE objectId <> '%s'" % OWN)
            r = check(arm)
            assert not r['pass'] and r['rows_with_an_expected_text_in_the_wrong_place'] == expected == r['expected_chunks_missing'], ('wrong chunk number', r)
            # One PDF whose rows are there but whose answer time the client did not record: not every PDF was compared.
            restore()
            r = check(arm, answered=None, answer_times=True)
            assert r['pass'], r
            cell = cell_of(arm)
            kept_rows = [x for x in (cell / 'per_doc.jsonl').read_text().splitlines() if json.loads(x)['doc'] != measured[5]]
            (cell / 'per_doc.jsonl').write_text('\n'.join(kept_rows) + '\n')
            subprocess.run([sys.executable, str(HERE / 'harness/dbcheck.py'), str(cell)] + PREFIX, check=True, capture_output=True)
            r = json.loads((cell / 'db-check.json').read_text())
            assert not r['pass'] and not r['commit_times_checked'] and r['documents_compared_with_their_answer'] == len(measured) - 1, ('one PDF not compared', r)
            psql('TRUNCATE %s' % table)
            psql('INSERT INTO %s SELECT * FROM keep' % table)
            psql(stray[arm])
            r = check(arm)
            assert not r['pass'] and r['rows_unexplained'] == 1 and r['unexplained_rows_by_document'] == {'stray.pdf': 1}, ('unexplained row', r)
            assert r['expected_chunks_missing'] == 0 and r['expected_chunks_stored_too_often'] == 0, r
            # A table with nothing in it but the store's own row: the check fails, it does not break.
            psql('TRUNCATE %s' % table)
            psql("INSERT INTO %s SELECT * FROM keep WHERE content = ''" % table)
            r = check(arm)
            assert not r['pass'] and r['expected_chunks_missing'] == expected and r['rows_total'] == (1 if arm == 'rr' else 0), ('empty table', r)
            psql('DROP TABLE keep')
    print('ok  LangChain store: %d measured documents, %d rows, 32 threads; second table creation raises %s' % (len(measured), sum(counts), second))
    print('ok  the step before the timed requests removes the %d warm-up rows and the fixture, keeps the store\'s own row, '
          'and stops on a row it cannot explain' % warm_rows)
    print('ok  dbcheck passes on both tables (%d expected chunks) and fails on a changed vector, a missing chunk, a chunk '
          'stored twice, a text under another PDF or chunk number, an unexplained row with or without a vector, a document '
          'committed after its answer, a row committed after the clock stopped, commit times that could not be checked for '
          'every PDF, and an empty table' % expected)
    print('ok  the index scan counter rises by one for a search through the index and not at all for a search past it; '
          'a recorded statement with placeholders gets its plan')
    print('ok  store report:', json.dumps({k: info[k] for k in ['class', 'write', 'table', 'driver', 'pool']}))
    print('ALL OK')


if __name__ == '__main__':
    main()
