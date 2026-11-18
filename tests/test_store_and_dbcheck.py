"""Test of the store code and of the read-back check against a real Postgres with pgvector. No Docker, no model.

  1. The LangGraph store code (workload/document/store_pg.py) writes through LangChain's PGVectorStore, from 32
     threads at once, the way the service's thread pool calls it.
  2. A table with RocketRide's own definition is filled the way its node writes (delete the document's rows, one
     INSERT per chunk, one commit per document).
  3. dbcheck.py passes on both tables against a reference, and fails when a chunk is missing, stored twice or its
     vector is changed.

Needs: a scratch Postgres on 127.0.0.1:54329 with database ragbench and the vector extension, user postgres
without a password (trust authentication); a Python with langchain-postgres and psycopg (see
langgraph/requirements.txt). The embedding model is not loaded: vectors are made up. The test drops and creates
the tables lc_chunks, rr_chunks and keep in that database.

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
APP, PSQL = HERE / 'langgraph/app', (sys.argv[1] if len(sys.argv) > 1 else 'psql')
PREFIX = [PSQL, '-h', '127.0.0.1', '-p', '54329', '-U', 'postgres', '-d', 'ragbench']
DIM = 384
RR_DDL = ('CREATE TABLE IF NOT EXISTS rr_chunks (id bigserial PRIMARY KEY, content text, objectId text, nodeId text, '
          'parent text, permissionId int, isDeleted boolean, chunkId int, isTable boolean, tableId int, vectorSize int, '
          'modelName text, embedding vector(384));')


def psql(sql):
    return subprocess.run(PREFIX + ['-At', '-v', 'ON_ERROR_STOP=1', '-c', sql], check=True, capture_output=True, text=True).stdout


def f32(values):
    return list(array.array('f', values))


def main():
    rng = random.Random(7)
    docs = {}
    for d in range(12):
        chunks = []
        for i in range(rng.randint(1, 30)):
            text = 'doc %d chunk %d %s\twith a tab, a "quote" and a \\ backslash\nand a new line' % (d, i, 'x' * rng.randint(0, 3000))
            chunks.append({'index': i, 'text': text, 'vector': f32([rng.uniform(-1, 1) for _ in range(DIM)])})
        docs['doc-%02d.pdf' % d] = chunks
    docs['doc-11.pdf'].append(dict(docs['doc-11.pdf'][0], index=len(docs['doc-11.pdf'])))   # the same text twice in one document

    for table in ['lc_chunks', 'rr_chunks', 'keep']:
        psql('DROP TABLE IF EXISTS %s' % table)

    # 1. LangGraph side: the real store code, 32 threads.
    os.environ.update(RAG_PG_HOST='127.0.0.1', RAG_PG_PORT='54329', RAG_PG_USER='postgres', RAG_PG_PASSWORD='unused',
                      RAG_PG_DATABASE='ragbench')
    sys.path.insert(0, str(APP))
    from workload.document import store_pg
    assert not store_pg.is_loaded()
    store_pg.load_store()
    with ThreadPoolExecutor(32) as pool:
        counts = list(pool.map(lambda kv: store_pg.store_chunks(kv[0], [{'index': c['index'], 'text': c['text']} for c in kv[1]],
                                                                 [c['vector'] for c in kv[1]]), docs.items()))
    assert counts == [len(c) for c in docs.values()], counts
    info = store_pg.store_info()
    assert 'unused' not in json.dumps(info) and info['loaded'], info
    try:
        store_pg._engine.init_vectorstore_table(table_name=store_pg.TABLE, vector_size=DIM)
        raise SystemExit('a second table creation must fail')
    except SystemExit:
        raise
    except Exception as e:
        second = type(e).__name__

    # 2. RocketRide's table, written the way its node writes.
    import psycopg
    psql(RR_DDL)
    with psycopg.connect('host=127.0.0.1 port=54329 user=postgres dbname=ragbench') as conn:
        with conn.cursor() as cur:
            cur.execute('INSERT INTO rr_chunks (content, objectId, vectorSize, modelName) VALUES (%s, %s, %s, %s)', ('', 'control', DIM, 'm'))
            conn.commit()
            for name, chunks in docs.items():
                cur.execute('DELETE FROM rr_chunks WHERE objectId = ANY(%s)', ([name],))
                for c in chunks:
                    cur.execute('INSERT INTO rr_chunks (content, objectId, parent, chunkId, vectorSize, modelName, embedding) '
                                'VALUES (%s, %s, %s, %s, %s, %s, %s::vector)', (c['text'], name, '/' + name, c['index'], DIM, 'm', str(c['vector'])))
                conn.commit()

    # 3. dbcheck against a reference made from the same documents, plus rows it must ignore.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'source').mkdir()
        ref = root / 'cells/screen-rr'
        ref.mkdir(parents=True)
        measured = sorted(docs)[:10]                      # the cell covers 10 documents; the other 2 are "warm-up" rows
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
        (root / 'source/corpus-manifest.json').write_text(json.dumps({'measured': measured}))
        (root / 'plan.json').write_text(json.dumps({'db': {'rr_table': 'rr_chunks', 'lg_table': 'lc_chunks', 'user': 'postgres',
                                                          'database': 'ragbench'}, 'reference_cell': 'cells/screen-rr'}))

        def check(arm):
            cell = root / 'cells' / (arm + '-test')
            cell.mkdir(parents=True, exist_ok=True)
            (cell / 'config.json').write_text(json.dumps({'label': arm + '-test', 'arm': arm, 'documents': len(measured)}))
            subprocess.run([sys.executable, str(HERE / 'harness/dbcheck.py'), str(cell)] + PREFIX, check=True, capture_output=True)
            return json.loads((cell / 'db-check.json').read_text())

        expected = sum(len(docs[n]) for n in measured)
        others = sum(len(docs[n]) for n in sorted(docs)[10:])
        for arm in ['lg', 'rr']:
            r = check(arm)
            assert r['pass'] and r['expected_chunks'] == expected == r['rows_matching_an_expected_chunk'] == r['vectors_compared'], r
            assert r['rows_other'] == others + (1 if arm == 'rr' else 0), r
            assert r['vectors_outside_tolerance'] == 0 and r['max_abs_difference'] < 1e-6, r

        victim = hashlib.sha256(docs[measured[0]][0]['text'].encode()).hexdigest()
        where = "encode(sha256(convert_to(content, 'UTF8')), 'hex') = '%s'" % victim
        for arm, table, key in [('lg', 'lc_chunks', 'langchain_id'), ('rr', 'rr_chunks', 'id')]:
            psql('CREATE TABLE keep AS SELECT * FROM %s' % table)
            psql("UPDATE %s SET embedding = (SELECT embedding FROM %s WHERE NOT (%s) AND embedding IS NOT NULL LIMIT 1) WHERE %s" % (table, table, where, where))
            r = check(arm)
            assert not r['pass'] and r['vectors_outside_tolerance'] == 1, ('changed vector', r)
            psql('DELETE FROM %s WHERE %s' % (table, where))
            r = check(arm)
            assert not r['pass'] and r['expected_chunks_missing'] == 1, ('missing chunk', r)
            psql('TRUNCATE %s' % table)
            psql('INSERT INTO %s SELECT * FROM keep' % table)
            cols = 'content, embedding' + (', langchain_id' if arm == 'lg' else '')
            vals = 'content, embedding' + (', gen_random_uuid()' if arm == 'lg' else '')
            psql('INSERT INTO %s (%s) SELECT %s FROM keep WHERE %s' % (table, cols, vals, where))
            r = check(arm)
            assert not r['pass'] and r['expected_chunks_stored_too_often'] == 1, ('stored twice', r)
            psql('DROP TABLE keep')
    print('ok  LangChain store: %d documents, %d rows, 32 threads; second table creation raises %s' % (len(docs), sum(counts), second))
    print('ok  dbcheck passes on both tables (%d expected chunks, %d other rows) and fails on a changed vector, a missing chunk and a chunk stored twice' % (expected, others))
    print('ok  store report:', json.dumps({k: info[k] for k in ['class', 'write', 'table', 'driver', 'pool']}))
    print('ALL OK')


if __name__ == '__main__':
    main()
