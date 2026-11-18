"""Ingestion client for one cell: four warm-up PDFs, then the measured PDFs as rolling requests.

Runs in the client container on the client CPUs. A fixed number of PDFs is kept in flight: as soon as one
finishes, the next is sent. Timing covers only those requests. Validation and writing happen after the clock
has stopped.

  RocketRide   one task from use() on the ingestion pipe; every PDF is one pipe() object: open, write, close
  LangGraph    one HTTP request per PDF: POST /v1/process/<pipeline>

Usage: driver.py <cell label>
"""
import array
import asyncio
import collections
import hashlib
import json
import math
import os
import sys
import time
import traceback
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/campaign')
CORPUS = Path('/corpus')
PIPELINES = Path('/pipelines')      # rocketride/pipelines of this repository
DIM = 384
QUERIES = None          # a screening run collects questions here


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def find_documents(obj):
    """Documents list of a RocketRide result, or None when the lane is absent."""
    if isinstance(obj, dict):
        if isinstance(obj.get('documents'), list):
            return obj['documents']
        for key in ('result', 'data', 'output', 'json'):
            if key in obj:
                found = find_documents(obj[key])
                if found is not None:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_documents(item)
            if found is not None:
                return found
    return None


def chunks_and_vectors(arm, result):
    """Ordered chunk texts and their vectors, in the same shape for both arms."""
    if arm == 'rr':
        docs = find_documents(result)
        if docs is None:
            return [], [], False
        return [d['page_content'] for d in docs], [d['embedding'] for d in docs], True
    output = result['output']
    return [c['text'] for c in output['chunks']], output['vectors'], True


def percentile(values, p):
    values = sorted(values)
    point = (len(values) - 1) * p
    i = int(point)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (point - i)


def counters():
    groups = {}
    for p in sorted(Path('/groups').iterdir()):
        stat = dict(line.split() for line in (p / 'cpu.stat').read_text().splitlines())
        groups[p.name] = {'cpu_s': int(stat['usage_usec']) / 1e6,
                          'memory_current': int((p / 'memory.current').read_text())}
    assert groups, 'no service cgroups mounted'
    return {'epoch': time.time(), 'cpu_s': sum(g['cpu_s'] for g in groups.values()), 'groups': groups}


def lg_request(port, name, body, pipeline):
    boundary = 'ragbench-' + uuid.uuid4().hex
    payload = b''.join([
        ('--%s\r\nContent-Disposition: form-data; name="request_id"\r\n\r\n%s\r\n' % (boundary, name)).encode(),
        ('--%s\r\nContent-Disposition: form-data; name="file"; filename="%s"\r\n'
         'Content-Type: application/pdf\r\n\r\n' % (boundary, name)).encode(),
        body, ('\r\n--%s--\r\n' % boundary).encode()])
    request = urllib.request.Request(
        'http://127.0.0.1:%d/v1/process/%s' % (port, pipeline), data=payload,
        headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
    with urllib.request.urlopen(request, timeout=1800) as response:
        raw = response.read()
        meta = {'response_bytes': len(raw),
                'identity_ok': response.headers.get('X-Request-ID') == name,
                'server_timing': response.headers.get('Server-Timing'),
                'sha_ok': response.headers.get('X-Output-SHA256') == hashlib.sha256(raw).hexdigest()}
    return json.loads(raw), meta


def persist(out, arm, names, results, manifest):
    """Validate every response and write per_doc.jsonl plus the vectors as float32.

    The services return float32 values, so float32 storage is lossless.
    """
    order = {n: i for i, n in enumerate(names)}
    results = sorted(results, key=lambda r: order[r[0]])
    vectors = array.array('f')
    rows = []
    for name, result, meta in results:
        texts, vecs, lane = chunks_and_vectors(arm, result)
        ok = len(texts) == len(vecs) and all(len(v) == DIM and all(math.isfinite(x) for x in v) for v in vecs)
        if arm == 'lg':
            ok = ok and meta['identity_ok'] and meta['sha_ok'] and result['input']['filename'] == name
        if not texts:
            (out / 'empty-responses').mkdir(exist_ok=True)
            save(out / 'empty-responses' / (name + '.json'), result)
        rows.append(dict(meta, doc=name, ok=bool(ok), documents_lane=lane, chunks=len(texts),
                         chars=sum(len(t) for t in texts), vector_offset=len(vectors) // DIM,
                         input_sha256=manifest['sha256'][name],
                         text_sha256=[hashlib.sha256(t.encode()).hexdigest() for t in texts]))
        if texts and QUERIES is not None:
            # One question per document: 12 words from the middle of one of its chunks, chosen by the file name.
            j = int(hashlib.sha256(name.encode()).hexdigest(), 16) % len(texts)
            words = texts[j].split()
            if len(words) >= 24:
                start = (len(words) - 12) // 2
                QUERIES.append({'doc': name, 'chunk_sha256': hashlib.sha256(texts[j].encode()).hexdigest(),
                                'text': ' '.join(words[start:start + 12])})
        for v in vecs:
            vectors.extend(v)
    with (out / 'vectors.f32').open('wb') as f:
        vectors.tofile(f)
    (out / 'per_doc.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows


def rr_pipeline(plan, store):
    """The RocketRide pipeline of this cell, from the .pipe file in rocketride/pipelines.

    store      ingest.pipe            webhook -> parse -> preprocessor_langchain -> embedding_transformer -> postgres
    no store   ingest_no_store.pipe   the same with response_documents in place of the store (screening runs)

    The store's connection settings are taken from the plan. The password stays the ${ROCKETRIDE_PG_PASSWORD}
    placeholder of the file: the engine fills it in from the values sent with use().
    """
    rr, db = plan['rr'], plan['db']
    pipe = json.loads((PIPELINES / (rr['ingest_pipeline'] if store else rr['screen_pipeline'])).read_text())
    if store:
        stores = [c for c in pipe['components'] if c['provider'] == 'postgres']
        assert len(stores) == 1 and not any(c['provider'].startswith('response') for c in pipe['components'])
        local = stores[0]['config']['local']
        assert local['password'] == '${ROCKETRIDE_PG_PASSWORD}'
        local.update(host=db['host'], port=db['port'], user=db['user'], database=db['database'], collection=db['rr_table'])
    return pipe


def persist_stored(out, arm, names, results, manifest):
    """A store run returns no chunks or vectors: the rows are in the database, and dbcheck.py reads them back there.

    Only what the service answered is recorded here. LangGraph answers with the number of rows it wrote.
    RocketRide's ingestion pipe has no response component, so the keys of whatever close() returned are kept.
    """
    order = {n: i for i, n in enumerate(names)}
    rows = []
    for name, result, meta in sorted(results, key=lambda r: order[r[0]]):
        if arm == 'lg':
            output = result.get('output') or {}
            ok = (meta['identity_ok'] and meta['sha_ok'] and result['input']['filename'] == name
                  and isinstance(output.get('stored'), int) and output['stored'] == output.get('chunks'))
            row = dict(meta, doc=name, ok=bool(ok), chunks=output.get('stored'))
        else:
            ok = isinstance(result, dict) and not result.get('error') and not result.get('errors')
            row = dict(meta, doc=name, ok=bool(ok), chunks=None,
                       response_keys=sorted(result) if isinstance(result, dict) else str(type(result)))
        row['input_sha256'] = manifest['sha256'][name]
        rows.append(row)
    (out / 'per_doc.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows


async def rr_request(client, token, name, body):
    pipe = await client.pipe(token, objinfo={'name': name, 'size': len(body)}, mime_type='application/pdf')
    await pipe.open()
    for offset in range(0, len(body), 1048576):
        await pipe.write(body[offset:offset + 1048576])
    return await pipe.close(), {}


async def main(label):
    out = ROOT / 'cells' / label
    cfg = json.loads((out / 'config.json').read_text())
    plan = json.loads((ROOT / 'plan.json').read_text())
    arm = cfg['arm']
    assert not (out / 'measurement-start.json').exists(), 'refusing to overwrite a measured run'
    concurrency = cfg.get('concurrency', plan['concurrency'])
    store = bool(cfg.get('store'))      # the pipeline ends in the vector store and returns no vectors
    if cfg.get('make_queries'):
        globals()['QUERIES'] = []
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=max(32, concurrency)))

    manifest = json.loads((ROOT / 'source/corpus-manifest.json').read_text())
    names = manifest['measured'][:cfg['documents']]
    warm = manifest['warm']
    for name in names + warm:
        assert hashlib.sha256((CORPUS / name).read_bytes()).hexdigest() == manifest['sha256'][name], name

    client = token = None
    if arm == 'rr':
        from rocketride import RocketRideClient
        client = RocketRideClient(uri='ws://127.0.0.1:%d/task/service' % plan['rr']['port'], auth='local-dev')
        await client.connect()
        pipe = rr_pipeline(plan, store)
        save(out / 'pipeline-used.json', pipe)
        try:
            save(out / 'validate.json', await client.validate(pipe))
        except Exception as e:                      # recorded, not fatal: use() refuses a pipe it cannot run
            save(out / 'validate.json', {'error': repr(e)})
        extra = {}
        if store:
            extra['env'] = {'ROCKETRIDE_PG_PASSWORD': os.environ['ROCKETRIDE_PG_PASSWORD']}
        used = await client.use(pipeline=pipe, threads=plan['workers'], **extra)
        token = used['token']
        save(out / 'task.json', {'token': token, 'workers': plan['workers']})

    async def request(name):
        started = time.perf_counter()
        epoch = time.time()
        body = (CORPUS / name).read_bytes()
        if arm == 'rr':
            result, meta = await rr_request(client, token, name, body)
        else:
            result, meta = await asyncio.to_thread(lg_request, plan['lg']['port'], name, body,
                                                   plan['lg']['ingest_pipeline' if store else 'screen_pipeline'])
        meta.update(latency_s=time.perf_counter() - started, start_epoch=epoch)
        return name, result, meta

    try:
        warmed = await asyncio.gather(*(asyncio.wait_for(request(n), 1800) for n in warm))
        assert len(warmed) == 4 and (store or all(chunks_and_vectors(arm, r)[0] for _, r, _ in warmed)), 'warm-up returned no chunks'
        save(out / 'sample-response.json', warmed[0][1])
        save(out / 'ready.json', {'epoch': time.time(), 'warm': warm})
        print('WARM_READY', label, flush=True)

        # The host records what the service is and clears the database statistics before it lets the run start.
        deadline = time.monotonic() + 600
        while not (out / 'go.json').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('the host did not release the run')
            await asyncio.sleep(0.05)

        queue = collections.deque(names)
        results = []

        failures = []

        async def worker():
            while queue:
                name = queue.popleft()
                try:
                    results.append(await asyncio.wait_for(request(name), cfg.get('request_timeout_s', cfg['timeout_s'])))
                except Exception as e:
                    # A screening run records a PDF that fails and goes on; every other run stops.
                    if not cfg.get('tolerant'):
                        raise
                    failures.append({'doc': name, 'error': repr(e)[:300]})

        before = counters()
        begin = time.perf_counter()
        save(out / 'measurement-start.json', before)
        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - begin
        after = counters()
        save(out / 'measurement-end.json', after)

        # Everything below is outside the measured time.
        if failures:
            save(out / 'failures.json', failures)
        rows = (persist_stored if store else persist)(out, arm, names, results, manifest)
        if QUERIES is not None:
            (out / 'queries-all.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in QUERIES))

        cpu = after['cpu_s'] - before['cpu_s']
        latencies = [r['latency_s'] for r in rows]
        result = {
            'label': label, 'arm': arm, 'stage': cfg['stage'],
            'documents': len(rows), 'success': sum(r['ok'] for r in rows),
            'empty_documents': sum(r.get('chunks') == 0 for r in rows),
            'chunks': sum(r.get('chunks') or 0 for r in rows), 'chars': sum(r.get('chars') or 0 for r in rows),
            'store': store, 'failures': len(failures),
            'wall_s': wall, 'docs_per_s': len(rows) / wall, 'cpu_s': cpu, 'cpu_s_per_doc': cpu / len(rows),
            'average_logical_cpus': cpu / wall,
            'cpu_by_service_s': {k: after['groups'][k]['cpu_s'] - v['cpu_s'] for k, v in before['groups'].items()},
            'latency_s': {'p50': percentile(latencies, .5), 'p95': percentile(latencies, .95), 'max': max(latencies)},
            'start_epoch': before['epoch'], 'end_epoch': after['epoch'],
            'concurrency': concurrency,
            'client': '%d PDFs in flight, one request each; RocketRide task threads %d' % (concurrency, plan['workers']),
        }
        save(out / 'result.json', result)
        print('RESULT', json.dumps({k: result[k] for k in ['label', 'documents', 'wall_s', 'docs_per_s', 'cpu_s']}), flush=True)
        assert len(rows) == cfg['documents'] - len(failures), 'a request is missing from the results'
        assert cfg.get('tolerant') or all(r['ok'] for r in rows), 'invalid output census or vectors'
    finally:
        if client:
            if token:
                try:
                    await asyncio.wait_for(client.terminate(token), 30)
                except Exception as e:
                    print('TERMINATE', repr(e), flush=True)
            await client.disconnect()


if __name__ == '__main__':
    cell_dir = ROOT / 'cells' / sys.argv[1]
    try:
        asyncio.run(main(sys.argv[1]))
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        save(cell_dir / 'finished.json', {'ok': False, 'error': error, 'epoch': time.time()})
        raise
    else:
        save(cell_dir / 'finished.json', {'ok': True, 'epoch': time.time()})
