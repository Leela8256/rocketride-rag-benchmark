"""Client for the search side of one cell: questions in, the nearest stored chunks out.

Runs in the client container on the client CPUs, after the cell's ingestion and its read-back check, against the
table that ingestion filled. The same questions go to both systems:

  RocketRide   client.chat() on rocketride/pipelines/search.pipe:
               chat -> embedding_transformer -> postgres -> response_answers
  LangGraph    POST /v1/process/search-pdf-rag-v1 with the question in the request's options

Each question is a run of 12 words taken from one stored chunk, so the chunk it came from is known. Both systems
are asked for the K nearest chunks. The questions are sent at several numbers in flight, one level after the other,
and each level sends every question once. Timing covers only those requests.

Usage: search_driver.py <cell label> <phase>      phase names the state of the table (for example exact or hnsw)
"""
import asyncio
import collections
import hashlib
import json
import os
import sys
import time
import traceback
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/campaign')
PIPELINES = Path('/pipelines')      # rocketride/pipelines of this repository


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def percentile(values, p):
    values = sorted(values)
    point = (len(values) - 1) * p
    i = int(point)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (point - i)


def counters():
    groups = {}
    for p in sorted(Path('/groups').iterdir()):
        stat = dict(line.split() for line in (p / 'cpu.stat').read_text().splitlines())
        groups[p.name] = int(stat['usage_usec']) / 1e6
    assert groups, 'no service cgroups mounted'
    return {'epoch': time.time(), 'cpu_s': sum(groups.values())}


def sha(text):
    return hashlib.sha256((text or '').encode()).hexdigest()


def rr_pipeline(plan):
    """RocketRide's search pipeline, from the .pipe file, with the store's own answers lane as the response.

    The store's connection settings are taken from the plan; the password stays the placeholder of the file.

    `score` is the store's similarity threshold. It is set to the store's fixed floor (0.20) so the node returns
    every result it can; its default of 0.5 would drop results that LangChain returns.
    """
    db = plan['db']
    pipe = json.loads((PIPELINES / plan['rr']['search_pipeline']).read_text())
    stores = [c for c in pipe['components'] if c['provider'] == 'postgres']
    assert len(stores) == 1
    local = stores[0]['config']['local']
    assert local['password'] == '${ROCKETRIDE_PG_PASSWORD}'
    local.update(host=db['host'], port=db['port'], user=db['user'], database=db['database'], collection=db['rr_table'],
                 score=plan['search']['rr_score'])
    return pipe


def find_docs(obj):
    """The first list of result documents inside a RocketRide response, wherever it sits."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) and 'page_content' in x for x in obj):
            return obj
        for item in obj:
            found = find_docs(item)
            if found is not None:
                return found
    elif isinstance(obj, dict):
        for value in obj.values():
            found = find_docs(value)
            if found is not None:
                return found
    elif isinstance(obj, str) and obj[:1] in '[{' and 'page_content' in obj:
        try:
            return find_docs(json.loads(obj))
        except ValueError:
            return None
    return None


def rr_hits(response):
    docs = find_docs(response) or []
    return [{'sha': sha(d.get('page_content')), 'score': d.get('score'),
             'doc': (d.get('metadata') or {}).get('parent'), 'chunk': (d.get('metadata') or {}).get('chunkId')} for d in docs]


def lg_request(port, pipeline, rid, question, k):
    boundary = 'ragbench-' + uuid.uuid4().hex
    options = json.dumps({'question': question, 'k': k})
    payload = ''.join('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n' % (boundary, name, value)
                      for name, value in [('request_id', rid), ('options', options)]) + '--%s--\r\n' % boundary
    request = urllib.request.Request('http://127.0.0.1:%d/v1/process/%s' % (port, pipeline), data=payload.encode(),
                                     headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
    with urllib.request.urlopen(request, timeout=600) as response:
        raw = response.read()
        ok = response.headers.get('X-Request-ID') == rid and response.headers.get('X-Output-SHA256') == hashlib.sha256(raw).hexdigest()
        timing = response.headers.get('Server-Timing')
    return json.loads(raw), ok, timing


def lg_hits(result):
    return [{'sha': sha(d['text']), 'score': 1 - d['distance'], 'doc': d.get('source'), 'chunk': d.get('chunk')}
            for d in result['output']['documents']]


async def main(label, phase):
    out = ROOT / 'cells' / label
    cfg = json.loads((out / 'config.json').read_text())
    plan = json.loads((ROOT / 'plan.json').read_text())
    search, arm = plan['search'], cfg['arm']
    k = search['k']
    wanted = search['queries']
    queries = [json.loads(x) for x in (ROOT / 'source/queries.jsonl').read_text().splitlines() if x.strip()][:wanted]
    assert len(queries) == wanted, 'not enough questions prepared'
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=max(32, max(search['levels']))))

    client = token = None
    if arm == 'rr':
        from rocketride import RocketRideClient
        from rocketride.schema import DocFilter, Question
        client = RocketRideClient(uri='ws://127.0.0.1:%d/task/service' % plan['rr']['port'], auth='local-dev')
        await client.connect()
        pipe = rr_pipeline(plan)
        save(out / ('search-%s-pipeline.json' % phase), pipe)
        try:
            save(out / ('search-%s-validate.json' % phase), await client.validate(pipe))
        except Exception as e:
            save(out / ('search-%s-validate.json' % phase), {'error': repr(e)})
        used = await client.use(pipeline=pipe, threads=plan['workers'], env={'ROCKETRIDE_PG_PASSWORD': os.environ['ROCKETRIDE_PG_PASSWORD']})
        token = used['token']

    async def ask(q):
        started = time.perf_counter()
        if arm == 'rr':
            question = Question(filter=DocFilter(limit=k))
            question.addQuestion(q['text'])
            response = await client.chat(token=token, question=question)
            hits, ok, timing = rr_hits(response), True, None
            raw = response
        else:
            raw, ok, timing = await asyncio.to_thread(lg_request, plan['lg']['port'], plan['lg']['search_pipeline'],
                                                      'q%d-%s' % (q['id'], uuid.uuid4().hex[:8]), q['text'], k)
            hits = lg_hits(raw)
        return {'id': q['id'], 'latency_s': time.perf_counter() - started, 'hits': hits, 'ok': ok, 'timing': timing}, raw

    try:
        # Warm-up, not timed: the first questions also show what a response looks like.
        warm = queries[:search['warm']]
        warm_limit = asyncio.Semaphore(8)

        async def warm_one(q):
            async with warm_limit:
                return await asyncio.wait_for(ask(q), 600)

        warmed = await asyncio.gather(*(warm_one(q) for q in warm))
        save(out / ('search-%s-sample-response.json' % phase), warmed[0][1])
        assert all(r['ok'] for r, _ in warmed), 'a warm-up answer failed its identity check'
        assert any(r['hits'] for r, _ in warmed), 'warm-up questions returned no results; see the sample response'

        levels, first = [], None
        for in_flight in search['levels']:
            queue = collections.deque(queries)
            results = []

            async def worker():
                while queue:
                    q = queue.popleft()
                    results.append((await asyncio.wait_for(ask(q), 600))[0])

            before = counters()
            begin = time.perf_counter()
            await asyncio.gather(*(worker() for _ in range(in_flight)))
            wall = time.perf_counter() - begin
            after = counters()
            results.sort(key=lambda r: r['id'])
            by_id = {r['id']: [h['sha'] for h in r['hits']] for r in results}
            if first is None:
                first = by_id
                with (out / ('search-%s-outputs.jsonl' % phase)).open('w') as f:
                    for r in results:
                        f.write(json.dumps({'id': r['id'], 'sha': [h['sha'] for h in r['hits']],
                                            'score': [h['score'] for h in r['hits']], 'timing': r['timing']}) + '\n')
            source = {q['id']: q['chunk_sha256'] for q in queries}
            latencies = [r['latency_s'] * 1000 for r in results]
            cpu = after['cpu_s'] - before['cpu_s']
            levels.append({
                'in_flight': in_flight, 'questions': len(results), 'wall_s': wall, 'questions_per_s': len(results) / wall,
                'latency_ms': {'p50': percentile(latencies, .5), 'p95': percentile(latencies, .95), 'p99': percentile(latencies, .99),
                               'mean': sum(latencies) / len(latencies)},
                'cpu_s': cpu, 'cpu_ms_per_question': 1000 * cpu / len(results), 'average_cpus': cpu / wall,
                'start_epoch': before['epoch'], 'end_epoch': after['epoch'],
                'results_per_question': sum(len(v) for v in by_id.values()) / len(by_id),
                'questions_with_no_result': sum(1 for v in by_id.values() if not v),
                'source_chunk_first': sum(1 for i, v in by_id.items() if v[:1] == [source[i]]) / len(by_id),
                'source_chunk_in_results': sum(1 for i, v in by_id.items() if source[i] in v) / len(by_id),
                'answers_failing_identity_check': sum(1 for r in results if not r['ok']),
                'questions_differing_from_first_level': sum(1 for i, v in by_id.items() if v != first[i]),
            })
            print('LEVEL', json.dumps({x: levels[-1][x] for x in ['in_flight', 'questions_per_s', 'wall_s']}), flush=True)
        save(out / ('search-%s.json' % phase), {'label': label, 'arm': arm, 'phase': phase, 'k': k,
                                                'questions': len(queries), 'levels': levels})
        assert all(l['questions'] == len(queries) and not l['answers_failing_identity_check'] for l in levels)
    finally:
        if client:
            if token:
                try:
                    await asyncio.wait_for(client.terminate(token), 30)
                except Exception as e:
                    print('TERMINATE', repr(e), flush=True)
            await client.disconnect()


if __name__ == '__main__':
    marker = ROOT / 'cells' / sys.argv[1] / ('search-%s-finished.json' % sys.argv[2])
    try:
        asyncio.run(main(sys.argv[1], sys.argv[2]))
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        save(marker, {'ok': False, 'error': error, 'epoch': time.time()})
        raise
    else:
        save(marker, {'ok': True, 'epoch': time.time()})
