"""Client for the search side of one cell: questions in, the nearest stored chunks out.

Runs in the client container on the client CPUs, after the cell's ingestion and its read-back check, against the
table that ingestion filled. The same questions go to both systems:

  RocketRide   client.chat() on rocketride/pipelines/search.pipe:
               chat -> embedding_transformer -> postgres -> response_answers
  LangGraph    POST /v1/process/search-pdf-rag-v1 with the question in the request's options

Each question is a run of 12 words taken from one stored chunk, so the chunk it came from is known. Both systems
are asked for the K nearest chunks. The questions are sent at several numbers in flight, one level after the other,
and each level sends every question once. Timing covers only those requests.

Every answer is checked before it is counted. An answer with no results is a valid answer; a response that is not
an answer (an error, another shape) is not, and one such response fails the run. The rates this client reports
are valid answers per second.

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
NO_SUCH_DOCUMENT = 'ragbench-no-such-document'      # a document id no table holds
KEEP = 5                                            # responses kept for inspection, per kind


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


def rr_answer(response):
    """The result documents of a RocketRide search answer, or why the response is not an answer.

    Returns (documents, None) or (None, reason).

    client.chat() on the search pipeline returns an object whose `answers` list holds one entry: the result
    documents of the store's answers lane, as JSON text. A valid answer is exactly that. An empty list of
    documents is a valid answer too: the store found nothing above its similarity floor. Every other response (an
    error, no `answers` entry or several, entries that are not documents) is not an answer.
    """
    if not isinstance(response, dict):
        return None, 'the response is a %s, not an object' % type(response).__name__
    for key in ('error', 'errors'):
        if response.get(key):
            return None, 'the response reports %s: %s' % (key, str(response[key])[:200])
    answers = response.get('answers')
    if not isinstance(answers, list) or len(answers) != 1:
        return None, 'the response does not hold exactly one answer (its keys: %s)' % ', '.join(sorted(str(k) for k in response))
    docs = answers[0]
    if isinstance(docs, str):
        try:
            docs = json.loads(docs)
        except ValueError:
            return None, 'the answer is text, not a list of result documents: %r' % docs[:80]
    if not isinstance(docs, list):
        return None, 'the answer is a %s, not a list of result documents' % type(docs).__name__
    if not all(isinstance(d, dict) and isinstance(d.get('page_content'), str) for d in docs):
        return None, 'an entry of the answer is not a result document'
    return docs, None


def rr_hits(docs):
    return [{'sha': sha(d['page_content']), 'score': d.get('score'),
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


def lg_answer(result, identity_ok):
    """The result documents of a LangGraph search answer, or why the response is not an answer. Same contract as rr_answer()."""
    if not identity_ok:
        return None, 'the response does not carry its request id and the hash of its own body'
    output = result.get('output') if isinstance(result, dict) else None
    docs = output.get('documents') if isinstance(output, dict) else None
    if not isinstance(docs, list):
        return None, 'the response holds no list of result documents'
    if not all(isinstance(d, dict) and isinstance(d.get('text'), str) and isinstance(d.get('distance'), (int, float)) for d in docs):
        return None, 'an entry of the answer is not a result document'
    return docs, None


def lg_hits(docs):
    return [{'sha': sha(d['text']), 'score': 1 - d['distance'], 'doc': d.get('source'), 'chunk': d.get('chunk')} for d in docs]


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

    client = token = DocFilter = Question = None
    kept = {'not_an_answer': [], 'no_result': []}       # the first responses of each kind, for inspection
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

    async def ask(q, doc_filter=None):
        """One question. A failing request raises and stops the run; a response that is not an answer is returned as such."""
        started = time.perf_counter()
        if arm == 'rr':
            question = Question(filter=doc_filter or DocFilter(limit=k))
            question.addQuestion(q['text'])
            raw = await client.chat(token=token, question=question)
            docs, problem = rr_answer(raw)
            hits, timing = rr_hits(docs or []), None
        else:
            raw, identity_ok, timing = await asyncio.to_thread(lg_request, plan['lg']['port'], plan['lg']['search_pipeline'],
                                                               'q%d-%s' % (q['id'], uuid.uuid4().hex[:8]), q['text'], k)
            docs, problem = lg_answer(raw, identity_ok)
            hits = lg_hits(docs or [])
        return {'id': q['id'], 'latency_s': time.perf_counter() - started, 'hits': hits, 'ok': problem is None,
                'problem': problem, 'timing': timing}, raw

    try:
        # Warm-up, not timed: the first questions also show what a response looks like.
        warm = queries[:search['warm']]
        warm_limit = asyncio.Semaphore(8)

        async def warm_one(q):
            async with warm_limit:
                return await asyncio.wait_for(ask(q), 600)

        warmed = await asyncio.gather(*(warm_one(q) for q in warm))
        save(out / ('search-%s-sample-response.json' % phase), warmed[0][1])
        problems = sorted({r['problem'] for r, _ in warmed if not r['ok']})
        assert not problems, 'a warm-up response is not an answer: %s' % '; '.join(problems)
        assert any(r['hits'] for r, _ in warmed), 'warm-up questions returned no results; see the sample response'

        empty_form = None
        if arm == 'rr':
            # One question that cannot match: it is limited to a document id no table holds. It shows what this
            # engine's answer without results looks like, and that rr_answer() reads it as a valid, empty answer.
            try:
                probe, raw = await asyncio.wait_for(ask(warm[0], DocFilter(limit=k, objectIds=[NO_SUCH_DOCUMENT])), 600)
            except Exception as e:      # the engine refused the question: the form stays unconfirmed, the run goes on
                empty_form = {'confirmed': False, 'why': 'the question could not be asked: %s' % repr(e)[:200]}
            else:
                save(out / ('search-%s-empty-answer.json' % phase), raw)
                assert probe['ok'], ('the answer to a question that cannot match is not read as a valid answer (%s); '
                                     'see search-%s-empty-answer.json' % (probe['problem'], phase))
                empty_form = {'confirmed': not probe['hits'], 'results_returned': len(probe['hits'])}

        levels, first = [], None
        for in_flight in search['levels']:
            queue = collections.deque(queries)
            results = []

            async def worker():
                while queue:
                    q = queue.popleft()
                    r, raw = await asyncio.wait_for(ask(q), 600)
                    results.append(r)
                    kind = 'not_an_answer' if not r['ok'] else 'no_result' if not r['hits'] else None
                    if kind and len(kept[kind]) < KEEP:
                        kept[kind].append({'in_flight': in_flight, 'id': r['id'], 'problem': r['problem'], 'response': raw})

            before = counters()
            own = time.process_time()
            begin = time.perf_counter()
            await asyncio.gather(*(worker() for _ in range(in_flight)))
            wall = time.perf_counter() - begin
            own = time.process_time() - own
            after = counters()
            results.sort(key=lambda r: r['id'])
            valid = [r for r in results if r['ok']]
            by_id = {r['id']: [h['sha'] for h in r['hits']] for r in valid}
            if first is None:
                first = by_id
                with (out / ('search-%s-outputs.jsonl' % phase)).open('w') as f:
                    for r in valid:
                        f.write(json.dumps({'id': r['id'], 'sha': [h['sha'] for h in r['hits']],
                                            'score': [h['score'] for h in r['hits']], 'timing': r['timing']}) + '\n')
            source = {q['id']: q['chunk_sha256'] for q in queries}
            latencies = [r['latency_s'] * 1000 for r in valid]
            cpu = after['cpu_s'] - before['cpu_s']
            n = max(1, len(valid))
            levels.append({
                'in_flight': in_flight, 'questions': len(results), 'wall_s': wall,
                # Only an answer that passed its check counts. A response that is not an answer also fails the run below.
                'answers_valid': len(valid), 'answers_not_valid': len(results) - len(valid),
                'questions_per_s': len(valid) / wall,
                'latency_ms': None if not valid else {
                    'p50': percentile(latencies, .5), 'p95': percentile(latencies, .95), 'p99': percentile(latencies, .99),
                    'mean': sum(latencies) / len(latencies)},
                'cpu_s': cpu, 'cpu_ms_per_question': 1000 * cpu / n, 'average_cpus': cpu / wall,
                # CPU time of this client process (all its threads) during the level.
                'client_cpu_s': own, 'client_average_cpus': own / wall,
                'start_epoch': before['epoch'], 'end_epoch': after['epoch'],
                'results_per_question': sum(len(v) for v in by_id.values()) / n,
                'questions_with_no_result': sum(1 for v in by_id.values() if not v),
                'source_chunk_first': sum(1 for i, v in by_id.items() if v[:1] == [source[i]]) / n,
                'source_chunk_in_results': sum(1 for i, v in by_id.items() if source[i] in v) / n,
                'questions_differing_from_first_level': sum(1 for i, v in by_id.items() if v != first.get(i)),
            })
            print('LEVEL', json.dumps({x: levels[-1][x] for x in ['in_flight', 'questions_per_s', 'wall_s', 'answers_not_valid']}), flush=True)
        save(out / ('search-%s.json' % phase), {'label': label, 'arm': arm, 'phase': phase, 'k': k,
                                                'questions': len(queries), 'empty_answer_form': empty_form, 'levels': levels})
        assert all(l['questions'] == len(queries) for l in levels), 'a question is missing from the results'
        bad = sum(l['answers_not_valid'] for l in levels)
        assert not bad, '%d responses were not answers; see search-%s-answers-to-inspect.json' % (bad, phase)
    finally:
        # Written here so that it is there also when a later level failed.
        if kept['not_an_answer'] or kept['no_result']:
            (out / ('search-%s-answers-to-inspect.json' % phase)).write_text(json.dumps(kept, indent=2, default=repr) + '\n')
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
