"""Test of the search client's answer check. No Docker, no network, no engine: a stand-in for the client SDK
answers the questions. Standard library only.

  1. rr_answer() and lg_answer() accept a real answer and an answer without results, and refuse every response
     that is not an answer.
  2. A run whose answers are all valid passes; an answer without results is counted as valid and as empty.
  3. A run whose warm-up is fine and whose timed responses are not answers fails. Not one of them is counted in
     the rate, and the responses are kept for inspection.
  4. A run fails before any timed question when the engine's answer to a question that cannot match is not read
     as a valid answer.
  5. When a later level fails outright, the responses kept from the earlier levels are still written.
  6. The same rules hold for the LangGraph side: a response without its identity or without a list of result
     documents is not counted and fails the run.

Usage: python3 tests/test_search_validation.py
"""
import asyncio
import json
import os
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'harness'))
import search_driver as sd  # noqa: E402

DOC = {'type': 'Document', 'page_content': 'a stored chunk', 'score': 0.9, 'metadata': {'parent': 'a.pdf', 'chunkId': 0}}


def answer(docs):
    """What client.chat() returns for the search pipeline: one answer, the result documents as JSON text."""
    return {'answers': [json.dumps(docs)], 'name': 'Question 1', 'path': '', 'result_types': {'answers': 'answers'}, 'objectId': 'x'}


def stand_in(timed, probe):
    """A module that looks like the client SDK. `timed(n)` answers the n-th timed question, `probe` the question
    that cannot match."""
    state = {'asked': 0}

    class Client:
        def __init__(self, **kw):
            pass

        async def connect(self):
            pass

        async def validate(self, pipe):
            return {}

        async def use(self, **kw):
            return {'token': 't'}

        async def chat(self, token, question):
            if question.filter.objectIds:
                assert question.filter.objectIds == [sd.NO_SUCH_DOCUMENT]
                return probe
            state['asked'] += 1
            return answer([DOC]) if state['asked'] <= 4 else timed(state['asked'] - 4)

        async def terminate(self, token):
            pass

        async def disconnect(self):
            pass

    class DocFilter:
        def __init__(self, limit=10, objectIds=None):
            self.limit, self.objectIds = limit, objectIds

    class Question:
        def __init__(self, filter=None):
            self.filter, self.text = filter, []

        def addQuestion(self, text):
            self.text.append(text)

    package, schema = types.ModuleType('rocketride'), types.ModuleType('rocketride.schema')
    package.RocketRideClient, schema.DocFilter, schema.Question = Client, DocFilter, Question
    sys.modules['rocketride'], sys.modules['rocketride.schema'] = package, schema


def run(root, timed=None, probe=None, label='rr-1'):
    """One search phase of a cell: 20 questions, 4 of them for the warm-up, at 1 and 8 in flight."""
    cell = root / 'cells' / label
    for old in cell.glob('search-*'):
        old.unlink()
    if label == 'rr-1':
        stand_in(timed, probe)
    try:
        asyncio.run(sd.main(label, 'exact'))
        error = None
    except (AssertionError, RuntimeError) as e:
        error = '%s: %s' % (type(e).__name__, e)
    path = cell / 'search-exact.json'
    return error, json.loads(path.read_text()) if path.exists() else None


def main():
    # 1. What is an answer.
    docs, problem = sd.rr_answer(answer([DOC] * 10))
    assert problem is None and len(docs) == 10 and sd.rr_hits(docs)[0]['doc'] == 'a.pdf'
    assert sd.rr_answer(answer([])) == ([], None), 'an answer without results is a valid answer'
    assert sd.rr_answer({'answers': [[DOC]]})[1] is None, 'the documents may also arrive as a list'
    for bad in [None, 'text', [], {}, {'unexpected': ['shape']}, {'answers': []}, {'answers': ['[]', '[]']},
                {'answers': ['not json']}, {'answers': ['{"a": 1}']}, {'answers': [json.dumps([{'no': 'text'}])]},
                {'answers': [None]}, dict(answer([DOC]), error='the pipeline failed'), dict(answer([]), errors=['x'])]:
        docs, problem = sd.rr_answer(bad)
        assert docs is None and problem, ('counted as an answer', bad)
    good = {'output': {'documents': [{'text': 'a', 'distance': 0.25, 'source': 'a.pdf', 'chunk': 0}]}}
    assert sd.lg_answer(good, True)[1] is None and sd.lg_hits(sd.lg_answer(good, True)[0])[0]['score'] == 0.75
    assert sd.lg_answer({'output': {'documents': []}}, True) == ([], None)
    for bad, identity in [(good, False), ({}, True), ({'output': None}, True), ({'output': {'documents': None}}, True),
                          ({'output': {'documents': [{'text': 'a'}]}}, True), ([], True)]:
        assert sd.lg_answer(bad, identity)[0] is None and sd.lg_answer(bad, identity)[1], ('counted as an answer', bad)
    print('ok  an answer, an answer without results and 19 responses that are not answers are told apart')

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'cells/rr-1').mkdir(parents=True)
        (root / 'source').mkdir()
        (root / 'groups/service').mkdir(parents=True)
        (root / 'groups/service/cpu.stat').write_text('usage_usec 1000\n')
        sd.ROOT, sd.PIPELINES = root, REPO / 'rocketride/pipelines'
        sd.counters = lambda: {'epoch': 0.0, 'cpu_s': 0.0}
        os.environ['ROCKETRIDE_PG_PASSWORD'] = 'not-a-real-password'
        (root / 'cells/lg-1').mkdir()
        (root / 'cells/lg-1/config.json').write_text(json.dumps({'arm': 'lg', 'label': 'lg-1'}))
        (root / 'plan.json').write_text(json.dumps({
            'rr': {'port': 5565, 'search_pipeline': 'search.pipe'}, 'workers': 4,
            'lg': {'port': 8100, 'search_pipeline': 'search-pdf-rag-v1'},
            'db': {'host': '127.0.0.1', 'port': 5432, 'user': 'postgres', 'database': 'ragbench', 'rr_table': 'rr_chunks'},
            'search': {'k': 10, 'queries': 20, 'warm': 4, 'levels': [1, 8], 'rr_score': 0.2}}))
        (root / 'cells/rr-1/config.json').write_text(json.dumps({'arm': 'rr', 'label': 'rr-1'}))
        (root / 'source/queries.jsonl').write_text(''.join(
            json.dumps({'id': i, 'text': 'question %d' % i, 'chunk_sha256': sd.sha(DOC['page_content'])}) + '\n' for i in range(20)))

        # 2. Every answer valid, one of them without results.
        error, result = run(root, lambda n: answer([] if n == 3 else [DOC]), answer([]))
        assert error is None, error
        assert result['empty_answer_form'] == {'confirmed': True, 'results_returned': 0}, result['empty_answer_form']
        first, second = result['levels']
        assert first['questions'] == first['answers_valid'] == 20 and first['answers_not_valid'] == 0, first
        assert first['questions_with_no_result'] == 1 and second['questions_with_no_result'] == 0, (first, second)
        assert first['client_cpu_s'] >= 0 and first['client_average_cpus'] >= 0 and first['source_chunk_in_results'] == 0.95
        kept = json.loads((root / 'cells/rr-1/search-exact-answers-to-inspect.json').read_text())
        assert len(kept['no_result']) == 1 and not kept['not_an_answer'], kept
        print('ok  a run of valid answers passes; the one answer without results is counted as valid and kept for inspection')

        # 3. A good warm-up, then responses that are not answers: the wrong shape, then an error.
        for name, response in [('another shape', {'unexpected': ['shape']}), ('an error', {'error': 'the pipeline failed'})]:
            error, result = run(root, lambda n: response, answer([]))
            assert error and 'were not answers' in error, (name, error)
            for level in result['levels']:
                assert level['questions'] == 20 and level['answers_valid'] == 0 and level['answers_not_valid'] == 20, (name, level)
                assert level['questions_per_s'] == 0 and level['questions_with_no_result'] == 0 and level['latency_ms'] is None, (name, level)
            kept = json.loads((root / 'cells/rr-1/search-exact-answers-to-inspect.json').read_text())
            assert len(kept['not_an_answer']) == sd.KEEP and kept['not_an_answer'][0]['problem'], kept
        print('ok  a run whose timed responses are not answers fails, with a rate of 0 valid answers and the responses kept')

        # 4. The engine's answer to a question that cannot match must be read as a valid answer before timing starts.
        error, result = run(root, lambda n: answer([DOC]), {'unexpected': ['shape']})
        assert error and 'cannot match' in error and result is None, (error, result)
        error, result = run(root, lambda n: answer([DOC]), answer([DOC]))       # the engine ignored the restriction
        assert error is None and result['empty_answer_form'] == {'confirmed': False, 'results_returned': 1}, (error, result)
        print('ok  the form of an answer without results is checked against the engine before the first timed question')

        # 5. The first level holds one answer without results; the second level fails outright on its 6th question.
        def fails_late(n):
            if n > 25:
                raise RuntimeError('the connection is gone')
            return answer([] if n == 3 else [DOC])
        error, result = run(root, fails_late, answer([]))
        assert error and 'the connection is gone' in error and result is None, (error, result)
        kept = json.loads((root / 'cells/rr-1/search-exact-answers-to-inspect.json').read_text())
        assert len(kept['no_result']) == 1 and kept['no_result'][0]['in_flight'] == 1, kept
        print('ok  responses kept from an earlier level are written although a later level failed')

        # 6. The LangGraph side, through the same run.
        calls = {'n': 0}
        hit = {'output': {'documents': [{'text': DOC['page_content'], 'distance': 0.1, 'source': 'a.pdf', 'chunk': 0}]}}

        def lg(bad):
            def request(port, pipeline, rid, question, k):
                calls['n'] += 1
                return (hit, True, 'embed;dur=1.0') if calls['n'] <= 4 else bad(calls['n'] - 4)
            calls['n'] = 0
            return request
        sd.lg_request = lg(lambda n: ({'output': {'documents': []}}, True, None) if n == 3 else (hit, True, None))
        error, result = run(root, label='lg-1')
        assert error is None and result['empty_answer_form'] is None, (error, result)
        assert result['levels'][0]['answers_valid'] == 20 and result['levels'][0]['questions_with_no_result'] == 1, result['levels'][0]
        for name, bad in [('no identity', (hit, False, None)), ('no list of documents', ({'output': {}}, True, None))]:
            sd.lg_request = lg(lambda n: bad)
            error, result = run(root, label='lg-1')
            assert error and 'were not answers' in error, (name, error)
            assert all(l['answers_valid'] == 0 and l['answers_not_valid'] == 20 and l['questions_per_s'] == 0 for l in result['levels']), (name, result)
        print('ok  the LangGraph side: valid answers pass, a response without identity or without documents fails the run')
    print('ALL OK')


if __name__ == '__main__':
    main()
