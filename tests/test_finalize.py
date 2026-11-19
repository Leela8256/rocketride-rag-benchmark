"""Test of the step that fixes the PDFs of the timed runs after the two screening runs. Standard library only.

A made-up campaign folder holds two screening runs over seven candidates. finalize.py must keep a PDF only when
both runs returned it with the same chunk texts and with vectors that are equal within tolerance, and must say
why it left every other one out. A vector file that is shorter than the chunks it belongs to must stop it: two
missing vectors must never compare as equal.

Before that, the function that reads what a screening run returned for one PDF: a response that is an error or has
another shape is not a result, and is never taken for a PDF without text. A PDF without text is a result of its own.

Usage: python3 tests/test_finalize.py
"""
import array
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIM = 384
sys.path.insert(0, str(REPO / 'harness'))
import driver  # noqa: E402


def screening_results():
    """What the ingestion client records for the responses of a run without the store."""
    v = [0.5] * DIM
    meta = {'latency_s': 1.0, 'start_epoch': 0.0, 'identity_ok': True, 'sha_ok': True}
    chunk = {'page_content': 'a chunk', 'embedding': v, 'metadata': {'chunkId': 7}}
    rr = [('a chunk with its vector', {'documents': [chunk]}, 'chunks'),
          ('a PDF without text: the normal object, no documents', {'metadata': {'Content-Type': 'application/pdf'}}, 'empty'),
          ('an error', {'error': 'parse failed'}, 'not_a_result'),
          ('a list of errors', {'errors': ['boom'], 'metadata': {}}, 'not_a_result'),
          ('an empty object', {}, 'not_a_result'),
          ('a string', 'not an object', 'not_a_result'),
          ('nothing', None, 'not_a_result'),
          ('documents that are not chunks', {'documents': [{'text': 'x'}]}, 'not_a_result'),
          ('a vector of the wrong length', {'documents': [dict(chunk, embedding=[0.5] * 3)]}, 'not_a_result')]
    ok_lg = {'input': {'filename': '0.pdf'}, 'output': {'chunks': [{'index': 0, 'text': 'a chunk'}], 'vectors': [v]}}
    lg = [('a chunk with its vector', ok_lg, 'chunks'),
          ('a PDF without text', {'input': {'filename': '1.pdf'}, 'output': {'chunks': [], 'vectors': []}}, 'empty'),
          ('an error', {'error': {'code': 'processing_failed'}}, 'not_a_result'),
          ('no output', {'input': {'filename': '3.pdf'}}, 'not_a_result'),
          ('another file name', {'input': {'filename': 'other.pdf'}, 'output': {'chunks': [], 'vectors': []}}, 'not_a_result')]
    with tempfile.TemporaryDirectory() as tmp:
        for arm, cases in [('rr', rr), ('lg', lg)]:
            out = Path(tmp) / arm
            out.mkdir()
            names = ['%d.pdf' % i for i in range(len(cases))]
            rows = driver.persist(out, arm, names, [(n, r, dict(meta)) for n, (_, r, _) in zip(names, cases)],
                                  {'sha256': {n: 'x' for n in names}})
            for (label, _, outcome), row in zip(cases, rows):
                assert row['outcome'] == outcome and row['ok'] == (outcome != 'not_a_result'), (arm, label, row)
                assert (row['problem'] is None) == row['ok'] and (row['chunks'] > 0) == (outcome == 'chunks'), (arm, label, row)
            assert rows[0]['chunk_numbers'] == ([7] if arm == 'rr' else [0]), rows[0]
    print('ok  a result with chunks, a PDF without text and %d responses that are not results are told apart' % (
        sum(1 for c in rr + lg if c[2] == 'not_a_result')))


def write_run(cell, docs, not_a_result=()):
    """docs: {name: [(text, vector), ...]} in the layout the ingestion client writes for a run without the store."""
    cell.mkdir(parents=True)
    vectors, rows = array.array('f'), []
    for name, chunks in docs.items():
        rows.append({'doc': name, 'ok': name not in not_a_result, 'chunks': len(chunks), 'vector_offset': len(vectors) // DIM,
                     'text_sha256': [hashlib.sha256(t.encode()).hexdigest() for t, _ in chunks]})
        for _, v in chunks:
            vectors.extend(v)
    (cell / 'per_doc.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with (cell / 'vectors.f32').open('wb') as f:
        vectors.tofile(f)


def main():
    screening_results()

    def vector(seed, shift=0.0):
        return [((seed * 31 + i) % 97) / 97.0 + (shift if i == 5 else 0.0) for i in range(DIM)]

    names = ['a.pdf', 'b.pdf', 'c.pdf', 'd.pdf', 'e.pdf', 'f.pdf', 'g.pdf', 'h.pdf', 'i.pdf']
    rr = {n: [('%s chunk %d' % (n, i), vector(k * 10 + i)) for i in range(2)] for k, n in enumerate(names)}
    rr['h.pdf'] = []                                                # RocketRide answered with an error, LangGraph found no text
    rr['i.pdf'] = []                                                # a PDF without text in both systems: kept
    lg = {n: [(t, list(v)) for t, v in chunks] for n, chunks in rr.items()}
    lg['b.pdf'][1] = (lg['b.pdf'][1][0], vector(11, 1e-7))          # within tolerance: kept
    lg['d.pdf'][1] = (lg['d.pdf'][1][0], vector(31, 0.01))          # outside tolerance: left out
    lg['e.pdf'][0] = ('another text', lg['e.pdf'][0][1])            # another chunk text: left out
    del lg['f.pdf']                                                 # failed in LangGraph: left out

    def folder(root):
        (root / 'source').mkdir(parents=True)
        (root / 'plan.json').write_text(json.dumps({'documents': 3, 'questions': 2}))
        (root / 'source/corpus-manifest.json').write_text(json.dumps(
            {'measured': names, 'warm': ['w.pdf'], 'stage': 'candidates', 'duplicates_left_out': ['x.pdf']}))
        write_run(root / 'cells/screen-rr', rr, not_a_result={'h.pdf'})
        write_run(root / 'cells/screen-lg', lg)
        (root / 'cells/screen-rr/queries-all.jsonl').write_text(''.join(
            json.dumps({'doc': n, 'chunk_sha256': 'x', 'text': 'twelve words from ' + n}) + '\n' for n in names))
        (root / 'cells/screen-lg/failures.json').write_text(json.dumps(
            [{'doc': 'f.pdf', 'error': "RuntimeError('Already borrowed')"}]))
        return [sys.executable, str(REPO / 'harness/finalize.py'), str(root)]

    with tempfile.TemporaryDirectory() as tmp:
        # Both vector files cut short by the same amount: the last documents have no vectors in either run.
        short = Path(tmp) / 'short'
        command = folder(short)
        for run in ['screen-rr', 'screen-lg']:
            path = short / 'cells' / run / 'vectors.f32'
            path.write_bytes(path.read_bytes()[:DIM * 4 * 3])       # the vectors of the first chunks only
        cut = subprocess.run(command, capture_output=True, text=True)
        assert cut.returncode != 0 and 'shorter than its chunks' in cut.stderr, cut.stderr[-300:]
        assert json.loads((short / 'source/corpus-manifest.json').read_text())['stage'] == 'candidates'

        root = Path(tmp) / 'run'
        subprocess.run(folder(root), check=True, capture_output=True, text=True)
        summary = json.loads((root / 'finalize.json').read_text())
        manifest = json.loads((root / 'source/corpus-manifest.json').read_text())
        assert summary['left_out'] == {'d.pdf': 'the two systems returned different vectors',
                                       'e.pdf': 'the two systems returned different chunk texts',
                                       'f.pdf': 'failed or timed out in LangGraph',
                                       'h.pdf': 'not a valid result in RocketRide'}, summary['left_out']
        assert summary['left_out_by_reason'] == {'the two systems returned different vectors': 1,
                                                 'the two systems returned different chunk texts': 1,
                                                 'failed or timed out in LangGraph': 1,
                                                 'not a valid result in RocketRide': 1}, summary['left_out_by_reason']
        # g.pdf and i.pdf are kept too; i.pdf is the PDF without text. The plan asks for 3, so the first 3 are measured.
        assert summary['kept_by_screening'] == 5 and manifest['measured'] == ['a.pdf', 'b.pdf', 'c.pdf'] and manifest['stage'] == 'final'
        assert summary['screening_errors'] == {'rr': [], 'lg': [["RuntimeError('Already borrowed')", 1]]}, summary['screening_errors']
        assert summary['duplicates_left_out_by_the_plan'] == 1
        questions = [json.loads(x) for x in (root / 'source/queries.jsonl').read_text().splitlines()]
        assert len(questions) == 2 and {q['doc'] for q in questions} <= set(manifest['measured'])

        # A fixed list: every kept PDF is measured, and the given questions are used where they can be.
        fixed = Path(tmp) / 'fixed'
        command = folder(fixed)
        digest = hashlib.sha256(b'a.pdf chunk 1').hexdigest()
        given = [{'id': 4, 'doc': 'a.pdf', 'text': 'from a', 'chunk_sha256': digest, 'extra': 1},
                 {'id': 5, 'doc': 'd.pdf', 'text': 'from a PDF the screening left out', 'chunk_sha256': 'x'},
                 {'id': 6, 'doc': 'b.pdf', 'text': 'from a chunk b.pdf does not have', 'chunk_sha256': digest}]
        (fixed / 'source/questions-given.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in given))
        (fixed / 'plan.json').write_text(json.dumps({
            'documents': 9, 'questions': 3, 'measure_all_kept': True, 'questions_given': 'source/questions-given.jsonl',
            'search': {'queries': 3, 'warm': 3},
            'cells': [{'label': 'screen-rr', 'stage': 'screening', 'documents': 9},
                      {'label': 'rr-1', 'stage': 'campaign', 'documents': 9},
                      {'label': 'rr-g4-1', 'stage': 'campaign', 'documents': 9}]}))
        subprocess.run(command, check=True, capture_output=True, text=True)
        summary, plan = json.loads((fixed / 'finalize.json').read_text()), json.loads((fixed / 'plan.json').read_text())
        manifest = json.loads((fixed / 'source/corpus-manifest.json').read_text())
        assert manifest['measured'] == ['a.pdf', 'b.pdf', 'c.pdf', 'g.pdf', 'i.pdf'] and summary['measured'] == 5
        assert plan['documents'] == 5 and plan['documents_planned'] == 9 and plan['questions'] == 1
        assert plan['search'] == {'queries': 1, 'warm': 1}
        assert [c['documents'] for c in plan['cells']] == [9, 5, 5], 'the screening entry keeps its number'
        questions = [json.loads(x) for x in (fixed / 'source/queries.jsonl').read_text().splitlines()]
        assert questions == [{'id': 4, 'doc': 'a.pdf', 'text': 'from a', 'chunk_sha256': digest}], questions
        assert summary['questions_given'] == {'given': 3, 'used': 1, 'not_used_by_reason': {
            'its PDF is not among the measured PDFs': 1,
            'its chunk is not among the chunks the reference run returned for its PDF': 1}}, summary['questions_given']
    print('ok  a fixed list: every kept PDF is measured and nothing fills up; a given question is used only when its '
          'PDF is measured and its chunk is in the reference')
    print('ok  kept: same chunks and vectors within tolerance, and a PDF without text in both systems; left out with its '
          'reason: different vectors, different text, failed, not a valid result')
    print('ok  vector files shorter than their chunks stop the step')
    print('ALL OK')


if __name__ == '__main__':
    main()
