"""Test of the step that fixes the PDFs of the timed runs after the two screening runs. Standard library only.

A made-up campaign folder holds two screening runs over seven candidates. finalize.py must keep a PDF only when
both runs returned it with the same chunk texts and with vectors that are equal within tolerance, and must say
why it left every other one out. A vector file that is shorter than the chunks it belongs to must stop it: two
missing vectors must never compare as equal.

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


def write_run(cell, docs):
    """docs: {name: [(text, vector), ...]} in the layout the ingestion client writes for a run without the store."""
    cell.mkdir(parents=True)
    vectors, rows = array.array('f'), []
    for name, chunks in docs.items():
        rows.append({'doc': name, 'ok': True, 'chunks': len(chunks), 'vector_offset': len(vectors) // DIM,
                     'text_sha256': [hashlib.sha256(t.encode()).hexdigest() for t, _ in chunks]})
        for _, v in chunks:
            vectors.extend(v)
    (cell / 'per_doc.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with (cell / 'vectors.f32').open('wb') as f:
        vectors.tofile(f)


def main():
    def vector(seed, shift=0.0):
        return [((seed * 31 + i) % 97) / 97.0 + (shift if i == 5 else 0.0) for i in range(DIM)]

    names = ['a.pdf', 'b.pdf', 'c.pdf', 'd.pdf', 'e.pdf', 'f.pdf', 'g.pdf']
    rr = {n: [('%s chunk %d' % (n, i), vector(k * 10 + i)) for i in range(2)] for k, n in enumerate(names)}
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
        write_run(root / 'cells/screen-rr', rr)
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
            path.write_bytes(path.read_bytes()[:DIM * 4 * 3])
        cut = subprocess.run(command, capture_output=True, text=True)
        assert cut.returncode != 0 and 'shorter than its chunks' in cut.stderr, cut.stderr[-300:]
        assert json.loads((short / 'source/corpus-manifest.json').read_text())['stage'] == 'candidates'

        root = Path(tmp) / 'run'
        subprocess.run(folder(root), check=True, capture_output=True, text=True)
        summary = json.loads((root / 'finalize.json').read_text())
        manifest = json.loads((root / 'source/corpus-manifest.json').read_text())
        assert summary['left_out'] == {'d.pdf': 'the two systems returned different vectors',
                                       'e.pdf': 'the two systems returned different chunk texts',
                                       'f.pdf': 'failed or timed out in LangGraph'}, summary['left_out']
        assert summary['left_out_by_reason'] == {'the two systems returned different vectors': 1,
                                                 'the two systems returned different chunk texts': 1,
                                                 'failed or timed out in LangGraph': 1}, summary['left_out_by_reason']
        assert summary['kept_by_screening'] == 4 and manifest['measured'] == ['a.pdf', 'b.pdf', 'c.pdf'] and manifest['stage'] == 'final'
        assert summary['screening_errors'] == {'rr': [], 'lg': [["RuntimeError('Already borrowed')", 1]]}, summary['screening_errors']
        assert summary['duplicates_left_out_by_the_plan'] == 1
        questions = [json.loads(x) for x in (root / 'source/queries.jsonl').read_text().splitlines()]
        assert len(questions) == 2 and {q['doc'] for q in questions} <= set(manifest['measured'])
    print('ok  kept: same chunks and vectors within tolerance; left out with its reason: different vectors, different text, failed')
    print('ok  vector files shorter than their chunks stop the step')
    print('ALL OK')


if __name__ == '__main__':
    main()
