"""After the two screening runs: fix the PDFs and the questions of the timed runs.

A candidate PDF is kept when both screening runs returned a valid result for it, the chunk texts are the same in
RocketRide and in LangGraph, and so are the vectors within the tolerance the read-back uses. The first N kept
candidates, in candidate order, are the campaign's PDFs (N is the plan's `documents`). Every candidate left out is
listed with its reason, and the errors of the screening runs are counted by message.

The questions were cut by the RocketRide screening run from the chunk texts it returned: for every document,
12 consecutive words from the middle of one of its chunks. The plan's number of questions is taken, spread evenly
over the kept PDFs. So for every question the chunk it came from is known.

Two options of the plan change this:

  measure_all_kept   (a fixed list of PDFs was given) every kept candidate is measured, however many those are.
                     Nothing fills up for a candidate that was left out. The plan's number of PDFs is set to the
                     number kept.
  questions_given    the given questions are used in place of the ones cut here. A given question is used when
                     its PDF was kept and its chunk is one of the chunks the reference run returned for that PDF.

Writes source/corpus-manifest.json (stage "final"), source/queries.jsonl and finalize.json. Standard library only.

Usage: python3 harness/finalize.py <campaign folder>
"""
import array
import collections
import json
import sys
from pathlib import Path

ATOL, RTOL, DIM = 1e-5, 1e-4, 384        # the tolerance of dbcheck.py


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def floats(path):
    data = array.array('f')
    with open(path, 'rb') as f:
        data.fromfile(f, Path(path).stat().st_size // data.itemsize)
    return data


def same_vectors(a, b):
    """Two runs of float32 values, equal within the tolerance. Equal bytes are the common case and are settled first."""
    return a == b or (len(a) == len(b) and all(abs(x - y) <= ATOL + RTOL * abs(y) for x, y in zip(a, b)))


def errors(path):
    """The errors of a screening run, counted by message, most frequent first."""
    if not path.exists():
        return []
    return collections.Counter(f['error'] for f in json.loads(path.read_text())).most_common(5)


def main():
    root = Path(sys.argv[1]).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    want, questions = plan['documents'], plan['questions']
    path = root / 'source/corpus-manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest.get('stage') == 'candidates', 'the manifest was already finalized'
    candidates = manifest['measured']
    rr = {r['doc']: r for r in lines(root / 'cells/screen-rr/per_doc.jsonl')}
    lg = {r['doc']: r for r in lines(root / 'cells/screen-lg/per_doc.jsonl')}
    rr_vectors, lg_vectors = floats(root / 'cells/screen-rr/vectors.f32'), floats(root / 'cells/screen-lg/vectors.f32')

    def vectors(row, data):
        part = data[row['vector_offset'] * DIM:(row['vector_offset'] + len(row['text_sha256'])) * DIM]
        assert len(part) == len(row['text_sha256']) * DIM, 'the vector file of a screening run is shorter than its chunks: %s' % row['doc']
        return part

    why = {}
    good = []
    for name in candidates:
        if name not in rr or name not in lg:
            why[name] = 'failed or timed out in %s' % ' and '.join(s for s, d in [('RocketRide', rr), ('LangGraph', lg)] if name not in d)
        elif not rr[name]['ok'] or not lg[name]['ok']:
            # An error or a response of another shape. It is never taken for a PDF without text.
            why[name] = 'not a valid result in %s' % ' and '.join(s for s, d in [('RocketRide', rr), ('LangGraph', lg)] if not d[name]['ok'])
        elif rr[name]['text_sha256'] != lg[name]['text_sha256']:
            why[name] = 'the two systems returned different chunk texts'
        elif not same_vectors(vectors(lg[name], lg_vectors), vectors(rr[name], rr_vectors)):
            # Left out here, or the read-back of every timed LangGraph run would fail on it at the end.
            why[name] = 'the two systems returned different vectors'
        else:
            good.append(name)
    planned = want
    if plan.get('measure_all_kept'):
        want = len(good)
        assert want, 'no candidate passed the screening'
    assert len(good) >= want, 'only %d candidates passed the screening; %d are needed' % (len(good), want)
    measured = good[:want]
    keep = set(measured)
    pool = [q for q in lines(root / 'cells/screen-rr/queries-all.jsonl') if q['doc'] in keep]
    given_out = None
    if plan.get('questions_given'):
        given = lines(root / plan['questions_given'])
        picked, given_out = [], collections.Counter()
        for q in given:
            if q['doc'] not in keep:
                given_out['its PDF is not among the measured PDFs'] += 1
            elif q['chunk_sha256'] not in rr[q['doc']]['text_sha256']:
                given_out['its chunk is not among the chunks the reference run returned for its PDF'] += 1
            else:
                picked.append({k: q[k] for k in ['id', 'doc', 'text', 'chunk_sha256']})
        assert picked, 'none of the given questions can be used'
        questions = len(picked)
    else:
        assert len(pool) >= questions, 'only %d questions available; %d are needed' % (len(pool), questions)
        step = len(pool) / questions
        picked = [dict(pool[int(i * step)], id=i) for i in range(questions)]
    (root / 'source/queries.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in picked))
    if want != plan['documents'] or questions != plan['questions']:
        # The plan follows what the screening left: every timed run ingests `want` PDFs and asks `questions` questions.
        plan.update(documents=want, questions=questions, documents_planned=planned)
        plan['search'].update(queries=questions, warm=min(plan['search']['warm'], questions))
        for c in plan['cells']:
            if c['stage'] == 'campaign':
                c['documents'] = want
        tmp = root / 'plan.json.tmp'
        tmp.write_text(json.dumps(plan, indent=2) + '\n')
        tmp.replace(root / 'plan.json')
    summary = {'candidates': len(candidates), 'kept_by_screening': len(good), 'left_out': why,
               'left_out_by_reason': dict(collections.Counter(why.values()).most_common()),
               'screening_errors': {'rr': errors(root / 'cells/screen-rr/failures.json'),
                                    'lg': errors(root / 'cells/screen-lg/failures.json')},
               'duplicates_left_out_by_the_plan': len(manifest.get('duplicates_left_out', [])),
               'measured': len(measured), 'measure_all_kept': bool(plan.get('measure_all_kept')),
               'questions_given': None if given_out is None else {
                   'given': len(given), 'used': len(picked), 'not_used_by_reason': dict(given_out)},
               'chunks_of_measured': sum(rr[n]['chunks'] for n in measured),
               # PDFs both systems read as holding no text. They are in the timed runs and store no rows.
               'documents_without_chunks': sum(1 for n in measured if not rr[n]['chunks']),
               'questions_available': len(pool), 'questions': len(picked),
               'screening_failures': {s: len(json.loads(p.read_text())) if p.exists() else 0
                                      for s, p in [('rr', root / 'cells/screen-rr/failures.json'),
                                                   ('lg', root / 'cells/screen-lg/failures.json')]}}
    (root / 'finalize.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest.update(measured=measured, stage='final', candidates=candidates)
    path.write_text(json.dumps(manifest) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'left_out'}))
    if why:
        print('left out (first 8):', json.dumps(dict(list(why.items())[:8])))


if __name__ == '__main__':
    main()
