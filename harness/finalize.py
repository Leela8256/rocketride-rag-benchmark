"""After the two screening runs: fix the PDFs and the questions of the timed runs.

A candidate PDF is kept when both screening runs returned a valid result for it and the chunk texts are the same
in RocketRide and in LangGraph. The first N kept candidates, in candidate order, are the campaign's PDFs (N is the
plan's `documents`).

The questions were cut by the RocketRide screening run from the chunk texts it returned: for every document,
12 consecutive words from the middle of one of its chunks. The plan's number of questions is taken, spread evenly
over the kept PDFs. So for every question the chunk it came from is known.

Writes source/corpus-manifest.json (stage "final"), source/queries.jsonl and finalize.json. Standard library only.

Usage: python3 harness/finalize.py <campaign folder>
"""
import json
import sys
from pathlib import Path


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


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
    why = {}
    good = []
    for name in candidates:
        if name not in rr or name not in lg:
            why[name] = 'failed or timed out in %s' % ' and '.join(s for s, d in [('RocketRide', rr), ('LangGraph', lg)] if name not in d)
        elif not rr[name]['ok'] or not lg[name]['ok']:
            why[name] = 'invalid result'
        elif rr[name]['text_sha256'] != lg[name]['text_sha256']:
            why[name] = 'the two systems returned different chunk texts'
        else:
            good.append(name)
    assert len(good) >= want, 'only %d candidates passed the screening; %d are needed' % (len(good), want)
    measured = good[:want]
    keep = set(measured)
    pool = [q for q in lines(root / 'cells/screen-rr/queries-all.jsonl') if q['doc'] in keep]
    assert len(pool) >= questions, 'only %d questions available; %d are needed' % (len(pool), questions)
    step = len(pool) / questions
    picked = [dict(pool[int(i * step)], id=i) for i in range(questions)]
    (root / 'source/queries.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in picked))
    summary = {'candidates': len(candidates), 'kept_by_screening': len(good), 'left_out': why, 'measured': len(measured),
               'chunks_of_measured': sum(rr[n]['chunks'] for n in measured),
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
