"""Write REPORT.md for a campaign from its analysis.json. Every number in the report comes from that file.

Usage: python3 harness/analyze.py <campaign folder> && python3 harness/report.py <campaign folder>
"""
import json
import sys
from pathlib import Path

NAMES = {'rr': 'RocketRide', 'lg': 'LangGraph'}
PHASES = {'exact': 'No vector index (the table as each system leaves it)',
          'hnsw': 'With a vector index (pgvector HNSW, built by the harness)'}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def num(x, digits=1):
    return 'n/a' if x is None else '%.*f' % (digits, x)


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(header) - 1)) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows])


def main():
    root = Path(sys.argv[1]).resolve()
    a = json.loads((root / 'analysis.json').read_text())
    runs = [r for r in a['runs'].values() if r['valid']]
    by_arm = {arm: [r for r in runs if r['arm'] == arm] for arm in NAMES}
    plan = a['plan']
    levels = plan['search']['levels']
    doc = ['# RAG benchmark: RocketRide and LangGraph, ingestion and search', '',
           'Campaign `%s`. Written by `harness/report.py` from `analysis.json`, which `harness/analyze.py` recomputes '
           'from the files of the runs.' % a['tag'], '']

    doc += ['## What was run', '',
            '- **PDFs per timed run:** %d, with %d in flight during ingestion.' % (plan['documents'], plan['concurrency']),
            '- **Search:** %d questions, %d results asked for per question, sent at %s in flight.' % (
                plan['search']['queries'], plan['search']['k'], ', '.join(str(x) for x in levels)),
            '- **Service under test:** CPUs `%s`, memory %s. **Database:** `%s` on CPUs `%s`, memory %s. **Client:** CPUs `%s`.' % (
                plan['service_cpus'], plan['service_memory'], a['db']['image'], a['db']['cpus'], a['db']['memory'], plan['client_cpus']),
            '- **RocketRide task threads:** %d. Both systems run as shipped: nothing inside either is changed.' % plan['workers'],
            '- **Timed runs, in the order taken:** %s.' % ', '.join('`%s`' % l for l in a['campaign']), '']
    if a['not_run'] or (a['status'].get('failed') or a['status'].get('skipped')):
        doc += ['**Not every run completed.** Not run: %s. Failed: %s. Skipped: %s.' % (
            a['not_run'] or 'none', a['status'].get('failed') or 'none', a['status'].get('skipped') or 'none'), '']
    invalid = [r for r in a['runs'].values() if not r['valid']]
    if invalid:
        doc += ['**Runs without a valid result:** ' + '; '.join('`%s` (%s)' % (r['label'], r.get('error')) for r in invalid), '']

    if a['screening']:
        doc += ['## Screening', '', 'Every candidate PDF through both systems without the store.', '']
        rows = []
        for label, s in a['screening'].items():
            ref = s.get('against_reference')
            rows.append([label, s['documents'], s['failures'], s['chunks'], num(s['docs_per_s'], 3),
                         'the reference' if not ref else '%d documents differ in text, %s' % (
                             ref['documents_with_different_chunk_text'],
                             '%d vector values outside tolerance' % ref['vector_values_outside_tolerance'] if ref['vectors_compared']
                             else 'vectors not compared (vector files not in this copy)')])
        doc += [table(['Run', 'PDFs with a result', 'PDFs that failed', 'Chunks', 'PDFs per second', 'Against the reference'], rows), '']
        fin = a.get('finalize')
        if fin:
            doc += ['%d of %d candidates gave the same chunks in both systems. The timed runs use the first %d: %d chunks, '
                    '%d of the PDFs have no text. %d questions.' % (
                        fin['kept_by_screening'], fin['candidates'], fin['measured'], fin['chunks_of_measured'],
                        fin['documents_without_chunks'], fin['questions']), '']

    doc += ['## Ingestion', '']
    base = mean([r['docs_per_s'] for r in by_arm['lg']])
    rows = []
    for arm in NAMES:
        rs = by_arm[arm]
        if not rs:
            continue
        rate = mean([r['docs_per_s'] for r in rs])
        rows.append([NAMES[arm], ' / '.join(num(r['docs_per_s'], 3) for r in rs), num(rate, 3),
                     'n/a' if not base else '%.0f %%' % (100 * rate / base),
                     num(mean([r['cpu_s_per_doc'] for r in rs])), num(mean([r.get('service_peak_own_memory_GiB') for r in rs])),
                     num(mean([r['p50_s'] for r in rs])), num(mean([r.get('db_ms_per_doc') for r in rs])),
                     ' / '.join(str(r.get('commits')) for r in rs)])
    doc += [table(['System', 'PDFs per second, each run', 'Mean', 'Share of LangGraph', 'CPU-seconds per PDF',
                   'Peak own memory, GiB', 'Time per PDF, middle value, s', 'Database time per PDF, ms', 'Commits'], rows), '']
    steps = by_arm['lg'][0].get('langgraph_step_ms_per_doc') if by_arm['lg'] else None
    if steps:
        doc += ['LangGraph, mean time per PDF and graph step (first run), ms: ' +
                ', '.join('%s %s' % (k, num(v)) for k, v in steps.items()) + '.', '']

    for phase, title in PHASES.items():
        if not any(phase in (r.get('search') or {}) for r in runs):
            continue
        doc += ['## Search: ' + title[0].lower() + title[1:], '']
        header = ['System']
        for n in levels:
            header += ['%d in flight, questions per second' % n, 'Middle latency, ms']
        header += ['Database CPUs busy at %d in flight' % levels[-1], 'Time inside the database per search, ms',
                   'Service CPU per question at %d in flight, ms' % levels[0]]
        rows = []
        for arm in NAMES:
            rs = [r['search'][phase] for r in by_arm[arm] if phase in (r.get('search') or {})]
            if not rs:
                continue
            row = [NAMES[arm]]
            for i in range(len(levels)):
                row += [num(mean([s['levels'][i]['questions_per_s'] for s in rs])),
                        num(mean([s['levels'][i]['latency_ms']['p50'] for s in rs]), 0)]
            row += [num(mean([s['levels'][-1]['db_average_cpus'] for s in rs]), 2), num(mean([s['db_ms_per_search'] for s in rs])),
                    num(mean([s['levels'][0]['cpu_ms_per_question'] for s in rs]), 0)]
            rows.append(row)
        doc += [table(header, rows), '', 'Mean of the runs of each system.', '']
        index = next((r['search'][phase]['index'] for r in runs if (r.get('search') or {}).get(phase, {}).get('index')), None)
        if index:
            doc += ['The index took %.0f s to build and is %.0f MB on %d rows (first run).' % (
                index['build_s'], index['index_bytes'] / 1e6, index['rows']), '']

    if any(r.get('agreement') or r.get('index_against_exact') for r in runs):
        doc += ['## Do the two systems return the same results', '',
                ('Each run compared with `%s`, question by question.' % a['search_reference'] if a['search_reference']
                 else 'There is no valid LangGraph run to compare with.') +
                ' Results under similarity 0.20 are left out on both sides, because RocketRide\'s store never returns them.', '']
        rows = []
        for r in runs:
            ag, own = r.get('agreement') or {}, r.get('index_against_exact')
            first = (r.get('search') or {}).get('exact') or {}
            with_index = (r.get('search') or {}).get('hnsw') or {}
            rows.append([
                r['label'],
                'the reference' if r['label'] == a['search_reference'] else
                ('n/a' if not ag.get('exact') else '%d of %d' % (ag['exact']['identical_lists'], ag['exact']['questions'])),
                'the reference' if r['label'] == a['search_reference'] else
                ('n/a' if not ag.get('hnsw') else '%.0f %%' % (100 * ag['hnsw']['mean_overlap'])),
                'n/a' if not own else '%.0f %%' % (100 * own['mean_overlap']),
                'n/a' if not first else '%.0f %%' % (100 * first['levels'][0]['source_chunk_in_results']),
                'n/a' if not with_index else '%.0f %%' % (100 * with_index['levels'][0]['source_chunk_in_results'])])
        doc += [table(['Run', 'No index: questions with the same result list', 'With the index: share of results in common',
                       'With the index against its own exact scan', 'Source chunk among the results, no index',
                       'Source chunk among the results, with the index'], rows), '']

    doc += ['## Checks', '']
    stored = [r for r in runs if 'check_pass' in r]
    doc += ['- **Read-back of the stored rows:** %d of %d runs passed. A run passes when every expected chunk is in the '
            'table exactly as often as expected and every stored vector equals the reference vector within tolerance.' % (
                sum(1 for r in stored if r['check_pass']), len(stored)), '']

    doc += ['## Every run', '']
    rows = []
    for r in a['runs'].values():
        if not r['valid']:
            rows.append([r['label'], NAMES[r['arm']], 'no valid result', '', '', '', '', '', '', ''])
            continue
        s = r.get('search') or {}

        def rates(phase):
            return 'n/a' if phase not in s else ' / '.join(num(l['questions_per_s']) for l in s[phase]['levels'])
        rows.append([r['label'], NAMES[r['arm']], num(r['docs_per_s'], 3), num(r['cpu_s_per_doc']),
                     num(r.get('service_peak_own_memory_GiB')), num(r.get('db_ms_per_doc')), r.get('commits', 'n/a'),
                     rates('exact'), rates('hnsw'), 'passed' if r.get('check_pass') else 'FAILED'])
    doc += [table(['Run', 'System', 'Ingestion, PDFs per second', 'CPU-seconds per PDF', 'Peak own memory, GiB',
                   'Database time per PDF, ms', 'Commits', 'Search, no index (questions per second at each level)',
                   'Search, with index', 'Read-back'], rows), '',
            'Rates from different sittings on the same machine do not compare reliably. Compare systems within one campaign.', '']
    (root / 'REPORT.md').write_text('\n'.join(doc))
    print('wrote', root / 'REPORT.md')


if __name__ == '__main__':
    main()
