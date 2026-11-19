"""Write REPORT.md for a campaign from its analysis.json. Every number in the report comes from that file.

Usage: python3 harness/analyze.py <campaign folder> && python3 harness/report.py <campaign folder>
"""
import collections
import json
import sys
from pathlib import Path

NAMES = {'rr': 'RocketRide', 'lg': 'LangGraph'}
PHASES = {'exact': 'No vector index (the table as each system leaves it)',
          'hnsw': 'With a vector index (pgvector HNSW, built by the harness)'}
CLIENT_NEAR_LIMIT = 0.8     # CPUs. Each client does its Python work on one CPU at a time.
MANY_EMPTY = 0.01           # share of valid answers without a result above which a level is flagged


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def num(x, digits=1):
    return 'n/a' if x is None else '%.*f' % (digits, x)


def spread(values, digits=1):
    """Lowest and highest of the runs of one system."""
    values = [v for v in values if v is not None]
    if not values:
        return 'n/a'
    if len(values) == 1:
        return 'one run'
    return '%.*f to %.*f' % (digits, min(values), digits, max(values))


def highest(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def ranges_overlap(a, b):
    """Do the runs of two systems overlap? None when a system has fewer than two runs: there is no spread to judge."""
    a, b = [v for v in a if v is not None], [v for v in b if v is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    return max(min(a), min(b)) <= min(max(a), max(b))


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(header) - 1)) + '|']
    return '\n'.join(out + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows])


def verdict(what, by_arm_values):
    """One sentence that describes the spread: do the runs of the two systems overlap. It describes, it does not
    decide: the table of run pairs shows every pair."""
    over = ranges_overlap(by_arm_values.get('rr', []), by_arm_values.get('lg', []))
    if over is None:
        return None
    return '%s: %s' % (what, 'the lowest and highest runs of the two systems overlap' if over
                       else 'the lowest and highest runs of the two systems do not overlap')


def pairs(runs):
    """[(n, RocketRide run n, LangGraph run n)] for every n both systems have a valid run of."""
    by_label = {r['label']: r for r in runs}
    numbers = sorted({int(label.split('-')[1]) for label in by_label if label.split('-')[-1].isdigit()})
    return [(n, by_label['rr-%d' % n], by_label['lg-%d' % n]) for n in numbers if 'rr-%d' % n in by_label and 'lg-%d' % n in by_label]


def share(a, b):
    return 'n/a' if not a or not b else '%.0f %%' % (100 * a / b)


def main():
    root = Path(sys.argv[1]).resolve()
    a = json.loads((root / 'analysis.json').read_text())
    runs = [r for r in a['runs'].values() if r['valid']]
    by_arm = {arm: [r for r in runs if r['arm'] == arm] for arm in NAMES}
    plan = a['plan']
    levels = plan['search']['levels']
    host, harness = a.get('host') or {}, a.get('harness') or {}
    doc = ['# RAG benchmark: RocketRide and LangGraph, ingestion and search', '',
           'Campaign `%s`. Written by `harness/report.py` from `analysis.json`, which `harness/analyze.py` recomputes '
           'from the files of the runs.' % a['tag'], '']

    # The harness commit is recorded with every run; the plan's is the fallback for a campaign without valid runs.
    revisions = [r['harness'] for r in runs if (r.get('harness') or {}).get('commit')] or ([harness] if harness.get('commit') else [])
    commits = sorted({h['commit'] for h in revisions})
    dirty = any(h.get('uncommitted_changes') for h in revisions)

    def known(value, unit=''):
        return 'not recorded' if value is None else '%s%s' % (value, unit)
    doc += ['## What was run', '',
            '- **PDFs per timed run:** %d, with %d in flight during ingestion.' % (plan['documents'], plan['concurrency']),
            '- **Search:** %d questions, %d results asked for per question, sent at %s in flight.' % (
                plan['search']['queries'], plan['search']['k'], ', '.join(str(x) for x in levels)),
            '- **Service under test:** CPUs `%s`, memory %s. **Database:** `%s` on CPUs `%s`, memory %s. **Client:** CPUs `%s`.' % (
                plan['service_cpus'], plan['service_memory'], a['db']['image'], a['db']['cpus'], a['db']['memory'], plan['client_cpus']),
            '- **Machine:** %s.' % ('not recorded' if not host else 'CPU %s; logical CPUs %s; physical cores %s; memory %s; kernel %s; Docker %s' % (
                ' / '.join(host.get('cpu_model') or ['not recorded']), known(host.get('logical_cpus')),
                known(host.get('physical_cores')), known(host.get('memory_GiB'), ' GiB'), known(host.get('kernel')),
                known(host.get('docker')))),
            '- **Harness:** %s.' % ('commit not recorded' if not commits else
                                    ('commit `%s`' % commits[0] if len(commits) == 1 else
                                     '**the runs were taken with %d different commits**: %s' % (len(commits), ', '.join(
                                         '`%s` %s' % (r['label'], (r.get('harness') or {}).get('commit')) for r in runs)))
                                    + (', with uncommitted changes in at least one run' if dirty else '')),
            '- **RocketRide task threads:** %d. Both systems run as shipped: nothing inside either is changed.' % plan['workers'],
            '- **Memory limits:** the limit of the service under test covers all of RocketRide, parser included. On the '
            'LangGraph side it covers the service; its parser runs in a container of its own without a limit.',
            '- **Load clients:** RocketRide through its client SDK on one websocket; LangGraph over HTTP connections that '
            'stay open, one per worker thread of the client.',
            '- **Timed runs, in the order taken:** %s.' % ', '.join('`%s`' % l for l in a['campaign']), '']
    if a['not_run'] or (a['status'].get('failed') or a['status'].get('skipped')):
        doc += ['**Not every run completed.** Not run: %s. Failed: %s. Skipped: %s.' % (
            a['not_run'] or 'none', a['status'].get('failed') or 'none', a['status'].get('skipped') or 'none'), '']
    invalid = [r for r in a['runs'].values() if not r['valid']]
    if invalid:
        doc += ['**Runs without a valid result:** ' + '; '.join('`%s` (%s)' % (r['label'], r.get('error')) for r in invalid), '']
    if a.get('failed_attempts'):
        doc += ['**Attempts that failed and were run again** (their files are kept in the campaign folder): ' +
                ', '.join('`%s`' % x for x in a['failed_attempts']) + '.', '']

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
            doc += ['%d of %d candidates gave the same chunks and vectors in both systems. The timed runs use the first %d: '
                    '%d chunks, %d of the PDFs have no text. %d questions.' % (
                        fin['kept_by_screening'], fin['candidates'], fin['measured'], fin['chunks_of_measured'],
                        fin['documents_without_chunks'], fin['questions']), '']
            reasons = fin.get('left_out_by_reason') or {}
            doc += ['The timed runs therefore cover only PDFs that both systems handle the same way. Candidates left out: ' +
                    ('none' if not reasons else '; '.join('%d %s' % (n, why) for why, n in reasons.items())) +
                    '. Files left out of the candidate list as copies of an earlier file: %s.' % fin.get('duplicates_left_out_by_the_plan', 'n/a'), '']
            for arm, found in (fin.get('screening_errors') or {}).items():
                if found:
                    doc += ['Most frequent errors of the %s screening run: ' % NAMES[arm] +
                            '; '.join('%d times `%s`' % (n, str(e)[:160].replace('`', "'").replace('|', '/')) for e, n in found) + '.', '']

    doc += ['## Ingestion', '']
    base = mean([r['docs_per_s'] for r in by_arm['lg']])
    rows = []
    for arm in NAMES:
        rs = by_arm[arm]
        if not rs:
            continue
        rate = mean([r['docs_per_s'] for r in rs])
        rows.append([NAMES[arm], ' / '.join(num(r['docs_per_s'], 3) for r in rs), num(rate, 3),
                     spread([r['docs_per_s'] for r in rs], 3),
                     'n/a' if not base else '%.0f %%' % (100 * rate / base),
                     # chunks that passed the read-back, per second
                     num(mean([r.get('chunks_per_s') for r in rs]), 1),
                     num(mean([r['cpu_s_per_doc'] for r in rs])), num(mean([r.get('service_peak_own_memory_GiB') for r in rs])),
                     num(mean([r['p50_s'] for r in rs])), num(mean([r.get('db_ms_per_doc') for r in rs])),
                     ' / '.join(str(r.get('commits')) for r in rs),
                     num(highest([r.get('client_average_cpus') for r in rs]), 2)])
    doc += [table(['System', 'PDFs per second, each run', 'Mean', 'Lowest to highest', 'Share of LangGraph',
                   'Verified chunks per second', 'CPU-seconds per PDF',
                   'Peak own memory, GiB', 'Time per PDF, middle value, s', 'Database time per PDF, ms', 'Commits',
                   'Client CPUs busy, highest run'], rows), '']
    said = verdict('PDFs per second', {arm: [r['docs_per_s'] for r in by_arm[arm]] for arm in NAMES})
    if said:
        doc += [said + '.', '']
    paired = pairs(runs)
    if paired:
        # Every pair on its own: the n-th run of each system. A ratio that changes side from pair to pair is no result.
        header = ['Pair', 'Ingestion']
        for phase, title in PHASES.items():
            if any(phase in (r.get('search') or {}) for r in runs):
                header += ['Search %s, %d in flight' % ('without index' if phase == 'exact' else 'with index', n) for n in levels]
        rows = []
        for n, rr, lg in paired:
            row = ['rr-%d / lg-%d' % (n, n), share(rr['docs_per_s'], lg['docs_per_s'])]
            for phase in PHASES:
                if any(phase in (r.get('search') or {}) for r in runs):
                    ours, theirs = (rr.get('search') or {}).get(phase), (lg.get('search') or {}).get(phase)
                    row += [share(ours['levels'][i]['questions_per_s'], theirs['levels'][i]['questions_per_s']) if ours and theirs else 'n/a'
                            for i in range(len(levels))]
            rows.append(row)
        doc += ['**Every pair of runs: RocketRide\'s rate as a share of LangGraph\'s.**', '', table(header, rows), '']
    steps = by_arm['lg'][0].get('langgraph_step_ms_per_doc') if by_arm['lg'] else None
    if steps:
        doc += ['LangGraph, mean time per PDF and graph step (first run), ms: ' +
                ', '.join('%s %s' % (k, num(v)) for k, v in steps.items()) + '.', '']

    client_busy = [('ingestion', r['label'], r['client_average_cpus']) for r in runs if (r.get('client_average_cpus') or 0) >= CLIENT_NEAR_LIMIT]
    client_measured = sum(1 for r in runs if r.get('client_average_cpus') is not None)
    many_empty, empty_forms, index_not_used, index_not_recorded = [], [], [], []
    for phase, title in PHASES.items():
        if not any(phase in (r.get('search') or {}) for r in runs):
            continue
        doc += ['## Search: ' + title[0].lower() + title[1:], '',
                'A rate counts answers that passed their check only. Mean of the runs of each system, with the lowest and '
                'highest run.', '']
        rows, facts, said = [], [], []
        for i, n in enumerate(levels):
            rates = {}
            for arm in NAMES:
                rs = [r['search'][phase] for r in by_arm[arm] if phase in (r.get('search') or {})]
                if not rs:
                    continue
                ls = [s['levels'][i] for s in rs]
                rates[arm] = [l['questions_per_s'] for l in ls]
                rows.append([NAMES[arm], n, num(mean(rates[arm])), spread(rates[arm]),
                             num(mean([(l['latency_ms'] or {}).get('p50') for l in ls]), 0),
                             num(mean([(l['latency_ms'] or {}).get('p95') for l in ls]), 0),
                             num(mean([l['average_cpus'] for l in ls]), 1), num(mean([l['db_average_cpus'] for l in ls]), 2),
                             num(highest([l.get('client_average_cpus') for l in ls]), 2)])
            one = verdict('At %d in flight' % n, rates)
            if one:
                said.append(one)
        doc += [table(['System', 'In flight', 'Valid answers per second', 'Lowest to highest', 'Middle latency, ms',
                       '95th percentile, ms', 'Service CPUs busy', 'Database CPUs busy', 'Client CPUs busy, highest run'], rows), '']
        if said:
            doc += ['. '.join(said) + '.', '']
        for arm in NAMES:
            picked = [r for r in by_arm[arm] if phase in (r.get('search') or {})]
            rs = [r['search'][phase] for r in picked]
            if not rs:
                continue
            every = [l for s in rs for l in s['levels']]
            row = [NAMES[arm], sum(l['questions'] for l in every), sum(l['answers_not_valid'] for l in every),
                   sum(l['questions_with_no_result'] for l in every), num(mean([s['db_ms_per_search'] for s in rs])),
                   num(mean([s['levels'][0]['cpu_ms_per_question'] for s in rs]), 0)]
            if phase == 'hnsw':
                # Beside the rate: how much of the exact answer the index returns in the same run.
                row.append(num(mean([100 * r['index_against_exact']['mean_overlap'] for r in picked if r.get('index_against_exact')]), 0))
                # And whether Postgres used the index at all: index scans per timed search, from the database's counter.
                used = [(r['label'], (r['search'][phase].get('index') or {})) for r in picked]
                row.append(num(mean([i['index_scans_during_search'] / i['searches_timed'] for _, i in used
                                     if i.get('searches_timed') and i.get('index_scans_during_search') is not None]), 2))
                index_not_used += ['`%s`' % label for label, i in used if i.get('index_used') is False]
                index_not_recorded += ['`%s`%s' % (label, ' (its database sessions were still inside a transaction, so their '
                                                   'counts had not arrived)' if i.get('sessions_still_in_a_transaction') else '')
                                       for label, i in used if i.get('index_used') is None]
            facts.append(row)
            for r in picked:
                where = 'no index' if phase == 'exact' else 'with the index'
                many_empty += ['`%s` %s at %d in flight: %d of %d' % (r['label'], where, l['in_flight'], l['questions_with_no_result'], l['answers_valid'])
                               for l in r['search'][phase]['levels'] if l['questions_with_no_result'] > MANY_EMPTY * max(1, l['answers_valid'])]
                if arm == 'rr':
                    empty_forms.append((r['label'], where, r['search'][phase].get('empty_answer_form')))
                client_measured += sum(1 for l in r['search'][phase]['levels'] if l.get('client_average_cpus') is not None)
                client_busy += [('search, %s' % ('no index' if phase == 'exact' else 'with the index'), r['label'], l['client_average_cpus'])
                                for l in r['search'][phase]['levels'] if (l.get('client_average_cpus') or 0) >= CLIENT_NEAR_LIMIT]
        header = ['System', 'Answers checked', 'Not valid', 'Valid, without a result', 'Time inside the database per search, ms',
                  'Service CPU per question at %d in flight, ms' % levels[0]]
        if phase == 'hnsw':
            header += ['Results in common with the same run\'s search without the index, %', 'Index scans per timed search']
        doc += [table(header, facts), '']
        if phase == 'hnsw':
            if index_not_used:
                doc += ['**The index was present but Postgres did not use it in:** %s. Their rates in this section are '
                        'rates of a search without an index.' % ', '.join(index_not_used), '']
            elif index_not_recorded:
                doc += ['Whether Postgres used the index could not be read from its counter for: %s. The plan below and '
                        'the time inside the database per search are the other evidence.' % ', '.join(index_not_recorded), '']
            else:
                doc += ['Postgres used the index in every run of this section. The last column is its own count of index '
                        'scans per timed search; the warm-up questions are counted in, so a value a little above 1 means '
                        'every search went through the index.', '']
            plans = [(r['label'], (r['search'][phase].get('plan') or {})) for r in runs if phase in (r.get('search') or {})]
            shown = {}
            for label, plan in plans:
                if plan.get('plan'):
                    shown.setdefault(label.split('-')[0], (label, plan))
            for arm, (label, plan) in shown.items():
                doc += ['Plan of the search statement %s sent most often (`%s`, taken after the timed questions). It is the '
                        'plan for the statement with placeholders, which can differ from the plan of a search with its real '
                        'values; the count of index scans above is what shows whether the index was used:' % (NAMES[arm], label),
                        '', '```', plan['query'][:400], '', *plan['plan'][:12], '```', '']
        index = next((r['search'][phase]['index'] for r in runs if (r.get('search') or {}).get(phase, {}).get('index')), None)
        if index:
            doc += ['The index took %.0f s to build and is %.0f MB on %d rows (first run). It is approximate: the last column '
                    'above says how much of the exact answer it returns, and a rate with the index has to be read with it.' % (
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
    timed = [r for r in stored if r.get('commit_times_checked')]
    unconfirmed = [(label, where, form) for label, where, form in empty_forms if not (form or {}).get('confirmed')]
    doc += ['- **Read-back of the stored rows:** %d of %d runs passed. A run passes when every expected chunk is in the '
            'table under its own PDF and chunk number exactly as often as expected, every stored vector equals the '
            'reference vector within tolerance, the table holds no row that is not explained, and every document was '
            'committed before the client had its answer.' % (
                sum(1 for r in stored if r['check_pass']), len(stored)),
            '- **Rows that are not expected chunks:** %s.' % ('n/a' if not stored else '; '.join(
                '`%s` %d of the store itself, %d unexplained, %d with an expected text under another PDF or chunk number' % (
                    r['label'], r.get('rows_of_the_store_itself') or 0, r.get('rows_unexplained') or 0, r.get('rows_misfiled') or 0)
                for r in stored)),
            '- **Stored before the answer:** ' + (
                'not checked: no run recorded the commit time of its rows.' if not timed else
                'in %d of %d runs the last commit of every document was compared with the moment the client had that '
                'document\'s answer, and the commit time of every row with the instant the client\'s clock stopped. '
                'Documents committed after their answer: %d. Rows committed after the clock stopped: %d.' % (
                    len(timed), len(stored), sum(r['documents_committed_after_their_answer'] for r in timed),
                    sum(r['rows_committed_after_the_clock_stopped'] for r in timed))),
            '- **Search answers:** a response that is not an answer fails its run, so none is in a rate. '
            'Levels where more than %.0f %% of the valid answers held no result: %s.' % (
                100 * MANY_EMPTY, 'none' if not many_empty else '**' + '; '.join(many_empty) + '**. An answer without a result is quick, '
                'so read those rates with the number of empty answers'),
            '- **Form of a RocketRide answer without results:** ' + (
                'no RocketRide search phase in this campaign.' if not empty_forms else
                'confirmed against the engine before the timed questions in %d of %d search phases%s. A response in any other form '
                'is not counted as an answer in either case.' % (
                    len(empty_forms) - len(unconfirmed), len(empty_forms),
                    '' if not unconfirmed else '; not confirmed in ' + '; '.join('`%s` %s (%s)' % (
                        label, where, 'not recorded' if not form else form.get('why') or 'the engine returned %s results for a question that cannot match' % form.get('results_returned'))
                        for label, where, form in unconfirmed))),
            '- **Load client:** ' + (
                'its CPU use was not recorded.' if not client_measured else
                'in none of the %d timed parts did the client average %.1f CPUs or more. That makes a limit in the client '
                'unlikely; it does not rule one out.' % (client_measured, CLIENT_NEAR_LIMIT) if not client_busy else
                '**the client was near its limit** (each client does its Python work on one CPU at a time) in: ' +
                '; '.join('`%s` %s, %.2f CPUs' % (label, what, cpus) for what, label, cpus in client_busy) +
                '. Those rates may describe the client and not the system.'), '']

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
                   'Database time per PDF, ms', 'Commits', 'Search, no index (valid answers per second at each level)',
                   'Search, with index', 'Read-back'], rows), '']
    images = collections.OrderedDict()
    for r in runs:
        for name, image in (r.get('images') or {}).items():
            role = 'database' if name.endswith('-db') else 'parser of the LangGraph side' if name.endswith('-parser') else NAMES[r['arm']]
            images.setdefault(role, set()).add(image[:19])
    if images:
        doc += ['Images the runs used (image id, as Docker reports it): ' +
                '; '.join('%s `%s`' % (role, '`, `'.join(sorted(ids))) for role, ids in images.items()) + '.', '']
    doc += ['Rates from different sittings on the same machine do not compare reliably. Compare systems within one campaign.', '']
    (root / 'REPORT.md').write_text('\n'.join(doc))
    print('wrote', root / 'REPORT.md')


if __name__ == '__main__':
    main()
