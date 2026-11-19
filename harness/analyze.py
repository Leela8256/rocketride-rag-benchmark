"""Recompute every number of a campaign from the raw files of its runs and write analysis.json.

Standard library only. Nothing is measured here; every value comes from files the runs wrote.

For every timed run:
  ingestion   PDFs per second, time per PDF, CPU-seconds per PDF and average CPUs of the service under test,
              peak memory of the service, CPU and memory of the database, statements by kind (calls and time
              inside the database), commits, write-ahead log volume, rows and table size, and for LangGraph the
              time per graph step (from the Server-Timing header of every response)
  check       the read-back made by dbcheck.py (db-check.json)
  search      for every state of the table (exact scan, or with the HNSW index) and number of questions in
              flight: valid answers per second, latency, CPU of the service, of the database and of the load
              client, time inside the database per search, how often the chunk a question was cut from comes back
  agreement   the result lists of every run compared with the first LangGraph run, question by question.
              RocketRide's store never returns a result under similarity 0.20, so results under 0.20 are left
              out on both sides before comparing.

For the screening runs: what each kept and left out, and the LangGraph run's chunk texts and vectors compared
with the RocketRide run's.

Usage: python3 harness/analyze.py <campaign folder>
"""
import array
import collections
import json
import re
import sys
from pathlib import Path

ATOL, RTOL, DIM = 1e-5, 1e-4, 384
FLOOR = 0.2
OWN = ('pg_stat_database', 'pg_stat_wal')        # the harness's own two baseline queries, not the system under test
D = None                                         # the campaign folder


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []


def floats(path):
    data = array.array('f')
    with open(path, 'rb') as f:
        data.fromfile(f, Path(path).stat().st_size // data.itemsize)
    return data


def db_name(label):
    return 'ragbench-%s-db' % label


def compare_with_reference(cell, reference):
    """A run without the store returns its chunks and vectors: same chunk texts, vectors within tolerance.

    The vector files are large. When a copy of the campaign folder was made without them, only the texts are compared.
    """
    ref_rows = {r['doc']: r for r in lines(reference / 'per_doc.jsonl')}
    rows = lines(cell / 'per_doc.jsonl')
    have_vectors = (reference / 'vectors.f32').exists() and (cell / 'vectors.f32').exists()
    ref_vectors = floats(reference / 'vectors.f32') if have_vectors else None
    got = floats(cell / 'vectors.f32') if have_vectors else None
    text_mismatch = chunks = outside = 0
    worst = 0.0
    for row in rows:
        r = ref_rows.get(row['doc'])
        if r is None or r['text_sha256'] != row['text_sha256']:
            text_mismatch += 1
            continue
        if not have_vectors:
            continue
        n = len(row['text_sha256']) * DIM
        a = got[row['vector_offset'] * DIM:row['vector_offset'] * DIM + n]
        b = ref_vectors[r['vector_offset'] * DIM:r['vector_offset'] * DIM + n]
        chunks += len(row['text_sha256'])
        for x, y in zip(a, b):
            d = abs(x - y)
            if d > worst:
                worst = d
            if d > ATOL + RTOL * abs(y):
                outside += 1
    return {'documents': len(rows), 'documents_with_different_chunk_text': text_mismatch, 'vectors_compared': have_vectors,
            'chunks_compared': chunks, 'vector_values_outside_tolerance': outside if have_vectors else None,
            'max_abs_difference': worst if have_vectors else None}


def server_timing(rows):
    """Mean milliseconds per request and step, from LangGraph's Server-Timing header."""
    total = collections.Counter()
    n = 0
    for row in rows:
        header = row.get('server_timing') or row.get('timing')
        if not header:
            continue
        n += 1
        for name, value in re.findall(r'([a-z_]+);dur=([0-9.]+)', header):
            total[name] += float(value)
    return {k: v / n for k, v in total.items()} if n else None


def percentile(values, p):
    values = sorted(values)
    point = (len(values) - 1) * p
    i = int(point)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (point - i)


def gate_use(cell, limit, start, end):
    """What the gate of a run with a replaced lock recorded between two instants: one line per call into the model.

    wait: time a call stood at the gate. hold: time it was inside the model. `active` is the number of calls inside
    when the call entered, itself counted."""
    waits, holds, active = [], [], collections.Counter()
    path = cell / 'gate.jsonl'
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if start <= row['epoch_ns'] / 1e9 <= end:
                waits.append(row['wait_ns'] / 1e6)
                holds.append(row['hold_ns'] / 1e6)
                active[row['active']] += 1
    if not holds:
        return None
    n = len(holds)
    return {'limit': limit, 'calls': n, 'most_calls_inside_at_once': max(active),
            'share_of_calls_entering_at_the_limit': active[limit] / n,
            # time inside the model summed over all calls, per second of the run: the mean number of calls inside
            'mean_calls_inside': sum(holds) / 1000 / (end - start),
            'wait_ms': {'mean': sum(waits) / n, 'p50': percentile(waits, .5), 'p95': percentile(waits, .95)},
            'hold_ms': {'mean': sum(holds) / n, 'p50': percentile(holds, .5), 'p95': percentile(holds, .95)}}


def gpu_use(cell, start, end):
    """The GPU between two instants, from the samples nvidia-smi wrote during the run (process backend with a GPU).

    busy: the share of time the card was computing, as nvidia-smi reports it, all users of the card together.
    service memory: the memory held by every process on the card that is not the other tenant's. Only the service
    under test runs on the card besides that tenant.
    """
    rows = [r for r in lines(cell / 'gpu.jsonl') if start <= r['epoch'] <= end]
    if not rows:
        return None
    spec = read(cell / 'services.json') if (cell / 'services.json').exists() else {}
    tenant = spec.get('other_tenant_pid')
    ours, other = collections.Counter(), collections.Counter()
    for r in lines(cell / 'gpu-procs.jsonl'):
        if start <= r['epoch'] <= end:
            (other if r['pid'] == tenant else ours)[r['epoch']] += r['mib']
    busy, power = [r['util'] for r in rows], [r['power_w'] for r in rows]
    return {'samples': len(rows), 'busy_mean_percent': sum(busy) / len(busy), 'busy_p50_percent': percentile(busy, .5),
            'busy_p95_percent': percentile(busy, .95), 'busy_max_percent': max(busy),
            'memory_all_users_peak_GiB': max(r['mem_used_mib'] for r in rows) / 1024,
            'service_memory_peak_GiB': max(ours.values()) / 1024 if ours else None,
            'other_tenant_memory_peak_GiB': max(other.values()) / 1024 if other else None,
            'power_mean_w': sum(power) / len(power), 'power_max_w': max(power),
            'energy_Wh': sum(power) / len(power) * (end - start) / 3600}


def db_connections(cell, start, end):
    """What the system's database connections were doing between two instants (process backend): the campaign reads
    pg_stat_activity twice a second. One line per reading and state: epoch, state, wait type, wait event, connections.

    A share is a share of connection-readings: of all connections seen in all readings, how many were in the state.
    """
    path = cell / 'db-activity.tsv'
    if not path.exists():
        return None
    by_state, per_reading, active = collections.Counter(), collections.Counter(), collections.Counter()
    for line in path.read_text().splitlines():
        f = line.split('\t')
        try:
            epoch, state, n = float(f[0]), f[1], int(f[4])
        except (ValueError, IndexError):
            continue
        if start <= epoch <= end:
            by_state[state] += n
            per_reading[epoch] += n
            if state == 'active':
                active[epoch] += n
    total = sum(by_state.values())
    if not total:
        return None
    return {'readings': len(per_reading), 'most_connections_in_a_reading': max(per_reading.values()),
            'mean_connections': total / len(per_reading),
            'mean_connections_running_a_statement': sum(active.values()) / len(per_reading),
            'share_by_state': {k: v / total for k, v in sorted(by_state.items())}}


def ingestion(label, cfg):
    cell = D / 'cells' / label
    out = {'label': label, 'arm': cfg['arm'], 'system': cfg.get('system', cfg['arm']), 'gate': cfg.get('gate'),
           'documents': cfg['documents'], 'valid': False}
    failure = cell / 'failure.json'
    try:
        # cell-ok.json is written last by the campaign: ingestion, the read-back and every search phase passed.
        ok = read(cell / 'cell-ok.json')['ok'] and read(cell / 'finished.json')['ok'] and read(cell / 'runtime-check.json')['ok']
        result = read(cell / 'result.json')
        check = read(cell / 'db-check.json') if cfg.get('store') else None
    except (OSError, ValueError, KeyError):
        out['error'] = read(failure)['error'].strip().splitlines()[-1][:300] if failure.exists() else 'no complete result'
        return out
    if not ok or (check is not None and not check['pass']):
        out['error'] = 'the run did not finish cleanly' if not ok else 'the stored rows differ from the reference'
        return out
    start, end, wall = result['start_epoch'], result['end_epoch'], result['wall_s']
    requests = result['documents']
    images, harness = cell / 'images.json', cell / 'harness.json'
    out.update(valid=True, wall_s=wall, docs_per_s=result['docs_per_s'], requests=requests,
               cpu_s_per_doc=result['cpu_s_per_doc'], average_cpus=result['average_logical_cpus'],
               client_average_cpus=result.get('client_average_cpus'), images=read(images) if images.exists() else None,
               harness=read(harness) if harness.exists() else None,
               p50_s=result['latency_s']['p50'], p95_s=result['latency_s']['p95'], start_epoch=start)
    window = [s for s in lines(cell / 'samples.jsonl') if start <= s['epoch'] <= end]
    services = sorted({n for s in window for n in s['services'] if n != db_name(label)})
    runtime = read(cell / 'runtime.json') if (cell / 'runtime.json').exists() else {}
    out.update(gpu=gpu_use(cell, start, end), db_connections=db_connections(cell, start, end),
               cpu_by_service_s=result.get('cpu_by_service_s'),
               # LangGraph reports its model's device; for RocketRide the GPU memory it held after the warm-up is the evidence.
               model_device=runtime.get('model_device'), gpu_memory_after_warm_up=runtime.get('gpu_memory_after_warm_up'),
               engine_version=((runtime.get('version') or {}).get('data') or {}).get('version'),
               versions=runtime.get('versions'))
    tenant = [s['other_tenant_cpu_s'] for s in window if s.get('other_tenant_cpu_s') is not None]
    if len(tenant) > 1:
        # CPU time another tenant's container used while this run was timed: was the machine otherwise idle?
        out['other_tenant_cpu_s'] = tenant[-1] - tenant[0]
    if window:
        # memory_current counts the file cache too; anon_bytes is the memory the processes hold themselves.
        out['service_peak_memory_GiB'] = max(sum(s['services'][n]['memory_current'] for n in services if n in s['services']) for s in window) / 2 ** 30
        out['service_peak_own_memory_GiB'] = max(sum(s['services'][n]['anon_bytes'] for n in services if n in s['services']) for s in window) / 2 ** 30
        with_db = [s for s in window if db_name(label) in s['services']]
        if with_db:
            out['db_cpu_s'] = with_db[-1]['services'][db_name(label)]['cpu_s'] - with_db[0]['services'][db_name(label)]['cpu_s']
            out['db_average_cpus'] = out['db_cpu_s'] / wall
            out['db_peak_memory_GiB'] = max(s['services'][db_name(label)]['memory_current'] for s in with_db) / 2 ** 30
    if cfg.get('gate') is not None:
        out['gate_use'] = gate_use(cell, cfg['gate'], start, end)
        gate_file = cell / 'gate-check.json'
        out['gate_check'] = read(gate_file) if gate_file.exists() else None
    steps = server_timing(lines(cell / 'per_doc.jsonl'))
    if steps:
        out['langgraph_step_ms_per_doc'] = steps
    if check is not None:
        timed = check['during_timed_requests']
        kinds = collections.defaultdict(lambda: {'calls': 0, 'ms': 0.0})
        # Rebuilt from the statement list so the harness's own baseline queries are left out.
        own_ms = sum(s['ms'] for s in timed['top_statements'] if any(o in s['query'] for o in OWN))
        own_calls = sum(s['calls'] for s in timed['top_statements'] if any(o in s['query'] for o in OWN))
        for kind, v in timed['by_kind'].items():
            kinds[kind] = {'calls': v['calls'], 'ms': v['ms']}
        kinds['SELECT'] = {'calls': kinds['SELECT']['calls'] - own_calls, 'ms': round(kinds['SELECT']['ms'] - own_ms, 1)}
        db_ms = sum(v['ms'] for v in kinds.values())
        out.update(check_pass=check['pass'], expected_chunks=check['expected_chunks'], rows_total=check['rows_total'],
                   rows_other=check['rows_other'], rows_of_the_store_itself=check['rows_of_the_store_itself'],
                   rows_unexplained=check['rows_unexplained'],
                   rows_misfiled=check.get('rows_with_an_expected_text_in_the_wrong_place'),
                   http=result.get('http'),
                   rows_committed_after_the_clock_stopped=check['rows_committed_after_the_clock_stopped'],
                   commit_times_checked=check['commit_times_checked'],
                   documents_committed_after_their_answer=check['documents_committed_after_their_answer'],
                   missing=check['expected_chunks_missing'],
                   stored_too_often=check['expected_chunks_stored_too_often'],
                   vectors_outside=check['vectors_outside_tolerance'], max_abs_difference=check['max_abs_difference'],
                   indexes=check['database']['indexes'], table_bytes=check['database']['table_bytes'],
                   db_statements={k: v for k, v in sorted(kinds.items())}, db_ms_total=db_ms, db_ms_per_doc=db_ms / requests,
                   commits=kinds['COMMIT']['calls'], inserts=kinds['INSERT']['calls'], deletes=kinds['DELETE']['calls'],
                   selects=kinds['SELECT']['calls'], wal_MB=timed['wal_bytes'] / 1e6,
                   top_statements=[s for s in timed['top_statements'] if not any(o in s['query'] for o in OWN)][:5],
                   chunks_per_s=check['expected_chunks'] / wall)
    return out


def search(label, plan, gate=None):
    cell = D / 'cells' / label
    samples = lines(cell / 'samples.jsonl')
    out = {}
    for phase in plan['search']['phases']:
        path = cell / ('search-%s.json' % phase)
        if not path.exists() or not (cell / ('db-search-%s.json' % phase)).exists():
            continue
        statements = read(cell / ('db-search-%s.json' % phase))
        main = next((s for s in statements if s['query'].upper().startswith('SELECT') and 'embedding' in s['query']), None)
        levels = []
        for level in read(path)['levels']:
            window = [s for s in samples if level['start_epoch'] <= s['epoch'] <= level['end_epoch'] and db_name(label) in s['services']]
            db_cpu = window[-1]['services'][db_name(label)]['cpu_s'] - window[0]['services'][db_name(label)]['cpu_s'] if len(window) > 1 else None
            levels.append(dict(level, db_cpu_s=db_cpu, db_average_cpus=None if db_cpu is None else db_cpu / level['wall_s'],
                               gpu=gpu_use(cell, level['start_epoch'], level['end_epoch']),
                               db_connections=db_connections(cell, level['start_epoch'], level['end_epoch']),
                               gate_use=None if gate is None else gate_use(cell, gate, level['start_epoch'], level['end_epoch']),
                               db_cpu_ms_per_question=None if db_cpu is None else 1000 * db_cpu / level['questions']))
        out[phase] = {'levels': levels,
                      # RocketRide only: was the form of an answer without results confirmed against the engine
                      'empty_answer_form': read(path).get('empty_answer_form'),
                      'http': read(path).get('http'),
                      # the plan of the system's own search statement, taken after the timed questions
                      'plan': read(cell / ('db-search-%s-plan.json' % phase)) if (cell / ('db-search-%s-plan.json' % phase)).exists() else None,
                      # LangGraph only: mean time per step and question at the first level
                      'langgraph_step_ms': server_timing(lines(cell / ('search-%s-outputs.jsonl' % phase))),
                      'db_search_statement': main['query'][:160] if main else None,
                      'db_ms_per_search': float(main['ms']) / int(main['calls']) if main else None,
                      'db_searches': int(main['calls']) if main else None,
                      'index': read(cell / 'db-hnsw.json') if phase == 'hnsw' and (cell / 'db-hnsw.json').exists() else None}
    return out


def floored(row):
    return [s for s, score in zip(row['sha'], row['score']) if score is None or score >= FLOOR - 1e-9]


def shared(x, y):
    """Share of results two lists have in common. The same chunk text can be stored more than once, so counted with repeats."""
    if not x and not y:
        return 1.0
    return sum((collections.Counter(x) & collections.Counter(y)).values()) / max(len(x), len(y))


def agreement(label, reference, phase, reference_phase=None):
    a = {r['id']: r for r in lines(D / 'cells' / label / ('search-%s-outputs.jsonl' % phase))}
    b = {r['id']: r for r in lines(D / 'cells' / reference / ('search-%s-outputs.jsonl' % (reference_phase or phase)))}
    if not a or not b:
        return None
    same_raw = same = same_set = 0
    overlap = 0.0
    for i in a:
        x, y = floored(a[i]), floored(b[i])
        same_raw += a[i]['sha'] == b[i]['sha']
        same += x == y
        same_set += sorted(x) == sorted(y)
        overlap += shared(x, y)
    return {'against': reference, 'questions': len(a), 'identical_lists': same, 'identical_lists_before_the_floor': same_raw,
            'same_results_any_order': same_set, 'mean_overlap': overlap / len(a)}


def main():
    global D
    D = Path(sys.argv[1]).resolve()
    plan = read(D / 'plan.json')
    cells = {c['label']: c for c in plan['cells']}
    timed = [c['label'] for c in plan['cells'] if c['stage'] == 'campaign']
    runs = {}
    for label in timed:
        if not (D / 'cells' / label).exists():
            continue
        row = ingestion(label, cells[label])
        if row['valid'] and cells[label].get('search'):
            row['search'] = search(label, plan, cells[label].get('gate'))
        runs[label] = row
    reference = next((l for l in timed if l in runs and runs[l]['valid'] and runs[l]['arm'] == 'lg'), None)
    for label, row in runs.items():
        if not row['valid'] or 'search' not in row:
            continue
        if reference and reference != label:
            row['agreement'] = {phase: agreement(label, reference, phase) for phase in row['search']}
        if 'exact' in row['search'] and 'hnsw' in row['search']:
            # How much of the exact answer the index returns, in the same run.
            row['index_against_exact'] = agreement(label, label, 'hnsw', 'exact')
    screening = {}
    for label in [c['label'] for c in plan['cells'] if c['stage'] == 'screening']:
        path = D / 'cells' / label / 'result.json'
        if path.exists():
            r = read(path)
            screening[label] = {k: r.get(k) for k in ['documents', 'failures', 'chunks', 'wall_s', 'docs_per_s', 'cpu_s_per_doc']}
            if label != Path(plan['reference_cell']).name:
                screening[label]['against_reference'] = compare_with_reference(D / 'cells' / label, D / plan['reference_cell'])
    status = read(D / 'status.json') if (D / 'status.json').exists() else {}
    host = next((read(p) for p in [D / 'host-campaign.json', D / 'host-cells.json', D / 'host-screening.json'] if p.exists()), None)
    # Attempts that failed and were set aside when the run was started again.
    set_aside = sorted(p.name for p in (D / 'cells').glob('*.failed-*'))
    out = {'tag': plan['tag'], 'campaign': timed, 'runs': runs, 'not_run': [l for l in timed if l not in runs],
           'harness': plan.get('harness'), 'host': host, 'failed_attempts': set_aside,
           'search_reference': reference, 'screening': screening,
           'finalize': read(D / 'finalize.json') if (D / 'finalize.json').exists() else None,
           'plan': dict({k: plan[k] for k in ['documents', 'concurrency', 'workers', 'service_cpus', 'client_cpus', 'service_memory', 'search']},
                        backend=plan.get('backend', 'docker'), device=plan.get('device', 'cpu'), gpu=plan.get('gpu'),
                        measure_all_kept=plan.get('measure_all_kept'), manifest_given=plan.get('manifest_given'),
                        questions_given=plan.get('questions_given'), documents_planned=plan.get('documents_planned'),
                        gate=None if not (plan.get('rr') or {}).get('gate') else {k: plan['rr']['gate'].get(k) for k in [
                            'loader', 'line', 'original_sha256', 'changed_sha256', 'module_sha256']}),
           'db': {k: plan['db'][k] for k in ['image', 'cpus', 'memory']},
           'status': {k: status.get(k) for k in ['state', 'failed', 'skipped']}}
    (D / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n')

    def f(x):
        return 'n/a' if x is None else '%.2f' % x
    for label, r in runs.items():
        if not r['valid']:
            print('%-8s NOT VALID: %s' % (label, r.get('error')))
            continue
        print('%-8s ingest %6.3f PDFs/s  cpu/PDF %5.1f  own mem %5.1f GiB  db ms/PDF %6.1f  check %s' % (
            label, r['docs_per_s'], r['cpu_s_per_doc'], r.get('service_peak_own_memory_GiB', 0), r.get('db_ms_per_doc', 0), r.get('check_pass')))
        for phase, s in (r.get('search') or {}).items():
            ag = (r.get('agreement') or {}).get(phase)
            print('    %-5s ' % phase + ' | '.join('%d in flight: %6.1f valid answers/s p50 %5.0f ms' % (
                l['in_flight'], l['questions_per_s'], (l['latency_ms'] or {}).get('p50', float('nan'))) for l in s['levels'])
                + ' | db ms/search %s' % f(s['db_ms_per_search'])
                + ('' if not ag else ' | same lists as %s: %d of %d' % (ag['against'], ag['identical_lists'], ag['questions'])))
    print('runs analysed %d, valid %d, not run %d' % (len(runs), sum(r['valid'] for r in runs.values()), len(out['not_run'])))


if __name__ == '__main__':
    main()
