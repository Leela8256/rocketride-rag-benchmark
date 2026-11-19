"""Run the cells of a campaign, one at a time.

A cell is one run of one system. For every cell this script starts, on the CPUs the plan names:

  1. an empty Postgres with pgvector (when the cell stores),
  2. the system under test in a fresh container (RocketRide engine, or the LangGraph service with its parser),
  3. the load client in its own container,

then waits for the warm-up, removes the warm-up rows from the table, clears the database statistics, releases the
timed ingestion, reads the table back (dbcheck.py), runs the search phases (search_driver.py) and removes every
container.

Runs on the host with the standard library only. It needs Docker and cgroup v2.

Usage:
  python3 harness/campaign.py <campaign folder> --stage screening
  python3 harness/campaign.py <campaign folder> --stage campaign
  python3 harness/campaign.py <campaign folder> --only rr-1 lg-1
  python3 harness/campaign.py <campaign folder> --stage campaign --dry-run     print the docker commands only
"""
import argparse
import json
import os
import platform
import secrets
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO = HARNESS.parent
R = None        # the campaign folder
PLAN = None


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sh(args, timeout=None):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True, text=True,
                          timeout=timeout).stdout.strip()


def expand(cpus):
    """'0-11,16-27' -> [0, 1, ..., 11, 16, ..., 27]"""
    out = []
    for part in cpus.split(','):
        low, _, high = part.partition('-')
        out.extend(range(int(low), int(high or low) + 1))
    return out


def status(**values):
    path = R / 'status.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(values, epoch=time.time())
    save(path, state)
    print(json.dumps(values), flush=True)


def fetch(url, seconds):
    end = time.monotonic() + seconds
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.read()
        except Exception:
            if time.monotonic() > end:
                raise TimeoutError('not ready: ' + url)
            time.sleep(2)


def service_name(cfg):
    return 'ragbench-' + cfg['label']


def db_name(cfg):
    return service_name(cfg) + '-db'


def env_file(label):
    """Where a cell's database settings live while it runs: outside the campaign folder, deleted after the cell."""
    return str(Path(PLAN['db']['env_dir']) / (label + '.env'))


def db_command(cfg):
    """A fresh, empty Postgres with pgvector for this one cell, on its own CPUs."""
    db = PLAN['db']
    return ['docker', 'run', '-d', '--label', 'benchmark.campaign=' + PLAN['tag'], '--network', 'host',
            '--cpuset-cpus', db['cpus'], '--name', db_name(cfg), '--memory', db['memory'], '--memory-swap', db['memory'],
            '--shm-size', '1g', '--env-file', env_file(cfg['label']),
            '-e', 'POSTGRES_DB=' + db['database'], '-e', 'POSTGRES_USER=' + db['user'],
            '-v', '%s:/docker-entrypoint-initdb.d/init.sql:ro' % (REPO / 'infra/db-init.sql'), db['image']] + db['args']


def start_db(cfg, out):
    """Write the cell's settings file, start the database and wait until it answers with the extension in place."""
    db = PLAN['db']
    path = Path(env_file(cfg['label']))
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    password = secrets.token_hex(16)                     # for this cell only; never printed, never in the cell folder
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(''.join('%s=%s\n' % pair for pair in [
            ('POSTGRES_PASSWORD', password), ('ROCKETRIDE_PG_PASSWORD', password), ('RAG_PG_PASSWORD', password),
            ('RAG_PG_HOST', db['host']), ('RAG_PG_PORT', db['port']), ('RAG_PG_USER', db['user']),
            ('RAG_PG_DATABASE', db['database']), ('RAG_PG_TABLE', db['lg_table'])]))
    cmd = db_command(cfg)
    save(out / (db_name(cfg) + '-command.json'), cmd)
    sh(cmd)
    end = time.monotonic() + 180
    while True:
        # The image's first start runs its init scripts on the socket only; TCP answers once the real server is up.
        tcp = subprocess.run(['docker', 'exec', db_name(cfg), 'pg_isready', '-h', '127.0.0.1', '-p', str(db['port'])],
                             capture_output=True).returncode == 0
        if tcp:
            version = subprocess.run(['docker', 'exec', db_name(cfg), 'psql', '-U', db['user'], '-d', db['database'], '-Atc',
                                      "SELECT extversion FROM pg_extension WHERE extname = 'vector'"],
                                     capture_output=True, text=True).stdout.strip()
            if version:
                (out / (db_name(cfg) + '-ready.txt')).write_text('pgvector %s\n' % version)
                return
        if time.monotonic() > end:
            raise TimeoutError('the database did not come up')
        time.sleep(1)


def db_sql(cfg, sql):
    db = PLAN['db']
    return subprocess.run(['docker', 'exec', db_name(cfg), 'psql', '-U', db['user'], '-d', db['database'], '-At', '-F', '\t',
                           '-v', 'ON_ERROR_STOP=1', '-c', sql], check=True, capture_output=True, text=True).stdout.strip()


def table_of(cfg):
    return PLAN['db']['rr_table'] if cfg['arm'] == 'rr' else PLAN['db']['lg_table']


def quoted(text):
    return "'" + str(text).replace("'", "''") + "'"


def warm_up_rows(cfg, warm):
    """SQL for the rows the warm-up left in the table: (delete them, count the rows that then remain unexplained,
    name the documents of such rows).

    The warm-up has to write, or it would not warm the store up. Its rows must not stay: a search could return
    them, and LangGraph's start-up fixture is a document RocketRide never stored. RocketRide's store keeps one row
    for itself, which its own search never returns; that row stays.
    """
    table = table_of(cfg)
    if cfg['arm'] == 'rr':
        names = 'ARRAY[%s]' % ', '.join(quoted(n) for n in warm)
        rest = 'FROM %s WHERE objectid IS DISTINCT FROM %s' % (table, quoted(PLAN['rr']['own_object_id']))
        return ('DELETE FROM %s WHERE parent = ANY(%s)' % (table, names), 'SELECT count(*) ' + rest,
                "SELECT coalesce(string_agg(DISTINCT coalesce(parent, '(none)'), ', '), '') FROM (SELECT parent %s LIMIT 50) x" % rest)
    names = 'ARRAY[%s]' % ', '.join(quoted(n) for n in list(warm) + PLAN['lg']['fixture_documents'])
    return ("DELETE FROM %s WHERE langchain_metadata->>'source' = ANY(%s)" % (table, names),
            'SELECT count(*) FROM %s' % table,
            "SELECT coalesce(string_agg(DISTINCT coalesce(langchain_metadata->>'source', '(none)'), ', '), '') "
            "FROM (SELECT langchain_metadata FROM %s LIMIT 50) x" % table)


def db_prepare(cfg, out, sql=None):
    """Between the warm-up and the timed requests: empty the table of warm-up rows, settle it and clear the
    database's statistics.

    The table exists by now: both systems create it while they warm up.
    """
    sql = sql or (lambda text: db_sql(cfg, text))
    table = table_of(cfg)
    started = time.time()
    warm = json.loads((R / 'source/corpus-manifest.json').read_text())['warm']
    delete, remaining, whose = warm_up_rows(cfg, warm)
    rows_after_warm_up = int(sql('SELECT count(*) FROM %s' % table))
    sql(delete)
    unexplained = int(sql(remaining))
    assert unexplained == 0, ('%d rows of %s are neither warm-up rows nor the store\'s own row; a search could '
                              'return them. Their documents: %s' % (unexplained, table, sql(whose)[:400]))
    # A table that has held its rows for a while has no dead rows, has planner statistics and nothing waiting to
    # be written.
    sql('VACUUM (ANALYZE) %s' % table)
    sql('CHECKPOINT')
    done = {'table': table, 'rows_after_warm_up': rows_after_warm_up,
            'rows_before_timed_requests': int(sql('SELECT count(*) FROM %s' % table)),
            'table_bytes_before_timed_requests': int(sql("SELECT pg_total_relation_size('%s')" % table)),
            'prepare_s': time.time() - started}
    sql('SELECT pg_stat_statements_reset()')
    wal = sql('SELECT wal_records, wal_bytes FROM pg_stat_wal').split('\t')
    counters = sql('SELECT xact_commit, xact_rollback FROM pg_stat_database WHERE datname = current_database()').split('\t')
    done.update(wal_records=int(wal[0]), wal_bytes=int(wal[1]), xact_commit=int(counters[0]), xact_rollback=int(counters[1]))
    save(out / 'db-prepare.json', done)


def db_statements(cfg):
    """What the database ran since its statistics were last cleared: the costliest statements."""
    raw = db_sql(cfg, "SELECT s.calls, round(s.total_exec_time::numeric, 3), s.rows, left(regexp_replace(s.query, '\\s+', ' ', 'g'), 200) "
                      "FROM pg_stat_statements s JOIN pg_database d ON d.oid = s.dbid WHERE d.datname = current_database() "
                      "AND s.query NOT LIKE '%pg_stat_statements%' ORDER BY s.total_exec_time DESC LIMIT 8")
    return [dict(zip(['calls', 'ms', 'rows', 'query'], line.split('\t', 3))) for line in raw.splitlines()]


def index_scans(cfg, sql=None, at_least=None, wait_s=15):
    """How often Postgres has scanned the vector index the harness built, from the database's own counter.

    A database connection hands its numbers in when it has been idle for some seconds, so after a search phase the
    counter is read until it has reached `at_least` or `wait_s` seconds have passed.
    """
    sql = sql or (lambda text: db_sql(cfg, text))
    query = "SELECT coalesce(sum(idx_scan), 0) FROM pg_stat_user_indexes WHERE indexrelname = '%s_hnsw'" % table_of(cfg)
    end = time.monotonic() + wait_s
    while True:
        scans = int(sql(query) or 0)
        if at_least is None or scans >= at_least or time.monotonic() > end:
            return scans
        time.sleep(1)


def index_use(scans, open_transactions):
    """True: Postgres scanned the index. False: it did not. None: not determined, because a database session of the
    system under test was still inside a transaction, and a session hands its scan counts in only when it leaves
    its transaction or disconnects."""
    return True if scans > 0 else None if open_transactions else False


def search_plan(cfg, sql=None):
    """The plan Postgres makes for the search statement the system under test sent most often. Taken after the timed
    questions, from the statement as the database recorded it, so it is the system's own statement and not a copy."""
    sql = sql or (lambda text: db_sql(cfg, text))
    try:
        # The search statement: a SELECT that uses one of pgvector's distance operators. Of several, the one called
        # most often, and of those the one the database spent most time on.
        query = sql("SELECT s.query FROM pg_stat_statements s JOIN pg_database d ON d.oid = s.dbid "
                    "WHERE d.datname = current_database() AND s.query ILIKE 'select%' AND s.query ~ '<=>|<->|<#>' "
                    "AND s.query NOT ILIKE '%pg_stat%' ORDER BY s.calls DESC, s.total_exec_time DESC LIMIT 1")
    except subprocess.CalledProcessError:
        query = None
    if not query:
        return {'query': None, 'plan': None, 'note': 'the database recorded no search statement'}
    try:
        plan = sql('EXPLAIN (GENERIC_PLAN) ' + query.strip().rstrip(';'))
    except subprocess.CalledProcessError as e:
        return {'query': query[:800], 'plan': None, 'note': 'no plan: ' + ' '.join((e.stderr or str(e)).split())[-300:]}
    return {'query': query[:800], 'plan': plan.splitlines(), 'names_the_vector_index': '%s_hnsw' % table_of(cfg) in plan}


def search_phases(cfg, out, client_cmd):
    """The search side, on the table the cell's ingestion filled.

    exact   the table as the system under test left it: neither creates a vector index, so every question is an
            exact scan of all stored vectors
    hnsw    the same table after a vector search index (pgvector HNSW, its defaults) has been built on it. The
            index is built by this harness. It is not part of either system. An index that exists is not an index
            that is used: the number of index scans during the phase is read from the database and recorded.
    """
    table = table_of(cfg)
    for phase in PLAN['search']['phases']:
        if phase == 'hnsw':
            started = time.time()
            db_sql(cfg, 'CREATE INDEX %s_hnsw ON %s USING hnsw (embedding vector_cosine_ops)' % (table, table))
            db_sql(cfg, 'ANALYZE %s' % table)
            save(out / 'db-hnsw.json', {'build_s': time.time() - started,
                                        'index_bytes': int(db_sql(cfg, "SELECT pg_relation_size('%s_hnsw')" % table)),
                                        'rows': int(db_sql(cfg, 'SELECT count(*) FROM %s' % table))})
        db_sql(cfg, 'SELECT pg_stat_statements_reset()')
        scans_before = index_scans(cfg) if phase == 'hnsw' else None
        cmd = client_cmd[:-2] + ['/harness/search_driver.py', cfg['label'], phase]
        with (out / ('search-%s.log' % phase)).open('w') as log:
            done = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        finished = out / ('search-%s-finished.json' % phase)
        assert done.returncode == 0 and finished.exists() and json.loads(finished.read_text())['ok'], \
            'the search run failed; see search-%s.log' % phase
        save(out / ('db-search-%s.json' % phase), db_statements(cfg))
        # After the timed questions: the plan of the system's own search statement, and how often the index was scanned.
        save(out / ('db-search-%s-plan.json' % phase), search_plan(cfg))
        if phase == 'hnsw':
            searches = sum(l['questions'] for l in json.loads((out / 'search-hnsw.json').read_text())['levels'])
            scans = index_scans(cfg, at_least=scans_before + searches) - scans_before
            waiting = int(db_sql(cfg, "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                                      "AND pid <> pg_backend_pid() AND state LIKE 'idle in transaction%'"))
            save(out / 'db-hnsw.json', dict(json.loads((out / 'db-hnsw.json').read_text()), searches_timed=searches,
                                            index_scans_during_search=scans, sessions_still_in_a_transaction=waiting,
                                            index_used=index_use(scans, waiting)))


def service_commands(cfg):
    """[(container name, docker run command, readiness URL)] for one cell.

    Both systems run as their image ships. Nothing is mounted into the service containers, and no thread setting
    of a math library is passed to either.

    The one exception is a RocketRide cell with `gate` in its entry (not as shipped, and named so in the report):
    the same image, with the engine's embedding lock replaced by a gate of that many calls at once. Three mounts
    do it: the cell folder (the gate reads its size there and writes its records), the gate module, and the
    engine's loader file with its one changed line. Nothing else differs from the shipped cell.
    """
    base = ['docker', 'run', '-d', '--label', 'benchmark.campaign=' + PLAN['tag'], '--network', 'host',
            '--cpuset-cpus', PLAN['service_cpus']]
    limits = ['--memory', PLAN['service_memory'], '--memory-swap', PLAN['service_memory']]
    name = service_name(cfg)
    if cfg['arm'] == 'rr':
        rr = PLAN['rr']
        mounts = []
        if 'gate' in cfg:
            gate = rr['gate']
            mounts = ['-v', '%s:/cell' % (R / 'cells' / cfg['label']),
                      '-v', '%s/bench_gate.py:%s/bench_gate.py:ro' % (gate['dir'], rr['engine_dir']),
                      '-v', '%s/sentence_transformers.py:%s/%s:ro' % (gate['dir'], rr['engine_dir'], gate['loader'])]
        cmd = base + ['--name', name] + limits + mounts + [rr['image'], 'ai/eaas.py', '--host=127.0.0.1', '--port=%d' % rr['port']]
        return [(name, cmd, 'http://127.0.0.1:%d/version' % rr['port'])]
    lg = PLAN['lg']
    # RocketRide's own parser behind a small HTTP server, so that both systems parse a PDF to the same text. It
    # runs on the service CPUs and its CPU time is counted as LangGraph's. It has no memory limit of its own.
    parser = base + ['--name', name + '-parser', '--entrypoint', '/opt/rocketride/engine/java/jre/bin/java',
                     lg['parser_image'], '-Dbench.host=127.0.0.1', '-Dbench.port=%d' % lg['parser_port'],
                     '-cp', '/opt/rocketride/engine/java/lib/*:/bench-classes', 'com.rocketride.tika_api.BenchTextServer']
    api = base + ['--name', name] + limits + [
        '-e', 'EXTRACTOR=tika', '-e', 'TIKA_URL=http://127.0.0.1:%d' % lg['parser_port']]
    if cfg.get('store'):
        # The database settings come from the cell's settings file.
        api += ['--env-file', env_file(cfg['label']),
                '-e', 'PIPELINES=%s,%s' % (lg['ingest_pipeline'], lg['search_pipeline'])]
    else:
        api += ['-e', 'PIPELINES=' + lg['screen_pipeline']]
    api += [lg['image'], 'uvicorn', 'lg_rag_runtime:app', '--host', '127.0.0.1', '--port', str(lg['port']), '--workers', '1']
    return [(name + '-parser', parser, 'http://127.0.0.1:%d/version' % lg['parser_port']),
            (name, api, 'http://127.0.0.1:%d/health/ready' % lg['port'])]


def client_command(cfg, groups):
    cmd = ['docker', 'run', '--rm', '--name', 'ragbench-client-' + cfg['label'], '--label',
           'benchmark.campaign=' + PLAN['tag'], '--network', 'host', '--cpuset-cpus', PLAN['client_cpus'],
           '-v', '%s:/campaign' % R, '-v', '%s:/harness:ro' % HARNESS,
           '-v', '%s:/pipelines:ro' % (REPO / 'rocketride/pipelines'), '-v', PLAN['corpus'] + ':/corpus:ro']
    if cfg.get('store'):
        cmd += ['--env-file', env_file(cfg['label'])]      # the database password, for the pipeline placeholder
    for name, group in groups.items():
        cmd += ['-v', '%s:/groups/%s:ro' % (group, name)]
    return cmd + [PLAN['client_image'], '/harness/driver.py', cfg['label']]


def record_runtime(cfg, out, main):
    """Record what the service under test is, and check the few things a valid run needs."""
    problems = {}
    if cfg['arm'] == 'rr':
        runtime = {'version': json.loads(fetch('http://127.0.0.1:%d/version' % PLAN['rr']['port'], 30))}
        # The engine files of this container against the hashes taken from the release when the image was built.
        check = subprocess.run(['docker', 'exec', main, 'sha256sum', '-c', '/opt/rocketride/release-files.sha256'],
                               capture_output=True, text=True)
        runtime['release_files_unchanged'] = check.returncode == 0
        problems.update(gate_problems(cfg, out, main, check, runtime))
    else:
        runtime = json.loads(fetch('http://127.0.0.1:%d/__bench/runtime' % PLAN['lg']['port'], 30))
        wanted = {'model_device': 'cpu', 'yappi_running': False, 'affinity': expand(PLAN['service_cpus'])}
        problems = {k: {'found': runtime.get(k), 'wanted': v} for k, v in wanted.items() if runtime.get(k) != v}
    save(out / 'runtime.json', runtime)
    save(out / 'runtime-check.json', {'ok': not problems, 'problems': problems})
    assert not problems, 'the service under test is not what the cell asked for: %s' % json.dumps(problems)


def gate_records(out):
    return [json.loads(p.read_text()) for p in sorted(out.glob('gate-runtime-*.json'))]


def gate_problems(cfg, out, main, check, runtime):
    """Is the engine of this cell what the cell asks for: the release untouched, or the release with the one
    changed loader file and the gate at the cell's size. Called after the warm-up, when the model is loaded."""
    rr, problems = PLAN['rr'], {}

    def in_container(*args):
        return subprocess.run(['docker', 'exec', main] + list(args), capture_output=True, text=True)
    records = gate_records(out)
    if 'gate' not in cfg:
        # As shipped: every release file unchanged, no gate module in the container, no gate record written.
        if check.returncode != 0:
            problems['release_files'] = (check.stdout + check.stderr).strip()[-400:]
        if in_container('test', '-e', '%s/bench_gate.py' % rr['engine_dir']).returncode == 0:
            problems['gate_module'] = 'present in a cell that runs the engine as shipped'
        if records:
            problems['gate_records'] = 'a cell that runs the engine as shipped wrote %d gate records' % len(records)
        return problems
    gate = rr['gate']
    # Every file the image recorded a hash for that is not reported as OK: changed, missing or unreadable.
    changed = sorted(line.rsplit(': ', 1)[0] for line in check.stdout.splitlines() if line.strip() and not line.endswith(': OK'))
    runtime.update(release_files_changed=changed, gate=cfg['gate'])
    if changed != [gate['loader']]:
        problems['release_files'] = 'changed files %s; exactly the loader %s is expected' % (changed, gate['loader'])
    # The changed loader must have been made from the loader of this very engine: the hash the image recorded for
    # its released loader is the hash of the file the change was made from.
    recorded = {}
    for line in in_container('cat', '/opt/rocketride/release-files.sha256').stdout.splitlines():
        digest, _, name = line.partition(' ')
        recorded[name.strip().lstrip('*')] = digest
    if recorded.get(gate['loader']) != gate['original_sha256']:
        problems['loader_origin'] = {'released_loader_of_this_image': recorded.get(gate['loader']),
                                     'the_change_was_made_from': gate['original_sha256']}
    for path, key in [('%s/%s' % (rr['engine_dir'], gate['loader']), 'changed_sha256'),
                      ('%s/bench_gate.py' % rr['engine_dir'], 'module_sha256')]:
        found = in_container('sha256sum', path).stdout.split()
        if not found or found[0] != gate[key]:
            problems[path] = {'found': found[:1], 'wanted': gate[key]}
    # One task has loaded the model so far (the ingestion task), so one record exists.
    if len(records) != 1:
        problems['gate_records'] = '%d gate records after the warm-up; one is expected' % len(records)
    for record in records:
        wanted = {'limit': cfg['gate'], 'affinity': expand(PLAN['service_cpus'])}
        wrong = {k: {'found': record.get(k), 'wanted': v} for k, v in wanted.items() if record.get(k) != v}
        # The engine sets TOKENIZERS_PARALLELISM itself. Any other thread setting would be a different experiment.
        extra = sorted(set(record.get('math_environment') or {}) - {'TOKENIZERS_PARALLELISM'})
        if extra:
            wrong['math_environment'] = {'found': record['math_environment'], 'wanted': 'no thread setting'}
        if wrong:
            problems['gate_record_%s' % record.get('pid')] = wrong
    runtime['gate_records'] = records
    return problems


def gate_end_check(cfg, out):
    """After the last phase: every task of the cell (ingestion and each search phase) ran with the cell's gate
    size, and no call was ever inside the model with more calls than the gate allows."""
    records = gate_records(out)
    if 'gate' not in cfg:
        assert not records and not (out / 'gate.jsonl').exists(), 'a cell that runs the engine as shipped wrote gate records'
        return
    assert records and all(r['limit'] == cfg['gate'] for r in records), 'a task of this cell ran with another gate size'
    highest = calls = 0
    with (out / 'gate.jsonl').open() as f:
        for line in f:
            calls += 1
            highest = max(highest, json.loads(line)['active'])
    assert calls and highest <= cfg['gate'], '%d calls were inside the model at once; the gate allows %d' % (highest, cfg['gate'])
    save(out / 'gate-check.json', {'gate': cfg['gate'], 'tasks': len(records), 'calls': calls, 'most_calls_inside_at_once': highest})


def run_cell(cfg, deadline):
    label = cfg['label']
    out = R / 'cells' / label
    out.mkdir(parents=True)
    save(out / 'config.json', cfg)
    save(out / 'harness.json', harness_revision())
    status(state='running', current=label)
    services = service_commands(cfg)
    names = [name for name, _, _ in services]
    store = bool(cfg.get('store'))
    if store:
        names = [db_name(cfg)] + names          # first, so that names[-1] stays the service under test
    sampler = client = None
    try:
        if store:
            start_db(cfg, out)
        for name, cmd, url in services:
            save(out / (name + '-command.json'), cmd)
            sh(cmd)
            (out / (name + '-ready.txt')).write_bytes(fetch(url, 900)[:4096])
        groups, images = {}, {}
        for name in names:
            meta = json.loads(sh(['docker', 'inspect', name]))[0]
            assert meta['State']['Running'] and meta['HostConfig']['CpusetCpus'] == (
                PLAN['db']['cpus'] if name == db_name(cfg) else PLAN['service_cpus'])
            images[name] = meta['Image']
            # Environment values are deliberately not stored: the database password is one of them.
            save(out / (name + '-container.json'), {k: meta[k] for k in ['Id', 'Image', 'State', 'Mounts', 'HostConfig']})
            lines = Path('/proc/%d/cgroup' % meta['State']['Pid']).read_text().splitlines()
            group = next(line[3:] for line in lines if line.startswith('0::'))
            groups[name] = str(Path('/sys/fs/cgroup') / group.lstrip('/'))
        save(out / 'cgroups.json', groups)
        save(out / 'images.json', images)
        main = names[-1]
        # A record of which files the service ran, not a check: a file a release does not have is noted, not fatal.
        files = PLAN['rr']['files'] + (['bench_gate.py'] if 'gate' in cfg else []) if cfg['arm'] == 'rr' else PLAN['lg']['files']
        hashed = subprocess.run(['docker', 'exec', main, 'sha256sum'] + files, capture_output=True, text=True)
        (out / 'service-files.sha256').write_text(hashed.stdout + hashed.stderr)

        sampler = subprocess.Popen([sys.executable, str(HARNESS / 'sampler.py'), str(out)])
        # The client reads the CPU of the service under test only; the database is sampled on the host.
        cmd = client_command(cfg, {k: v for k, v in groups.items() if k != db_name(cfg)})
        save(out / 'client-command.json', cmd)
        with (out / 'driver.log').open('w') as log:
            client = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            end = time.monotonic() + 1800
            while not (out / 'ready.json').exists():
                if client.poll() is not None:
                    raise RuntimeError('warm-up failed; see driver.log')
                if time.monotonic() > end:
                    raise TimeoutError('warm-up timed out')
                time.sleep(.5)
            record_runtime(cfg, out, main)
            if store:
                db_prepare(cfg, out)
            save(out / 'go.json', {'epoch': time.time()})
            client.wait(timeout=max(60, min(cfg['timeout_s'], deadline - time.time())))
        assert client.returncode == 0, 'client failed; see driver.log'
        assert json.loads((out / 'finished.json').read_text())['ok']
        if store:
            # Read the table back while the database is still up, and compare it with the reference run.
            checked = subprocess.run([sys.executable, str(HARNESS / 'dbcheck.py'), str(out)], capture_output=True, text=True)
            (out / 'db-check.log').write_text(checked.stdout + checked.stderr)
            assert checked.returncode == 0, 'the database check could not run; see db-check.log'
            assert json.loads((out / 'db-check.json').read_text())['pass'], 'the stored rows differ from the reference; see db-check.json'
        if cfg.get('search'):
            search_phases(cfg, out, cmd)
        if cfg['arm'] == 'rr':
            gate_end_check(cfg, out)
        result = json.loads((out / 'result.json').read_text())
        # Written last: a cell counts as done only when ingestion, the read-back and every search phase passed.
        save(out / 'cell-ok.json', {'ok': True, 'epoch': time.time()})
        status(last_completed=label, last_docs_per_s=result['docs_per_s'], last_wall_s=result['wall_s'])
        return result
    finally:
        if store:
            Path(env_file(label)).unlink(missing_ok=True)
        (out / 'sampler.stop').touch()
        if sampler:
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()
        for name in names:
            with (out / (name + '.log')).open('w') as log:
                subprocess.run(['docker', 'logs', name], stdout=log, stderr=subprocess.STDOUT)
        for name in names + ['ragbench-client-' + label]:
            subprocess.run(['docker', 'rm', '-f', '-v', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if client and client.poll() is None:
            client.kill()


def run_quietly(args):
    try:
        return subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def harness_revision():
    """The commit of this repository. Recorded with every run: a campaign that is started again after a change of
    the harness holds runs of two commits, and the report says so."""
    head = run_quietly(['git', '-C', REPO, 'rev-parse', 'HEAD'])
    return {'commit': head, 'uncommitted_changes': None if head is None else bool(
        run_quietly(['git', '-C', REPO, 'status', '--porcelain', '--untracked-files=no']))}


def host_record():
    """The machine of this sitting, for the report. Every value is read, none is assumed."""
    def text(path):
        try:
            return Path(path).read_text()
        except OSError:
            return ''
    run = run_quietly
    cpu = text('/proc/cpuinfo')
    models = sorted({line.split(':', 1)[1].strip() for line in cpu.splitlines() if line.startswith('model name')})
    cores = {(block.split('physical id', 1)[1].split('\n', 1)[0], block.split('core id', 1)[1].split('\n', 1)[0])
             for block in cpu.split('\n\n') if 'physical id' in block and 'core id' in block}
    memory = next((int(line.split()[1]) for line in text('/proc/meminfo').splitlines() if line.startswith('MemTotal')), None)
    return {'cpu_model': models, 'logical_cpus': os.cpu_count(), 'physical_cores': len(cores) or None,
            'memory_GiB': None if memory is None else round(memory / 2 ** 20, 1),
            'kernel': platform.release(), 'machine': platform.machine(),
            'docker': run(['docker', 'version', '--format', '{{.Server.Version}}']), 'epoch': time.time()}


def finished(label):
    out = R / 'cells' / label
    try:
        return json.loads((out / 'cell-ok.json').read_text())['ok'] and (out / 'result.json').exists()
    except (OSError, ValueError, KeyError):
        return False


def set_aside(label):
    """Keep a failed attempt for inspection and free the label for a rerun."""
    out = R / 'cells' / label
    n = 1
    while (R / 'cells' / ('%s.failed-%d' % (label, n))).exists():
        n += 1
    out.rename(R / 'cells' / ('%s.failed-%d' % (label, n)))


def main():
    global R, PLAN
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('campaign', help='campaign folder made by make_plan.py')
    parser.add_argument('--stage', choices=['screening', 'campaign'], help='run every cell of this stage')
    parser.add_argument('--only', nargs='+', default=[], help='run these cells only')
    parser.add_argument('--hours', type=float, default=24.0, help='no cell starts when it could not finish before this')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.stage and not args.only:
        parser.error('give --stage or --only')
    R = Path(args.campaign).resolve()
    PLAN = json.loads((R / 'plan.json').read_text())
    assert PLAN['root'] == str(R), 'this folder was planned as %s; make a new plan for a new place' % PLAN['root']
    cells = [c for c in PLAN['cells'] if (c['label'] in args.only if args.only else c['stage'] == args.stage)]
    unknown = set(args.only) - {c['label'] for c in PLAN['cells']}
    if unknown:
        parser.error('not in the plan: %s' % ' '.join(sorted(unknown)))

    if args.dry_run:
        for cfg in cells:
            print('\n# %s  documents=%d in_flight=%d store=%s search=%s%s' % (
                cfg['label'], cfg['documents'], PLAN['concurrency'], bool(cfg.get('store')), bool(cfg.get('search')),
                '  gate=%d (lock replaced, not as shipped)' % cfg['gate'] if 'gate' in cfg else ''))
            groups = {}
            if cfg.get('store'):
                print(' '.join(db_command(cfg)))
            for name, cmd, url in service_commands(cfg):
                print(' '.join(cmd))
                groups[name] = '<cgroup of %s>' % name
            print(' '.join(client_command(cfg, groups)))
        print('\n%d cells' % len(cells))
        return

    # This script, the sampler and the docker commands it starts stay on the client CPUs, off the CPUs of the
    # system under test and of the database.
    os.sched_setaffinity(0, expand(PLAN['client_cpus']))
    manifest = json.loads((R / 'source/corpus-manifest.json').read_text())
    if any(c['stage'] == 'campaign' for c in cells):
        assert manifest.get('stage') == 'final', 'run the screening stage and finalize.py before the timed runs'
    assert not sh(['docker', 'ps', '-q']), 'another container is running on this machine; a timed run needs it to itself'
    save(R / ('host-%s.json' % (args.stage or 'cells')), host_record())
    deadline = time.time() + args.hours * 3600
    done, failed, skipped = [], [], []
    status(state='starting', current=None, deadline_epoch=deadline)
    try:
        consecutive, stop_reason = {}, {}
        for cfg in cells:
            # Failures are counted per system: a RocketRide variant with a replaced lock is a system of its own.
            arm, label = cfg.get('system', cfg['arm']), cfg['label']
            if finished(label):
                done.append(label)
                continue
            if stop_reason.get(arm):
                skipped.append({'cell': label, 'reason': stop_reason[arm]})
                continue
            if time.time() + cfg['timeout_s'] > deadline:
                skipped.append({'cell': label, 'reason': 'not enough time left before the deadline (see --hours)'})
                continue
            if (R / 'cells' / label).exists():
                set_aside(label)
            try:
                run_cell(cfg, deadline)
                done.append(label)
                consecutive[arm] = 0
            except Exception:
                error = traceback.format_exc()
                print(error, flush=True)
                (R / 'cells' / label).mkdir(parents=True, exist_ok=True)
                save(R / 'cells' / label / 'failure.json', {'error': error, 'epoch': time.time()})
                failed.append({'cell': label, 'error': error.strip().splitlines()[-1]})
                consecutive[arm] = consecutive.get(arm, 0) + 1
                if consecutive[arm] >= 2:
                    stop_reason[arm] = 'two cells of this system failed in a row'
            status(completed=done, failed=failed, skipped=skipped)
    finally:
        status(state='complete' if not failed and not skipped else 'complete_with_limits', current=None,
               completed=done, failed=failed, skipped=skipped)
    sys.exit(1 if failed or skipped else 0)


if __name__ == '__main__':
    main()
