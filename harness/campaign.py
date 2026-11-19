"""Run the cells of a campaign, one at a time.

A cell is one run of one system. For every cell this script starts, on the CPUs the plan names:

  1. an empty Postgres with pgvector (when the cell stores),
  2. the system under test in a fresh container (RocketRide engine, or the LangGraph service with its parser),
  3. the load client in its own container,

then waits for the warm-up, removes the warm-up rows from the table, clears the database statistics, releases the
timed ingestion, reads the table back (dbcheck.py), runs the search phases (search_driver.py) and removes every
container.

Runs on the host with the standard library only. It needs Docker and cgroup v2.

On a machine where the services cannot run in containers, a plan made with `--backend process` runs the same
steps with plain processes: each service is started with `taskset` on the CPUs the plan names, its CPU time and
memory are read from /proc (procs.py), and the database is a Postgres started from unpacked packages. Such a plan
can also put the embedding model of both systems on a GPU (`--device cuda`); the campaign then records the GPU
during every run and stops a run when the GPU's memory passes the plan's limit.

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
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
import procs  # noqa: E402
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


def fetch(url, seconds, proc=None):
    end = time.monotonic() + seconds
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.read()
        except Exception:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError('the process behind %s exited' % url)
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


def write_settings(cfg):
    """Write the cell's settings file (the database password among its values). Returns the password."""
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
    return password


def start_db(cfg, out):
    """Write the cell's settings file, start the database and wait until it answers with the extension in place."""
    db = PLAN['db']
    write_settings(cfg)
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


def process_backend():
    return PLAN.get('backend') == 'process'


def psql_prefix(cfg):
    """The psql command that reaches the cell's database as its superuser, over the local socket."""
    db = PLAN['db']
    if process_backend():
        return [str(Path(PLAN['process']['pg_bin']) / 'psql'), '-h', PLAN['process']['socket_dir'], '-p', str(db['port']),
                '-U', db['user'], '-d', db['database']]
    return ['docker', 'exec', db_name(cfg), 'psql', '-U', db['user'], '-d', db['database']]


def pg_env():
    """Environment of the database's own programs. Process backend only: they are unpacked, not installed."""
    return dict(os.environ, LD_LIBRARY_PATH=PLAN['process']['pg_lib']) if process_backend() else None


def db_sql(cfg, sql):
    return subprocess.run(psql_prefix(cfg) + ['-At', '-F', '\t', '-v', 'ON_ERROR_STOP=1', '-c', sql], check=True,
                          capture_output=True, text=True, env=pg_env()).stdout.strip()


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


def search_phases(cfg, out, client_cmd, script='/harness/search_driver.py', run=None):
    """The search side, on the table the cell's ingestion filled.

    `script` is the search client as the client process sees it, and `run(command, log, phase)` starts it and
    returns its exit code; the defaults are those of the client container.

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
        cmd = client_cmd[:-2] + [script, cfg['label'], phase]
        with (out / ('search-%s.log' % phase)).open('w') as log:
            code = run(cmd, log, phase) if run else subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=3600).returncode
        finished = out / ('search-%s-finished.json' % phase)
        assert code == 0 and finished.exists() and json.loads(finished.read_text())['ok'], \
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


# ---------------------------------------------------------------------------------------------------------------
# Process backend: the same cell with plain processes in place of containers (plan `backend: process`).

def clean_env(extra=None):
    """The environment of a process the campaign starts: what a container of its image has, nothing of the shell
    that started the campaign. No thread setting of a math library is in it."""
    env = {'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'HOME': os.environ.get('HOME', '/'),
           'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'}
    env.update(extra or {})
    return env


def settings_values(label):
    """The values of a cell's settings file (the database password among them), for a process's environment."""
    return dict(line.split('=', 1) for line in Path(env_file(label)).read_text().splitlines() if '=' in line)


def listening(port):
    out = subprocess.run(['ss', '-tln'], capture_output=True, text=True).stdout
    return any(line.split()[3].endswith(':%d' % port) for line in out.splitlines()[1:] if len(line.split()) > 3)


def cell_ports(cfg):
    ports = [PLAN['rr']['port']] if cfg['arm'] == 'rr' else [PLAN['lg']['parser_port'], PLAN['lg']['port']]
    return ports + ([PLAN['db']['port']] if cfg.get('store') else [])


def cell_tmp(cfg):
    return Path(PLAN['process']['run_dir']) / 'tmp' / cfg['label']


def start_db_process(cfg, out, data, live):
    """A fresh, empty Postgres with pgvector for this one cell, on its own CPUs, with its files in `data`. The
    process is entered in `live` the moment it exists, so that a failure further down still stops it.

    What the database image does at its first start is done here by hand: a new cluster owned by the plan's user,
    password logins over TCP, trust on the local socket, the plan's database, and infra/db-init.sql.
    """
    db, p = PLAN['db'], PLAN['process']
    pgbin, sock = Path(p['pg_bin']), Path(p['socket_dir'])
    shutil.rmtree(data, ignore_errors=True)
    sock.mkdir(parents=True, exist_ok=True)
    password = write_settings(cfg)
    pw = Path(env_file(cfg['label']) + '.pw')            # beside the settings file, outside the campaign folder
    with os.fdopen(os.open(pw, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as f:
        f.write(password + '\n')
    try:
        init = subprocess.run([str(pgbin / 'initdb'), '-D', str(data), '-U', db['user'], '--pwfile', str(pw), '--auth-local=trust',
                               '--auth-host=scram-sha-256', '--encoding=UTF8', '--locale=en_US.UTF-8'],
                              capture_output=True, text=True, env=pg_env())
        (out / (db_name(cfg) + '-initdb.log')).write_text(init.stdout + init.stderr)
        assert init.returncode == 0, 'initdb failed; see the initdb log of the cell'
    finally:
        pw.unlink(missing_ok=True)
    cmd = ['taskset', '-c', db['cpus'], str(pgbin / 'postgres'), '-D', str(data), '-c', 'unix_socket_directories=' + str(sock)] + db['args']
    save(out / (db_name(cfg) + '-command.json'), cmd)
    proc = live[db_name(cfg)] = subprocess.Popen(cmd, stdout=(out / (db_name(cfg) + '.log')).open('w'), stderr=subprocess.STDOUT,
                                                 stdin=subprocess.DEVNULL, env=pg_env(), start_new_session=True)
    end = time.monotonic() + 180
    while subprocess.run([str(pgbin / 'pg_isready'), '-h', db['host'], '-p', str(db['port'])], capture_output=True, env=pg_env()).returncode != 0:
        assert proc.poll() is None, 'the database exited; see its log in the cell folder'
        assert time.monotonic() < end, 'the database did not come up'
        time.sleep(.5)
    base = ['-h', str(sock), '-p', str(db['port']), '-U', db['user']]
    subprocess.run([str(pgbin / 'createdb')] + base + [db['database']], check=True, capture_output=True, env=pg_env())
    subprocess.run([str(pgbin / 'psql')] + base + ['-d', db['database'], '-v', 'ON_ERROR_STOP=1', '-f', str(REPO / 'infra/db-init.sql')],
                   check=True, capture_output=True, env=pg_env())
    (out / (db_name(cfg) + '-ready.txt')).write_text('pgvector %s\n%s\n' % (
        db_sql(cfg, "SELECT extversion FROM pg_extension WHERE extname = 'vector'"), db_sql(cfg, 'SELECT version()')))
    save(out / 'db-settings.json', dict(line.split('\t') for line in db_sql(
        cfg, "SELECT name, setting FROM pg_settings WHERE name IN ('shared_buffers','max_connections','maintenance_work_mem','work_mem',"
             "'max_parallel_workers_per_gather','max_worker_processes','fsync','synchronous_commit','wal_level','max_wal_size',"
             "'effective_cache_size','huge_pages','lc_collate','server_encoding','hnsw.ef_search','jit','track_commit_timestamp')").splitlines()))


def process_services(cfg):
    """[(name, command, readiness URL, working folder, environment)] for one cell, as plain processes.

    The same programs with the same arguments as the containers of service_commands(), pinned to the service CPUs
    with taskset. Nothing is added to either system; the one value the process backend adds is the device of the
    LangGraph side's embedding model (the plan's `device`), which RocketRide chooses for itself.
    """
    p, name = PLAN['process'], service_name(cfg)
    tmp = str(cell_tmp(cfg))
    pin = ['taskset', '-c', PLAN['service_cpus']]
    if cfg['arm'] == 'rr':
        rr = PLAN['rr']
        assert 'gate' not in cfg, 'a run with a replaced lock is not available with the process backend'
        env = clean_env({'ROCKETRIDE_APIKEY': 'local-dev', 'TMPDIR': tmp})
        if p.get('rr_ld_library_path'):
            env['LD_LIBRARY_PATH'] = p['rr_ld_library_path']
        cmd = pin + ['./engine', 'ai/eaas.py', '--host=127.0.0.1', '--port=%d' % rr['port']]
        return [(name, cmd, 'http://127.0.0.1:%d/version' % rr['port'], p['rr_dist'], env)]
    lg = PLAN['lg']
    parser = pin + ['%s/java/jre/bin/java' % p['rr_dist'], '-Dbench.host=127.0.0.1', '-Dbench.port=%d' % lg['parser_port'],
                    '-Djava.io.tmpdir=' + tmp, '-cp', '%s/java/lib/*:%s' % (p['rr_dist'], p['parser_classes']),
                    'com.rocketride.tika_api.BenchTextServer']
    env = clean_env(dict(p['lg_env'], EXTRACTOR='tika', TIKA_URL='http://127.0.0.1:%d' % lg['parser_port'],
                         TEMP_DIRECTORY=tmp, TMPDIR=tmp, EMBED_DEVICE=PLAN['device']))
    if cfg.get('store'):
        env.update(settings_values(cfg['label']), PIPELINES='%s,%s' % (lg['ingest_pipeline'], lg['search_pipeline']))
    else:
        env['PIPELINES'] = lg['screen_pipeline']
    api = pin + [p['lg_python'], '-m', 'uvicorn', 'lg_rag_runtime:app', '--host', '127.0.0.1', '--port', str(lg['port']), '--workers', '1']
    return [(name + '-parser', parser, 'http://127.0.0.1:%d/version' % lg['parser_port'], p['rr_dist'], clean_env({'TMPDIR': tmp})),
            (name, api, 'http://127.0.0.1:%d/health/ready' % lg['port'], p['lg_app'], env)]


def process_client(cfg, out):
    """(command, environment) of the load client: the same script with the client's own Python, on the client CPUs
    (it inherits them from this script)."""
    p = PLAN['process']
    env = clean_env({'RAGBENCH_CAMPAIGN': str(R), 'RAGBENCH_CORPUS': PLAN['corpus'],
                     'RAGBENCH_PIPELINES': str(REPO / 'rocketride/pipelines'), 'RAGBENCH_SERVICES': str(out / 'services.json'),
                     'TMPDIR': str(cell_tmp(cfg))})
    if cfg.get('store'):
        env.update(settings_values(cfg['label']))       # the database password, for the pipeline placeholder
    return [p['client_python'], '-u', str(HARNESS / 'driver.py'), cfg['label']], env


def stop_tree(proc):
    """End a service: its root, every descendant and every process of its session."""
    pids = procs.members(procs.table(), proc.pid) or [proc.pid]
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        alive = []
        for pid in reversed(pids):
            try:
                os.kill(pid, sig)
                alive.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        end = time.monotonic() + wait
        while alive and time.monotonic() < end:
            proc.poll()
            alive = [x for x in alive if Path('/proc/%d' % x).exists()]
            time.sleep(.3)
        pids = alive
        if not pids:
            break
    proc.poll()


def db_series_row(cfg):
    """The database's statement statistics since they were last cleared, by kind of statement, and the row counters
    of the system's table. Reads statistics views only; it does not touch the table."""
    kinds = {}
    for line in db_sql(cfg, "SELECT upper(split_part(ltrim(s.query), ' ', 1)), sum(s.calls), round(sum(s.total_exec_time)::numeric, 1) "
                            "FROM pg_stat_statements s JOIN pg_database d ON d.oid = s.dbid WHERE d.datname = current_database() "
                            "AND s.query NOT LIKE '%pg_stat_%' GROUP BY 1").splitlines():
        kind, calls, ms = line.split('\t')
        kinds[kind] = {'calls': int(calls), 'ms': float(ms)}
    rows = db_sql(cfg, "SELECT n_tup_ins, n_tup_del, n_live_tup FROM pg_stat_user_tables WHERE relname = '%s'" % table_of(cfg)).split('\t')
    return {'epoch': time.time(), 'by_kind': kinds, 'rows_inserted': int(rows[0]), 'rows_deleted': int(rows[1]), 'rows_live': int(rows[2])}


def wait_client(client, seconds, what, out, series=None):
    """Wait for the client process. A run of ours is stopped when the GPU's memory, all its users together, passes
    the plan's limit: the card can be shared. With `series` (a cell), the database's statement statistics are
    appended to db-series.jsonl every 30 s."""
    limit = (PLAN.get('gpu') or {}).get('abort_above_mib')
    end = time.monotonic() + seconds
    next_row = time.monotonic()
    while client.poll() is None:
        if series is not None and time.monotonic() >= next_row:
            next_row += 30
            try:
                with (out / 'db-series.jsonl').open('a') as f:
                    f.write(json.dumps(db_series_row(series)) + '\n')
            except Exception as e:
                print('db series', repr(e)[:200], flush=True)
        if time.monotonic() > end:
            client.kill()
            raise TimeoutError(what + ' timed out')
        if limit:
            used = procs.gpu_memory_mib()
            if used > limit:
                client.kill()
                (out / 'gpu-watchdog.txt').write_text('%s GPU memory %d MiB over the limit %d: run stopped\n' % (
                    time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), used, limit))
                raise RuntimeError('GPU memory over the limit; the run was stopped')
        time.sleep(5)
    return client.returncode


def nvidia_csv_to_jsonl(out):
    """nvidia-smi writes local wall-clock times; turn them into epochs here, on the machine, where the time zone is
    right. gpu.jsonl: the card every half second. gpu-procs.jsonl: memory per process every 5 s."""
    def epoch(text):
        return time.mktime(time.strptime(text[:19], '%Y/%m/%d %H:%M:%S')) + float('0' + text[19:])
    rows = []
    if (out / 'gpu.csv').exists():
        for line in (out / 'gpu.csv').read_text().splitlines()[1:]:
            f = [x.strip() for x in line.split(',')]
            try:
                rows.append({'epoch': epoch(f[0]), 'util': float(f[1]), 'mem_used_mib': float(f[2]), 'power_w': float(f[3])})
            except (ValueError, IndexError):
                continue
        (out / 'gpu.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    rows = []
    if (out / 'gpu-procs.csv').exists():
        for line in (out / 'gpu-procs.csv').read_text().splitlines():
            f = [x.strip() for x in line.split(',')]
            try:
                rows.append({'epoch': epoch(f[0]), 'pid': int(f[1]), 'mib': float(f[2])})
            except (ValueError, IndexError):
                continue
        (out / 'gpu-procs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


def record_runtime_process(cfg, out, live):
    """After the warm-up, before the timed requests: is the service under test what the cell asked for?

    Every process of every service runs on its CPUs. RocketRide: the files of the release are unchanged, no gate
    module is present and the engine was started without a model server, so its models run inside it. LangGraph: the
    service reports the device the plan names. On a GPU plan the service under test must hold GPU memory by now (the
    warm-up has embedded four PDFs); on a CPU plan it must hold none.
    """
    p, main = PLAN['process'], service_name(cfg)
    table = procs.table()
    trees = {name: procs.members(table, proc.pid) for name, proc in live.items()}
    problems, report = {}, {}
    for name, proc in live.items():
        cpus = PLAN['db']['cpus'] if name == db_name(cfg) else PLAN['service_cpus']
        allowed, off = set(expand(cpus)), []
        for pid in trees[name]:
            try:
                if set(os.sched_getaffinity(pid)) != allowed:
                    off.append(pid)
            except ProcessLookupError:
                pass
        report[name] = {'root_pid': proc.pid, 'processes': len(trees[name]), 'cpus': cpus, 'processes_off_their_cpus': off,
                        'threads': sum(table[x][4] for x in trees[name])}
        if off or not trees[name]:
            problems[name] = 'not running on its CPUs, or gone'
    apps = procs.gpu_apps() if PLAN.get('gpu') else {}
    under_test = [x for name in trees if name != db_name(cfg) for x in trees[name]]
    held = sum(m for pid, m in apps.items() if pid in under_test)
    gpu = {'service_under_test_mib': held, 'other_processes_mib': {str(k): v for k, v in apps.items() if k not in under_test}}
    if PLAN.get('device', 'cpu') == 'cuda' and not held:
        problems['gpu'] = 'the service under test holds no GPU memory after the warm-up: its embedding model is not on the GPU'
    if PLAN.get('device', 'cpu') == 'cpu' and held:
        problems['gpu'] = 'the service under test holds GPU memory in a plan that embeds on CPU'
    if cfg['arm'] == 'rr':
        runtime = {'version': json.loads(fetch('http://127.0.0.1:%d/version' % PLAN['rr']['port'], 30))}
        check = subprocess.run(['sha256sum', '-c', p['release_files']], cwd=p['rr_dist'], capture_output=True, text=True)
        runtime['release_files_unchanged'] = check.returncode == 0
        if check.returncode != 0:
            problems['release_files'] = (check.stdout + check.stderr).strip()[-400:]
        if (Path(p['rr_dist']) / 'bench_gate.py').exists():
            problems['gate_module'] = 'present in a cell that runs the engine as shipped'
        if gate_records(out):
            problems['gate_records'] = 'a cell that runs the engine as shipped wrote gate records'
        command = Path('/proc/%d/cmdline' % live[main].pid).read_bytes().replace(b'\0', b' ').decode().strip()
        runtime['command'] = command
        runtime['model_server'] = '--modelserver' in command
        if runtime['model_server']:
            problems['mode'] = 'the engine was started with a model server; this plan runs its models inside the engine'
    else:
        runtime = json.loads(fetch('http://127.0.0.1:%d/__bench/runtime' % PLAN['lg']['port'], 30))
        wanted = {'model_device': 'cuda:0' if PLAN.get('device') == 'cuda' else 'cpu', 'yappi_running': False,
                  'affinity': expand(PLAN['service_cpus'])}
        problems.update({k: {'found': runtime.get(k), 'wanted': v} for k, v in wanted.items() if runtime.get(k) != v})
    runtime.update(processes=report, gpu_memory_after_warm_up=gpu)
    save(out / 'runtime.json', runtime)
    save(out / 'runtime-check.json', {'ok': not problems, 'problems': problems})
    assert not problems, 'the service under test is not what the cell asked for: %s' % json.dumps(problems)
    return {x for pids in trees.values() for x in pids}


def gpu_quiet(out, ours, name):
    """No process but ours may be working the GPU when a timed part starts. Tried for ten minutes, then the cell
    fails. The readings are kept."""
    if not PLAN.get('gpu'):
        return
    for attempt in range(20):
        busy = procs.gpu_busy_processes(ours)
        save(out / name, {'epoch': time.time(), 'attempt': attempt + 1, 'other_processes_working_the_gpu': busy})
        if not busy:
            return
        time.sleep(30)
    raise RuntimeError('another process kept working the GPU; the timed part was not started')


def run_cell_process(cfg, deadline):
    """run_cell() with plain processes. The steps and their order are the same."""
    label = cfg['label']
    out = R / 'cells' / label
    out.mkdir(parents=True)
    save(out / 'config.json', cfg)
    save(out / 'harness.json', harness_revision())
    status(state='running', current=label)
    p, store, main = PLAN['process'], bool(cfg.get('store')), service_name(cfg)
    for port in cell_ports(cfg):
        assert not listening(port), 'port %d is in use' % port
    cell_tmp(cfg).mkdir(parents=True, exist_ok=True)
    live, side, sampler, client = {}, [], None, None
    data = Path(p['run_dir']) / ('pgdata-' + label)
    try:
        if PLAN.get('gpu'):
            (out / 'gpu-memory-before-mib.txt').write_text('%d\n' % procs.gpu_memory_mib())
        if store:
            start_db_process(cfg, out, data, live)
        for name, cmd, url, cwd, env in process_services(cfg):
            # Values are not stored for names that hold a password.
            save(out / (name + '-command.json'), {'command': cmd, 'working_folder': cwd, 'environment': {
                k: ('(not stored)' if 'PASSWORD' in k else v) for k, v in sorted(env.items())}})
            live[name] = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=(out / (name + '.log')).open('w'),
                                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            (out / (name + '-ready.txt')).write_bytes(fetch(url, 900, live[name])[:4096])
        table = procs.table()
        ours = {x for proc in live.values() for x in procs.members(table, proc.pid)}
        tenant, tenant_pid = procs.foreign_cgroup(ours) if PLAN.get('gpu') else (None, None)
        save(out / 'services.json', {
            'services': {name: {'pid': proc.pid, 'under_test': name != db_name(cfg),
                                'cpus': PLAN['db']['cpus'] if name == db_name(cfg) else PLAN['service_cpus']}
                         for name, proc in live.items()},
            # Another tenant's container that holds GPU memory: only its CPU time is read, to show whether it was idle.
            'other_tenant_cgroup': tenant, 'other_tenant_pid': tenant_pid, 'utc_offset_s': -time.timezone})
        # A record of which files the service ran, not a check.
        files, where = (PLAN['rr']['files'], p['rr_dist']) if cfg['arm'] == 'rr' else (
            [f.replace('/app/', '', 1) for f in PLAN['lg']['files']], p['lg_app'])
        hashed = subprocess.run(['sha256sum'] + files, cwd=where, capture_output=True, text=True)
        (out / 'service-files.sha256').write_text(hashed.stdout + hashed.stderr)

        sampler = subprocess.Popen([sys.executable, str(HARNESS / 'sampler.py'), str(out)])
        if PLAN.get('gpu'):
            side.append(subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,power.draw',
                                          '--format=csv,nounits', '-lms', '500'], stdout=(out / 'gpu.csv').open('w'), stderr=subprocess.DEVNULL))
            side.append(subprocess.Popen(['nvidia-smi', '--query-compute-apps=timestamp,pid,used_memory', '--format=csv,noheader,nounits',
                                          '-l', '5'], stdout=(out / 'gpu-procs.csv').open('w'), stderr=subprocess.DEVNULL))
        if store:
            # Twice a second, what the system's database connections are doing (active, idle in transaction, idle).
            # A statistics view is read; nothing of the system is touched.
            watch = subprocess.Popen(psql_prefix(cfg) + ['-At', '-F', '\t', '-q'], stdin=subprocess.PIPE, text=True, env=pg_env(),
                                     stdout=(out / 'db-activity.tsv').open('w'), stderr=subprocess.DEVNULL)
            watch.stdin.write("SELECT round(extract(epoch from clock_timestamp())::numeric, 2), state, coalesce(wait_event_type, ''), "
                              "coalesce(wait_event, ''), count(*) FROM pg_stat_activity WHERE datname = current_database() "
                              "AND pid <> pg_backend_pid() AND backend_type = 'client backend' GROUP BY 2, 3, 4 \\watch 0.5\n")
            watch.stdin.close()
            side.append(watch)

        cmd, env = process_client(cfg, out)
        save(out / 'client-command.json', cmd)
        with (out / 'driver.log').open('w') as log:
            client = subprocess.Popen(cmd, cwd=HARNESS, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            end = time.monotonic() + 1800
            while not (out / 'ready.json').exists():
                if client.poll() is not None:
                    raise RuntimeError('warm-up failed; see driver.log')
                if time.monotonic() > end:
                    raise TimeoutError('warm-up timed out')
                time.sleep(.5)
            ours = record_runtime_process(cfg, out, live)
            if store:
                db_prepare(cfg, out)
            gpu_quiet(out, ours, 'gpu-quiet-before-ingestion.json')
            save(out / 'go.json', {'epoch': time.time()})
            code = wait_client(client, max(60, min(cfg['timeout_s'], deadline - time.time())), 'the ingestion', out,
                               series=cfg if store else None)
            if store:
                with (out / 'db-series.jsonl').open('a') as f:
                    f.write(json.dumps(db_series_row(cfg)) + '\n')
        assert code == 0, 'client failed; see driver.log'
        assert json.loads((out / 'finished.json').read_text())['ok']
        if store:
            # Read the table back while the database is still up, and compare it with the reference run.
            checked = subprocess.run([sys.executable, str(HARNESS / 'dbcheck.py'), str(out)] + psql_prefix(cfg),
                                     capture_output=True, text=True, env=pg_env())
            (out / 'db-check.log').write_text(checked.stdout + checked.stderr)
            assert checked.returncode == 0, 'the database check could not run; see db-check.log'
            assert json.loads((out / 'db-check.json').read_text())['pass'], 'the stored rows differ from the reference; see db-check.json'
        if cfg.get('search'):
            def run_search(command, log, phase):
                gpu_quiet(out, ours, 'gpu-quiet-before-search-%s.json' % phase)
                search = subprocess.Popen(command, cwd=HARNESS, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                return wait_client(search, 3600, 'the search run', out)
            search_phases(cfg, out, cmd, script=str(HARNESS / 'search_driver.py'), run=run_search)
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
        if client and client.poll() is None:
            client.kill()
        if sampler:
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()
        for proc in side:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if PLAN.get('gpu'):
            (out / 'gpu-memory-end-mib.txt').write_text('%d\n' % procs.gpu_memory_mib())
        for name in reversed(list(live)):
            if name == db_name(cfg):
                subprocess.run([str(Path(p['pg_bin']) / 'pg_ctl'), 'stop', '-D', str(data), '-m', 'fast', '-w', '-t', '60'],
                               capture_output=True, env=pg_env())
            stop_tree(live[name])
        shutil.rmtree(data, ignore_errors=True)
        shutil.rmtree(cell_tmp(cfg), ignore_errors=True)
        nvidia_csv_to_jsonl(out)
        time.sleep(5)
        left = [port for port in cell_ports(cfg) if listening(port)]
        save(out / 'after-the-cell.json', {'ports_still_listening': left,
                                           'gpu_memory_mib': procs.gpu_memory_mib() if PLAN.get('gpu') else None})


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
            'docker': None if process_backend() else run(['docker', 'version', '--format', '{{.Server.Version}}']),
            'backend': PLAN.get('backend', 'docker'), 'gpu': procs.gpu_record() if PLAN.get('gpu') else None,
            'epoch': time.time()}


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
            if process_backend():
                if cfg.get('store'):
                    print('a fresh Postgres from %s on CPUs %s' % (PLAN['process']['pg_bin'], PLAN['db']['cpus']))
                    path = Path(env_file(cfg['label']))
                    placeholder = not path.exists()
                    if placeholder:      # the settings file exists only while a cell runs
                        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        path.write_text('')
                for name, cmd, url, cwd, env in process_services(cfg):
                    print('(in %s) %s' % (cwd, ' '.join(cmd)))
                print(' '.join(process_client(cfg, R / 'cells' / cfg['label'])[0]))
                if cfg.get('store') and placeholder:
                    path.unlink()
                continue
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
    if process_backend():
        # The machine can have other tenants. What a run needs is its own ports; a busy GPU stops a cell before its
        # timed part (gpu_quiet), and the other tenant's CPU time is recorded with every sample.
        busy = sorted({port for c in cells for port in cell_ports(c) if listening(port)})
        assert not busy, 'ports %s are in use' % busy
    else:
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
                (run_cell_process if process_backend() else run_cell)(cfg, deadline)
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
