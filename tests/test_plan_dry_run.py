"""Test of the plan and of the docker commands a campaign would run. No Docker, no network, standard library only.

  1. make_plan.py writes a plan and a candidate list for a made-up corpus, and leaves out a file that has the
     same bytes as an earlier one.
  2. campaign.py --dry-run prints one database, one service and one client command per cell, on the planned CPUs.
  3. Neither system's service container gets a mount or a thread setting.
  4. The three RocketRide pipeline files are well formed and are the pipelines the drivers send.

Usage: python3 tests/test_plan_dry_run.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable] + [str(a) for a in args], check=True, capture_output=True, text=True).stdout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_list_and_gates(tmp, corpus):
    """A plan from a given list of PDFs and given questions, with RocketRide runs whose lock is replaced by a gate."""
    names = sorted(x.name for x in corpus.glob('*.pdf'))
    # The list has its own order and holds two files with the same bytes (0010 and 0020): both stay.
    warm, measured = names[50:54], ['0030.pdf', '0010.pdf', '0020.pdf', '0002.pdf', '0041.pdf']
    listed = {'warm': warm, 'measured': measured, 'sha256': {n: sha(corpus / n) for n in warm + measured}}
    (tmp / 'list.json').write_text(json.dumps(listed))
    questions = [{'id': 7, 'doc': '0030.pdf', 'text': 'twelve words', 'chunk_sha256': 'x'},
                 {'id': 9, 'doc': '0002.pdf', 'text': 'twelve more', 'chunk_sha256': 'y'}]
    (tmp / 'questions.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in questions))
    gate = tmp / 'gate'
    gate.mkdir()
    (gate / 'bench_gate.py').write_text('# the gate module\n')
    (gate / 'sentence_transformers.py').write_text('# the loader with its one changed line\n')
    record = {'loader': 'ai/common/models/transformers/sentence_transformers.py', 'line': 75, 'original_sha256': 'o',
              'changed_sha256': sha(gate / 'sentence_transformers.py'), 'module_sha256': sha(gate / 'bench_gate.py')}
    (gate / 'gate-files.json').write_text(json.dumps(record))
    out = tmp / 'fixed'
    make = [REPO / 'harness/make_plan.py', '--corpus', corpus, '--manifest', tmp / 'list.json', '--questions-file',
            tmp / 'questions.jsonl', '--rr-gates', '4,8', '--gate-dir', gate, '--repeats', 1, '--engine-threads', 64,
            '--db-env-dir', tmp / 'env']
    run(*make, '--out', out)
    plan = json.loads((out / 'plan.json').read_text())
    manifest = json.loads((out / 'source/corpus-manifest.json').read_text())
    assert manifest['measured'] == measured and manifest['warm'] == warm, 'a given list is used in its own order'
    assert plan['measure_all_kept'] and plan['documents'] == plan['candidates'] == 5 and plan['workers'] == 64
    assert plan['questions'] == 2 and plan['search']['queries'] == 2 and plan['questions_given'] == 'source/questions-given.jsonl'
    assert [json.loads(x)['id'] for x in (out / 'source/questions-given.jsonl').read_text().splitlines()] == [7, 9]
    cells = {c['label']: c for c in plan['cells']}
    assert list(cells) == ['screen-rr', 'screen-lg', 'rr-1', 'lg-1', 'rr-g4-1', 'rr-g8-1'], list(cells)
    assert 'gate' not in cells['rr-1'] and 'gate' not in cells['lg-1'] and 'gate' not in cells['screen-rr']
    assert (cells['rr-g4-1']['gate'], cells['rr-g4-1']['arm'], cells['rr-g4-1']['system']) == (4, 'rr', 'rr-g4')
    assert (cells['rr-g8-1']['gate'], cells['rr-g8-1']['system']) == (8, 'rr-g8')
    timed = run(REPO / 'harness/campaign.py', out, '--stage', 'campaign', '--dry-run')
    services = {}
    for line in timed.splitlines():
        name = line.split('--name ')[1].split()[0] if line.startswith('docker run') else ''
        if name and not name.endswith('-db') and 'ragbench-client-' not in name:
            services[name] = line
    assert set(services) == {'ragbench-rr-1', 'ragbench-lg-1', 'ragbench-lg-1-parser', 'ragbench-rr-g4-1', 'ragbench-rr-g8-1'}
    for name, line in services.items():
        mounts = [part.split()[0] for part in line.split(' -v ')[1:]]
        for banned in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TORCH_']:
            assert banned not in line, (banned, line)
        if '-g' not in name:
            assert not mounts, 'a run as shipped gets no mount: %s' % line
            continue
        label = name[len('ragbench-'):]
        engine = plan['rr']['engine_dir']
        assert mounts == ['%s:/cell' % (out / 'cells' / label), '%s/bench_gate.py:%s/bench_gate.py:ro' % (gate, engine),
                          '%s/sentence_transformers.py:%s/%s:ro' % (gate, engine, record['loader'])], mounts
        # Apart from the three mounts the command is the command of the run as shipped.
        plain = line
        for m in mounts:
            plain = plain.replace(' -v ' + m, '')
        assert plain.replace(label, 'rr-1') == services['ragbench-rr-1'], (plain, services['ragbench-rr-1'])
    # A file that is not the file the list names stops the plan, and so does a gate file that was changed.
    listed['sha256']['0041.pdf'] = '0' * 64
    (tmp / 'list.json').write_text(json.dumps(listed))
    bad = subprocess.run([sys.executable] + [str(a) for a in make] + ['--out', str(tmp / 'bad')], capture_output=True, text=True)
    assert bad.returncode != 0 and 'SHA-256 differs' in bad.stderr, bad.stderr[-300:]
    listed['sha256']['0041.pdf'] = sha(corpus / '0041.pdf')
    (tmp / 'list.json').write_text(json.dumps(listed))
    (gate / 'sentence_transformers.py').write_text('# another change\n')
    bad = subprocess.run([sys.executable] + [str(a) for a in make] + ['--out', str(tmp / 'bad2')], capture_output=True, text=True)
    assert bad.returncode != 0 and 'not the file gate-files.json describes' in bad.stderr, bad.stderr[-300:]
    print('ok  a given list and given questions are used as they are; runs with a replaced lock get three mounts, '
          'runs as shipped none')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp).resolve()
        corpus = tmp / 'corpus'
        corpus.mkdir()
        for i in range(60):
            (corpus / ('%04d.pdf' % i)).write_bytes(b'%%PDF-1.4 made up %d' % i)
        # Copies: 0007 of a warm-up file, 0020 and 0021 of the earlier candidate 0010.
        copies = {'0007.pdf': '0001.pdf', '0020.pdf': '0010.pdf', '0021.pdf': '0010.pdf'}
        for copy, original in copies.items():
            (corpus / copy).write_bytes((corpus / original).read_bytes())
        out = tmp / 'run'
        run(REPO / 'harness/make_plan.py', '--corpus', corpus, '--out', out, '--smoke', '--db-env-dir', tmp / 'env')
        plan = json.loads((out / 'plan.json').read_text())
        manifest = json.loads((out / 'source/corpus-manifest.json').read_text())
        assert manifest['stage'] == 'candidates' and len(manifest['measured']) == 44 == len(set(manifest['measured']))
        assert len(manifest['warm']) == 4 and not set(manifest['warm']) & set(manifest['measured'])
        assert set(manifest['sha256']) == set(manifest['measured'] + manifest['warm'])
        # No two files of a run have the same bytes, and every copy the selection met is listed.
        assert len(set(manifest['sha256'].values())) == 48, 'two files with the same bytes were selected'
        assert not set(copies) & set(manifest['measured']) and set(manifest['duplicates_left_out']) == set(copies), manifest['duplicates_left_out']
        assert manifest['measured'] == sorted(manifest['measured'])
        labels = [c['label'] for c in plan['cells']]
        assert labels == ['screen-rr', 'screen-lg', 'rr-1', 'lg-1'], labels
        assert plan['documents'] == 40 and plan['search']['queries'] == 30 and plan['search']['levels'] == [1, 8]
        assert plan['rr']['own_object_id'] == 'schema' and plan['lg']['fixture_documents'] == ['warmup.pdf']
        assert set(plan['harness']) == {'commit', 'uncommitted_changes'}

        screening = run(REPO / 'harness/campaign.py', out, '--stage', 'screening', '--dry-run')
        timed = run(REPO / 'harness/campaign.py', out, '--stage', 'campaign', '--dry-run')
        assert screening.count('docker run') == 2 + 3 and 'pgvector' not in screening, screening      # rr+client, parser+lg+client
        assert timed.count('docker run') == 3 + 4, timed                                              # db+rr+client, db+parser+lg+client
        for line in (screening + timed).splitlines():
            if not line.startswith('docker run'):
                continue
            if 'ragbench-client-' in line:
                assert '--cpuset-cpus %s ' % plan['client_cpus'] in line and '/harness:ro' in line and '/corpus:ro' in line
            elif line.split('--name ')[1].split()[0].endswith('-db'):
                assert '--cpuset-cpus %s ' % plan['db']['cpus'] in line and 'POSTGRES_PASSWORD' not in line
                assert '-c track_commit_timestamp=on' in line, 'the read-back needs the commit time of every row'
            else:
                # A service under test: on the service CPUs, as its image ships.
                assert '--cpuset-cpus %s ' % plan['service_cpus'] in line, line
                for banned in [' -v ', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TORCH_']:
                    assert banned not in line, (banned, line)
        assert '--env-file' in timed and str(tmp / 'env') in timed

        # The same plan cannot be made twice, and a folder cannot be run from another place.
        again = subprocess.run([sys.executable, str(REPO / 'harness/make_plan.py'), '--corpus', str(corpus), '--out', str(out), '--smoke'],
                               capture_output=True, text=True)
        assert again.returncode != 0 and 'already has a plan' in again.stderr

        # Without --smoke: three timed runs per system, in alternating order.
        run(REPO / 'harness/make_plan.py', '--corpus', corpus, '--out', tmp / 'full', '--documents', 20, '--questions', 10,
            '--db-env-dir', tmp / 'env')
        full = json.loads((tmp / 'full/plan.json').read_text())
        assert [c['label'] for c in full['cells']] == ['screen-rr', 'screen-lg', 'rr-1', 'lg-1', 'lg-2', 'rr-2', 'rr-3', 'lg-3']

        fixed_list_and_gates(tmp, corpus)

    pipes = {}
    for name in ['ingest.pipe', 'ingest_no_store.pipe', 'search.pipe']:
        text = (REPO / 'rocketride/pipelines' / name).read_text()
        pipe = json.loads(text)
        assert list(pipe)[0] == 'components' and pipe['version'] == 1 and len(pipe['project_id']) == 36
        ids = [c['id'] for c in pipe['components']]
        assert len(set(ids)) == len(ids) and pipe['source'] in ids
        for c in pipe['components']:
            assert all(i['from'] in ids for i in c.get('input', [])), (name, c['id'])
        pipes[name] = [c['provider'] for c in pipe['components']]
        if 'postgres' in pipes[name]:
            local = next(c for c in pipe['components'] if c['provider'] == 'postgres')['config']['local']
            assert local['password'] == '${ROCKETRIDE_PG_PASSWORD}', 'the pipeline file must hold the placeholder, never a password'
    assert pipes['ingest.pipe'] == ['webhook', 'parse', 'preprocessor_langchain', 'embedding_transformer', 'postgres']
    assert pipes['ingest_no_store.pipe'] == ['webhook', 'parse', 'preprocessor_langchain', 'embedding_transformer', 'response_documents']
    assert pipes['search.pipe'] == ['chat', 'embedding_transformer', 'postgres', 'response_answers']
    assert len({json.loads((REPO / 'rocketride/pipelines' / n).read_text())['project_id'] for n in pipes}) == 3
    print('ALL OK')


if __name__ == '__main__':
    main()
