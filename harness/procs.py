"""Counters of a service that runs as plain processes, and the few GPU readings the campaign needs.

Used by the process backend (plan `backend: process`), for a machine where the services cannot run in containers.
A service is its root process, every descendant, and every process of the root's session. Its CPU time is the
user and system time of those processes plus the time of their children that have already exited (which Linux
adds to the parent), so work done by a helper process that has ended is still counted, as a cgroup counts it.

Memory is the sum of each process's anonymous resident memory (the cgroup's `anon`). Shared memory, such as the
database's buffer pool, is counted once (the largest share any one process maps) and added in `memory_current`.

Standard library only.
"""
import json
import os
import subprocess
import time
from pathlib import Path

TICK = os.sysconf('SC_CLK_TCK')


def table():
    """pid -> (parent, session, own ticks, exited children's ticks, threads) for every process on the machine."""
    procs = {}
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        try:
            with open('/proc/%s/stat' % name) as f:
                raw = f.read()
        except OSError:
            continue
        f = raw.rpartition(')')[2].split()
        procs[int(name)] = (int(f[1]), int(f[3]), int(f[11]) + int(f[12]), int(f[13]) + int(f[14]), int(f[17]))
    return procs


def members(procs, root):
    """The pids of one service: the root, its descendants, and the processes of the session the root leads."""
    kids = {}
    for pid, row in procs.items():
        kids.setdefault(row[0], []).append(pid)
    found, stack = set(), [root]
    while stack:
        pid = stack.pop()
        if pid in procs and pid not in found:
            found.add(pid)
            stack.extend(kids.get(pid, []))
    if root in procs and procs[root][1] == root:
        found.update(pid for pid, row in procs.items() if row[1] == root)
    return sorted(found)


def memory(pid):
    out = {}
    try:
        with open('/proc/%d/status' % pid) as f:
            for line in f:
                if line.startswith(('RssAnon:', 'RssShmem:')):
                    key, value = line.split(':')
                    out[key] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return out


def usage(roots, procs=None):
    """{service name: counters} for {service name: root pid}. A service whose root is gone is left out."""
    procs = procs or table()
    services = {}
    for name, root in roots.items():
        pids = members(procs, root)
        if root not in procs or not pids:
            continue
        mem = [memory(p) for p in pids]
        anon = sum(m.get('RssAnon', 0) for m in mem)
        services[name] = {'cpu_s': sum(procs[p][2] + procs[p][3] for p in pids) / TICK,
                          'anon_bytes': anon,
                          'memory_current': anon + max((m.get('RssShmem', 0) for m in mem), default=0),
                          'os_threads': sum(procs[p][4] for p in pids), 'processes': len(pids)}
    return services


def roots(path, under_test_only=False):
    """{service name: root pid} from a cell's services.json."""
    spec = json.loads(Path(path).read_text())
    return {name: s['pid'] for name, s in spec['services'].items() if s['under_test'] or not under_test_only}


def counters(path):
    """The load client's view at the edges of a timed part: CPU seconds of the service under test."""
    wanted = roots(path, under_test_only=True)
    groups = usage(wanted)
    assert groups and len(groups) == len(wanted), 'a process of the service under test is gone'
    return {'epoch': time.time(), 'cpu_s': sum(g['cpu_s'] for g in groups.values()), 'groups': groups}


def cgroup_cpu_s(path):
    """CPU seconds of a cgroup this campaign does not own (another tenant of the machine), read only."""
    try:
        with open(os.path.join(path, 'cpu.stat')) as f:
            return int(dict(line.split() for line in f)['usage_usec']) / 1e6
    except (OSError, KeyError, ValueError):
        return None


def gpu_memory_mib():
    """Memory in use on the first GPU, all processes together."""
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def gpu_apps():
    """{pid: MiB} of every process that holds GPU memory."""
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (line.split(', ') for line in out.splitlines() if line.strip())}


def gpu_busy_processes(ours):
    """Processes that are working the GPU right now and are not ours: [{pid, sm_percent}]."""
    out = subprocess.run(['nvidia-smi', 'pmon', '-c', '1', '-s', 'u'], capture_output=True, text=True).stdout
    busy = []
    for line in out.splitlines():
        f = line.split()
        if line.startswith('#') or len(f) < 4 or not f[1].isdigit():
            continue
        if f[3].isdigit() and int(f[3]) > 0 and int(f[1]) not in ours:
            busy.append({'pid': int(f[1]), 'sm_percent': int(f[3])})
    return busy


def gpu_record():
    """What the GPU is, for the machine record."""
    out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return [dict(zip(['name', 'driver', 'memory_total_mib'], [x.strip() for x in line.split(',')])) for line in out]


def foreign_cgroup(ours):
    """(cgroup path, pid) of a container that holds GPU memory and is not ours, to read its CPU time. (None, None)
    when there is none."""
    for pid in gpu_apps():
        if pid in ours:
            continue
        try:
            line = next(x for x in Path('/proc/%d/cgroup' % pid).read_text().splitlines() if x.startswith('0::'))
        except (OSError, StopIteration):
            continue
        path = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
        if 'docker' in str(path) and (path / 'cpu.stat').exists():
            return str(path), pid
    return None, None
