"""Host-side counters of every service of a cell every 0.5 s. No profiler, tracing or process injection.

Containers: the kernel's cgroup counters of each container (cgroups.json).
Process backend: the same numbers read from /proc for each service's process tree (services.json, see procs.py),
plus the CPU time of another tenant's container when the campaign found one sharing the GPU.

Usage: sampler.py <cell directory>
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import procs

out = Path(sys.argv[1])
spec = json.loads((out / 'services.json').read_text()) if (out / 'services.json').exists() else None
groups = {} if spec else {k: Path(v) for k, v in json.loads((out / 'cgroups.json').read_text()).items()}
roots = {name: s['pid'] for name, s in spec['services'].items()} if spec else {}


def pairs(path):
    return dict(line.split() for line in path.read_text().splitlines())


def from_cgroups():
    services = {}
    for name, p in groups.items():
        try:
            services[name] = {'cpu_s': int(pairs(p / 'cpu.stat')['usage_usec']) / 1e6,
                              'anon_bytes': int(pairs(p / 'memory.stat')['anon']),
                              'memory_current': int((p / 'memory.current').read_text()),
                              'os_threads': len((p / 'cgroup.threads').read_text().split())}
        except (OSError, KeyError, ValueError):
            continue
    return services


with (out / 'samples.jsonl').open('w') as f:
    while not (out / 'sampler.stop').exists():
        services = procs.usage(roots) if spec else from_cgroups()
        if services:
            row = {'epoch': time.time(), 'services': services}
            for key in ['cpu_s', 'anon_bytes', 'memory_current', 'os_threads']:
                row[key] = sum(s[key] for s in services.values())
            if spec and spec.get('other_tenant_cgroup'):
                row['other_tenant_cpu_s'] = procs.cgroup_cpu_s(spec['other_tenant_cgroup'])
            f.write(json.dumps(row) + '\n')
            f.flush()
        time.sleep(.5)
