"""Host-side cgroup counters every 0.5 s. No profiler, tracing or process injection.

Usage: sampler.py <cell directory>
"""
import json
import sys
import time
from pathlib import Path

out = Path(sys.argv[1])
groups = {k: Path(v) for k, v in json.loads((out / 'cgroups.json').read_text()).items()}


def pairs(path):
    return dict(line.split() for line in path.read_text().splitlines())


with (out / 'samples.jsonl').open('w') as f:
    while not (out / 'sampler.stop').exists():
        services = {}
        for name, p in groups.items():
            try:
                services[name] = {'cpu_s': int(pairs(p / 'cpu.stat')['usage_usec']) / 1e6,
                                  'anon_bytes': int(pairs(p / 'memory.stat')['anon']),
                                  'memory_current': int((p / 'memory.current').read_text()),
                                  'os_threads': len((p / 'cgroup.threads').read_text().split())}
            except (OSError, KeyError, ValueError):
                continue
        if services:
            row = {'epoch': time.time(), 'services': services}
            for key in ['cpu_s', 'anon_bytes', 'memory_current', 'os_threads']:
                row[key] = sum(s[key] for s in services.values())
            f.write(json.dumps(row) + '\n')
            f.flush()
        time.sleep(.5)
