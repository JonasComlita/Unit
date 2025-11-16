#!/usr/bin/env python3
"""Automation wrapper to run `scripts/benchmark_gpu.py` across configurations
and collect results into JSON files under `experiments/benchmarks/`.
"""
import argparse
import json
import os
import subprocess
import datetime
import ast


def run_benchmark():
    cmd = ["python3", "scripts/benchmark_gpu.py"]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out = proc.stdout
    # the benchmark prints Python dicts; parse lines that look like dicts
    results = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                d = ast.literal_eval(line)
                results.append(d)
            except Exception:
                continue
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='experiments/benchmarks')
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    results = run_benchmark()
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_path = os.path.join(args.out_dir, f'benchmark_{timestamp}.json')
    with open(out_path, 'w') as f:
        json.dump({'timestamp': timestamp, 'results': results}, f, indent=2)
    print('Wrote', out_path)


if __name__ == '__main__':
    main()
