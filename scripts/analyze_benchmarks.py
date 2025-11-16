#!/usr/bin/env python3
"""Analyze benchmark results under experiments/benchmarks and produce recommended defaults.

Usage: python3 scripts/analyze_benchmarks.py --out docs/recommendations.md
"""
import argparse
import json
import os
import glob
from statistics import mean


def load_results(dirpath='experiments/benchmarks'):
    files = glob.glob(os.path.join(dirpath, '*.json'))
    results = []
    for f in files:
        try:
            j = json.load(open(f))
            results.extend(j.get('results', []))
        except Exception:
            continue
    return results


def recommend(results):
    rec = {}
    # forward benchmarks: choose batch with lowest gpu_sec_per_iter
    fwd = [r for r in results if 'gpu_sec_per_iter' in r]
    if fwd:
        by_bs = {}
        for r in fwd:
            bs = r.get('batch_size')
            by_bs.setdefault(bs, []).append(r['gpu_sec_per_iter'])
        avg = {bs: mean(v) for bs, v in by_bs.items()}
        best_bs = min(avg, key=avg.get)
        rec['inference_batch_size'] = int(best_bs)

    # training benchmarks: choose batch with lowest sec_per_batch
    train = [r for r in results if 'sec_per_batch' in r]
    if train:
        by_bs = {}
        for r in train:
            bs = r.get('batch_size')
            by_bs.setdefault(bs, []).append(r['sec_per_batch'])
        avg = {bs: mean(v) for bs, v in by_bs.items()}
        best_bs = min(avg, key=avg.get)
        rec['training_batch_size'] = int(best_bs)

    # AMP: if AMP runs faster on average for same batch sizes, recommend True
    amp_yes = [r for r in train if r.get('use_amp')]
    amp_no = [r for r in train if not r.get('use_amp')]
    if amp_yes and amp_no:
        # compare averages for batch sizes present in both
        bs_common = set(r['batch_size'] for r in amp_yes) & set(r['batch_size'] for r in amp_no)
        if bs_common:
            diffs = []
            for bs in bs_common:
                a = mean([r['sec_per_batch'] for r in amp_yes if r['batch_size'] == bs])
                b = mean([r['sec_per_batch'] for r in amp_no if r['batch_size'] == bs])
                diffs.append(a - b)
            rec['use_amp_recommended'] = mean(diffs) < 0

    return rec


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--bench-dir', default='experiments/benchmarks')
    p.add_argument('--out', default='docs/recommendations.md')
    args = p.parse_args(argv)

    results = load_results(args.bench_dir)
    rec = recommend(results)

    with open(args.out, 'w') as f:
        f.write('# Benchmark-derived recommended defaults\n\n')
        for k, v in rec.items():
            f.write(f'- {k}: {v}\n')

    print('Wrote recommendations to', args.out)


if __name__ == '__main__':
    main()
