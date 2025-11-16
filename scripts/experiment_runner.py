#!/usr/bin/env python3
"""Run self-play generator experiments with real DB writes and capture Prometheus metrics.

This script starts the generator as a subprocess with a metrics HTTP server, polls
the /metrics endpoint until a target number of saved games is reached, then sends
SIGINT for graceful shutdown and collects final metrics.

Usage:
  python scripts/experiment_runner.py --db-url POSTGRES_URL --concurrent 10 --target-saved 100 --metrics-port 9000

"""
import argparse
import subprocess
import time
import requests
import signal
import sys
import os


def parse_metrics(text):
    """Parse Prometheus text format into a dict of metric -> value (last seen)."""
    d = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            try:
                val = float(parts[1])
            except Exception:
                continue
            d[key] = val
    return d


def wait_for_target(url, target_saved, timeout=600):
    start = time.time()
    last = None
    while True:
        try:
            r = requests.get(url, timeout=5)
            metrics = parse_metrics(r.text)
            saved = int(metrics.get('unit_games_saved_total', 0))
            gen = int(metrics.get('unit_games_generated_total', 0)) if 'unit_games_generated_total' in metrics else None
            queue = metrics.get('unit_write_queue_length')
            db_latency_count = metrics.get('unit_db_write_latency_seconds_count')
            db_latency_sum = metrics.get('unit_db_write_latency_seconds_sum')
            last = dict(saved=saved, generated=gen, queue=queue, db_count=db_latency_count, db_sum=db_latency_sum)
            print(f"metrics: saved={saved} generated={gen} queue={queue} db_count={db_latency_count}")
            if saved >= target_saved:
                return last
        except Exception as e:
            print('metrics fetch error', e)

        if time.time() - start > timeout:
            raise RuntimeError('timeout waiting for target saved games')
        time.sleep(1)


def run_experiment(db_url, concurrent, target_saved, metrics_port, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    metrics_url = f'http://127.0.0.1:{metrics_port}/metrics'

    cmd = [sys.executable, 'self_play_system.py',
           '--concurrent-games', str(concurrent),
           '--log-level', 'INFO',
           '--metrics-port', str(metrics_port),
           '--db-url', db_url]

    print('Starting generator:', ' '.join(cmd))
    env = os.environ.copy()
    # ensure python path so module imports find project
    env['PYTHONPATH'] = env.get('PYTHONPATH', '.')

    with open(os.path.join(log_dir, f'generator_concurrent_{concurrent}.log'), 'wb') as logfile:
        p = subprocess.Popen(cmd, stdout=logfile, stderr=logfile, env=env)
        try:
            metrics = wait_for_target(metrics_url, target_saved)
            # Request graceful shutdown
            print('Target reached, sending SIGINT')
            p.send_signal(signal.SIGINT)
            # wait for process to exit
            p.wait(timeout=30)
        except Exception as e:
            print('Experiment error:', e)
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        finally:
            # fetch final metrics if possible
            try:
                r = requests.get(metrics_url, timeout=2)
                final_metrics = parse_metrics(r.text)
            except Exception:
                final_metrics = {}

    return {'concurrent': concurrent, 'target_saved': target_saved, 'metrics_snapshot': metrics, 'final_metrics': final_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url', required=True)
    parser.add_argument('--concurrent', type=int, nargs='+', default=[5,10,20])
    parser.add_argument('--target-saved', type=int, default=100)
    parser.add_argument('--metrics-port', type=int, default=9000)
    parser.add_argument('--out-dir', default='experiment_results')
    args = parser.parse_args()

    results = []
    for i, c in enumerate(args.concurrent):
        port = args.metrics_port + i
        print('\n=== Running experiment concurrent=', c, 'metrics_port=', port)
        res = run_experiment(args.db_url, c, args.target_saved, port, args.out_dir)
        results.append(res)
        # short cooldown
        time.sleep(2)

    # write summary
    summary_file = os.path.join(args.out_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        for r in results:
            f.write(str(r) + '\n')
    print('Experiments complete. Results saved to', args.out_dir)


if __name__ == '__main__':
    main()
