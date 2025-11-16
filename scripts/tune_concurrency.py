#!/usr/bin/env python3
"""Run quick dry-run self-play experiments to measure generation throughput.

This script imports the generator in-process and runs batch-only dry runs
to measure games generated per second for different concurrency and batch sizes.

Usage: python scripts/tune_concurrency.py --concurrent 5 --games 100 --trials 3
"""
import argparse
import time
import asyncio
from typing import Optional
from self_play_system import SelfPlayConfig, SelfPlayGenerator


async def run_trial(concurrent, games_per_batch):
    cfg = SelfPlayConfig(
        concurrent_games=concurrent,
        games_per_batch=games_per_batch,
        # default is dry-run (no DB writes); tune runner CLI can override
        dry_run=run_trial.dry_run if hasattr(run_trial, 'dry_run') else True,
        batch_only=True,
    )
    # Optional DB config hooks set by caller
    if hasattr(run_trial, 'database_url') and run_trial.database_url:
        cfg.database_url = run_trial.database_url
    if hasattr(run_trial, 'db_writer_workers'):
        cfg.db_writer_workers = run_trial.db_writer_workers
    if hasattr(run_trial, 'db_pool_min'):
        cfg.db_pool_min_size = run_trial.db_pool_min
    if hasattr(run_trial, 'db_pool_max'):
        cfg.db_pool_max_size = run_trial.db_pool_max
    gen = SelfPlayGenerator(cfg)
    await gen.initialize()

    # The generator produces `concurrent_games` per internal batch. Produce
    # enough batches to reach (or slightly exceed) the requested games_per_batch.
    import math
    num_batches = math.ceil(games_per_batch / cfg.concurrent_games)

    start = time.time()
    for _ in range(num_batches):
        # Use the internal batch generator which enqueues games for saving
        await gen._generate_batch()
    elapsed = time.time() - start

    # Return elapsed and actual generated count from the generator metrics
    return elapsed, gen.metrics.games_generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrent', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--games', type=int, nargs='+', default=[100])
    parser.add_argument('--trials', type=int, default=2)
    parser.add_argument('--db-url', type=str, default=None, help='Postgres URL to enable DB writes')
    parser.add_argument('--no-dry-run', dest='dry_run', action='store_false', help='Disable dry-run (allow DB writes)')
    parser.add_argument('--db-writers', type=int, default=None, help='Number of DB writer workers to use')
    parser.add_argument('--db-pool-min', type=int, default=None, help='DB pool min size')
    parser.add_argument('--db-pool-max', type=int, default=None, help='DB pool max size')
    parser.add_argument('--enable-batching', action='store_true', help='Enable batched DB writes')
    parser.add_argument('--batch-games', type=int, default=None, help='Batch size for batched DB writes')
    args = parser.parse_args()

    results = []

    for c in args.concurrent:
        for g in args.games:
            for t in range(args.trials):
                print(f'Running trial concurrent={c} games={g} trial={t+1}')
                try:
                    # Attach runtime options to the async helper function so it can pick them up
                    run_trial.dry_run = args.dry_run
                    run_trial.database_url = args.db_url
                    run_trial.db_writer_workers = args.db_writers
                    run_trial.db_pool_min = args.db_pool_min
                    run_trial.db_pool_max = args.db_pool_max
                    run_trial.enable_batching = args.enable_batching
                    run_trial.batch_games = args.batch_games

                    elapsed, generated = asyncio.run(run_trial(c, g))
                    rate = generated / elapsed if elapsed > 0 else 0
                    print(f'  elapsed={elapsed:.2f}s generated={generated} rate={rate:.2f} games/s')
                    results.append({'concurrent': c, 'games': g, 'trial': t+1, 'elapsed': elapsed, 'generated': generated, 'rate': rate})
                except Exception as e:
                    print('  trial failed:', e)

    print('\nSummary:')
    for r in results:
        print(r)


if __name__ == '__main__':
    main()
