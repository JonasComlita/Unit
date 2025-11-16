#!/usr/bin/env python3
"""Run a short file-backed writer pilot and report throughput and shard sizes."""
import asyncio
import os
import time
from pathlib import Path

from self_play_system import SelfPlayConfig, SelfPlayGenerator


async def run_pilot(games=2000, concurrent=10, shard_dir='shards', shard_format='jsonl', shard_trim_states=False, shard_move_mode='full'):
    config = SelfPlayConfig(
        concurrent_games=concurrent,
        games_per_batch=concurrent,
        file_writer_enabled=True,
        shard_dir=shard_dir,
        shard_games=500,
        shard_format=shard_format,
        shard_move_mode=shard_move_mode,
        trim_states=shard_trim_states,
        write_queue_maxsize=20000,
        enqueue_timeout=1.0,
        dry_run=False,
        batch_only=False,
    )

    gen = SelfPlayGenerator(config)
    await gen.initialize()
    # ensure file writer background task started
    try:
        await gen.db_writer.start_writer()
    except Exception:
        # start_writer may not exist on some writer implementations
        pass

    generated = 0
    start = time.time()

    try:
        while generated < games:
            to_run = min(config.concurrent_games, games - generated)
            tasks = [asyncio.create_task(gen.play_single_game(game_id=generated + i)) for i in range(to_run)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    # count as game error
                    gen.metrics.game_errors += 1
                else:
                    await gen.db_writer.enqueue_game(r)
                    generated += 1

            # small backoff to allow writer to catch up
            await asyncio.sleep(0)

    finally:
        # shutdown and wait for drain
        await gen.shutdown()

    elapsed = time.time() - start

    # compute shard stats
    total_bytes = 0
    shard_count = 0
    if os.path.exists(config.shard_dir):
        if config.shard_format == 'jsonl':
            patterns = ['shard_*.jsonl']
        else:
            patterns = ['shard_*.parquet']
        for pat in patterns:
            for p in Path(config.shard_dir).glob(pat):
                if p.is_file():
                    shard_count += 1
                    total_bytes += p.stat().st_size

    print(f"elapsed={elapsed:.2f}s generated={generated} rate={generated/elapsed:.2f} games/s")
    print("Shard files:", shard_count, "Total bytes:", total_bytes)
    print({
        'concurrent': config.concurrent_games,
        'games': games,
        'elapsed': elapsed,
        'generated': generated,
        'rate': generated/elapsed if elapsed>0 else None,
        'shard_count': shard_count,
        'total_bytes': total_bytes,
    })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=2000)
    parser.add_argument('--concurrent', type=int, default=10)
    parser.add_argument('--shard-dir', type=str, default='shards')
    parser.add_argument('--shard-format', type=str, default='jsonl', choices=['jsonl', 'parquet'])
    parser.add_argument('--trim-states', action='store_true', help='Trim state_before/state_after from games when writing shards')
    parser.add_argument('--shard-move-mode', type=str, default='full', choices=['full', 'compressed', 'compact'], help='How to store moves in shards')
    args = parser.parse_args()

    asyncio.run(run_pilot(games=args.games, concurrent=args.concurrent, shard_dir=args.shard_dir, shard_format=args.shard_format, shard_trim_states=args.trim_states, shard_move_mode=args.shard_move_mode))
