#!/usr/bin/env python3
"""Run a safe, reproducible pilot for file-backed self-play generation.

This script computes games-per-batch from total_games and batches_per_gpu (and GPUs detected),
constructs a `self_play_system.py` invocation with sensible shard/parquet flags, and prints it.

By default this script is a dry-run (prints the command). Pass --execute to actually run it.

Usage examples:
  # dry-run (default)
  python3 scripts/run_pilot_filewriter.py --total-games 10000 --batches-per-gpu 128 --shard-dir shards/pilot_10k

  # execute
  python3 scripts/run_pilot_filewriter.py --total-games 10000 --batches-per-gpu 128 --shard-dir shards/pilot_10k --execute

"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time


def detect_num_gpus() -> int:
    # Prefer explicit override
    env = os.environ.get("NUM_GPUS")
    if env:
        try:
            return max(1, int(env))
        except Exception:
            pass

    # Try torch if available
    try:
        import torch

        n = torch.cuda.device_count()
        return max(1, n if n > 0 else 1)
    except Exception:
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pilot runner for self-play file writer (parquet shards)")
    parser.add_argument("--total-games", type=int, default=10000, help="Total number of games to generate for the pilot")
    parser.add_argument("--batches-per-gpu", type=int, default=128, help="Number of batches to run per GPU")
    parser.add_argument("--num-gpus", type=int, default=None, help="Override GPU count detection")
    parser.add_argument("--shard-dir", type=str, default=None, help="Directory to write shards into (default: shards/pilot_<ts>)")
    parser.add_argument("--shard-games", type=int, default=1000, help="How many games per parquet shard file")
    parser.add_argument("--board-layout", type=str, default=None, help="Comma-separated board layout (e.g. 3,5,7,5,3)")
    parser.add_argument("--game-version", type=str, default="pilot-10k", help="Game version stamp to put into metadata")
    parser.add_argument("--execute", action="store_true", help="If set, actually execute the generated self_play_system command")
    parser.add_argument("--extra-args", type=str, default="", help="Extra args to append to the self_play_system command")

    args = parser.parse_args(argv)

    num_gpus = args.num_gpus if args.num_gpus is not None else detect_num_gpus()
    total_batches = args.batches_per_gpu * max(1, num_gpus)
    games_per_batch = math.ceil(args.total_games / total_batches)

    timestamp = int(time.time())
    shard_dir = args.shard_dir or f"shards/pilot_{args.total_games}_{timestamp}"
    os.makedirs(shard_dir, exist_ok=True)

    cmd = [sys.executable, "self_play_system.py"]
    # Make the run batch-only and file-writer with parquet shards
    cmd += [
        "--batch-only",
        "--file-writer",
        "--shard-format",
        "parquet",
        "--shard-games",
        str(args.shard_games),
        "--shard-dir",
        shard_dir,
        "--games-per-batch",
        str(games_per_batch),
        "--concurrent-games",
        str(games_per_batch),
        "--game-version",
        args.game_version,
    ]

    if args.board_layout:
        # pass board layout as comma-separated string
        cmd += ["--board-layout", args.board_layout]

    if args.extra_args:
        cmd += args.extra_args.split()

    # Always show what we will run
    print("Pilot configuration:")
    print(f"  total_games: {args.total_games}")
    print(f"  batches_per_gpu: {args.batches_per_gpu}")
    print(f"  num_gpus: {num_gpus}")
    print(f"  total_batches (computed): {total_batches}")
    print(f"  games_per_batch (computed): {games_per_batch}")
    print(f"  shard_games: {args.shard_games}")
    print(f"  shard_dir: {shard_dir}")
    print("")

    print("Generated command:")
    print(" ".join(map(lambda s: f'"{s}"' if ' ' in s else s, cmd)))
    print("")

    if args.execute:
        print("Executing pilot (this will run until complete)...")
        # Use subprocess.run so the output is streamed to the terminal
        ret = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)) or ".")
        return ret.returncode

    print("Dry-run mode: command not executed. Re-run with --execute to actually run the pilot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Run a file-backed pilot generating N games using the Python generator.

This script creates a SelfPlayConfig that enables the FileWriter, sets the
requested board layout and game_version, and runs batches until total_games
are generated. It is intended for quick parity checks and small pilots.
"""
import asyncio
import math
import os
from pathlib import Path

from self_play_system import SelfPlayConfig, SelfPlayGenerator


async def run_pilot(total_games: int = 10_000, concurrent_games: int = 100, shard_dir: str = "shards/pilot_v1", board_layout=None, game_version: str = "v1.0"):
    cfg = SelfPlayConfig(
        concurrent_games=concurrent_games,
        games_per_batch=concurrent_games,
        batch_only=False,
        dry_run=False,
        file_writer_enabled=True,
        shard_dir=shard_dir,
        shard_games=1000,
        board_layout=board_layout,
    )
    # attach version
    setattr(cfg, 'game_version', game_version)

    gen = SelfPlayGenerator(cfg)
    await gen.initialize()

    batches = math.ceil(total_games / concurrent_games)
    print(f"Starting pilot: total_games={total_games}, concurrent_games={concurrent_games}, batches={batches}")

    try:
        for i in range(batches):
            print(f"Running batch {i+1}/{batches}")
            await gen._generate_batch()
    finally:
        print("Shutting down generator...")
        await gen.shutdown()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run a file-backed pilot using SelfPlayGenerator')
    parser.add_argument('--total-games', type=int, default=10000)
    parser.add_argument('--concurrent-games', type=int, default=100)
    parser.add_argument('--shard-dir', type=str, default='shards/pilot_v1')
    parser.add_argument('--board-layout', type=str, default='3,5,7,5,3')
    parser.add_argument('--game-version', type=str, default='v1.0')

    args = parser.parse_args()
    # ensure shard dir exists
    Path(args.shard_dir).mkdir(parents=True, exist_ok=True)
    board_layout = [int(x) for x in args.board_layout.split(',')] if args.board_layout else None

    asyncio.run(run_pilot(total_games=args.total_games, concurrent_games=args.concurrent_games, shard_dir=args.shard_dir, board_layout=board_layout, game_version=args.game_version))
