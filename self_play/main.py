"""./.venv311/bin/python -m self_play.main --file-writer --shard-format parquet --shard-move-mode compressed --trim-states --random-start --shard-dir shards/v1_model_data --use-model --model
-path checkpoints/best_model.pt --model-device cuda --game-version v1-nn"""

import asyncio
import os
import signal
import sys
import logging
from self_play.config import SelfPlayConfig
from self_play.self_play_generator import SelfPlayGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s'
)

async def main():
    """Main entry point with CLI argument parsing."""
    import argparse

    # Load environment variables from .env when available.
    # Prefer python-dotenv if installed; fall back to a lightweight loader.
    try:
        from dotenv import load_dotenv
        # load_dotenv reads a .env file and does not overwrite existing env vars by default
        load_dotenv()
    except Exception:
        env_path = os.path.join(os.getcwd(), '.env')
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if '=' not in line:
                            continue
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        # Don't overwrite existing environment variables
                        if os.getenv(k) is None:
                            os.environ[k] = v
            except Exception:
                logger.debug('Failed to load .env file', exc_info=True)

    parser = argparse.ArgumentParser(
        description='Self-play training data generator'
    )
    parser.add_argument(
        '--concurrent-games',
        type=int,
        default=32,
        help='Number of concurrent games (default: 32)'
    )
    parser.add_argument(
        '--games-per-batch',
        type=int,
        default=32,
        help='Games per batch (default: 32)'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default=None,
        help='PostgreSQL connection URL (or use DATABASE_URL env var)'
    )
    parser.add_argument(
        '--search-depth',
        type=int,
        default=4,
        help='Engine search depth (default: 4)'
    )
    parser.add_argument(
        '--exploration-rate',
        type=float,
        default=0.1,
        help='Random move exploration rate 0-1 (default: 0.1)'
    )
    parser.add_argument(
        '--random-start',
        action='store_true',
        help='Randomize starting player for each game'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature for stochastic policies (default: 1.0)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without database writes'
    )
    parser.add_argument(
        '--batch-only',
        action='store_true',
        help='Run single batch and exit'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--metrics-port',
        type=int,
        default=None,
        help='Port to expose Prometheus metrics (default: none)'
    )
    parser.add_argument(
        '--use-model',
        action='store_true',
        help='Enable neural model inference for move selection (requires torch and model)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to torch model state_dict to load'
    )
    parser.add_argument(
        '--model-device',
        type=str,
        default=None,
        help="Device string for the model (e.g. 'cuda' or 'cpu'). If omitted, auto-selects CUDA when available."
    )
    parser.add_argument(
        '--inference-batch-size',
        type=int,
        default=32,
        help='Maximum batch size for batched inference'
    )
    parser.add_argument(
        '--inference-batch-timeout',
        type=float,
        default=0.02,
        help='Maximum wait time (seconds) before flushing an inference batch'
    )
    parser.add_argument(
        '--board-layout',
        type=str,
        default=None,
        help='Comma-separated board layout layers, e.g. 3,5,7,5,3 (overrides default canonical layout)'
    )
    parser.add_argument(
        '--game-version',
        type=str,
        default=None,
        help='Optional version tag to stamp into generated game metadata'
    )
    parser.add_argument(
        '--file-writer',
        action='store_true',
        help='Enable local file-backed shard writer (Parquet/JSONL)'
    )
    parser.add_argument(
        '--shard-format',
        type=str,
        default='parquet',
        choices=['parquet', 'jsonl'],
        help='Shard file format when using file-writer (default: parquet)'
    )
    parser.add_argument(
        '--shard-games',
        type=int,
        default=1024,
        help='Number of games per shard file (default: 1024)'
    )
    parser.add_argument(
        '--trim-states',
        action='store_true',
        help='Trim state_before/state_after blobs when writing shards to save space'
    )
    parser.add_argument(
        '--shard-move-mode',
        type=str,
        default='compressed',
        choices=['full', 'compressed', 'compact'],
        help='How to store moves in shards (default: compressed)'
    )
    parser.add_argument(
        '--shard-dir',
        type=str,
        default='shards',
        help='Directory to write shard files to (default: shards)'
    )

    args = parser.parse_args()
    # Only override file_writer if ALLOW_FILE_WRITER is set, otherwise respect CLI
    env_override = os.getenv('ALLOW_FILE_WRITER', '').lower()
    if env_override in ('1', 'true', 'yes'):
        args.file_writer = True
    # Log the actual config being used
    logger.info(
        'Runtime config: '
        f'inference_batch_size={args.inference_batch_size}, '
        f'concurrent_games={args.concurrent_games}, '
        f'trim_states={args.trim_states}, '
        f'use_model={args.use_model}, '
        f'model_device={args.model_device}, '
        f'file_writer={args.file_writer}'
    )

    # Build config from args
    config = SelfPlayConfig(
        concurrent_games=args.concurrent_games,
        games_per_batch=args.games_per_batch,
        database_url=args.db_url or os.getenv(
            'DATABASE_URL',
            'postgresql://user:pass@localhost/unitgame'
        ),
        search_depth=args.search_depth,
        exploration_rate=args.exploration_rate,
        random_start=args.random_start,
        temperature=args.temperature,
        metrics_port=args.metrics_port,
        dry_run=args.dry_run,
        batch_only=args.batch_only,
        log_level=args.log_level,
        use_model=args.use_model,
        model_path=args.model_path,
        model_device=args.model_device,
        inference_batch_size=args.inference_batch_size,
        inference_batch_timeout=args.inference_batch_timeout,
        board_layout=[int(x) for x in args.board_layout.split(',')] if args.board_layout else None,
        file_writer_enabled=args.file_writer,
        shard_format=args.shard_format,
        shard_games=args.shard_games,
        trim_states=args.trim_states,
        shard_move_mode=args.shard_move_mode,
        shard_dir=args.shard_dir,
    )

    # stamp generator/game version into config for metadata
    if args.game_version:
        # attach to config so play_single_game can include it
        setattr(config, 'game_version', args.game_version)
    else:
        setattr(config, 'game_version', os.getenv('GAME_VERSION'))

    # Note: simplified/lightweight initializer has been removed; the
    # generator defaults to the canonical board layout. No safety gate is
    # required here.

    # Create generator
    generator = SelfPlayGenerator(config)

    # Setup signal handlers for graceful shutdown
    # Prefer asyncio's add_signal_handler when running in an event loop
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(
            signal.SIGINT, lambda: asyncio.create_task(generator.shutdown())
        )
        loop.add_signal_handler(
            signal.SIGTERM, lambda: asyncio.create_task(generator.shutdown())
        )
    except NotImplementedError:
        # Fallback for platforms where add_signal_handler isn't implemented
        def _signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating shutdown...")
            asyncio.create_task(generator.shutdown())

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    try:
        await generator.initialize()
        await generator.generate_training_data()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await generator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())