"""
Production-ready self-play system for generating training data.

This module generates game training data asynchronously with:
- Async database operations using asyncpg
- Connection pooling with retry logic
- Graceful shutdown handling
- CLI configuration
- Structured logging and metrics
- Comprehensive error handling

"""

import asyncio
import json
import logging
import random
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import struct
import base64

import asyncpg
import numpy as np
from asyncpg.pool import Pool
from metrics import get_registry_and_start, create_metrics
# Optional model/batching imports
try:
    from services.inference_batcher import InferenceBatcher
except Exception:
    InferenceBatcher = None

try:
    # Import model utilities if available; keep optional so file works without torch
    from neural_network_model import UnitGameNet, state_to_tensor
    import torch
except Exception:
    UnitGameNet = None
    state_to_tensor = None
    torch = None
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False

# Optional compact serialization / compression
try:
    import msgpack
    import zstandard as zstd
    MSGPACK_AVAILABLE = True
except Exception:
    msgpack = None
    zstd = None
    MSGPACK_AVAILABLE = False

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play generator."""
    games_per_batch: int = 100
    search_depth: int = 4
    exploration_rate: float = 0.1
    concurrent_games: int = 10
    database_url: str = field(
        default_factory=lambda: os.getenv(
            'DATABASE_URL',
            'postgresql://user:pass@localhost/unitgame'
        )
    )
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20
    db_retry_attempts: int = 3
    db_retry_delay: float = 1.0
    dry_run: bool = False
    batch_only: bool = False
    log_level: str = 'INFO'
    random_start: bool = False
    temperature: float = 1.0
    # Optional override for board layout layers; if None the canonical
    # 5-layer layout is used for training.
    board_layout: Optional[List[int]] = None
    metrics_port: Optional[int] = None
    # Number of concurrent async writer workers consuming the write queue.
    # Increasing this allows parallel DB writes (must be balanced with pool size).
    db_writer_workers: int = 4
    # Max size for the in-memory write queue (provide backpressure).
    # Set to a large number by default so tests and local runs don't block.
    write_queue_maxsize: int = 1000
    # Timeout when trying to enqueue a game before considering it dropped (seconds)
    enqueue_timeout: float = 0.5
    # How long to wait for writer to drain on shutdown before forcing a flush (seconds)
    shutdown_grace_period: float = 5.0
    # Maximum backoff cap for DB retries (seconds)
    db_retry_backoff_cap: float = 30.0
    # Batch write options: when enabled, writer workers will attempt to write
    # multiple games in a single DB transaction to reduce per-game overhead.
    enable_batch_writes: bool = False
    batch_games: int = 25
    batch_timeout: float = 0.5
    # File writer options (JSONL shards)
    file_writer_enabled: bool = False
    shard_dir: str = 'shards'
    shard_games: int = 1000
    # whether to compress shards (not implemented yet)
    shard_compress: bool = False
    # shard_format: 'jsonl' or 'parquet'
    shard_format: str = 'jsonl'
    # Trim large state blobs (state_before/state_after) when writing shards.
    # Set to True to reduce shard size for production training data exports.
    trim_states: bool = False
    # How to store moves in shards: 'full' (store complete moves array),
    # 'compressed' (gzip-compress moves JSON into a binary column),
    # or 'compact' (store a reduced representation - not implemented here).
    shard_move_mode: str = 'full'
    # Optional neural inference settings - disabled by default
    use_model: bool = False
    model_path: Optional[str] = None
    model_device: Optional[str] = None  # e.g. 'cuda', 'cpu', or 'cuda:0'
    inference_batch_size: int = 32
    inference_batch_timeout: float = 0.02
    # State serialization strategy: 'none'|'json'|'binary'|'delta'
    # 'none' will NOT store per-move states (store initial_state once) - recommended for training
    state_serialization: str = 'none'
    # When True, evaluate candidate next-states from the mover's perspective
    # (fixes an evaluation-perspective bias where evaluate_position used
    # the new state's currentPlayerId). This can be toggled for A/B tests.
    evaluate_from_mover: bool = False
    # Instrumentation: when True, get_engine_move will log candidate moves
    # and their evaluated scores to help debug policy/eval issues.
    instrument: bool = False


@dataclass
class Metrics:
    """Track system metrics."""
    games_generated: int = 0
    games_saved: int = 0
    db_errors: int = 0
    game_errors: int = 0
    start_time: float = field(default_factory=time.time)

    def log_summary(self):
        """Log metrics summary."""
        elapsed = time.time() - self.start_time
        logger.info(
            f"Metrics Summary - Games Generated: {self.games_generated}, "
            f"Games Saved: {self.games_saved}, "
            f"DB Errors: {self.db_errors}, "
            f"Game Errors: {self.game_errors}, "
            f"Elapsed: {elapsed:.1f}s, "
            f"Rate: {self.games_generated / elapsed if elapsed > 0 else 0:.2f} games/s"
        )


class DatabaseWriter:
    """Handles asynchronous database operations with connection pooling."""

    def __init__(self, config: SelfPlayConfig, metrics: Metrics, prom_metrics: Optional[dict] = None):
        self.config = config
        self.metrics = metrics
        self.pool: Optional[Pool] = None
        # Use a bounded queue to provide backpressure between generator and DB writer.
        maxsize = getattr(self.config, 'write_queue_maxsize', 0) or 0
        # asyncio.Queue treats 0 as infinite, so pass maxsize directly.
        self.write_queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        # support multiple concurrent writer tasks for higher write throughput
        self.writer_tasks: List[asyncio.Task] = []
        # backward-compatible single-task reference used by some tests
        self.writer_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        # internal flag to stop accepting new enqueues during shutdown
        self._accepting = True

        # Prometheus metrics: if a dict of pre-created metric objects is supplied
        # (via `prom_metrics`), bind to those. Otherwise metric ops are no-ops.
        if prom_metrics:
            self._p_games_generated = prom_metrics.get('games_generated')
            self._p_games_saved = prom_metrics.get('games_saved')
            self._p_db_errors = prom_metrics.get('db_errors')
            self._p_game_errors = prom_metrics.get('game_errors')
            self._p_queue_length = prom_metrics.get('queue_length')
            self._p_db_latency = prom_metrics.get('db_latency')
        else:
            self._p_games_generated = None
            self._p_games_saved = None
            self._p_db_errors = None
            self._p_game_errors = None
            self._p_queue_length = None
            self._p_db_latency = None

    async def initialize(self):
        """Initialize database connection pool."""
        if self.config.dry_run:
            logger.info("Dry run mode - skipping database initialization")
            return

        logger.info("Initializing database connection pool...")
        retry_count = 0
        last_error = None

        while retry_count < self.config.db_retry_attempts:
            try:
                # Some tests patch `asyncpg.create_pool` with a plain object
                # or AsyncMock that may not be awaitable. Call it and handle
                # both awaitable and non-awaitable returns to be robust.
                pool_candidate = asyncpg.create_pool(
                    self.config.database_url,
                    min_size=self.config.db_pool_min_size,
                    max_size=self.config.db_pool_max_size,
                    command_timeout=60
                )

                # If the factory returned an awaitable/coroutine, await it.
                if hasattr(pool_candidate, '__await__'):
                    self.pool = await pool_candidate  # real asyncpg returns a coroutine
                else:
                    # Tests may return AsyncMock or a plain object directly.
                    self.pool = pool_candidate
                logger.info(
                    f"Database pool initialized (min={self.config.db_pool_min_size}, "
                    f"max={self.config.db_pool_max_size})"
                )
                # Ensure metadata table exists (lightweight, non-destructive)
                try:
                    # Some pool mocks may not support async context manager; handle both.
                    acquire_candidate = self.pool.acquire()

                    async def _create_metadata_table(conn):
                        await conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS game_metadata (
                                game_id VARCHAR(50) PRIMARY KEY REFERENCES games(game_id),
                                metadata JSONB,
                                created_at TIMESTAMP DEFAULT NOW()
                            )
                            """
                        )

                    if hasattr(acquire_candidate, '__aenter__'):
                        async with acquire_candidate as conn:
                            await _create_metadata_table(conn)
                    else:
                        conn = await acquire_candidate
                        await _create_metadata_table(conn)
                except Exception:
                    # Non-fatal: metadata table creation failure shouldn't stop initialization.
                    logger.debug("Could not ensure game_metadata table exists (continuing)", exc_info=True)

                return
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.config.db_retry_attempts:
                    delay = self.config.db_retry_delay * (2 ** (retry_count - 1))
                    logger.warning(
                        f"Database connection failed (attempt {retry_count}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)

        logger.error(f"Failed to initialize database after {retry_count} attempts")
        raise last_error

    async def start_writer(self):
        """Start the database writer worker."""
        if self.config.dry_run:
            logger.info("Dry run mode - writer not started")
            return
        # If workers already running, skip
        if any(not t.done() for t in self.writer_tasks):
            logger.debug("Writer workers already running")
            return

        workers = max(1, int(getattr(self.config, 'db_writer_workers', 1)))
        self.writer_tasks = [asyncio.create_task(self._writer_worker()) for _ in range(workers)]
        logger.info(f"Database writer workers started (count={workers})")

    async def _writer_worker(self):
        """Worker that consumes games from queue and writes to database."""
        logger.info("Writer worker running")
        
        while not self.shutdown_event.is_set() or not self.write_queue.empty():
            try:
                if getattr(self.config, 'enable_batch_writes', False):
                    # Attempt to gather a batch of games
                    batch = []
                    try:
                        first = await asyncio.wait_for(self.write_queue.get(), timeout=1.0)
                        batch.append(first)
                    except asyncio.TimeoutError:
                        # nothing to do right now
                        continue

                    # Try to quickly drain up to batch size without waiting long
                    batch_size = max(1, int(getattr(self.config, 'batch_games', 25)))
                    batch_deadline = time.time() + float(getattr(self.config, 'batch_timeout', 0.5))

                    while len(batch) < batch_size and time.time() < batch_deadline:
                        try:
                            item = self.write_queue.get_nowait()
                            batch.append(item)
                        except asyncio.QueueEmpty:
                            # give a tiny sleep to allow quick arrivals
                            await asyncio.sleep(0)
                            break

                    # Save the batch with retry semantics
                    await self._save_games_with_retry(batch)

                    # mark done for all items
                    for _ in batch:
                        try:
                            self.write_queue.task_done()
                        except Exception:
                            pass

                    try:
                        if getattr(self, '_p_queue_length', None):
                            self._p_queue_length.set(self.write_queue.qsize())
                    except Exception:
                        pass
                else:
                    # Non-batched path (legacy)
                    game = await asyncio.wait_for(self.write_queue.get(), timeout=1.0)
                    await self._save_game_with_retry(game)
                    self.write_queue.task_done()
                    try:
                        if getattr(self, '_p_queue_length', None):
                            self._p_queue_length.set(self.write_queue.qsize())
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Writer worker error: {e}", exc_info=True)
                self.metrics.db_errors += 1
                try:
                    if getattr(self, '_p_db_errors', None):
                        self._p_db_errors.inc()
                except Exception:
                    pass

        logger.info("Writer worker shutting down")

    async def _force_drain_queue(self):
        """Synchronous drain: try to write remaining items directly to DB.

        This is used as a last-resort during shutdown if the writer didn't
        finish within the grace period. It attempts best-effort writes and
        logs failures without raising.
        """
        logger.info("Force-draining write queue (%d items)...", self.write_queue.qsize())
        while not self.write_queue.empty():
            try:
                game = self.write_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                await self._save_game_with_retry(game)
            except Exception as e:
                logger.exception("Failed to force-write game %s: %s", game.get('game_id'), e)
            finally:
                try:
                    self.write_queue.task_done()
                except Exception:
                    pass

    async def _save_game_with_retry(self, game: Dict[str, Any]):
        """Save game to database with retry logic."""
        retry_count = 0
        last_error = None

        while retry_count < self.config.db_retry_attempts:
            try:
                # measure DB latency via histogram if available
                db_latency = getattr(self, '_p_db_latency', None)
                if db_latency:
                    with db_latency.time():
                        await self._save_game_to_db(game)
                else:
                    await self._save_game_to_db(game)
                self.metrics.games_saved += 1
                try:
                    if getattr(self, '_p_games_saved', None):
                        self._p_games_saved.inc()
                except Exception:
                    pass
                return
            except Exception as e:
                last_error = e
                retry_count += 1
                self.metrics.db_errors += 1
                try:
                    if getattr(self, '_p_db_errors', None):
                        self._p_db_errors.inc()
                except Exception:
                    pass
                
                if retry_count < self.config.db_retry_attempts:
                    # Exponential backoff with a small jitter to avoid thundering herd
                    base = self.config.db_retry_delay * (2 ** (retry_count - 1))
                    delay = min(self.config.db_retry_backoff_cap, base)
                    jitter = random.uniform(0, min(0.2 * delay, 1.0))
                    delay += jitter
                    logger.warning(
                        f"DB write failed (attempt {retry_count}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)

        logger.error(
            f"Failed to save game {game['game_id']} after "
            f"{retry_count} attempts: {last_error}"
        )

    async def _save_game_to_db(self, game: Dict[str, Any]):
        """Save game and moves to PostgreSQL."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        # Prefer game-provided timestamps if available, otherwise use now.
        start_ts = int(game.get('start_time') or int(time.time() * 1000))
        end_ts = int(game.get('end_time') or int(time.time() * 1000))
        # Acquire may return either an async context manager (real pool)
        # or an awaitable that yields a connection (some test mocks). Handle both.
        acquire_candidate = self.pool.acquire()

        # Helper to perform DB operations given a connection object.
        async def _perform_ops(conn):
            async with conn.transaction():
                # Insert game record
                await conn.execute(
                    """
                    INSERT INTO games (
                        game_id, start_time, end_time, winner, 
                        total_moves, platform, initial_state
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (game_id) DO NOTHING
                    """,
                    game['game_id'],
                    start_ts,
                    end_ts,
                    game.get('winner'),
                    game.get('total_moves'),
                    'selfplay',
                    game.get('initial_state')
                )

                # Save game metadata if present and metadata table exists
                metadata = game.get('metadata')
                if metadata is not None:
                    try:
                        await conn.execute(
                            """
                            INSERT INTO game_metadata (game_id, metadata)
                            VALUES ($1, $2)
                            ON CONFLICT (game_id) DO UPDATE SET metadata = EXCLUDED.metadata
                            """,
                            game['game_id'],
                            json.dumps(metadata)
                        )
                    except Exception:
                        # Don't fail game save just because metadata couldn't be saved
                        logger.debug("Failed to save metadata for game %s", game['game_id'], exc_info=True)

                # Batch insert moves
                now_ts = int(time.time() * 1000)
                move_records = [
                    (
                        game['game_id'],
                        move.get('move_number'),
                        move.get('player'),
                        (move.get('action') or {}).get('type'),
                        json.dumps(move.get('action', {})),
                        move.get('thinking_time_ms', 0),
                        int(move.get('timestamp') or now_ts)
                    )
                    for move in game.get('moves', [])
                ]

                if move_records:
                    await conn.executemany(
                        """
                        INSERT INTO moves (
                            game_id, move_number, player_id, action_type,
                            action_data, thinking_time_ms, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT DO NOTHING
                        """,
                        move_records
                    )

        # If acquire_candidate supports async context manager protocol, use it.
        if hasattr(acquire_candidate, '__aenter__'):
            async with acquire_candidate as conn:
                await _perform_ops(conn)
        else:
            # Otherwise, assume it's awaitable and yields a connection object.
            conn = await acquire_candidate
            # Perform operations using the plain connection object.
            await _perform_ops(conn)

    async def _save_games_with_retry(self, games: List[Dict[str, Any]]):
        """Save multiple games in a single transaction with retry logic."""
        retry_count = 0
        last_error = None

        while retry_count < self.config.db_retry_attempts:
            try:
                db_latency = getattr(self, '_p_db_latency', None)
                if db_latency:
                    with db_latency.time():
                        await self._save_games_to_db(games)
                else:
                    await self._save_games_to_db(games)

                # update metrics for all games in the batch
                self.metrics.games_saved += len(games)
                try:
                    if getattr(self, '_p_games_saved', None):
                        for _ in range(len(games)):
                            self._p_games_saved.inc()
                except Exception:
                    pass
                return
            except Exception as e:
                last_error = e
                retry_count += 1
                self.metrics.db_errors += 1
                try:
                    if getattr(self, '_p_db_errors', None):
                        self._p_db_errors.inc()
                except Exception:
                    pass

                if retry_count < self.config.db_retry_attempts:
                    base = self.config.db_retry_delay * (2 ** (retry_count - 1))
                    delay = min(self.config.db_retry_backoff_cap, base)
                    jitter = random.uniform(0, min(0.2 * delay, 1.0))
                    delay += jitter
                    logger.warning(
                        f"DB batch write failed (attempt {retry_count}), retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)

        logger.error(f"Failed to save batch of {len(games)} games after {retry_count} attempts: {last_error}")

    async def _save_games_to_db(self, games: List[Dict[str, Any]]):
        """Perform batched save of games and moves inside a single transaction."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        acquire_candidate = self.pool.acquire()

        async def _perform_ops(conn):
            async with conn.transaction():
                # Prepare game records
                game_records = [(
                    g['game_id'],
                    int(g.get('start_time') or int(time.time() * 1000)),
                    int(g.get('end_time') or int(time.time() * 1000)),
                    g.get('winner'),
                    g.get('total_moves'),
                    'selfplay',
                    g.get('initial_state')
                ) for g in games]

                if game_records:
                    await conn.executemany(
                        """
                        INSERT INTO games (
                            game_id, start_time, end_time, winner, 
                            total_moves, platform, initial_state
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (game_id) DO NOTHING
                        """,
                        game_records
                    )

                # Metadata inserts (best-effort)
                for g in games:
                    metadata = g.get('metadata')
                    if metadata is not None:
                        try:
                            await conn.execute(
                                """
                                INSERT INTO game_metadata (game_id, metadata)
                                VALUES ($1, $2)
                                ON CONFLICT (game_id) DO UPDATE SET metadata = EXCLUDED.metadata
                                """,
                                g['game_id'],
                                json.dumps(metadata)
                            )
                        except Exception:
                            logger.debug("Failed to save metadata for game %s", g['game_id'], exc_info=True)

                # Batch moves - flatten moves across all games
                move_records = []
                now_ts = int(time.time() * 1000)
                for g in games:
                    for move in g.get('moves', []):
                        move_records.append((
                            g['game_id'],
                            move.get('move_number'),
                            move.get('player'),
                            (move.get('action') or {}).get('type'),
                            json.dumps(move.get('action', {})),
                            move.get('thinking_time_ms', 0),
                            int(move.get('timestamp') or now_ts)
                        ))

                if move_records:
                    await conn.executemany(
                        """
                        INSERT INTO moves (
                            game_id, move_number, player_id, action_type,
                            action_data, thinking_time_ms, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT DO NOTHING
                        """,
                        move_records
                    )

        if hasattr(acquire_candidate, '__aenter__'):
            async with acquire_candidate as conn:
                await _perform_ops(conn)
        else:
            conn = await acquire_candidate
            await _perform_ops(conn)

    async def enqueue_game(self, game: Dict[str, Any]):
        """Add game to write queue."""
        if self.config.dry_run:
            logger.debug(f"[DRY RUN] Would save game: {game['game_id']}")
            return
        if not self._accepting:
            logger.debug("Writer not accepting new games (shutting down); dropping %s", game.get('game_id'))
            self.metrics.db_errors += 1
            return
        try:
            # Provide short backpressure window; if the queue is full we wait up to configured timeout.
            await asyncio.wait_for(self.write_queue.put(game), timeout=getattr(self.config, 'enqueue_timeout', 2.0))
            try:
                if getattr(self, '_p_queue_length', None):
                    self._p_queue_length.set(self.write_queue.qsize())
            except Exception:
                pass
        except asyncio.TimeoutError:
            # Queue is full and cannot accept new items quickly — drop and log.
            logger.warning(
                f"Write queue full, dropping game {game.get('game_id')} to avoid blocking"
            )
            self.metrics.db_errors += 1
            try:
                if getattr(self, '_p_db_errors', None):
                    self._p_db_errors.inc()
            except Exception:
                pass

    async def shutdown(self):
        """Gracefully shutdown writer and close pool."""
        logger.info("Shutting down database writer...")
        # Stop accepting new enqueues
        self._accepting = False
        self.shutdown_event.set()

        # Backwards-compat: some tests set a single writer task on `writer_task`.
        # Ensure it's included in `writer_tasks` so shutdown waits for it.
        if getattr(self, 'writer_task', None) and self.writer_task not in self.writer_tasks:
            self.writer_tasks.append(self.writer_task)

        if self.writer_tasks:
            # Wait for queue to drain with a grace period; if it doesn't finish
            # within the grace period, attempt a force-drain.
            try:
                await asyncio.wait_for(self.write_queue.join(), timeout=self.config.shutdown_grace_period)
            except asyncio.TimeoutError:
                logger.warning("Writer did not drain within %s seconds, force-draining remaining items", self.config.shutdown_grace_period)
                try:
                    await self._force_drain_queue()
                except Exception:
                    logger.exception("Error during forced drain")

            # Wait for workers to finish
            for task in self.writer_tasks:
                try:
                    await asyncio.wait_for(task, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("Writer task did not exit promptly after drain; cancelling")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            logger.info("Writer workers stopped")

        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")


class SelfPlayGenerator:
    """Generates self-play games for training data."""

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        # State serializer chooses how (and if) per-move states are stored.
        # Possible values: 'none', 'json', 'binary', 'delta'
        self._state_serialization = getattr(self.config, 'state_serialization', 'none')
        self.metrics = Metrics()
        # Initialize prometheus registry and metrics objects.
        reg = None
        try:
            reg = get_registry_and_start(getattr(self.config, 'metrics_port', None))
        except Exception:
            reg = None

        prom_metrics = None
        try:
            prom_metrics = create_metrics(reg)
        except Exception:
            prom_metrics = None

        # Prefer a file-backed writer when enabled; otherwise use DB writer.
        if getattr(self.config, 'file_writer_enabled', False):
            try:
                # Local import so scripts that don't have writers/ dir won't break.
                from writers.file_writer import FileWriter
                self.db_writer = FileWriter(config, self.metrics, prom_metrics=prom_metrics)
            except Exception:
                logger.exception("Failed to initialize FileWriter; falling back to DatabaseWriter")
                self.db_writer = DatabaseWriter(config, self.metrics, prom_metrics=prom_metrics)
        else:
            self.db_writer = DatabaseWriter(config, self.metrics, prom_metrics=prom_metrics)
        # Optional model and batched inference
        self._model = None
        self._inference_batcher = None
        if getattr(self.config, 'use_model', False):
            if UnitGameNet is None or state_to_tensor is None or torch is None or InferenceBatcher is None:
                logger.warning("Model or InferenceBatcher not available; continuing without model")
            else:
                # create model instance
                try:
                    device = self.config.model_device or ("cuda" if torch.cuda.is_available() else "cpu")
                    self._model = UnitGameNet()
                    if getattr(self.config, 'model_path', None):
                        try:
                            self._model.load_state_dict(torch.load(self.config.model_path, map_location='cpu'))
                        except Exception:
                            logger.exception("Failed to load model state dict from %s", self.config.model_path)
                    self._model.to(device)
                    self._model.eval()

                    # Create model_fn that accepts a list of game state dicts and returns list of (policy, value)
                    def model_fn(batch_states):
                        # batch_states: list[dict]
                        with torch.no_grad():
                            tensors = [state_to_tensor(s) for s in batch_states]
                            import numpy as _np
                            batch = _np.stack(tensors, axis=0).astype('float32')
                            x = torch.from_numpy(batch).to(device)
                            policy_pred, value_pred = self._model(x)
                            # move to cpu and numpy
                            policy_np = policy_pred.detach().cpu().numpy()
                            value_np = value_pred.detach().cpu().numpy()
                            results = []
                            for i in range(policy_np.shape[0]):
                                results.append((policy_np[i], float(value_np[i].squeeze())))
                            return results

                    self._inference_batcher = InferenceBatcher(model_fn, max_batch_size=int(self.config.inference_batch_size), timeout=float(self.config.inference_batch_timeout))
                except Exception:
                    logger.exception("Failed to initialize model/batcher; continuing without model")
        self.shutdown_requested = False

        # Set log level
        logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
        # Note: HTTP server is started by get_registry_and_start above when a port is provided.

    async def initialize(self):
        """Initialize generator and database."""
        await self.db_writer.initialize()
        await self.db_writer.start_writer()
        # start inference batcher if present
        if self._inference_batcher:
            try:
                await self._inference_batcher.start()
                logger.info("Inference batcher started (batch_size=%d timeout=%.3f)", self.config.inference_batch_size, self.config.inference_batch_timeout)
            except Exception:
                logger.exception("Failed to start inference batcher")

    async def generate_training_data(self):
        """Generate games continuously or for one batch."""
        logger.info(
            f"Starting self-play generation - "
            f"concurrent_games={self.config.concurrent_games}, "
            f"batch_size={self.config.games_per_batch}, "
            f"dry_run={self.config.dry_run}"
        )

        try:
            if self.config.batch_only:
                await self._generate_batch()
            else:
                await self._generate_continuous()
        finally:
            self.metrics.log_summary()

    async def _generate_continuous(self):
        """Generate games continuously until shutdown."""
        last_stats_time = time.time()
        stats_interval = 30  # Log stats every 30 seconds

        while not self.shutdown_requested:
            await self._generate_batch()

            # Periodic stats logging
            if time.time() - last_stats_time > stats_interval:
                self.metrics.log_summary()
                last_stats_time = time.time()

    async def _generate_batch(self):
        """Generate one batch of games."""
        batch_start = time.time()
        
        # Create tasks for concurrent games
        tasks = [
            self.play_single_game(game_id=i)
            for i in range(self.config.concurrent_games)
        ]

        # Gather with error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        games = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Game generation error: {result}", exc_info=result)
                self.metrics.game_errors += 1
            else:
                games.append(result)
                self.metrics.games_generated += 1

        # Save to database
        for game in games:
            await self.db_writer.enqueue_game(game)

        # Print statistics
        self._print_batch_statistics(games, batch_start)

    def _print_batch_statistics(self, games: List[Dict], batch_start: float):
        """Print statistics about generated games."""
        if not games:
            logger.warning("No games generated in batch")
            return

        batch_time = time.time() - batch_start
        avg_moves = np.mean([g['total_moves'] for g in games])
        p1_wins = sum(1 for g in games if g['winner'] == 'Player1')
        p2_wins = sum(1 for g in games if g['winner'] == 'Player2')
        draws = len(games) - p1_wins - p2_wins

        logger.info(
            f"Batch complete: {len(games)} games in {batch_time:.1f}s "
            f"({len(games)/batch_time:.2f} games/s)"
        )
        logger.info(
            f"Stats - Avg moves: {avg_moves:.1f}, "
            f"P1 wins: {p1_wins} ({p1_wins/len(games)*100:.1f}%), "
            f"P2 wins: {p2_wins} ({p2_wins/len(games)*100:.1f}%), "
            f"Draws: {draws}"
        )
        # Starting player distribution if metadata available
        starts = [g.get('metadata', {}).get('starting_player', 'UNKNOWN') for g in games]
        if any(s != 'UNKNOWN' for s in starts):
            from collections import Counter
            c = Counter(starts)
            parts = [f"{k}:{v}" for k, v in c.items()]
            logger.info(f"Starting player distribution: {', '.join(parts)}")

    async def play_single_game(self, game_id: int) -> Dict[str, Any]:
        """
        Simulate one complete game.
        
        Args:
            game_id: Unique identifier for this game
            
        Returns:
            Dictionary containing game data and move history
        """
        # Record start timestamp for the game (ms)
        start_ms = int(time.time() * 1000)
        game_state = self.initialize_game()
        # Keep a copy of the initial state; we may store it once per-game
        try:
            import copy as _copy
            _initial_state = _copy.deepcopy(game_state)
        except Exception:
            _initial_state = game_state.copy()

        # Per-game random seed and starting player selection
        seed = int.from_bytes(os.urandom(4), 'big') % (2 ** 31)
        if getattr(self.config, 'random_start', False):
            starting_player = 'Player1' if (seed % 2 == 0) else 'Player2'
            game_state['currentPlayerId'] = starting_player
        else:
            starting_player = game_state.get('currentPlayerId', 'Player1')

        move_history = []
        move_count = 0
        max_moves = 500

        while not game_state['winner'] and move_count < max_moves:
            current_player = game_state['currentPlayerId']

            # Exploration vs exploitation
            if np.random.random() < self.config.exploration_rate:
                move = self.get_random_move(game_state)
            else:
                if self._inference_batcher:
                    try:
                        move = await self.get_model_move(game_state)
                    except Exception:
                        logger.exception("Model inference failed; falling back to engine move")
                        move = self.get_engine_move(game_state, self.config.search_depth)
                else:
                    move = self.get_engine_move(game_state, self.config.search_depth)

            # Capture state_before (may be omitted based on strategy)
            state_before_raw = game_state

            # Apply move
            game_state = self.apply_move(game_state, move)

            # Capture state_after raw
            state_after_raw = game_state

            # Build move record according to selected serialization strategy
            move_rec: Dict[str, Any] = {
                'move_number': move_count,
                'player': current_player,
                'action': move,
                'timestamp': int(time.time() * 1000)
            }

            strat = getattr(self.config, 'state_serialization', 'none')
            if strat == 'json':
                move_rec['state_before'] = self.serialize_state(state_before_raw)
                move_rec['state_after'] = self.serialize_state(state_after_raw)
            elif strat == 'binary':
                # compact binary representation, base64 encoded
                try:
                    packed_before = self._compact_binary_state(state_before_raw)
                    packed_after = self._compact_binary_state(state_after_raw)
                    move_rec['state_before'] = base64.b64encode(packed_before).decode('ascii')
                    move_rec['state_after'] = base64.b64encode(packed_after).decode('ascii')
                except Exception:
                    # fallback to JSON if binary fails
                    move_rec['state_before'] = self.serialize_state(state_before_raw)
                    move_rec['state_after'] = self.serialize_state(state_after_raw)
            elif strat == 'delta':
                # Store only the delta between before/after
                try:
                    move_rec['state_delta'] = self._compute_state_delta(state_before_raw, state_after_raw)
                except Exception:
                    move_rec['state_before'] = self.serialize_state(state_before_raw)
                    move_rec['state_after'] = self.serialize_state(state_after_raw)
            else:
                # 'none' or unknown: do not store per-move states (recommended for training)
                pass

            move_history.append(move_rec)

            move_count += 1

        end_ms = int(time.time() * 1000)
        duration_ms = end_ms - start_ms
        # Attach game schema metadata so dataset consumers can detect which
        # board layout and actions were used when generating this game.
        board_layout = self._get_board_layout()
        num_vertices = sum([s * s for s in board_layout])
        actions_supported = ['place', 'infuse', 'move', 'attack', 'pincer', 'endTurn']

        # Clarify terminology: `moves` is a list of action events. Each action
        # can be 'place','infuse','move','attack','pincer','endTurn'. We also
        # compute turn-level counts so downstream consumers can distinguish
        # actions vs turns.
        total_actions = move_count
        total_turns = sum(1 for m in move_history if (m.get('action') or {}).get('type') in ('endTurn', 'attack', 'pincer'))
        avg_actions_per_turn = (total_actions / total_turns) if total_turns > 0 else total_actions

        return {
            'game_id': f'selfplay_{game_id}_{end_ms}',
            'moves': move_history,  # legacy name: sequence of action events
            'winner': game_state.get('winner'),
            'total_moves': total_actions,  # legacy field (keeps previous behavior)
            'total_actions': total_actions,
            'total_turns': total_turns,
            'avg_actions_per_turn': avg_actions_per_turn,
            'start_time': start_ms,
            'end_time': end_ms,
            'initial_state': self._serialize_initial_state(_initial_state),
            'game_duration_ms': duration_ms,
            'metadata': {
                'seed': seed,
                'starting_player': starting_player,
                'exploration_rate': self.config.exploration_rate,
                'search_depth': self.config.search_depth,
                'temperature': getattr(self.config, 'temperature', None),
                    'game_schema': {
                        'board_layout': board_layout,
                        'num_vertices': num_vertices,
                        'actions_supported': actions_supported,
                    }
            },
            'timestamp': datetime.fromtimestamp(end_ms / 1000.0).isoformat(),
        }
    
    def _serialize_initial_state(self, state: Optional[Dict]) -> Optional[str]:
        """Serialize initial state (stored once per game)."""
        if state is None or self._state_serialization == 'none':
            # For 'none' strategy, we still store initial state (cheap!)
            # Training needs it to reconstruct all positions
            return json.dumps(state, separators=(',', ':'))
        
        if self._state_serialization == 'binary':
            binary = self._compact_binary_state(state)
            return base64.b64encode(binary).decode('ascii')
        
        return json.dumps(state, separators=(',', ':'))

    def get_engine_move(self, state: Dict, depth: int) -> Dict:
        """
        Get best move using simplified evaluation.
        
        In production, this would call the TypeScript engine via API.
        """
        legal_moves = self.get_legal_moves(state)

        if not legal_moves:
            return {'type': 'endTurn'}

        best_move = None
        best_score = -float('inf')
        current_player = state.get('currentPlayerId')

        for move in legal_moves:
            # Shallow copy for evaluation
            new_state = self.apply_move(state.copy(), move)
            # Choose perspective based on config: either evaluate from the
            # new state's currentPlayerId (legacy) or from the mover's
            # perspective to avoid evaluation flips when turns end.
            if getattr(self.config, 'evaluate_from_mover', False):
                score = self.evaluate_position(new_state, perspective=current_player)
            else:
                score = self.evaluate_position(new_state)

            if getattr(self.config, 'instrument', False):
                logger.debug(
                    "Eval candidate move %s -> score=%.2f (perspective=%s)",
                    move, score, current_player
                )

            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move else {'type': 'endTurn'}

    # ------------------------- State helpers -------------------------
    def _compact_binary_state(self, state: Dict[str, Any]) -> bytes:
        """Create a compact binary summary of the game state.

        This intentionally keeps only the small pieces of information needed
        to reconstruct vertex-level occupancy (owner, stack size, energy)
        and a couple of turn flags. It's not a full engine snapshot.
        """
        try:
            vertices = state.get('vertices', {}) if isinstance(state, dict) else {}
            out = bytearray()
            # Header: magic + version
            out.extend(struct.pack('BB', 0x55, 1))
            # vertex count (unsigned short)
            out.extend(struct.pack('H', len(vertices)))
            # Per-vertex: index, stack_count, energy (clamped to 0-255), owner (0/1/2)
            for idx, (vid, v) in enumerate(vertices.items()):
                stack = v.get('stack', []) if isinstance(v, dict) else []
                energy = int(v.get('energy', 0)) if isinstance(v, dict) else 0
                owner = 0
                if stack:
                    try:
                        owner = 1 if stack[0].get('player') == 'Player1' else 2
                    except Exception:
                        owner = 0
                out.extend(struct.pack('BBBB', idx & 0xFF, len(stack) & 0xFF, energy & 0xFF, owner & 0xFF))
            # Current player and turn flags
            cur = 1 if state.get('currentPlayerId') == 'Player1' else 2
            turn = state.get('turn', {}) if isinstance(state, dict) else {}
            flags = (1 if turn.get('hasPlaced') else 0) | (2 if turn.get('hasInfused') else 0) | (4 if turn.get('hasMoved') else 0)
            out.extend(struct.pack('BB', cur & 0xFF, flags & 0xFF))
            return bytes(out)
        except Exception:
            # On any error, raise to allow caller to fallback
            raise

    def _compute_state_delta(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Return a small dict describing what changed between two states.

        Only include vertices whose stack or energy changed, plus currentPlayerId
        and turn changes.
        """
        delta: Dict[str, Any] = {'changed_vertices': {}}
        vb = before.get('vertices', {}) if isinstance(before, dict) else {}
        va = after.get('vertices', {}) if isinstance(after, dict) else {}
        keys = set(vb.keys()) | set(va.keys())
        for k in keys:
            b = vb.get(k, {})
            a = va.get(k, {})
            if b.get('stack') != a.get('stack') or b.get('energy') != a.get('energy'):
                delta['changed_vertices'][k] = {
                    'stack': a.get('stack', []),
                    'energy': a.get('energy', 0)
                }
        if before.get('currentPlayerId') != after.get('currentPlayerId'):
            delta['currentPlayerId'] = after.get('currentPlayerId')
        if before.get('turn') != after.get('turn'):
            delta['turn'] = after.get('turn')
        return delta

    def _get_board_layout(self) -> List[int]:
        """
        Determine board layout: prefer explicit override in config, otherwise
        fall back to the canonical 5-layer layout used for training.
        """
        if getattr(self.config, 'board_layout', None):
            return list(self.config.board_layout)
        # canonical 5-layer layout
        return [3, 5, 7, 5, 3]

    async def get_model_move(self, state: Dict) -> Dict:
        """Use the batched model to pick a move for the given state.

        The model returns a policy vector shaped [num_vertices * 4]. We map
        legal moves to indices in that vector and pick the highest-scoring
        legal move. If mapping fails or the model produces an invalid result,
        fall back to a random legal move.
        """
        legal_moves = self.get_legal_moves(state)

        if not legal_moves:
            return {'type': 'endTurn'}

        # Ask batcher for policy/value
        res = await self._inference_batcher.predict(state, timeout=max(1.0, self.config.inference_batch_timeout * 10))
        # res is (policy_array, value)
        try:
            policy_array, _ = res
        except Exception:
            # Unexpected response shape
            logger.warning("Invalid model response shape; falling back to random move")
            return self.get_random_move(state)

        # policy_array expected shape [num_vertices * 4]
        # Map each legal move to an index into policy_array
        best_move = None
        best_score = -float('inf')
        for mv in legal_moves:
            try:
                action_type = (mv.get('type') or mv.get('action') or {}).get('type') if isinstance(mv.get('action', None), dict) else mv.get('type')
            except Exception:
                action_type = mv.get('type')

            # determine vertex index
            vertex_id = mv.get('vertexId') or mv.get('fromId') or mv.get('toId')
            # map vertex id to index using current state's vertices ordering
            vertices = list(state.get('vertices', {}).keys())
            try:
                vertex_idx = vertices.index(vertex_id) if vertex_id in vertices else None
            except Exception:
                vertex_idx = None

            if vertex_idx is None:
                # if we can't map by id, skip
                continue

            action_offset = {
                'place': 0,
                'infuse': 1,
                'move': 2,
                'attack': 3
            }.get(action_type, 0)

            idx = vertex_idx * 4 + action_offset
            if idx < 0 or idx >= len(policy_array):
                continue

            score = float(policy_array[idx])
            if score > best_score:
                best_score = score
                best_move = mv

        if best_move is None:
            # fallback to random legal move
            return self.get_random_move(state)
        return best_move

    def get_random_move(self, state: Dict) -> Dict:
        """
        Get a random legal move for exploration.
        
        Args:
            state: Current game state
            
        Returns:
            Random legal move or endTurn if no moves available
        """
        legal_moves = self.get_legal_moves(state)
        if not legal_moves:
            return {'type': 'endTurn'}
        return legal_moves[np.random.randint(len(legal_moves))]

    def evaluate_position(self, state: Dict, perspective: Optional[str] = None) -> float:
        """
        Simple heuristic position evaluation.
        
        Args:
            state: Game state to evaluate
            
        Returns:
            Score (positive favors current player)
        """
        score = 0.0
        # Allow caller to specify the perspective (moving player) to avoid
        # evaluation flips when apply_move changes currentPlayerId for turn-ending
        # moves. If no perspective is provided, fall back to the state's currentPlayerId.
        current_player = perspective if perspective is not None else state.get('currentPlayerId')

        vertices = state.get('vertices', {})
        for vertex in vertices.values():
            if vertex.get('stack'):
                owner = vertex['stack'][0]['player']
                piece_count = len(vertex['stack'])
                energy = vertex.get('energy', 0)

                value = piece_count * 10 + energy * 15

                if owner == current_player:
                    score += value
                else:
                    score -= value

        return score

    def initialize_game(self) -> Dict:
        """
            Create initial game state using configured board layout (canonical
            5-layer layout is the default for training).

            For production, replace this with full game initialization
            or call the TypeScript engine API.
            """
        # Determine board layout (allow override via config)
        board_layout = self._get_board_layout()

        # Using the provided board layout (or canonical layout by default).
        # The previous tiny simplified initializer ([3,5,3]) has been removed
        # and is no longer an option for training.
        vertices = {}
        
        vertex_id = 0
        for layer_idx, size in enumerate(board_layout):
            for x in range(size):
                for z in range(size):
                    vid = f"v{vertex_id}"
                    vertices[vid] = {
                        'id': vid,
                        'layer': layer_idx,
                        'x': x,
                        'z': z,
                        'stack': [],
                        'energy': 0,
                        'adjacencies': []  # Set after all vertices created
                    }
                    vertex_id += 1
        
        # Set up adjacencies (simplified 4-connected grid)
        for vid, vertex in vertices.items():
            adj = []
            layer_verts = [v for v in vertices.values() if v['layer'] == vertex['layer']]
            
            # Same layer neighbors
            for other in layer_verts:
                if other['id'] != vid:
                    dx = abs(other['x'] - vertex['x'])
                    dz = abs(other['z'] - vertex['z'])
                    if (dx == 1 and dz == 0) or (dx == 0 and dz == 1):
                        adj.append(other['id'])
            
            vertex['adjacencies'] = adj
        
        # Find corner vertices for home positions
        layer0 = [v for v in vertices.values() if v['layer'] == 0]
        corners_p1 = [v['id'] for v in layer0 if v['x'] == 0 and v['z'] == 0]
        corners_p2 = [v['id'] for v in layer0 if v['x'] == 2 and v['z'] == 2]
        
        # compute number of vertices for metadata
        num_vertices = sum([s * s for s in board_layout])

        return {
            'vertices': vertices,
            'currentPlayerId': 'Player1',
            'winner': None,
            'turn': {
                'hasPlaced': False,
                'hasInfused': False,
                'hasMoved': False,
                'turnNumber': 1
            },
            'players': {
                'Player1': {'reinforcements': 3},
                'Player2': {'reinforcements': 3}
            },
            'homeCorners': {
                'Player1': corners_p1 if corners_p1 else [list(vertices.keys())[0]],
                'Player2': corners_p2 if corners_p2 else [list(vertices.keys())[-1]]
            }
        }

    def get_legal_moves(self, state: Dict) -> List[Dict]:
        """
        Generate all legal moves for current player.
        
        Returns list of move dictionaries with 'type' and relevant fields.
        """
        moves = []
        current_player = state['currentPlayerId']
        turn = state['turn']
        vertices = state['vertices']
        
        # Placement moves
        if not turn['hasPlaced'] and state['players'][current_player]['reinforcements'] > 0:
            for corner_id in state['homeCorners'][current_player]:
                moves.append({'type': 'place', 'vertexId': corner_id})
        
        # Infusion moves
        if not turn['hasInfused']:
            for vid, vertex in vertices.items():
                if vertex['stack'] and vertex['stack'][0]['player'] == current_player:
                    # Check force cap (simplified)
                    if vertex['energy'] < 10:
                        moves.append({'type': 'infuse', 'vertexId': vid})
        
        # Movement moves
        if not turn['hasMoved']:
            for vid, vertex in vertices.items():
                if vertex['stack'] and vertex['stack'][0]['player'] == current_player:
                    for target_id in vertex['adjacencies']:
                        target = vertices[target_id]
                        # Can move to empty spaces
                        if not target['stack']:
                            moves.append({'type': 'move', 'fromId': vid, 'toId': target_id})
        
        # Attack moves
        for vid, vertex in vertices.items():
            if vertex['stack'] and vertex['stack'][0]['player'] == current_player:
                if len(vertex['stack']) >= 1 and vertex['energy'] >= 1:
                    for target_id in vertex['adjacencies']:
                        target = vertices[target_id]
                        if target['stack'] and target['stack'][0]['player'] != current_player:
                            moves.append({'type': 'attack', 'vertexId': vid, 'targetId': target_id})

        # Pincer moves (special multi-target attack) - available when vertex has enough energy
        for vid, vertex in vertices.items():
            if vertex['stack'] and vertex['stack'][0]['player'] == current_player and vertex.get('energy', 0) >= 2:
                adjacent_enemies = [t for t in vertex['adjacencies'] if vertices[t]['stack'] and vertices[t]['stack'][0]['player'] != current_player]
                for target_id in adjacent_enemies:
                    moves.append({'type': 'pincer', 'vertexId': vid, 'targetId': target_id})
        
        # End turn (if mandatory actions done)
        if turn['hasPlaced'] and turn['hasInfused'] and turn['hasMoved']:
            moves.append({'type': 'endTurn'})
        
        # Always allow end turn if no other moves
        if not moves:
            moves.append({'type': 'endTurn'})
        
        return moves

    def apply_move(self, state: Dict, move: Dict) -> Dict:
        """
        Apply move to state and return new state.
        
        This is a simplified implementation. For production, integrate
        with the full TypeScript game engine via API.
        """
        import copy
        new_state = copy.deepcopy(state)
        
        move_type = move.get('type')
        current_player = new_state['currentPlayerId']
        
        if move_type == 'place':
            vertex_id = move['vertexId']
            vertex = new_state['vertices'][vertex_id]
            vertex['stack'].append({'player': current_player, 'id': f'p{len(vertex["stack"])}'})
            new_state['players'][current_player]['reinforcements'] -= 1
            new_state['turn']['hasPlaced'] = True
        
        elif move_type == 'infuse':
            vertex_id = move['vertexId']
            new_state['vertices'][vertex_id]['energy'] += 1
            new_state['turn']['hasInfused'] = True
        
        elif move_type == 'move':
            from_id = move['fromId']
            to_id = move['toId']
            source = new_state['vertices'][from_id]
            target = new_state['vertices'][to_id]
            
            # Transfer stack and energy
            target['stack'] = source['stack']
            target['energy'] = source['energy']
            source['stack'] = []
            source['energy'] = 0
            new_state['turn']['hasMoved'] = True
        
        elif move_type == 'attack':
            attacker_id = move['vertexId']
            defender_id = move['targetId']
            attacker = new_state['vertices'][attacker_id]
            defender = new_state['vertices'][defender_id]
            
            # Simplified combat: compare stack size + energy
            attacker_strength = len(attacker['stack']) * 10 + attacker['energy'] * 15
            defender_strength = len(defender['stack']) * 10 + defender['energy'] * 15
            
            if attacker_strength > defender_strength:
                # Attacker wins: defender gets attacker's pieces
                defender['stack'] = attacker['stack']
                defender['energy'] = max(0, attacker['energy'] - defender['energy'])
            else:
                # Defender wins or draw: defender keeps position
                defender['energy'] = max(0, defender['energy'] - attacker['energy'])
            
            # Attacker position is emptied
            attacker['stack'] = []
            attacker['energy'] = 0
            
            # Attack ends turn
            new_state['currentPlayerId'] = 'Player2' if current_player == 'Player1' else 'Player1'
            new_state['players'][new_state['currentPlayerId']]['reinforcements'] += 1
            new_state['turn'] = {
                'hasPlaced': False,
                'hasInfused': False,
                'hasMoved': False,
                'turnNumber': new_state['turn']['turnNumber'] + 1
            }
        
        elif move_type == 'pincer':
            # Pincer: a focused attack that costs more energy but has a bonus
            attacker_id = move.get('vertexId')
            defender_id = move.get('targetId')
            attacker = new_state['vertices'].get(attacker_id)
            defender = new_state['vertices'].get(defender_id)

            if not attacker or not defender:
                # malformed move, ignore
                return new_state

            # require attacker stack and minimum energy
            if not attacker.get('stack') or attacker.get('energy', 0) < 2:
                # invalid pincer - do nothing
                return new_state

            attacker_strength = len(attacker['stack']) * 10 + attacker['energy'] * 15
            defender_strength = len(defender['stack']) * 10 + defender['energy'] * 15

            # pincer gets a small bonus for coordinated action
            attacker_strength += 10

            if attacker_strength > defender_strength:
                defender['stack'] = attacker['stack']
                defender['energy'] = max(0, attacker['energy'] - defender['energy'] - 1)
            else:
                defender['energy'] = max(0, defender['energy'] - attacker['energy'])

            # Attacker position is emptied and energy consumed
            attacker['stack'] = []
            attacker['energy'] = 0

            # Pincer ends the turn
            new_state['currentPlayerId'] = 'Player2' if current_player == 'Player1' else 'Player1'
            new_state['players'][new_state['currentPlayerId']]['reinforcements'] += 1
            new_state['turn'] = {
                'hasPlaced': False,
                'hasInfused': False,
                'hasMoved': False,
                'turnNumber': new_state['turn']['turnNumber'] + 1
            }
        
        elif move_type == 'endTurn':
            # Switch players
            new_state['currentPlayerId'] = 'Player2' if current_player == 'Player1' else 'Player1'
            new_state['players'][new_state['currentPlayerId']]['reinforcements'] += 1
            new_state['turn'] = {
                'hasPlaced': False,
                'hasInfused': False,
                'hasMoved': False,
                'turnNumber': new_state['turn']['turnNumber'] + 1
            }
        
        # Check win condition: control opponent's home corners
        for player in ['Player1', 'Player2']:
            opponent = 'Player2' if player == 'Player1' else 'Player1'
            opponent_corners = new_state['homeCorners'][opponent]
            
            if opponent_corners and all(
                new_state['vertices'][cid]['stack'] and 
                new_state['vertices'][cid]['stack'][0]['player'] == player
                for cid in opponent_corners
            ):
                new_state['winner'] = player
                break
        
        return new_state

    def serialize_state(self, state: Dict) -> str:
        """Convert state to compact string representation.

        By default this returns a compact JSON string (no whitespace).
        If `self.config.trim_states` is True, a trimmed representation is
        returned to reduce size (useful for large shard exports). The
        function intentionally preserves the return type (str) so existing
        callers/tests which expect JSON strings continue to work.
        """
        # Default fast path: no trimming
        if not getattr(self, 'config', None) or not getattr(self.config, 'trim_states', False):
            return json.dumps(state, separators=(',', ':'))

        # Trim large blobs: keep minimal vertex info (top owner, count, energy)
        trimmed: Dict[str, Any] = {}
        for k, v in state.items():
            if k == 'vertices' and isinstance(v, dict):
                tv: Dict[str, Any] = {}
                for vid, vert in v.items():
                    # vert may contain 'stack', 'energy', 'layer', etc. Keep a compact summary.
                    stack = vert.get('stack', []) if isinstance(vert, dict) else []
                    top_owner = stack[0].get('player') if stack and isinstance(stack[0], dict) else None
                    tv[vid] = {
                        'owner': top_owner,
                        'count': len(stack),
                        'energy': vert.get('energy') if isinstance(vert, dict) else None,
                    }
                trimmed['vertices'] = tv
            elif k in ('state_before', 'state_after'):
                # Replace full state blobs with a short summary to save space
                trimmed[k] = {'summary': 'trimmed'}
            else:
                # Keep other metadata unchanged (small fields)
                trimmed[k] = v

        return json.dumps(trimmed, separators=(',', ':'))

    def serialize_for_shard(self, obj: Dict[str, Any]) -> bytes:
        """Serialize an object for shard writing.

        Returns bytes. If `shard_compress` is enabled and msgpack/zstd are
        available, use msgpack + zstd for compact binary storage. Otherwise
        return UTF-8 encoded compact JSON.
        """
        # Prefer msgpack+zstd for compactness when configured and available
        try:
            if getattr(self, 'config', None) and getattr(self.config, 'shard_compress', False) and MSGPACK_AVAILABLE:
                packed = msgpack.packb(obj, use_bin_type=True)
                cctx = zstd.ZstdCompressor(level=3)
                return cctx.compress(packed)
        except Exception:
            # Fall back to JSON if compression fails
            logger.exception("Shard compression failed, falling back to JSON")

        # Default: compact JSON bytes
        return self.serialize_state(obj).encode('utf-8')

    async def shutdown(self):
        """Gracefully shutdown the generator."""
        logger.info("Shutdown requested...")
        self.shutdown_requested = True
        # stop inference batcher first so no more model inferences are attempted
        if getattr(self, '_inference_batcher', None):
            try:
                await self._inference_batcher.stop()
            except Exception:
                logger.exception("Error stopping inference batcher")
        await self.db_writer.shutdown()
        self.metrics.log_summary()
        logger.info("Shutdown complete")


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
        default=10,
        help='Number of concurrent games (default: 10)'
    )
    parser.add_argument(
        '--games-per-batch',
        type=int,
        default=100,
        help='Games per batch (default: 100)'
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
        default=1000,
        help='Number of games per shard file (default: 1000)'
    )
    parser.add_argument(
        '--trim-states',
        action='store_true',
        help='Trim state_before/state_after blobs when writing shards to save space'
    )
    parser.add_argument(
        '--shard-move-mode',
        type=str,
        default='full',
        choices=['full', 'compressed', 'compact'],
        help='How to store moves in shards (default: full)'
    )
    parser.add_argument(
        '--shard-dir',
        type=str,
        default='shards',
        help='Directory to write shard files to (default: shards)'
    )

    args = parser.parse_args()

    # Enforce tuned GPU+file-writer configuration discovered by experiments.
    # These settings are intentionally locked so runs are reproducible and use
    # the measured sweet-spot (inference_batch_size=32, concurrent_games=64,
    # trim_states=False, use_model=True, model_device='cuda').
    # NOTE: This intentionally overrides any CLI-provided values.
    args.inference_batch_size = 32
    args.concurrent_games = 64
    args.trim_states = False
    args.use_model = True
    args.model_device = 'cuda'
    # Default to writing to the database for production runs (disable file writer)
    # This ensures new runs target the fresh DB by default even if --file-writer
    # was passed previously in older scripts or CI invocations.
    args.file_writer = False
    logger.info(
        'Enforcing tuned runtime config: '
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