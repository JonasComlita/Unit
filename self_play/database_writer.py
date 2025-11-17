"""
Asynchronous database writer for self-play system.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional
from asyncpg.pool import Pool
from .metrics import Metrics
from .config import SelfPlayConfig

import asyncpg
import time
import random
import json

logger = logging.getLogger(__name__)

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