import asyncio
import json
import logging
import os
import time
import uuid
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional parquet support
try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    pd = None
    _HAVE_PANDAS = False

# Prefer pyarrow for parquet writes if available (lighter-weight than going through pandas)
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_PYARROW = True
except Exception:
    pa = None
    pq = None
    _HAVE_PYARROW = False


class FileWriter:
    """Simple file-backed JSONL writer with shard rotation.

    Interface mirrors DatabaseWriter enough for `enqueue_game()` and
    `shutdown()` to be used by SelfPlayGenerator.
    """

    def __init__(self, config, metrics, prom_metrics: Optional[dict] = None):
        self.config = config
        self.metrics = metrics
        maxsize = getattr(self.config, 'write_queue_maxsize', 0) or 0
        self.write_queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.worker_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self._accepting = True
        # shard rotation parameters
        self.shard_dir = getattr(self.config, 'shard_dir', 'shards')
        self.shard_games = int(getattr(self.config, 'shard_games', 1000))
        self.shard_format = getattr(self.config, 'shard_format', 'jsonl') or 'jsonl'
        # internal state
        self._current_shard_path = None
        self._current_shard_count = 0
        self._shard_seq = 0
        # buffer for parquet mode
        self._parquet_buffer = []

        # Prometheus hooks (optional)
        if prom_metrics:
            self._p_queue_length = prom_metrics.get('queue_length')
            self._p_games_saved = prom_metrics.get('games_saved')
            # shard metrics
            self._p_shards_written = prom_metrics.get('shards_written')
            self._p_shard_bytes = prom_metrics.get('shard_bytes')
        else:
            self._p_queue_length = None
            self._p_games_saved = None
            self._p_shards_written = None
            self._p_shard_bytes = None

    async def initialize(self):
        os.makedirs(self.shard_dir, exist_ok=True)
        # create first shard file lazily in worker

    async def start_writer(self):
        if self.worker_task and not self.worker_task.done():
            return
        self.worker_task = asyncio.create_task(self._writer())
        logger.info("File writer started (dir=%s, shard_games=%d)", self.shard_dir, self.shard_games)

    async def _open_new_shard(self):
        t = int(time.time())
        ext = 'jsonl'
        if self.shard_format == 'parquet':
            ext = 'parquet'
        path = os.path.join(self.shard_dir, f"shard_{t}_{self._shard_seq}.{ext}")
        self._shard_seq += 1

        if self.shard_format == 'jsonl':
            f = open(path, 'a', encoding='utf-8')
            self._current_shard_file = f
            logger.info("Opened new shard %s", path)
        else:
            # parquet: buffer in memory until flush
            self._current_shard_file = None
            self._parquet_buffer = []
            logger.info("Started new parquet shard buffer %s", path)

        self._current_shard_path = path
        self._current_shard_count = 0

    async def _close_shard(self):
        try:
            if self.shard_format == 'jsonl' and getattr(self, '_current_shard_file', None):
                self._current_shard_file.flush()
                self._current_shard_file.close()
                # measure size and update shard metrics
                try:
                    size = os.path.getsize(self._current_shard_path)
                except Exception:
                    size = None
                logger.info("Closed shard %s (games=%d) size=%s", self._current_shard_path, self._current_shard_count, size)
                try:
                    if self._p_shards_written:
                        self._p_shards_written.inc()
                    if self._p_shard_bytes and size is not None:
                        self._p_shard_bytes.inc(size)
                except Exception:
                    pass
            elif self.shard_format == 'parquet' and self._parquet_buffer:
                await self._flush_parquet_shard()
        except Exception:
            pass
        self._current_shard_path = None
        self._current_shard_file = None
        self._current_shard_count = 0

    async def _writer(self):
        logger.info("File writer running")
        while not self.shutdown_event.is_set() or not self.write_queue.empty():
            try:
                game = await asyncio.wait_for(self.write_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                # open shard lazily
                if not getattr(self, '_current_shard_path', None):
                    await self._open_new_shard()

                # optionally trim large state payloads (only drop state blobs; keep moves unless move mode says otherwise)
                if getattr(self.config, 'trim_states', True):
                    if isinstance(game, dict):
                        game = dict(game)
                        game.pop('state_before', None)
                        game.pop('state_after', None)

                # handle move storage modes (only relevant for parquet mode)
                move_mode = getattr(self.config, 'shard_move_mode', 'compressed')

                if self.shard_format == 'jsonl':
                    # write json line
                    line = json.dumps(game, ensure_ascii=False)
                    self._current_shard_file.write(line + '\n')
                    self._current_shard_count += 1
                    # try to account for bytes written for metrics
                    try:
                        b = len((line + '\n').encode('utf-8'))
                        if self._p_games_saved:
                            # games_saved already incremented below; track bytes separately via prom_metrics if provided
                            pass
                    except Exception:
                        pass
                else:
                    # parquet buffering
                    # normalize fields to simple types; handle moves according to mode
                    rec = dict(game)
                    rec['metadata'] = json.dumps(rec.get('metadata', {}))
                    rec['initial_state'] = game.get('initial_state')
                    moves_val = rec.get('moves', [])
                    if move_mode == 'full':
                        # store moves as JSON string
                        rec['moves'] = json.dumps(moves_val)
                    elif move_mode == 'compressed':
                        # store gzip-compressed JSON bytes for moves
                        try:
                            import gzip
                            moves_json = json.dumps(moves_val)
                            comp = gzip.compress(moves_json.encode('utf-8'))
                            # store as bytes; pandas/pyarrow will write binary column
                            rec['moves'] = comp
                        except Exception:
                            # fallback to JSON string
                            rec['moves'] = json.dumps(moves_val)
                    else:
                        # fallback/compact: store moves count only
                        try:
                            if isinstance(moves_val, (list, tuple)):
                                rec['moves_count'] = len(moves_val)
                            else:
                                rec['moves_count'] = None
                        except Exception:
                            rec['moves_count'] = None
                        rec.pop('moves', None)

                    self._parquet_buffer.append(rec)
                    self._current_shard_count += 1

                # update metrics
                self.metrics.games_saved += 1
                try:
                    if self._p_games_saved:
                        self._p_games_saved.inc()
                except Exception:
                    pass

                # record shard-level metrics when writing parquet buffer (we'll increment bytes/shard at flush time)

                # rotate if needed
                if self._current_shard_count >= self.shard_games:
                    if self.shard_format == 'jsonl':
                        await self._close_shard()
                    else:
                        # flush parquet buffer to file
                        await self._flush_parquet_shard()

            except Exception as e:
                logger.exception("File writer failed to write game: %s", e)
            finally:
                try:
                    self.write_queue.task_done()
                except Exception:
                    pass

            try:
                if self._p_queue_length:
                    self._p_queue_length.set(self.write_queue.qsize())
            except Exception:
                pass

        # after loop, close any open shard
        if getattr(self, '_current_shard_path', None):
            try:
                await self._close_shard()
            except Exception:
                pass

        logger.info("File writer exiting")

    async def enqueue_game(self, game: Dict[str, Any]):
        if not self._accepting:
            logger.debug("FileWriter not accepting new games; dropping %s", game.get('game_id'))
            return
        try:
            await asyncio.wait_for(self.write_queue.put(game), timeout=getattr(self.config, 'enqueue_timeout', 2.0))
            try:
                if self._p_queue_length:
                    self._p_queue_length.set(self.write_queue.qsize())
            except Exception:
                pass
        except asyncio.TimeoutError:
            logger.warning("File write queue full, dropping game %s", game.get('game_id'))

    async def _flush_parquet_shard(self):
        """Flush the in-memory parquet buffer to a parquet file on disk.

        Requires pandas + pyarrow. If unavailable, falls back to JSONL file save.
        """
        if not self._parquet_buffer:
            return

        # generate unique target path per flush and write atomically
        filename = f"shard_{int(time.time())}_{uuid.uuid4().hex}.parquet"
        path = os.path.join(self.shard_dir, filename)
        temp_path = path + ".tmp"
        try:
            # Prefer pyarrow Table -> parquet write (no pandas roundtrip)
            if _HAVE_PYARROW:
                # Build columnar dict: collect all keys
                all_keys = set()
                for r in self._parquet_buffer:
                    all_keys.update(r.keys())

                col_dict = {}
                for k in sorted(all_keys):
                    vals = []
                    for r in self._parquet_buffer:
                        v = r.get(k, None)
                        # ensure metadata is a string
                        if k == 'metadata' and v is not None and not isinstance(v, (str, bytes)):
                            try:
                                v = json.dumps(v)
                            except Exception:
                                v = str(v)
                        # moves might be bytes (compressed) or string; leave bytes alone
                        if k == 'moves' and isinstance(v, memoryview):
                            # convert memoryview to bytes
                            v = bytes(v)
                        vals.append(v)
                    col_dict[k] = vals

                # create a pyarrow Table and write parquet with snappy
                try:
                    table = pa.Table.from_pydict(col_dict)
                    pq.write_table(table, temp_path, compression='snappy')
                    try:
                        size = os.path.getsize(temp_path)
                    except Exception:
                        size = None
                    os.replace(temp_path, path)
                    logger.info("Wrote parquet shard %s (rows=%d) size=%s", path, len(self._parquet_buffer), size)
                    try:
                        if self._p_shards_written:
                            self._p_shards_written.inc()
                        if self._p_shard_bytes and size is not None:
                            self._p_shard_bytes.inc(size)
                    except Exception:
                        pass
                except Exception:
                    # fallback to pandas if pyarrow write fails
                    if _HAVE_PANDAS:
                        df = pd.DataFrame(self._parquet_buffer)
                        if 'metadata' in df.columns:
                            df['metadata'] = df['metadata'].astype(str)
                        if 'moves' in df.columns:
                            # if moves are bytes, convert to Python bytes -> pandas handles as object
                            df['moves'] = df['moves'].apply(lambda x: x if isinstance(x, (bytes, type(None))) else str(x))
                        df.to_parquet(temp_path, engine='pyarrow', compression='snappy')
                        try:
                            size = os.path.getsize(temp_path)
                        except Exception:
                            size = None
                        os.replace(temp_path, path)
                        logger.info("Wrote parquet shard %s via pandas fallback (rows=%d) size=%s", path, len(self._parquet_buffer), size)
                        try:
                            if self._p_shards_written:
                                self._p_shards_written.inc()
                            if self._p_shard_bytes and size is not None:
                                self._p_shard_bytes.inc(size)
                        except Exception:
                            pass
                    else:
                        # fallback: write JSONL to unique file
                        json_path = path.replace('.parquet', '.jsonl')
                        temp_json = json_path + ".tmp"
                        with open(temp_json, 'a', encoding='utf-8') as f:
                            for rec in self._parquet_buffer:
                                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                        try:
                            size = os.path.getsize(temp_json)
                        except Exception:
                            size = None
                        os.replace(temp_json, json_path)
                        logger.warning("PyArrow and pandas not available, wrote fallback JSONL shard for %s size=%s", json_path, size)
                        try:
                            if self._p_shards_written:
                                self._p_shards_written.inc()
                            if self._p_shard_bytes and size is not None:
                                self._p_shard_bytes.inc(size)
                        except Exception:
                            pass
            else:
                # pyarrow not available; try pandas path then JSONL fallback
                if _HAVE_PANDAS:
                    df = pd.DataFrame(self._parquet_buffer)
                    if 'metadata' in df.columns:
                        df['metadata'] = df['metadata'].astype(str)
                    if 'moves' in df.columns:
                        df['moves'] = df['moves'].apply(lambda x: x if isinstance(x, (bytes, type(None))) else str(x))
                    df.to_parquet(temp_path, engine='pyarrow', compression='snappy')
                    try:
                        size = os.path.getsize(temp_path)
                    except Exception:
                        size = None
                    os.replace(temp_path, path)
                    logger.info("Wrote parquet shard %s (rows=%d) size=%s", path, len(self._parquet_buffer), size)
                    try:
                        if self._p_shards_written:
                            self._p_shards_written.inc()
                        if self._p_shard_bytes and size is not None:
                            self._p_shard_bytes.inc(size)
                    except Exception:
                        pass
                else:
                    json_path = path.replace('.parquet', '.jsonl')
                    temp_json = json_path + ".tmp"
                    with open(temp_json, 'a', encoding='utf-8') as f:
                        for rec in self._parquet_buffer:
                            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    try:
                        size = os.path.getsize(temp_json)
                    except Exception:
                        size = None
                    os.replace(temp_json, json_path)
                    logger.warning("Pandas not available, wrote fallback JSONL shard for %s size=%s", json_path, size)
                    try:
                        if self._p_shards_written:
                            self._p_shards_written.inc()
                        if self._p_shard_bytes and size is not None:
                            self._p_shard_bytes.inc(size)
                    except Exception:
                        pass
        except Exception:
            logger.exception("Failed to flush parquet shard %s", path)
        finally:
            # clear buffer and reset current shard path/count
            self._parquet_buffer = []
            self._current_shard_count = 0
            # mark that there's no active shard path so next _open_new_shard creates one
            self._current_shard_path = None

    async def shutdown(self):
        logger.info("Shutting down FileWriter")
        self._accepting = False
        self.shutdown_event.set()
        if self.worker_task:
            try:
                await asyncio.wait_for(self.write_queue.join(), timeout=getattr(self.config, 'shutdown_grace_period', 5.0))
            except asyncio.TimeoutError:
                logger.warning("File writer did not drain in time; continuing shutdown")
            try:
                await asyncio.wait_for(self.worker_task, timeout=10.0)
            except asyncio.TimeoutError:
                try:
                    self.worker_task.cancel()
                except Exception:
                    pass
        logger.info("FileWriter shutdown complete")
