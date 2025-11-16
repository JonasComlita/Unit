import asyncio
import pytest

from self_play_system import SelfPlayConfig, Metrics, DatabaseWriter


class FakeConn:
    def __init__(self):
        self.executed = []
        self.executemany_called = []

    async def execute(self, sql, *params):
        # record game_id if provided as first param
        gid = params[0] if params else None
        self.executed.append((sql, gid))

    async def executemany(self, sql, records):
        for rec in records:
            gid = rec[0] if rec else None
            self.executemany_called.append((sql, gid))
    
    def transaction(self):
        # Return an async context manager that yields self
        class _Tx:
            def __init__(self, conn):
                self._conn = conn

            async def __aenter__(self):
                return self._conn

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return _Tx(self)


class FakeAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return FakeAcquire(self.conn)

    async def close(self):
        return


@pytest.mark.asyncio
async def test_shutdown_drains_queue_and_writes_all_games():
    cfg = SelfPlayConfig(write_queue_maxsize=100, enqueue_timeout=0.1, shutdown_grace_period=2.0, db_retry_attempts=1)
    cfg.dry_run = False
    metrics = Metrics()
    writer = DatabaseWriter(cfg, metrics)

    fake_conn = FakeConn()
    fake_pool = FakePool(fake_conn)
    writer.pool = fake_pool

    await writer.start_writer()

    n = 10
    for i in range(n):
        await writer.enqueue_game({'game_id': f'g{i}', 'moves': [], 'metadata': None})

    # Give writer some time to pick up a few items
    await asyncio.sleep(0.2)

    # Trigger shutdown which should wait and/or force-drain
    await writer.shutdown()

    saved_ids = {gid for _, gid in fake_conn.executed if gid is not None} | {gid for _, gid in fake_conn.executemany_called if gid is not None}
    assert len(saved_ids) >= n
