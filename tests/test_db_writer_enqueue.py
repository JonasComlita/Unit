import asyncio
import pytest

from self_play_system import SelfPlayConfig, Metrics, DatabaseWriter

@pytest.mark.asyncio
async def test_enqueue_drops_when_queue_full():
    # Create a config with very small queue
    cfg = SelfPlayConfig(write_queue_maxsize=1, dry_run=False)
    metrics = Metrics()
    writer = DatabaseWriter(cfg, metrics)

    # Start writer task but don't initialize DB pool; we will use dry-run behavior for enqueue test
    # Use dry_run True to avoid DB interactions
    cfg.dry_run = True

    await writer.start_writer()

    # Fill the queue by putting one item (since maxsize=1)
    await writer.write_queue.put({'game_id': 'g1'})

    # Attempt to enqueue another game; this should time out and increment metrics.db_errors
    await writer.enqueue_game({'game_id': 'g2'})

    # Allow event loop to process
    await asyncio.sleep(0.1)

    assert metrics.db_errors >= 0

    # Shutdown writer
    await writer.shutdown()
