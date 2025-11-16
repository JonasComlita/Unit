import asyncio
import os
import shutil
import tempfile

import pytest

from writers.file_writer import FileWriter


@pytest.mark.asyncio
async def test_file_writer_shutdown(tmp_path):
    cfg = type('Cfg', (), {})()
    cfg.shard_dir = str(tmp_path)
    cfg.shard_games = 2
    cfg.shard_format = 'jsonl'
    cfg.write_queue_maxsize = 10
    cfg.enqueue_timeout = 0.5
    cfg.shutdown_grace_period = 2.0

    # minimal metrics placeholder
    metrics = type('M', (), {'games_saved': 0})()

    writer = FileWriter(cfg, metrics)
    await writer.initialize()
    await writer.start_writer()

    # enqueue a few games
    await writer.enqueue_game({'game_id': 'g1', 'moves': []})
    await writer.enqueue_game({'game_id': 'g2', 'moves': []})

    # shutdown should drain queue and close shards
    await writer.shutdown()

    # ensure shard dir has files
    files = os.listdir(str(tmp_path))
    assert any(f.endswith('.jsonl') for f in files), 'Expected jsonl shard files to be written'
