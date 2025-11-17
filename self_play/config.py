"""
Configuration dataclasses and helpers for self-play system.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List

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
    board_layout: Optional[List[int]] = None
    metrics_port: Optional[int] = None
    db_writer_workers: int = 16
    write_queue_maxsize: int = 1000
    enqueue_timeout: float = 0.5
    shutdown_grace_period: float = 5.0
    db_retry_backoff_cap: float = 30.0
    enable_batch_writes: bool = True
    batch_games: int = 25
    batch_timeout: float = 0.5
    file_writer_enabled: bool = True
    shard_dir: str = 'shards'
    shard_games: int = 1000
    shard_compress: bool = False
    shard_format: str = 'parquet'
    trim_states: bool = True
    shard_move_mode: str = 'compressed'
    use_model: bool = False
    model_path: Optional[str] = None
    model_device: Optional[str] = 'cuda'
    inference_batch_size: int = 32
    inference_batch_timeout: float = 0.02
    state_serialization: str = 'none'
    evaluate_from_mover: bool = False
    instrument: bool = False
    instrument_sample_count: int = 200
    use_mcts: bool = True
    mcts_simulations: int = 8
    mcts_c_puct: float = 1.0
