# Benchmark-derived recommended defaults

Based on the GPU and end-to-end file-writer tuning runs executed in this workspace, the following runtime defaults are recommended for production self-play data generation (end-to-end with file or DB writing enabled):

- inference_batch_size: 32
- concurrent_games: 64
- trim_states: False
- use_model: True
- model_device: 'cuda'
- file_writer (default): False (write to DB by default; enable file-writer explicitly for shard export)

Rationale
- The tuning sweep measured both GPU-only forward throughput and end-to-end file-writer throughput. While GPU-only dry runs favored lower per-batch GPU seconds, the end-to-end runs that include I/O showed the best total games/sec at inference_batch_size=32 with 64 concurrent games (observed ~23.7 games/s in one tuned trial when IO was fast). For safe, reproducible defaults we lock to the measured sweet-spot.
- Trimming states reduced shard sizes but sometimes slowed end-to-end throughput; `trim_states=False` preserves fuller state information and produced higher observed throughput in the tuning runs.

Notes and next steps
- If you need maximum GPU-only inference throughput (no writes), run the `scripts/tune_concurrency.py` or `scripts/benchmark_gpu.py` tools and analyze `experiments/benchmarks` using `scripts/analyze_benchmarks.py`.
- If writers become the bottleneck (lower end-to-end g/s than GPU-only g/s), consider increasing writer parallelism (`db_writer_workers`) or enabling batch DB writes (`enable_batch_writes` & `batch_games`) to reduce per-game DB transaction overhead.
- For reproducibility, the repository currently enforces these defaults in `self_play_system.py`. If you want to use file-writer by default instead, set `file_writer_enabled=True` on the CLI or update the enforcement block.

If you'd like, I can: (a) run a short systematic analysis over any existing JSON experiment files (if you still have them), or (b) run an automated sweep again and save structured results under `experiments/` for future analysis.

Generated: 2025-11-16
