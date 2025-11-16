# Runbook — Self-Play Pipeline

This runbook documents common operational tasks for the self-play training
data pipeline: running pilots, exporting shards, backups, and storage
considerations.

1) Quick start — run a small pilot

```bash
# from repo root
PYTHONPATH=. python3 self_play_system.py --concurrent-games 8 --games-per-batch 64 --metrics-port 8000
```

2) Export shards to S3 (recommended flow)

- Generate shards with file writer enabled: `--file-writer-enabled` (or config)
- Upload shards to S3 in a separate process (don't block self-play). Example:

  aws s3 cp shards/ s3://my-bucket/unit-data/ --recursive --exclude "*.tmp"

3) Backup PostgreSQL

- Daily logical backup with pg_dump (example cron):

  0 2 * * * /usr/bin/pg_dump -Fc --dbname=$DATABASE_URL -f /backups/unit_$(date +\%F).dump

- Keep WAL shipping or point-in-time recovery as needed. Consider lifecycle rules in S3 (move to Glacier after 30 days).

4) Storage & retention

- Recommended shard format: Parquet with per-move compression (snappy + per-move gzip) for balance of size and fidelity.
- Suggested defaults: `shard_games=1000`, `shard_format=parquet`, `shard_move_mode=compressed`.
- Estimate: ~0.1 MB per game with parquet+compression (varies by move count); test with pilots to refine.

5) Troubleshooting

- If write queue grows, increase `db_writer_workers` or enable file-backed writer to offload DB.
- To recover partially written shards, look for `.tmp` files in shards directory and inspect them.

6) Next steps

- Automate S3 uploader as a background worker with retries and exponential backoff.
- Add lifecycle and cost tracking for storage.
