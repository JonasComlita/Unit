Development & operations
======================

Provide a `DATABASE_URL` environment variable or a `.env` file (see `.env.example`). Never commit real credentials.

To run the test suite locally use Python 3.11 (see `.python-version`) and install dependencies:

```bash
python -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt
pytest -q
```

To run a local Postgres for development, use the provided `docker-compose.yml`:

```bash
docker compose up -d db
# wait for db to be ready, then apply `backend_schema.sql`.
```

Backups: use `scripts/pg_backup.sh <DATABASE_URL>` to create timestamped dumps in `backups/`.
