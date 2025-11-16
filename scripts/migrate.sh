#!/usr/bin/env bash
set -euo pipefail

# Simple migration script: applies `backend_schema.sql` then ensures `game_metadata` exists.
# Usage: scripts/migrate.sh <DATABASE_URL>

DB_URL="${1:-${DATABASE_URL:-}}"
if [ -z "$DB_URL" ]; then
  echo "Usage: $0 <DATABASE_URL>" >&2
  exit 2
fi

echo "Applying backend_schema.sql to $DB_URL"
# Prefer alembic if available
if command -v alembic >/dev/null 2>&1; then
  echo "Alembic detected - attempting 'alembic upgrade head'"
  DATABASE_URL="$DB_URL" alembic upgrade head || {
    echo "Alembic upgrade failed, falling back to raw SQL"
    PGPASSWORD="${PGPASSWORD:-}" psql "$DB_URL" -v ON_ERROR_STOP=1 -f backend_schema.sql
  }
else
  PGPASSWORD="${PGPASSWORD:-}" psql "$DB_URL" -v ON_ERROR_STOP=1 -f backend_schema.sql
fi

echo "Ensuring game_metadata table exists"
PGPASSWORD="${PGPASSWORD:-}" psql "$DB_URL" -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE IF NOT EXISTS game_metadata (
  game_id VARCHAR(50) PRIMARY KEY REFERENCES games(game_id),
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);
SQL

echo "Migrations applied"
