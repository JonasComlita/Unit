#!/usr/bin/env bash
# Archive tagged games (game_metadata.metadata->>'archived' = 'true') into archive tables and delete from main tables.
# Usage: ./scripts/archive_tagged_games.sh "postgresql://unit_user:PASS@127.0.0.1:5432/unitgame"
set -euo pipefail
DB_URL="${1:-$DATABASE_URL}"
if [ -z "$DB_URL" ]; then
  echo "Provide DB URL as first arg or set DATABASE_URL" >&2
  exit 2
fi

# Create a backup first
BACKUP_FILE="unitgame_pre_archive_$(date +%Y%m%d_%H%M%S).dump"
echo "Creating pg_dump backup: $BACKUP_FILE"
PGPASSWORD="${PGPASSWORD:-}" pg_dump -Fc -f "$BACKUP_FILE" "$DB_URL"

echo "Running archive+delete transaction"
psql "$DB_URL" <<'SQL'
BEGIN;

CREATE TABLE IF NOT EXISTS games_archive (LIKE games INCLUDING ALL);
CREATE TABLE IF NOT EXISTS moves_archive (LIKE moves INCLUDING ALL);
CREATE TABLE IF NOT EXISTS game_metadata_archive (LIKE game_metadata INCLUDING ALL);

-- Copy games
INSERT INTO games_archive
SELECT g.* FROM games g
LEFT JOIN game_metadata gm ON g.game_id = gm.game_id
WHERE gm.metadata->>'archived' = 'true';

-- Copy associated moves
INSERT INTO moves_archive
SELECT m.* FROM moves m
WHERE m.game_id IN (
  SELECT g.game_id FROM games g
  LEFT JOIN game_metadata gm ON g.game_id = gm.game_id
  WHERE gm.metadata->>'archived' = 'true'
);
-- Copy associated metadata
INSERT INTO game_metadata_archive
SELECT gm.* FROM game_metadata gm
WHERE gm.metadata->>'archived' = 'true';

-- Delete moves then metadata then games from primary tables
DELETE FROM moves
WHERE game_id IN (
  SELECT g.game_id FROM games g
  LEFT JOIN game_metadata gm ON g.game_id = gm.game_id
  WHERE gm.metadata->>'archived' = 'true'
);

-- Remove metadata rows after archiving them so FK constraints don't block deletes
DELETE FROM game_metadata
WHERE metadata->>'archived' = 'true';

DELETE FROM games
WHERE game_id IN (
  SELECT g.game_id FROM games g
  LEFT JOIN game_metadata gm ON g.game_id = gm.game_id
  WHERE gm.metadata->>'archived' = 'true'
);

COMMIT;
SQL

echo "Archive and delete completed. Backup file: $BACKUP_FILE"
