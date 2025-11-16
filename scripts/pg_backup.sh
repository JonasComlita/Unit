#!/usr/bin/env bash
set -euo pipefail

# Simple pg_dump wrapper that writes a timestamped compressed dump.
# Usage: scripts/pg_backup.sh <DATABASE_URL> [output_dir]

DB_URL="${1:-${DATABASE_URL:-}}"
OUT_DIR="${2:-./backups}"

if [ -z "$DB_URL" ]; then
  echo "Usage: $0 <DATABASE_URL> [output_dir]" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$OUT_DIR/unitgame_backup_${TS}.dump"

echo "Creating compressed pg_dump to $OUT_FILE"
PGPASSWORD="${PGPASSWORD:-}" pg_dump --format=custom --file="$OUT_FILE" "$DB_URL"

echo "Backup written: $OUT_FILE"

# Optional: prune old backups (keep last 7)
if command -v ls >/dev/null 2>&1; then
  (cd "$OUT_DIR" && ls -1t unitgame_backup_*.dump 2>/dev/null | sed -e '1,7d' | xargs -r rm -f)
fi
#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper for pg_dump with timestamped filename and optional retention
# Usage: scripts/pg_backup.sh "postgresql://user:pass@host:port/dbname" [retention_days]

DB_URL="${1:-${DATABASE_URL:-}}"
RETENTION_DAYS="${2:-14}"

if [ -z "$DB_URL" ]; then
  echo "Usage: $0 <DATABASE_URL> [retention_days]" >&2
  exit 2
fi

OUT_DIR="backups"
mkdir -p "$OUT_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$OUT_DIR/unitgame_backup_${STAMP}.dump"

echo "Creating pg_dump backup: $OUT_FILE"
PGPASSWORD="${PGPASSWORD:-}" pg_dump -Fc -f "$OUT_FILE" "$DB_URL"

echo "Pruning backups older than ${RETENTION_DAYS} days"
find "$OUT_DIR" -type f -name 'unitgame_backup_*.dump' -mtime +${RETENTION_DAYS} -print -delete || true

echo "Backup complete: $OUT_FILE"
