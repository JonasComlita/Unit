"""Create partitioned games parent table and sample monthly partitions

Revision ID: 0001_partition_games
Revises: 
Create Date: 2025-11-15 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
import time

# revision identifiers, used by Alembic.
revision = '0001_partition_games'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create partitioned parent table
    conn = op.get_bind()
    # Check whether `games` exists and whether it is partitioned.
    is_partitioned = False
    table_exists = False
    try:
        row = conn.execute(sa.text("SELECT relkind FROM pg_class WHERE relname='games';")).fetchone()
        if row:
            table_exists = True
            # relkind='p' indicates a partitioned table in Postgres
            is_partitioned = (row[0] == 'p')
    except Exception:
        # If querying pg_class fails for any reason, fall back to attempting
        # to create the partitioned parent (wrapped below). We keep going
        # cautiously to avoid leaving the transaction aborted.
        table_exists = False
        is_partitioned = False

    if not table_exists:
        # Safe to create partitioned parent when no conflicting table exists.
        op.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id VARCHAR(50) PRIMARY KEY,
            start_time BIGINT NOT NULL,
            end_time BIGINT NOT NULL,
            winner VARCHAR(32),
            total_moves INTEGER,
            platform VARCHAR(32)
        ) PARTITION BY RANGE (start_time);
        """)
    elif not is_partitioned:
        # Table exists but is not partitioned; skip creating partition parent
        # to avoid destructive changes. Partitions will not be created.
        pass

    # create a few monthly partitions starting from now
    now = int(time.time() * 1000)
    month_ms = 30 * 24 * 3600 * 1000
    for i in range(0, 12):
        start = now + i * month_ms
        end = now + (i + 1) * month_ms
        tbl = f"games_p_{i}"
        if is_partitioned or not table_exists:
            # Only attempt to create partitions if we have a partitioned parent
            op.execute(f"CREATE TABLE IF NOT EXISTS {tbl} PARTITION OF games FOR VALUES FROM ({start}) TO ({end});")
        else:
            # Skipping partition creation because games exists and is not partitioned
            continue


def downgrade():
    # Drop the example partitions and parent table (best-effort)
    for i in range(0, 12):
        tbl = f"games_p_{i}"
        op.execute(f"DROP TABLE IF EXISTS {tbl};")
    op.execute("DROP TABLE IF EXISTS games;")
