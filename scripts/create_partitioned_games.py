"""Helper script to create a partitioned `games` table (Postgres).

This is a best-effort helper; run in a controlled environment. It connects
to the DATABASE_URL and creates a partitioned parent table `games` which
partitions by RANGE on start_time (epoch ms).

Note: This script uses asyncpg and requires an accessible Postgres instance.
"""
import asyncio
import os
import time
import asyncpg

SQL_PARENT = """
CREATE TABLE IF NOT EXISTS games (
    game_id VARCHAR(50) PRIMARY KEY,
    start_time BIGINT NOT NULL,
    end_time BIGINT NOT NULL,
    winner VARCHAR(32),
    total_moves INTEGER,
    platform VARCHAR(32)
)
PARTITION BY RANGE (start_time);
"""

SQL_CREATE_PARTITION = """
CREATE TABLE IF NOT EXISTS games_p_{suffix} PARTITION OF games
    FOR VALUES FROM ({start}) TO ({end});
"""


async def main():
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        print('Please set DATABASE_URL')
        return 2

    conn = await asyncpg.connect(db_url)
    try:
        await conn.execute(SQL_PARENT)

        # Create partitions for the next 12 months (one per month)
        now = int(time.time() * 1000)
        month_ms = 30 * 24 * 3600 * 1000
        for i in range(0, 12):
            start = now + i * month_ms
            end = now + (i + 1) * month_ms
            suffix = i
            sql = SQL_CREATE_PARTITION.format(suffix=suffix, start=start, end=end)
            await conn.execute(sql)

        print('Partitioned games table and monthly partitions created (or already present)')
    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
