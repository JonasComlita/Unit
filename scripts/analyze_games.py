#!/usr/bin/env python3
"""Simple analysis script to compute win rates by starting player.

Usage:
  python scripts/analyze_games.py --last 1000 --db-url 'postgresql://...'
"""
import argparse
import asyncio
import os
import sys
from collections import Counter

import asyncpg


async def main_async(db_url: str, last: int):
    conn = await asyncpg.connect(db_url)
    try:
        # Attempt to query joined metadata if available
        try:
            rows = await conn.fetch(
                """
                SELECT g.winner AS winner, gm.metadata->>'starting_player' AS starting_player
                FROM games g
                LEFT JOIN game_metadata gm ON g.game_id = gm.game_id
                ORDER BY g.created_at DESC
                LIMIT $1
                """,
                last,
            )
        except asyncpg.UndefinedTableError:
            # game_metadata doesn't exist; only query games
            rows = await conn.fetch(
                """
                SELECT winner, NULL::text as starting_player
                FROM games
                ORDER BY created_at DESC
                LIMIT $1
                """,
                last,
            )

        total = len(rows)
        if total == 0:
            print("No games found")
            return 0

        win_counter = Counter()
        start_counter = Counter()
        start_win = Counter()

        for r in rows:
            winner = r.get('winner')
            start = r.get('starting_player') or 'UNKNOWN'
            win_counter[winner] += 1
            start_counter[start] += 1
            start_win[(start, winner)] += 1

        print(f"Analyzed {total} games (most recent)")
        print("Starting player distribution:")
        for k, v in start_counter.most_common():
            print(f"  {k}: {v} ({v/total*100:.1f}%)")

        print("Win distribution:")
        for k, v in win_counter.most_common():
            print(f"  {k}: {v} ({v/total*100:.1f}%)")

        print("Wins by starting player -> winner:")
        for (start, winner), cnt in start_win.items():
            print(f"  start={start} winner={winner}: {cnt} ({cnt/total*100:.1f}%)")

        return 0
    finally:
        await conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--last', type=int, default=1000, help='Number of recent games to analyze')
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'), help='DB URL')
    args = parser.parse_args()

    if not args.db_url:
        print('Provide --db-url or set DATABASE_URL environment variable', file=sys.stderr)
        sys.exit(2)

    sys.exit(asyncio.run(main_async(args.db_url, args.last)))
