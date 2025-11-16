#!/usr/bin/env python3
"""Analyze dataset statistics from the database.

Usage: python scripts/analyze_dataset.py --db-url POSTGRES_URL [--limit N]

Produces simple stats: total games, games by starting player, win rates, move length distribution.
"""
import argparse
import asyncpg
import asyncio
import statistics
import json


async def analyze(db_url, limit=None):
    conn = await asyncpg.connect(db_url)
    try:
        total = await conn.fetchval('SELECT COUNT(*) FROM games')
    except Exception:
        print('No games table or DB not initialized')
        await conn.close()
        return

    print(f'Total games: {total}')

    q = 'SELECT g.game_id, g.winner, gm.metadata FROM games g LEFT JOIN game_metadata gm USING (game_id)'
    if limit:
        q = q + f' LIMIT {int(limit)}'

    rows = await conn.fetch(q)
    starts = {}
    winners = {'Player1': 0, 'Player2': 0, 'DRAW': 0, None: 0}
    move_counts = []

    for r in rows:
        meta = r.get('metadata')
        if meta:
            if isinstance(meta, str):
                try:
                    md = json.loads(meta)
                except Exception:
                    md = {}
            else:
                md = meta
        else:
            md = {}

        sp = md.get('starting_player')
        starts[sp] = starts.get(sp, 0) + 1
        w = r.get('winner') or 'DRAW'
        winners[w] = winners.get(w, 0) + 1

        # Fetch move count
        mc = await conn.fetchval('SELECT COUNT(*) FROM moves WHERE game_id=$1', r['game_id'])
        move_counts.append(mc or 0)

    print('Starting player distribution:')
    for k, v in starts.items():
        print(f'  {k}: {v}')

    print('Winner counts:')
    for k, v in winners.items():
        print(f'  {k}: {v}')

    if move_counts:
        print('Move count stats:')
        print(f'  mean: {statistics.mean(move_counts):.2f}, median: {statistics.median(move_counts)}')
        print(f'  min: {min(move_counts)}, max: {max(move_counts)}')

    await conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url', required=True)
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()

    asyncio.run(analyze(args.db_url, args.limit))


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Lightweight dataset analysis for class imbalance and simple stats.

Usage: python scripts/analyze_dataset.py --db-url 'postgresql://...'
"""
import argparse
import asyncio
import json

import asyncpg


async def analyze(db_url: str, last: int = 1000):
    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            """
            SELECT g.game_id, g.winner, gm.metadata ->> 'starting_player' AS starting_player
            FROM games g
            LEFT JOIN game_metadata gm ON g.game_id = gm.game_id
            ORDER BY g.created_at DESC
            LIMIT $1
            """,
            last,
        )

        total = len(rows)
        wins = {}
        starts = {}
        for r in rows:
            wins[r['winner']] = wins.get(r['winner'], 0) + 1
            starts[r['starting_player']] = starts.get(r['starting_player'], 0) + 1

        print(f'Total sampled games: {total}')
        print('Wins by player:')
        print(json.dumps(wins, indent=2))
        print('Starting player distribution:')
        print(json.dumps(starts, indent=2))

    finally:
        await conn.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--db-url', required=True)
    p.add_argument('--last', type=int, default=1000)
    args = p.parse_args(argv)
    asyncio.run(analyze(args.db_url, last=args.last))


if __name__ == '__main__':
    main()
