#!/usr/bin/env python3
"""Export games to Parquet or JSONL excluding archived games by default.

Usage:
  python scripts/export_dataset.py --out data/games.parquet --limit 10000 --db-url 'postgresql://...'

By default archived games (game_metadata.metadata->>'archived' = 'true') are excluded.
"""
import argparse
import asyncio
import json
import os
import sys
from typing import List, Dict, Any, Optional

import asyncpg


def _infer_format(out_path: str, fmt_arg: Optional[str]) -> str:
    if fmt_arg:
        return fmt_arg.lower()
    if out_path.lower().endswith('.parquet'):
        return 'parquet'
    if out_path.lower().endswith('.jsonl') or out_path.lower().endswith('.ndjson'):
        return 'jsonl'
    return 'parquet'


async def fetch_games(db_url: str, limit: int, include_archived: bool) -> List[Dict[str, Any]]:
    conn = await asyncpg.connect(db_url)
    try:
        # Fetch core game rows
        if include_archived:
            rows = await conn.fetch(
                """
                SELECT g.game_id, g.start_time, g.end_time, g.winner, g.total_moves, g.platform
                FROM games g
                ORDER BY g.created_at DESC
                LIMIT $1
                """,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT g.game_id, g.start_time, g.end_time, g.winner, g.total_moves, g.platform
                FROM games g
                LEFT JOIN game_metadata gm ON g.game_id = gm.game_id
                WHERE gm.metadata->>'archived' IS NULL OR gm.metadata->>'archived' != 'true'
                ORDER BY g.created_at DESC
                LIMIT $1
                """,
                limit,
            )

        results = []
        for r in rows:
            game_id = r['game_id']
            moves = await conn.fetch(
                "SELECT move_number, player_id, action_type, action_data, thinking_time_ms, timestamp FROM moves WHERE game_id=$1 ORDER BY move_number",
                game_id,
            )
            # fetch metadata if exists
            metadata = await conn.fetchrow("SELECT metadata FROM game_metadata WHERE game_id=$1", game_id)
            meta = metadata['metadata'] if metadata and 'metadata' in metadata else None
            results.append({
                'game_id': game_id,
                'start_time': r.get('start_time'),
                'end_time': r.get('end_time'),
                'winner': r.get('winner'),
                'total_moves': r.get('total_moves'),
                'platform': r.get('platform'),
                'metadata': meta,
                'moves': [dict(m) for m in moves],
            })

        return results
    finally:
        await conn.close()


def write_jsonl(records: List[Dict[str, Any]], out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + '\n')


def write_parquet(records: List[Dict[str, Any]], out_path: str):
    try:
        import pandas as pd
    except Exception as e:
        print('To write Parquet you need pandas and pyarrow installed: pip install pandas pyarrow', file=sys.stderr)
        raise

    # Convert moves and metadata to JSON strings so Parquet can store them cleanly
    for r in records:
        r['moves_json'] = json.dumps(r.get('moves', []), default=str)
        r['metadata_json'] = json.dumps(r.get('metadata'), default=str) if r.get('metadata') is not None else None

    # Select columns to store
    df = pd.DataFrame.from_records(records)
    # Keep moves_json and metadata_json, drop large nested objects
    if 'moves' in df.columns:
        df = df.drop(columns=['moves'])

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_parquet(out_path, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='Output path (.parquet or .jsonl)')
    parser.add_argument('--limit', type=int, default=1000, help='Number of games to export')
    parser.add_argument('--db-url', type=str, default=os.getenv('DATABASE_URL'), help='DB URL')
    parser.add_argument('--include-archived', action='store_true', help='Include archived games')
    parser.add_argument('--format', type=str, default=None, choices=['parquet', 'jsonl'], help='Output format (optional)')

    args = parser.parse_args(argv)
    if not args.db_url:
        print('Provide --db-url or set DATABASE_URL environment variable', file=sys.stderr)
        return 2

    out_format = _infer_format(args.out, args.format)

    records = asyncio.run(fetch_games(args.db_url, args.limit, args.include_archived))
    if not records:
        print('No games to export')
        return 0

    if out_format == 'jsonl':
        write_jsonl(records, args.out)
    else:
        write_parquet(records, args.out)

    print(f'Wrote {len(records)} games to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
