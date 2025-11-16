#!/usr/bin/env python3
"""Generate train/val/test Parquet splits from the games database.

This script performs a simple stratified split by `starting_player` and `winner`.
It writes three Parquet files: train.parquet, val.parquet, test.parquet in the output directory.

Usage:
  python scripts/generate_splits.py --db-url POSTGRES_URL --out-dir data/splits --limit 10000

"""
import argparse
import asyncio
import asyncpg
import pandas as pd
import os
import json
import random
import math


async def fetch_games(db_url, limit=None):
    conn = await asyncpg.connect(db_url)
    try:
        q = 'SELECT g.game_id, g.winner, gm.metadata FROM games g LEFT JOIN game_metadata gm USING (game_id)'
        if limit:
            q = q + f' LIMIT {int(limit)}'
        rows = await conn.fetch(q)
    finally:
        await conn.close()

    records = []
    for r in rows:
        meta = r.get('metadata')
        if isinstance(meta, str):
            try:
                md = json.loads(meta)
            except Exception:
                md = {}
        else:
            md = meta or {}

        records.append({
            'game_id': r['game_id'],
            'winner': r['winner'],
            'starting_player': md.get('starting_player'),
            'metadata': md,
        })

    return records


def split_and_write(records, out_dir, train_frac=0.7, val_frac=0.15, seed=42):
    df = pd.DataFrame(records)
    if df.empty:
        print('No games found; nothing to write')
        return

    # Create a simple stratification key
    df['strata'] = df['starting_player'].astype(str) + '|' + df['winner'].astype(str)

    random.seed(seed)
    train_rows = []
    val_rows = []
    test_rows = []

    # Group by strata and split within each group to preserve proportions
    for strata_val, group in df.groupby('strata'):
        items = group.to_dict(orient='records')
        n = len(items)
        if n == 0:
            continue
        # Shuffle
        random.shuffle(items)
        n_train = int(math.floor(train_frac * n))
        n_val = int(math.floor(val_frac * n))
        n_test = n - n_train - n_val

        train_rows.extend(items[:n_train])
        val_rows.extend(items[n_train:n_train + n_val])
        test_rows.extend(items[n_train + n_val:])

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(os.path.join(out_dir, 'train.parquet'), index=False)
    pd.DataFrame(val_rows).to_parquet(os.path.join(out_dir, 'val.parquet'), index=False)
    pd.DataFrame(test_rows).to_parquet(os.path.join(out_dir, 'test.parquet'), index=False)

    print(f'Wrote splits: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} to {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url', required=True)
    parser.add_argument('--out-dir', default='data/splits')
    parser.add_argument('--limit', type=int, default=10000)
    args = parser.parse_args()

    records = asyncio.run(fetch_games(args.db_url, args.limit))
    split_and_write(records, args.out_dir)


if __name__ == '__main__':
    main()
