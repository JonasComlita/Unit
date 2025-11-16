#!/usr/bin/env python3
"""Simple CSV experiment logger for quick local experiment tracking.

This is intentionally lightweight. For full-featured experiment tracking use MLflow.
"""
import argparse
import csv
import os
from datetime import datetime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='experiments.csv')
    p.add_argument('--name', required=True)
    p.add_argument('--notes', default='')
    p.add_argument('--metrics', default='')
    args = p.parse_args()

    header = ['timestamp', 'name', 'notes', 'metrics']
    exists = os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([datetime.utcnow().isoformat(), args.name, args.notes, args.metrics])
    print(f'Logged experiment to {args.out}')


if __name__ == '__main__':
    main()
