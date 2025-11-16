#!/usr/bin/env python3
"""Simple CLI to upload existing shard files to S3 using boto3.

This is a one-shot uploader (not the background uploader) useful for
ad-hoc uploads from the command line or CI.
"""
import argparse
import os
import sys

try:
    import boto3
except Exception:
    boto3 = None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--bucket', required=True)
    p.add_argument('--shard-dir', default='shards')
    p.add_argument('--prefix', default='')
    p.add_argument('--delete-after-upload', action='store_true')
    args = p.parse_args(argv)

    if boto3 is None:
        print('boto3 not installed; install with pip install boto3', file=sys.stderr)
        return 2

    s3 = boto3.client('s3')
    for name in os.listdir(args.shard_dir):
        if name.endswith('.tmp'):
            continue
        if not (name.endswith('.parquet') or name.endswith('.jsonl')):
            continue
        path = os.path.join(args.shard_dir, name)
        key = os.path.join(args.prefix, name) if args.prefix else name
        print(f'Uploading {path} -> s3://{args.bucket}/{key}')
        s3.upload_file(Filename=path, Bucket=args.bucket, Key=key)
        if args.delete_after_upload:
            os.remove(path)


if __name__ == '__main__':
    sys.exit(main())
