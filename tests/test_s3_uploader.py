import os
import tempfile

import pytest

from services.s3_uploader import S3Uploader


class DummyS3:
    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls = 0

    def upload_file(self, Filename, Bucket, Key):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise Exception('simulated failure')
        # else succeed


def test_s3_uploader_retries_and_metrics(tmp_path, monkeypatch):
    # create a dummy file
    fpath = tmp_path / 'shard_test.jsonl'
    fpath.write_text('hello')

    # dummy prom metrics
    class Cnt:
        def __init__(self):
            self.count = 0
        def inc(self, n=1):
            self.count += n

    p = {'s3_uploads': Cnt(), 's3_upload_errors': Cnt()}

    uploader = S3Uploader(bucket='my-bucket', shard_dir=str(tmp_path), prom_metrics=p, max_retries=3, backoff_base=0.01)

    # replace s3 client with dummy that fails twice then succeeds
    uploader._s3 = DummyS3(fail_times=2)

    # call upload directly and ensure retries happen (no exception)
    uploader._upload_file(str(fpath), 'shard_test.jsonl')

    assert p['s3_uploads'].count == 1
    # should have recorded errors on failed attempts
    assert p['s3_upload_errors'].count >= 2
