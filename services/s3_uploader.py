import asyncio
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    _HAVE_BOTO3 = True
except Exception:
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception
    _HAVE_BOTO3 = False


class S3Uploader:
    """Simple background uploader that uploads shard files to S3.

    Uses blocking boto3 calls executed in an executor to avoid blocking
    the event loop. It watches a directory and uploads new files matching
    an extension (default .parquet/.jsonl) and deletes local files on
    successful upload if configured.
    """

    def __init__(self, bucket: str, shard_dir: str = 'shards', region: Optional[str] = None, delete_after_upload: bool = False, prom_metrics: Optional[dict] = None, max_retries: int = 3, backoff_base: float = 0.5):
        self.bucket = bucket
        self.shard_dir = shard_dir
        self.region = region
        self.delete_after_upload = delete_after_upload
        self._task: Optional[asyncio.Task] = None
        self._stop = False
        self._s3 = None
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)

        # Prometheus hooks
        if prom_metrics:
            self._p_s3_uploads = prom_metrics.get('s3_uploads')
            self._p_s3_upload_errors = prom_metrics.get('s3_upload_errors')
        else:
            self._p_s3_uploads = None
            self._p_s3_upload_errors = None

    def _ensure_client(self):
        if not _HAVE_BOTO3:
            raise RuntimeError('boto3 not installed')
        if self._s3 is None:
            session = boto3.session.Session()
            self._s3 = session.client('s3', region_name=self.region)

    async def start(self):
        self._stop = False
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stop = True
        if self._task:
            try:
                await self._task
            except Exception:
                pass

    async def _run(self):
        self._ensure_client()
        loop = asyncio.get_running_loop()
        while not self._stop:
            try:
                # scan directory for shard files
                for name in os.listdir(self.shard_dir):
                    if name.endswith('.tmp'):
                        continue
                    if not (name.endswith('.parquet') or name.endswith('.jsonl')):
                        continue
                    path = os.path.join(self.shard_dir, name)
                    # upload in threadpool
                    try:
                        await loop.run_in_executor(None, self._upload_file, path, name)
                        if self.delete_after_upload:
                            try:
                                os.remove(path)
                            except Exception:
                                logger.warning("Failed to delete %s after upload", path)
                    except Exception as e:
                        logger.debug("Upload failed for %s: %s", path, e)

                await asyncio.sleep(2.0)
            except Exception:
                logger.exception("Unexpected error in S3Uploader loop")
                await asyncio.sleep(5.0)

    def _upload_file(self, path: str, key: str):
        # blocking boto3 upload with retries and exponential backoff
        attempt = 0
        while True:
            try:
                attempt += 1
                self._s3.upload_file(Filename=path, Bucket=self.bucket, Key=key)
                logger.info("Uploaded %s to s3://%s/%s", path, self.bucket, key)
                try:
                    if self._p_s3_uploads:
                        self._p_s3_uploads.inc()
                except Exception:
                    pass
                return
            except (BotoCoreError, ClientError) as e:
                logger.warning("S3 upload attempt %d failed for %s: %s", attempt, path, e)
                try:
                    if self._p_s3_upload_errors:
                        self._p_s3_upload_errors.inc()
                except Exception:
                    pass
                if attempt >= self.max_retries:
                    logger.exception("S3 upload failed after %d attempts for %s", attempt, path)
                    raise
                # exponential backoff with jitter
                backoff = self.backoff_base * (2 ** (attempt - 1))
                jitter = min(0.1 * backoff, 1.0)
                sleep_for = backoff + (0.5 - os.urandom(1)[0] / 255.0) * jitter
                try:
                    # sleep a little before retrying
                    import time as _time
                    _time.sleep(max(0.1, sleep_for))
                except Exception:
                    pass
