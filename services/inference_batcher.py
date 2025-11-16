import asyncio
import inspect
from typing import Any, Callable, List, Optional, Tuple

"""Asynchronous batched inference helper.

This provides a small, dependency-free way to batch many concurrent
inference requests into a single model call. The model function may be
sync or async and must accept a list of inputs and return a list of
outputs of the same length.

Example usage:
    async def model_fn(batch_inputs):
        # return list of results
        return [inference(x) for x in batch_inputs]

    batcher = InferenceBatcher(model_fn, max_batch_size=32, timeout=0.02)
    await batcher.start()
    result = await batcher.predict(input_item)
    await batcher.stop()
"""


async def _maybe_await(v):
    if inspect.isawaitable(v):
        return await v
    return v


class InferenceBatcher:
    def __init__(self, model_fn: Callable[[List[Any]], List[Any]], max_batch_size: int = 32, timeout: float = 0.02):
        self.model_fn = model_fn
        self.max_batch_size = max_batch_size
        self.timeout = float(timeout)
        self._queue: "asyncio.Queue[Tuple[Any, asyncio.Future]]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._worker())

    async def stop(self):
        # request stop, wait for queue to drain, then cancel worker
        self._stopping = True
        if self._task:
            try:
                await self._queue.join()
            except Exception:
                pass
            try:
                self._task.cancel()
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def predict(self, input_item: Any, timeout: Optional[float] = None) -> Any:
        """Enqueue a single inference request and await the result.

        Returns whatever the model returns for that input. The call will
        wait until the batch is processed; optional timeout can be provided
        (seconds) to raise asyncio.TimeoutError.
        """
        fut = asyncio.get_event_loop().create_future()
        await self._queue.put((input_item, fut))
        if timeout is None:
            return await fut
        return await asyncio.wait_for(fut, timeout=timeout)

    async def _worker(self):
        while not self._stopping:
            batch_inputs: List[Any] = []
            futs: List[asyncio.Future] = []
            try:
                # Wait until at least one item arrives or timeout
                item, fut = await asyncio.wait_for(self._queue.get(), timeout=self.timeout)
                batch_inputs.append(item)
                futs.append(fut)

                # non-blocking drain up to max_batch_size
                while len(batch_inputs) < self.max_batch_size:
                    try:
                        item, fut = self._queue.get_nowait()
                        batch_inputs.append(item)
                        futs.append(fut)
                    except asyncio.QueueEmpty:
                        break

                # call model function (supports sync or async model_fn)
                try:
                    results = await _maybe_await(self.model_fn(batch_inputs))
                except Exception as e:
                    # fail all futures
                    for f in futs:
                        if not f.done():
                            f.set_exception(e)
                    # mark tasks done and continue
                    for _ in range(len(futs)):
                        try:
                            self._queue.task_done()
                        except Exception:
                            pass
                    continue

                # ensure results length matches
                if not isinstance(results, (list, tuple)):
                    exc = RuntimeError("model_fn must return a list/tuple of results matching batch size")
                    for f in futs:
                        if not f.done():
                            f.set_exception(exc)
                else:
                    for f, r in zip(futs, results):
                        if not f.done():
                            f.set_result(r)

                # mark queue items as done
                for _ in range(len(futs)):
                    try:
                        self._queue.task_done()
                    except Exception:
                        pass

            except asyncio.TimeoutError:
                # nothing arrived within timeout, loop and check _stopping
                continue
            except Exception:
                # unexpected exception in worker; keep running
                continue

