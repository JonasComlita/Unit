import asyncio
import time

import pytest

from services.inference_batcher import InferenceBatcher


@pytest.mark.asyncio
async def test_inference_batcher_batches():
    calls = []

    async def model_fn(inputs):
        # record batch size and return per-input result
        calls.append(len(inputs))
        return [f'result:{i}' for i in range(len(inputs))]

    batcher = InferenceBatcher(model_fn, max_batch_size=4, timeout=0.05)
    await batcher.start()

    # fire off 3 concurrent predictions
    async def req(i):
        return await batcher.predict(f'input{i}')

    tasks = [asyncio.create_task(req(i)) for i in range(3)]
    results = await asyncio.gather(*tasks)

    assert results == ['result:0', 'result:1', 'result:2']
    # ensure at least one batched call occurred with size >= 1
    assert any(c >= 1 for c in calls)

    await batcher.stop()
