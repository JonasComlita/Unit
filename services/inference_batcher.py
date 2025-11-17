"""
GPU-Optimized Inference Batcher for Unit Game

Save this file as: services/gpu_inference_batcher.py

This fixes GPU utilization by running model inference in a thread pool
to avoid blocking the asyncio event loop.
"""

import asyncio
import logging
import threading
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Delay torch import to avoid import errors when torch isn't available
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU batcher disabled")


class GPUInferenceBatcher:
    """Async batched inference optimized for GPU throughput.
    
    Key improvements over standard InferenceBatcher:
    - Runs model forward pass in thread pool (avoids blocking event loop)
    - Efficient GPU batching
    - Minimizes CPU-GPU transfer overhead
    """
    
    def __init__(
        self, 
        model_fn: Callable[[List[Any]], List[Any]], 
        max_batch_size: int = 32, 
        timeout: float = 0.02,
        device: str = 'cuda'
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for GPU inference batcher")
            
        self.model_fn = model_fn
        self.max_batch_size = max_batch_size
        self.timeout = float(timeout)
        self.device = device
        
        # Queue for incoming requests
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        
        # Thread pool for GPU operations (small pool to reduce overhead)
        self._executor = ThreadPoolExecutor(
            max_workers=2, 
            thread_name_prefix='gpu_inference'
        )
        
        # Stats for monitoring
        self._total_batches = 0
        self._total_requests = 0
        self._avg_batch_size = 0.0
        self._lock = threading.Lock()

    async def start(self):
        """Start the background worker."""
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._worker())
        logger.info(
            f"GPU inference batcher started "
            f"(device={self.device}, batch_size={self.max_batch_size}, "
            f"timeout={self.timeout:.4f})"
        )

    async def stop(self):
        """Stop the worker and shutdown thread pool."""
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
        
        # Shutdown thread pool
        self._executor.shutdown(wait=True)
        
        # Log final stats
        with self._lock:
            if self._total_batches > 0:
                logger.info(
                    f"GPU batcher stats: {self._total_requests} requests, "
                    f"{self._total_batches} batches, "
                    f"avg batch size: {self._avg_batch_size:.1f}"
                )

    async def predict(self, input_item: Any, timeout: Optional[float] = None) -> Any:
        """Enqueue inference request and await result.
        
        Args:
            input_item: Single input to pass to model (will be batched)
            timeout: Optional timeout in seconds
            
        Returns:
            Model output for this input
        """
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        await self._queue.put((input_item, fut))
        
        if timeout is None:
            return await fut
        return await asyncio.wait_for(fut, timeout=timeout)

    async def _worker(self):
        """Main worker loop that batches requests and dispatches to GPU."""
        loop = asyncio.get_event_loop()
        
        while not self._stopping:
            batch_inputs: List[Any] = []
            futs: List[asyncio.Future] = []
            
            try:
                # Wait for first item or timeout
                item, fut = await asyncio.wait_for(
                    self._queue.get(), 
                    timeout=self.timeout
                )
                batch_inputs.append(item)
                futs.append(fut)

                # Aggressively drain queue up to max_batch_size
                # This is critical for GPU utilization
                while len(batch_inputs) < self.max_batch_size:
                    try:
                        item, fut = self._queue.get_nowait()
                        batch_inputs.append(item)
                        futs.append(fut)
                    except asyncio.QueueEmpty:
                        break

                # Run GPU inference in thread pool to avoid blocking event loop
                try:
                    results = await loop.run_in_executor(
                        self._executor,
                        self._run_batch_inference,
                        batch_inputs
                    )
                    
                    # Update stats
                    with self._lock:
                        self._total_batches += 1
                        self._total_requests += len(batch_inputs)
                        self._avg_batch_size = self._total_requests / self._total_batches
                    
                    # Log large batches (indicates good GPU utilization)
                    if len(batch_inputs) >= self.max_batch_size * 0.8:
                        logger.debug(
                            f"Large GPU batch processed: {len(batch_inputs)} items "
                            f"(avg: {self._avg_batch_size:.1f})"
                        )
                    
                except Exception as e:
                    logger.error(f"Batch inference failed: {e}", exc_info=True)
                    # Fail all futures
                    for f in futs:
                        if not f.done():
                            f.set_exception(e)
                    # Mark tasks done
                    for _ in range(len(futs)):
                        try:
                            self._queue.task_done()
                        except Exception:
                            pass
                    continue

                # Distribute results to futures
                if not isinstance(results, (list, tuple)):
                    exc = RuntimeError(
                        f"model_fn must return list/tuple, got {type(results)}"
                    )
                    for f in futs:
                        if not f.done():
                            f.set_exception(exc)
                elif len(results) != len(futs):
                    exc = RuntimeError(
                        f"model_fn returned {len(results)} results for "
                        f"{len(futs)} inputs"
                    )
                    for f in futs:
                        if not f.done():
                            f.set_exception(exc)
                else:
                    for f, r in zip(futs, results):
                        if not f.done():
                            f.set_result(r)

                # Mark queue items done
                for _ in range(len(futs)):
                    try:
                        self._queue.task_done()
                    except Exception:
                        pass

            except asyncio.TimeoutError:
                # No items arrived, continue
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                continue

    def _run_batch_inference(self, batch_inputs: List[Any]) -> List[Any]:
        """Run batched inference on GPU (called in thread pool).
        
        This runs in a separate thread to avoid blocking the asyncio event loop.
        All GPU operations happen here.
        """
        try:
            # Call the model function with the batch
            # model_fn should handle conversion to tensors and GPU transfer
            results = self.model_fn(batch_inputs)
            
            # Ensure we return a list
            if not isinstance(results, (list, tuple)):
                results = [results]
            
            return results
            
        except Exception as e:
            logger.error(f"GPU inference error: {e}", exc_info=True)
            raise


class InferenceBatcher:
    """Async batched inference for CPU or generic model functions.

    Provides the same public interface as GPUInferenceBatcher used elsewhere in
    the codebase: start(), stop(), predict(input_item, timeout=None).

    Implementation details:
    - Batches incoming requests from an asyncio.Queue
    - Runs the user-provided model_fn in a ThreadPoolExecutor to avoid
      blocking the asyncio event loop (useful if model_fn is CPU-bound).
    - Guarantees model_fn returns a list/tuple of outputs with one output per
      input.
    """

    def __init__(
        self,
        model_fn: Callable[[List[Any]], List[Any]],
        max_batch_size: int = 32,
        timeout: float = 0.02,
        executor_workers: Optional[int] = None,
    ):
        self.model_fn = model_fn
        self.max_batch_size = max_batch_size
        self.timeout = float(timeout)

        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

        # Small thread pool for running model_fn without blocking the loop
        workers = executor_workers if executor_workers is not None else max(1, (os.cpu_count() or 1) // 2)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inference")

        # Stats
        self._total_batches = 0
        self._total_requests = 0
        self._avg_batch_size = 0.0
        self._lock = threading.Lock()

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._worker())
        logger.info(f"InferenceBatcher started (batch_size={self.max_batch_size}, timeout={self.timeout:.4f})")

    async def stop(self):
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

        self._executor.shutdown(wait=True)

        with self._lock:
            if self._total_batches > 0:
                logger.info(
                    f"Inference batcher stats: {self._total_requests} requests, {self._total_batches} batches, avg batch size: {self._avg_batch_size:.1f}"
                )

    async def predict(self, input_item: Any, timeout: Optional[float] = None) -> Any:
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        await self._queue.put((input_item, fut))

        if timeout is None:
            return await fut
        return await asyncio.wait_for(fut, timeout=timeout)

    async def _worker(self):
        loop = asyncio.get_event_loop()

        while not self._stopping:
            batch_inputs: List[Any] = []
            futs: List[asyncio.Future] = []

            try:
                item, fut = await asyncio.wait_for(self._queue.get(), timeout=self.timeout)
                batch_inputs.append(item)
                futs.append(fut)

                # drain up to max_batch_size
                while len(batch_inputs) < self.max_batch_size:
                    try:
                        item, fut = self._queue.get_nowait()
                        batch_inputs.append(item)
                        futs.append(fut)
                    except asyncio.QueueEmpty:
                        break

                # run model_fn in threadpool
                try:
                    results = await loop.run_in_executor(self._executor, self._run_batch_inference, batch_inputs)

                    with self._lock:
                        self._total_batches += 1
                        self._total_requests += len(batch_inputs)
                        self._avg_batch_size = self._total_requests / self._total_batches

                except Exception as e:
                    logger.error(f"Batch inference failed: {e}", exc_info=True)
                    for f in futs:
                        if not f.done():
                            f.set_exception(e)
                    for _ in range(len(futs)):
                        try:
                            self._queue.task_done()
                        except Exception:
                            pass
                    continue

                # distribute results
                if not isinstance(results, (list, tuple)):
                    exc = RuntimeError(f"model_fn must return list/tuple, got {type(results)}")
                    for f in futs:
                        if not f.done():
                            f.set_exception(exc)
                elif len(results) != len(futs):
                    exc = RuntimeError(f"model_fn returned {len(results)} results for {len(futs)} inputs")
                    for f in futs:
                        if not f.done():
                            f.set_exception(exc)
                else:
                    for f, r in zip(futs, results):
                        if not f.done():
                            f.set_result(r)

                for _ in range(len(futs)):
                    try:
                        self._queue.task_done()
                    except Exception:
                        pass

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                continue

    def _run_batch_inference(self, batch_inputs: List[Any]) -> List[Any]:
        try:
            results = self.model_fn(batch_inputs)
            if not isinstance(results, (list, tuple)):
                results = [results]
            return results
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            raise



def create_optimized_model_fn(model, device):
    """Create an optimized model function for GPU batching.
    
    This function:
    1. Batches state tensors efficiently
    2. Minimizes CPU-GPU transfers
    3. Uses torch.no_grad() for inference
    4. Returns results on CPU for downstream use
    
    Args:
        model: UnitGameNet instance
        device: torch device (e.g., 'cuda' or 'cpu')
        
    Returns:
        Callable that processes a batch of game states
    """
    
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for optimized model function")
    
    # Import state_to_tensor here to avoid circular imports
    from self_play.neural_network_model import state_to_tensor
    
    def model_fn(batch_states: List[dict]) -> List[Tuple[np.ndarray, float]]:
        """Process a batch of game states efficiently on GPU.
        
        Args:
            batch_states: List of game state dictionaries
            
        Returns:
            List of (policy_array, value) tuples
        """
        
        try:
            # Convert all states to tensors (on CPU first)
            tensors = [state_to_tensor(s) for s in batch_states]
            
            # Stack into batch tensor
            batch_tensor = np.stack(tensors, axis=0).astype('float32')
            
            # Move to GPU in one operation and run inference
            with torch.no_grad():
                x = torch.from_numpy(batch_tensor).to(device)
                
                # Run model forward pass
                policy_pred, value_pred = model(x)
                
                # Move results back to CPU
                policy_np = policy_pred.cpu().numpy()
                value_np = value_pred.cpu().numpy()
            
            # Return list of (policy, value) tuples
            results = []
            for i in range(len(batch_states)):
                policy = policy_np[i]
                value = float(value_np[i].squeeze())
                results.append((policy, value))
            
            return results
            
        except Exception as e:
            logger.error(f"model_fn error: {e}", exc_info=True)
            raise
    
    return model_fn