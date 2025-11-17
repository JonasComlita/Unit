import multiprocessing as mp
import threading
import queue
import time
import torch
import traceback

class InferenceRequest:
    def __init__(self, data, response_queue, meta=None):
        self.data = data
        self.response_queue = response_queue
        self.meta = meta

class InferenceResponse:
    def __init__(self, result, meta=None, error=None):
        self.result = result
        self.meta = meta
        self.error = error

class GPUInferenceServer(mp.Process):
    def __init__(self, model_path, device, batch_size=32, timeout=0.02):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.timeout = timeout
        self.request_queue = mp.Queue()
        self.shutdown_event = mp.Event()

    def load_model(self):
        model = torch.load(self.model_path, map_location=self.device)
        if hasattr(model, 'eval'):
            model.eval()
        return model

    def run(self):
        model = self.load_model()
        print(f"[GPUInferenceServer] Model loaded on {self.device}")
        while not self.shutdown_event.is_set():
            batch = []
            response_queues = []
            start = time.time()
            # Collect requests for batching
            while len(batch) < self.batch_size and (time.time() - start) < self.timeout:
                try:
                    req = self.request_queue.get(timeout=self.timeout)
                    batch.append(req.data)
                    response_queues.append((req.response_queue, req.meta))
                except queue.Empty:
                    break
            if batch:
                try:
                    batch_tensor = torch.stack(batch).to(self.device)
                    with torch.no_grad():
                        output = model(batch_tensor)
                    # Split output for each request
                    for i, (resp_queue, meta) in enumerate(response_queues):
                        result = output[i].cpu()
                        resp = InferenceResponse(result=result, meta=meta)
                        resp_queue.put(resp)
                except Exception as e:
                    tb = traceback.format_exc()
                    for resp_queue, meta in response_queues:
                        resp = InferenceResponse(result=None, meta=meta, error=str(e) + "\n" + tb)
                        resp_queue.put(resp)
        print("[GPUInferenceServer] Shutdown requested.")

    def shutdown(self):
        self.shutdown_event.set()

# Worker-side helper
class GPUInferenceClient:
    def __init__(self, request_queue):
        self.request_queue = request_queue
        self.response_queue = mp.Queue()

    def infer(self, data, meta=None):
        req = InferenceRequest(data, self.response_queue, meta)
        self.request_queue.put(req)
        resp = self.response_queue.get()
        if resp.error:
            raise RuntimeError(f"Inference error: {resp.error}")
        return resp.result

# Example usage:
# server = GPUInferenceServer(model_path, device='cuda', batch_size=32)
# server.start()
# client = GPUInferenceClient(server.request_queue)
# result = client.infer(input_tensor)
# server.shutdown()
