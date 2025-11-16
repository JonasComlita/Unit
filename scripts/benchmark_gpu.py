#!/usr/bin/env python3
"""Benchmark model forward and training speeds for various batch sizes and AMP settings."""
import time
import torch
import numpy as np
from neural_network_model import UnitGameNet, UnitGameTrainer


def forward_benchmark(batch_size=32, iters=100):
    model = UnitGameNet()
    x = torch.randn(batch_size, 83, 5)

    # CPU
    model.to('cpu')
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(x)
    t1 = time.perf_counter()
    cpu_time = (t1 - t0) / iters

    results = {'batch_size': batch_size, 'cpu_sec_per_iter': cpu_time}

    if torch.cuda.is_available():
        model.to('cuda')
        xg = x.to('cuda')
        # warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(xg)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters * 2):
            with torch.no_grad():
                _ = model(xg)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        results['gpu_sec_per_iter'] = (t1 - t0) / (iters * 2)

    return results


def training_benchmark(batch_size=32, iters=100, use_amp=False):
    model = UnitGameNet()
    trainer = UnitGameTrainer(model, learning_rate=1e-3, use_amp=use_amp)
    # synthetic data
    states = np.random.rand(batch_size, 83, 5).astype('float32')
    # policy targets: use per-vertex classification (flattened) — create dummy integer targets
    policy_targets = np.random.randint(0, 83 * 4, size=(batch_size,))
    value_targets = np.random.randn(batch_size).astype('float32')

    # quick warmup
    for _ in range(5):
        trainer.train_on_batch(states, policy_targets, value_targets)

    t0 = time.perf_counter()
    for _ in range(iters):
        trainer.train_on_batch(states, policy_targets, value_targets)
    t1 = time.perf_counter()
    return {'batch_size': batch_size, 'sec_per_batch': (t1 - t0) / iters, 'use_amp': use_amp}


if __name__ == '__main__':
    print('Forward benchmarks:')
    for bs in [8, 32, 128]:
        r = forward_benchmark(batch_size=bs, iters=50)
        print(r)

    print('\nTraining benchmarks:')
    for bs in [8, 32, 64]:
        r = training_benchmark(batch_size=bs, iters=20, use_amp=False)
        print(r)
        if torch.cuda.is_available():
            r2 = training_benchmark(batch_size=bs, iters=20, use_amp=True)
            print(r2)
