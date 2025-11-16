"""
Run model-enabled A/B test (dry-run) using SelfPlayGenerator.
Produces two arms:
 - baseline: evaluate_from_mover=False
 - fixed:    evaluate_from_mover=True
Each arm runs `games_per_arm` games in dry-run mode with the model enabled.

This script imports the generator directly to avoid the CLI-enforced overrides in
`main()` and therefore can control per-run config exactly.

Note: run this from repository root so imports resolve: `python3 scripts/run_model_ab_test.py`
"""

import asyncio
import time
from collections import Counter

try:
    import torch
except Exception:
    torch = None

from self_play_system import SelfPlayConfig, SelfPlayGenerator

async def run_arm(name: str, evaluate_from_mover: bool, games_per_arm: int, batch_size: int = 50):
    device = None
    if torch is not None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    config = SelfPlayConfig(
        dry_run=True,
        use_model=True,
        model_device=device,
        inference_batch_size=32,
        inference_batch_timeout=0.02,
        # We will call play_single_game directly in concurrent batches, so leave concurrent_games default
        # Keep trim_states False to match recommended defaults
        trim_states=False,
    )
    # Set A/B flag
    config.evaluate_from_mover = evaluate_from_mover

    gen = SelfPlayGenerator(config)
    await gen.initialize()

    tally = Counter()
    games_done = 0
    next_game_id = 0
    start = time.time()

    try:
        while games_done < games_per_arm:
            to_do = min(batch_size, games_per_arm - games_done)
            tasks = [asyncio.create_task(gen.play_single_game(next_game_id + i)) for i in range(to_do)]
            results = await asyncio.gather(*tasks)
            for g in results:
                winner = g.get('winner') or 'DRAW'
                tally[winner] += 1
            games_done += len(results)
            next_game_id += len(results)
            # optional small sleep to yield
            await asyncio.sleep(0)
    finally:
        await gen.shutdown()

    elapsed = time.time() - start
    return tally, elapsed

async def main():
    games_per_arm = 500
    print(f"Running model-enabled A/B dry-run with {games_per_arm} games per arm (no DB writes)")

    print("Running baseline (evaluate_from_mover=False) ...")
    base_tally, base_time = await run_arm('baseline', False, games_per_arm, batch_size=50)
    print(f"Baseline results: {base_tally} (elapsed={base_time:.1f}s)")

    print("Running fixed (evaluate_from_mover=True) ...")
    fixed_tally, fixed_time = await run_arm('fixed', True, games_per_arm, batch_size=50)
    print(f"Fixed results: {fixed_tally} (elapsed={fixed_time:.1f}s)")

    print('\nA/B summary:')
    print('baseline', dict(base_tally))
    print('fixed', dict(fixed_tally))
    print(f"Total elapsed: {base_time + fixed_time:.1f}s")

if __name__ == '__main__':
    asyncio.run(main())
