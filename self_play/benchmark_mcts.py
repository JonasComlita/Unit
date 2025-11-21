from self_play.benchmark import benchmark
from self_play.alpha_beta_agent import select_move as alpha_beta_select
from self_play.mcts import MCTSAgent

print("="*60)
print("Alpha-Beta vs MCTS Benchmarks")
print("="*60)

# Initialize MCTS agents
mcts_d3 = MCTSAgent(simulations=20, rollout_depth=3)
mcts_d6 = MCTSAgent(simulations=20, rollout_depth=6)

print("\n--- Alpha-Beta (Depth 2) vs MCTS (Depth 3) ---")
benchmark(alpha_beta_select, mcts_d3.select_move, rounds=4)

print("\n--- Alpha-Beta (Depth 2) vs MCTS (Depth 6) ---")
benchmark(alpha_beta_select, mcts_d6.select_move, rounds=4)
