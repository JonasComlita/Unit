from self_play.benchmark import benchmark
from self_play.random_algorithm import select_move as random_select
from self_play.greedy_aggressor import select_move as aggressor_select
from self_play.greedy_banker import select_move as banker_select
from self_play.greedy_spreader import select_move as spreader_select
from self_play.dynamic_agent import select_move as dynamic_select
from self_play.alpha_beta_agent import select_move as alpha_beta_select

print("="*60)
print("Advanced Agent Benchmarks")
print("="*60)

print("\n--- Dynamic Agent ---")
print("Dynamic vs Random (5 rounds):")
benchmark(dynamic_select, random_select, rounds=5)

print("Dynamic vs Aggressor (5 rounds):")
benchmark(dynamic_select, aggressor_select, rounds=5)

print("Dynamic vs Banker (5 rounds):")
benchmark(dynamic_select, banker_select, rounds=5)

print("Dynamic vs Spreader (5 rounds):")
benchmark(dynamic_select, spreader_select, rounds=5)

print("\n--- Alpha-Beta Agent (Depth 2) ---")
print("Alpha-Beta vs Random (2 rounds):")
benchmark(alpha_beta_select, random_select, rounds=2)

print("Alpha-Beta vs Aggressor (2 rounds):")
benchmark(alpha_beta_select, aggressor_select, rounds=2)

print("Alpha-Beta vs Dynamic (2 rounds):")
benchmark(alpha_beta_select, dynamic_select, rounds=2)
