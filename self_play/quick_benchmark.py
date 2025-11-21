from self_play.benchmark import benchmark
from self_play.random_algorithm import select_move as random_select
from self_play.greedy_aggressor import select_move as aggressor_select
from self_play.greedy_banker import select_move as banker_select
from self_play.greedy_spreader import select_move as spreader_select

print("Running quick benchmark...")
print("\nAggressor vs Random (2 rounds):")
benchmark(aggressor_select, random_select, rounds=2)

print("\nBanker vs Random (2 rounds):")
benchmark(banker_select, random_select, rounds=2)

print("\nSpreader vs Random (2 rounds):")
benchmark(spreader_select, random_select, rounds=2)
