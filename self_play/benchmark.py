"""
Small benchmark harness to pit agents against each other for quick smoke tests.
Usage (python -m self_play.benchmark or run file):
    from self_play import random_algorithm, greedy_algorithm, a_star_algorithm

This script runs short games and reports basic win stats.
"""
import time
from typing import Callable

from .agent_utils import initialize_game, apply_move, get_legal_moves
from .random_algorithm import select_move as random_select
from .greedy_algorithm import select_move as greedy_select
from .a_star_algorithm import select_move as a_star_select
from .greedy_banker import select_move as banker_select
from .greedy_spreader import select_move as spreader_select
from .greedy_aggressor import select_move as aggressor_select


AgentFn = Callable[[dict], dict]


def play_game(agent_white: AgentFn, agent_black: AgentFn, max_moves: int = 200):
    state = initialize_game()
    move_count = 0
    while not state.get('winner') and move_count < max_moves:
        current = state['currentPlayerId']
        agent = agent_white if current == 'Player1' else agent_black
        mv = agent(state)
        if not isinstance(mv, dict):
            # defensive: expect dict
            mv = {'type': 'endTurn'}
        state = apply_move(state, mv)
        move_count += 1
    return state.get('winner'), move_count


def benchmark(agent_a: AgentFn, agent_b: AgentFn, rounds: int = 20):
    stats = {'A': 0, 'B': 0, 'draw': 0, 'moves': 0}
    start = time.time()
    for i in range(rounds):
        # alternate starting player
        if i % 2 == 0:
            winner, moves = play_game(agent_a, agent_b)
            if winner == 'Player1':
                stats['A'] += 1
            elif winner == 'Player2':
                stats['B'] += 1
            else:
                stats['draw'] += 1
        else:
            winner, moves = play_game(agent_b, agent_a)
            if winner == 'Player1':
                stats['B'] += 1
            elif winner == 'Player2':
                stats['A'] += 1
            else:
                stats['draw'] += 1
        stats['moves'] += moves
    elapsed = time.time() - start
    print(f"Benchmark rounds={rounds} elapsed={elapsed:.2f}s avg_moves={stats['moves']/rounds:.1f}")
    print(f"A wins: {stats['A']}, B wins: {stats['B']}, draws: {stats['draw']}")


if __name__ == '__main__':
    print("=" * 60)
    print("Running comprehensive benchmarks")
    print("=" * 60)
    
    print("\n--- Random vs Greedy Variants ---")
    print("\nRandom vs Greedy (basic):")
    benchmark(random_select, greedy_select, rounds=10)
    
    print("\nRandom vs Banker:")
    benchmark(random_select, banker_select, rounds=10)
    
    print("\nRandom vs Spreader:")
    benchmark(random_select, spreader_select, rounds=10)
    
    print("\nRandom vs Aggressor:")
    benchmark(random_select, aggressor_select, rounds=10)
    
    print("\n--- Greedy Variants vs Each Other ---")
    print("\nBanker vs Spreader:")
    benchmark(banker_select, spreader_select, rounds=8)
    
    print("\nBanker vs Aggressor:")
    benchmark(banker_select, aggressor_select, rounds=8)
    
    print("\nSpreader vs Aggressor:")
    benchmark(spreader_select, aggressor_select, rounds=8)
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

