#!/usr/bin/env python3
"""
Comprehensive benchmark to find the crossover point where MCTS beats Alpha-Beta.
Tests various configurations of both algorithms.
"""

import sys
import os
from typing import Dict, Any, List, Tuple
import time

from .agent_utils import initialize_game, get_legal_moves, apply_move
from .alpha_beta_agent import select_move as alpha_beta_select, MAX_DEPTH as AB_DEFAULT_DEPTH
from .mcts import MCTSAgent

def play_game(agent1_fn, agent2_fn, agent1_name: str, agent2_name: str, max_turns: int = 300) -> Dict[str, Any]:
    """
    Play a single game between two agents.
    Returns game result with winner, turn count, and timing info.
    """
    state = initialize_game()
    turn_count = 0
    agent1_time = 0.0
    agent2_time = 0.0
    
    while turn_count < max_turns:
        current_player = state['currentPlayerId']
        
        # Select agent based on current player
        if current_player == 'Player1':
            start = time.time()
            move = agent1_fn(state, 'Player1')
            agent1_time += time.time() - start
        else:
            start = time.time()
            move = agent2_fn(state, 'Player2')
            agent2_time += time.time() - start
        
        if not move:
            # No valid moves, game over
            break
            
        state = apply_move(state, move)
        turn_count += 1
        
        # Check if game is over
        winner = state.get('winner')
        if winner:
            return {
                'winner': winner,
                'turns': turn_count,
                'agent1_time': agent1_time,
                'agent2_time': agent2_time,
                'agent1_avg_time': agent1_time / max(1, turn_count // 2),
                'agent2_avg_time': agent2_time / max(1, turn_count // 2),
            }
    
    # Max turns reached, no winner
    return {
        'winner': None,
        'turns': turn_count,
        'agent1_time': agent1_time,
        'agent2_time': agent2_time,
        'agent1_avg_time': agent1_time / max(1, turn_count // 2),
        'agent2_avg_time': agent2_time / max(1, turn_count // 2),
    }

def run_benchmark(config: Dict[str, Any], num_games: int = 10) -> Dict[str, Any]:
    """
    Run a benchmark with the given configuration.
    
    config should contain:
    - agent1_type: 'alpha_beta' or 'mcts'
    - agent1_params: dict of parameters
    - agent2_type: 'alpha_beta' or 'mcts'
    - agent2_params: dict of parameters
    """
    
    # Create agent functions
    def create_agent_fn(agent_type: str, params: Dict[str, Any]):
        if agent_type == 'alpha_beta':
            depth = params.get('depth', AB_DEFAULT_DEPTH)
            # Modify alpha_beta_agent to accept depth parameter
            def ab_fn(state, perspective):
                from . import alpha_beta_agent
                old_depth = alpha_beta_agent.MAX_DEPTH
                alpha_beta_agent.MAX_DEPTH = depth
                move = alpha_beta_select(state, perspective)
                alpha_beta_agent.MAX_DEPTH = old_depth
                return move
            return ab_fn
        elif agent_type == 'mcts':
            simulations = params.get('simulations', 100)
            rollout_depth = params.get('rollout_depth', 3)
            agent = MCTSAgent(simulations=simulations, rollout_depth=rollout_depth)
            return lambda state, perspective: agent.select_move(state)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent1_fn = create_agent_fn(config['agent1_type'], config['agent1_params'])
    agent2_fn = create_agent_fn(config['agent2_type'], config['agent2_params'])
    
    agent1_name = f"{config['agent1_type']}({config['agent1_params']})"
    agent2_name = f"{config['agent2_type']}({config['agent2_params']})"
    
    print(f"\n{'='*80}")
    print(f"Benchmark: {agent1_name} vs {agent2_name}")
    print(f"{'='*80}")
    
    results = {
        'agent1_wins': 0,
        'agent2_wins': 0,
        'draws': 0,
        'total_turns': 0,
        'agent1_total_time': 0.0,
        'agent2_total_time': 0.0,
        'games': []
    }
    
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}...", end=' ', flush=True)
        
        game_result = play_game(agent1_fn, agent2_fn, agent1_name, agent2_name)
        results['games'].append(game_result)
        
        if game_result['winner'] == 'Player1':
            results['agent1_wins'] += 1
            print(f"✓ {agent1_name} wins in {game_result['turns']} turns")
        elif game_result['winner'] == 'Player2':
            results['agent2_wins'] += 1
            print(f"✗ {agent2_name} wins in {game_result['turns']} turns")
        else:
            results['draws'] += 1
            print(f"= Draw after {game_result['turns']} turns")
        
        results['total_turns'] += game_result['turns']
        results['agent1_total_time'] += game_result['agent1_time']
        results['agent2_total_time'] += game_result['agent2_time']
    
    # Calculate statistics
    results['agent1_win_rate'] = results['agent1_wins'] / num_games
    results['agent2_win_rate'] = results['agent2_wins'] / num_games
    results['draw_rate'] = results['draws'] / num_games
    results['avg_turns'] = results['total_turns'] / num_games
    results['agent1_avg_time_per_game'] = results['agent1_total_time'] / num_games
    results['agent2_avg_time_per_game'] = results['agent2_total_time'] / num_games
    
    # Print summary
    print(f"\n{'-'*80}")
    print(f"RESULTS:")
    print(f"{'-'*80}")
    print(f"{agent1_name}: {results['agent1_wins']}/{num_games} wins ({results['agent1_win_rate']*100:.1f}%)")
    print(f"{agent2_name}: {results['agent2_wins']}/{num_games} wins ({results['agent2_win_rate']*100:.1f}%)")
    print(f"Draws: {results['draws']}/{num_games} ({results['draw_rate']*100:.1f}%)")
    print(f"Average game length: {results['avg_turns']:.1f} turns")
    print(f"{agent1_name} avg time/game: {results['agent1_avg_time_per_game']:.2f}s")
    print(f"{agent2_name} avg time/game: {results['agent2_avg_time_per_game']:.2f}s")
    
    return results

def main():
    """
    Run comprehensive benchmarks to find MCTS crossover point.
    """
    
    print("="*80)
    print("MCTS vs Alpha-Beta Crossover Point Analysis (Optimized MCTS)")
    print("="*80)
    
    num_games = 8  # Games per configuration
    
    # Test configurations
    # We'll test Alpha-Beta at depths 2, 3, 4
    # Against MCTS with varying simulations and rollout depths
    
    configurations = []
    
    # Alpha-Beta Depth 2 vs MCTS
    print("\n" + "="*80)
    print("PHASE 1: Alpha-Beta (Depth 2) vs MCTS")
    print("="*80)
    
    # User requested only 20, 50 simulations for speed
    for sims in [20, 50]:
        for rollout in [3, 5, 7]:
            configurations.append({
                'name': f'AB-D2 vs MCTS-S{sims}-R{rollout}',
                'agent1_type': 'alpha_beta',
                'agent1_params': {'depth': 2},
                'agent2_type': 'mcts',
                'agent2_params': {'simulations': sims, 'rollout_depth': rollout}
            })
    
    # Run all benchmarks
    all_results = []
    
    for config in configurations:
        result = run_benchmark(config, num_games=num_games)
        result['config'] = config
        all_results.append(result)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - MCTS Win Rates Against Alpha-Beta")
    print("="*80)
    
    print("\nAlpha-Beta Depth 2:")
    for r in all_results:
        if r['config']['agent1_params']['depth'] == 2:
            mcts_params = r['config']['agent2_params']
            print(f"  MCTS(sims={mcts_params['simulations']}, rollout={mcts_params['rollout_depth']}): "
                  f"{r['agent2_win_rate']*100:.1f}% wins, "
                  f"{r['draw_rate']*100:.1f}% draws, "
                  f"{r['agent2_avg_time_per_game']:.2f}s/game")
    
    # Find best MCTS configurations
    print("\n" + "="*80)
    print("BEST MCTS CONFIGURATIONS (by win rate)")
    print("="*80)
    
    sorted_results = sorted(all_results, key=lambda x: x['agent2_win_rate'], reverse=True)
    
    for i, r in enumerate(sorted_results[:10]):
        ab_depth = r['config']['agent1_params']['depth']
        mcts_params = r['config']['agent2_params']
        print(f"{i+1}. MCTS(sims={mcts_params['simulations']}, rollout={mcts_params['rollout_depth']}) "
              f"vs AB(depth={ab_depth}): "
              f"{r['agent2_win_rate']*100:.1f}% wins, "
              f"{r['agent2_avg_time_per_game']:.2f}s/game")

if __name__ == '__main__':
    main()
