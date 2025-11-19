"""
Quick smoke test to verify all algorithms can make moves.
"""
from .agent_utils import initialize_game
from .random_algorithm import select_move as random_select
from .greedy_algorithm import select_move as greedy_select
from .greedy_banker import select_move as banker_select
from .greedy_spreader import select_move as spreader_select
from .greedy_aggressor import select_move as aggressor_select

def test_algorithm(name, select_fn):
    """Test that an algorithm can select a move from initial state."""
    print(f"\nTesting {name}...")
    state = initialize_game()
    try:
        move = select_fn(state)
        print(f"  ✓ {name} selected move: {move.get('type', 'unknown')}")
        return True
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Basic Algorithm Smoke Tests")
    print("=" * 60)
    
    results = []
    results.append(test_algorithm("Random", random_select))
    results.append(test_algorithm("Greedy (basic)", greedy_select))
    results.append(test_algorithm("Banker", banker_select))
    results.append(test_algorithm("Spreader", spreader_select))
    results.append(test_algorithm("Aggressor", aggressor_select))
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All algorithms passed!")
    else:
        print("✗ Some algorithms failed")
    print("=" * 60)
