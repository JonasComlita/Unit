import heapq
import itertools
from typing import Dict, Any, List, Tuple, Optional

from .agent_utils import get_legal_moves, apply_move, evaluate_position


def a_star_search(game_state: Dict[str, Any], max_expansions: int = 2000, max_depth: int = 6) -> Optional[Dict]:
    """
    Perform a bounded A* search where the heuristic is the (negative)
    evaluation used by the greedy agent. We search for the path that
    maximizes the evaluation for the current mover.

    Returns the first move on the best-found path or None.
    """
    mover = game_state.get('currentPlayerId')

    # Each frontier item: (priority, depth, counter, state, path)
    # counter is a monotonic tie-breaker so heapq never attempts to compare dicts
    frontier: List[Tuple[float, int, int, Dict[str, Any], List[Dict[str, Any]]]] = []
    counter = itertools.count()
    # Initial heuristic is negative of evaluation so we treat smaller as better
    start_h = -evaluate_position(game_state, perspective=mover)
    heapq.heappush(frontier, (start_h, 0, next(counter), game_state, []))
    visited = 0
    best_path: Optional[List[Dict[str, Any]]] = None
    best_score = -float('inf')

    while frontier and visited < max_expansions:
        priority, depth, _ctr, state, path = heapq.heappop(frontier)
        # Evaluate terminal/winner
        if state.get('winner'):
            # Terminal reached; evaluate final position (from mover perspective)
            final_score = evaluate_position(state, perspective=mover)
            if final_score > best_score:
                best_score = final_score
                best_path = path
            visited += 1
            continue

        # Stop exploring deeper if depth limit reached
        if depth >= max_depth:
            # Use evaluation as leaf estimate
            leaf_score = evaluate_position(state, perspective=mover)
            if leaf_score > best_score:
                best_score = leaf_score
                best_path = path
            visited += 1
            continue

        legal = get_legal_moves(state)
        for mv in legal:
            next_state = apply_move(state, mv)
            # Use evaluation as heuristic (higher is better)
            h = -evaluate_position(next_state, perspective=mover)
            new_path = path + [mv]
            heapq.heappush(frontier, (h, depth + 1, next(counter), next_state, new_path))

        visited += 1

    if best_path and len(best_path) > 0:
        return best_path[0]
    # Fallback: pick highest-eval immediate move
    legal_now = get_legal_moves(game_state)
    if not legal_now:
        return {'type': 'endTurn'}
    best_mv = None
    best_mv_score = -float('inf')
    for mv in legal_now:
        s = evaluate_position(apply_move(game_state, mv), perspective=mover)
        if s > best_mv_score:
            best_mv_score = s
            best_mv = mv
    return best_mv if best_mv is not None else {'type': 'endTurn'}


def select_move(game_state: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper to select a move using A* with sensible defaults."""
    return a_star_search(game_state)
