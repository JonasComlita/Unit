from typing import Dict, Any, Optional

from .agent_utils import get_legal_moves, apply_move, evaluate_position


def select_move(game_state: Dict[str, Any], perspective: Optional[str] = None) -> Dict:
    """
    Greedy agent: evaluate the immediate resulting position for each legal
    move and pick the move with the highest evaluation.

    Parameters
    - game_state: state dict
    - perspective: optionally evaluate from this player's perspective (defaults to mover)

    Returns a move dict or {'type':'endTurn'} if no moves.
    """
    legal = get_legal_moves(game_state)
    if not legal:
        return {'type': 'endTurn'}

    mover = game_state.get('currentPlayerId')
    best_move = None
    best_score = -float('inf')
    for mv in legal:
        new_state = apply_move(game_state, mv)
        score = evaluate_position(new_state, perspective or mover)
        if score > best_score:
            best_score = score
            best_move = mv

    return best_move if best_move is not None else {'type': 'endTurn'}
