import random
from typing import Dict, Any

from .agent_utils import get_legal_moves


def select_move(game_state: Dict[str, Any]) -> Dict:
    """
    Select a uniformly random legal move for the current state.

    This function is defensive: it accepts raw state dictionaries produced
    by the self-play generator and will compute legal moves locally if the
    caller doesn't provide helper methods.

    Returns a move dict (e.g. {'type':'move', ...}) or {'type':'endTurn'} if
    no moves are available.
    """
    legal = get_legal_moves(game_state)
    if not legal:
        return {'type': 'endTurn'}
    return random.choice(legal)
