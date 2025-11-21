from typing import Dict, Any, Optional
from .agent_utils import get_legal_moves, apply_move
from .heuristics import evaluate_combined

AGGRESSOR_WEIGHTS = {
    'h_mat': 1.0,
    'h_force': 2.0,
    'h_terr': 0.5,
    'h_def': 0.5,
    'h_off': 5.0,
    'h_pin': 1.0
}

def select_move(game_state: Dict[str, Any], perspective: Optional[str] = None) -> Dict:
    legal = get_legal_moves(game_state)
    if not legal:
        return {'type': 'endTurn'}

    mover = perspective or game_state.get('currentPlayerId')
    best_move = None
    best_score = -float('inf')
    
    for mv in legal:
        new_state = apply_move(game_state, mv)
        score = evaluate_combined(new_state, mover, AGGRESSOR_WEIGHTS)
        if score > best_score:
            best_score = score
            best_move = mv
            
    return best_move if best_move is not None else {'type': 'endTurn'}
