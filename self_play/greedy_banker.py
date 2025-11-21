from typing import Dict, Any, Optional
from .agent_utils import get_legal_moves, apply_move
from .heuristics import evaluate_combined

BANKER_WEIGHTS = {
    'h_mat': 4.0,
    'h_force': 1.0,
    'h_terr': 0.5,
    'h_def': 3.0,
    'h_off': 1.0,
    'h_pin': 0.5
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
        score = evaluate_combined(new_state, mover, BANKER_WEIGHTS)
        if score > best_score:
            best_score = score
            best_move = mv
            
    return best_move if best_move is not None else {'type': 'endTurn'}
