from typing import Dict, Any, Optional
from .agent_utils import get_legal_moves, apply_move
from .heuristics import evaluate_combined
from .greedy_aggressor import AGGRESSOR_WEIGHTS
from .greedy_banker import BANKER_WEIGHTS
from .greedy_spreader import SPREADER_WEIGHTS

def get_game_phase(state: Dict[str, Any]) -> str:
    """Determine the current phase of the game."""
    turn = state.get('turn', {}).get('turnNumber', 1)
    
    if turn <= 10:
        return 'OPENING'
    elif turn >= 40:
        return 'ENDGAME'
    else:
        return 'MIDGAME'

def is_under_threat(state: Dict[str, Any], perspective: str) -> bool:
    """Check if home corners are threatened."""
    vertices = state.get('vertices', {})
    home_corners = state.get('homeCorners', {}).get(perspective, [])
    opponent = 'Player2' if perspective == 'Player1' else 'Player1'
    
    for cid in home_corners:
        if cid not in vertices: continue
        corner = vertices[cid]
        
        # Check for nearby enemies (Manhattan distance <= 3)
        cx, cz, cl = corner['x'], corner['z'], corner['layer']
        
        for v in vertices.values():
            if v.get('stack') and v['stack'][0]['player'] == opponent:
                dist = abs(v['x'] - cx) + abs(v['z'] - cz) + abs(v['layer'] - cl)
                if dist <= 3:
                    return True
    return False

def select_weights(state: Dict[str, Any], perspective: str) -> Dict[str, float]:
    """Select appropriate weights based on game state."""
    phase = get_game_phase(state)
    
    if phase == 'OPENING':
        # Opening: Expand and take territory
        return SPREADER_WEIGHTS
        
    elif phase == 'ENDGAME':
        # Endgame: Aggressive push to finish
        return AGGRESSOR_WEIGHTS
        
    else: # MIDGAME
        if is_under_threat(state, perspective):
            # Defend if threatened
            return BANKER_WEIGHTS
        else:
            # Default to balanced/aggressive in midgame
            # We can blend weights or just pick one. Let's pick Aggressor to keep pressure.
            return AGGRESSOR_WEIGHTS

def select_move(game_state: Dict[str, Any], perspective: Optional[str] = None) -> Dict:
    legal = get_legal_moves(game_state)
    if not legal:
        return {'type': 'endTurn'}

    mover = perspective or game_state.get('currentPlayerId')
    
    # Determine strategy for this turn
    weights = select_weights(game_state, mover)
    
    best_move = None
    best_score = -float('inf')
    
    for mv in legal:
        new_state = apply_move(game_state, mv)
        score = evaluate_combined(new_state, mover, weights)
        if score > best_score:
            best_score = score
            best_move = mv
            
    return best_move if best_move is not None else {'type': 'endTurn'}
