from typing import Dict, Any, Optional
from .agent_utils import get_legal_moves, apply_move

def evaluate_banker(state: Dict[str, Any], perspective: str) -> float:
    """
    Banker Heuristic:
    Maximize material, but weight it by layer gravity.
    Layer 2 (Center) = 1.5x
    Layers 1/3 = 1.2x
    Layers 0/4 = 1.0x
    """
    score = 0.0
    vertices = state.get('vertices', {})
    
    for vertex in vertices.values():
        stack = vertex.get('stack', [])
        if not stack:
            continue
            
        owner = stack[0]['player']
        if owner != perspective:
            continue
            
        count = len(stack)
        energy = vertex.get('energy', 0)
        layer = vertex.get('layer', 0)
        
        # Base material value
        material_value = count * 10 + energy * 1
        
        # Layer Multiplier
        multiplier = 1.0
        if layer == 2:
            multiplier = 1.5
        elif layer == 1 or layer == 3:
            multiplier = 1.2
            
        score += material_value * multiplier
        
    return score

def select_move(game_state: Dict[str, Any], perspective: Optional[str] = None) -> Dict:
    legal = get_legal_moves(game_state)
    if not legal:
        return {'type': 'endTurn'}

    mover = perspective or game_state.get('currentPlayerId')
    best_move = None
    best_score = -float('inf')
    
    for mv in legal:
        new_state = apply_move(game_state, mv)
        score = evaluate_banker(new_state, mover)
        if score > best_score:
            best_score = score
            best_move = mv
            
    return best_move if best_move is not None else {'type': 'endTurn'}
