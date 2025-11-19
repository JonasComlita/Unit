from typing import Dict, Any, Optional
from .agent_utils import get_legal_moves, apply_move

def evaluate_spreader(state: Dict[str, Any], perspective: str) -> float:
    """
    Spreader Heuristic:
    Maximize the number of occupied vertices.
    Score = NumOccupiedVertices * 50 + Material
    """
    occupied_count = 0
    material_score = 0.0
    
    vertices = state.get('vertices', {})
    for vertex in vertices.values():
        stack = vertex.get('stack', [])
        if not stack:
            continue
            
        owner = stack[0]['player']
        if owner == perspective:
            occupied_count += 1
            count = len(stack)
            energy = vertex.get('energy', 0)
            material_score += count * 10 + energy * 1
            
    score = occupied_count * 50.0 + material_score
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
        score = evaluate_spreader(new_state, mover)
        if score > best_score:
            best_score = score
            best_move = mv
            
    return best_move if best_move is not None else {'type': 'endTurn'}
