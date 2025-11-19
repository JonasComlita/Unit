from typing import Dict, Any, Optional
from .agent_utils import get_legal_moves, apply_move

def evaluate_aggressor(state: Dict[str, Any], perspective: str) -> float:
    """
    Aggressor Heuristic:
    Maximize the difference between my material and enemy material.
    Weights pieces higher than energy.
    """
    my_pieces = 0
    my_energy = 0
    enemy_pieces = 0
    enemy_energy = 0
    
    vertices = state.get('vertices', {})
    for vertex in vertices.values():
        stack = vertex.get('stack', [])
        if not stack:
            continue
            
        owner = stack[0]['player']
        count = len(stack)
        energy = vertex.get('energy', 0)
        
        if owner == perspective:
            my_pieces += count
            my_energy += energy
        else:
            enemy_pieces += count
            enemy_energy += energy
            
    # Heuristic from plan: (MyPieces - EnemyPieces) * 2.0 + (MyEnergy - EnemyEnergy) * 1.0
    # We multiply by 10 to keep it somewhat consistent with other scores if compared, 
    # but the relative value is what matters.
    score = (my_pieces - enemy_pieces) * 20.0 + (my_energy - enemy_energy) * 1.0
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
        score = evaluate_aggressor(new_state, mover)
        if score > best_score:
            best_score = score
            best_move = mv
            
    return best_move if best_move is not None else {'type': 'endTurn'}
