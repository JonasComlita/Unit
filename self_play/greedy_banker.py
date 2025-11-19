from typing import Dict, Any, Optional
from .agent_utils import get_legal_moves, apply_move

def _calculate_distance(v1: Dict[str, Any], v2: Dict[str, Any]) -> float:
    """Calculate Manhattan distance between two vertices."""
    dx = abs(v1['x'] - v2['x'])
    dz = abs(v1['z'] - v2['z'])
    # Add layer difference for 3D distance
    dy = abs(v1['layer'] - v2['layer'])
    return dx + dz + dy

def evaluate_banker(state: Dict[str, Any], perspective: str) -> float:
    """
    Banker Heuristic:
    - Rewards vertices with more force (pieces + energy)
    - Small bonus for forces closer to own home corners (defensive positioning)
    - Moderate penalty for enemies getting close to own home corners
    - Extra bonus for high-force stacks (consolidated power)
    - OFFENSIVE: Bonus for threatening enemy home corners
    """
    score = 0.0
    vertices = state.get('vertices', {})
    home_corners = state.get('homeCorners', {}).get(perspective, [])
    
    # Get opponent
    opponent = 'Player2' if perspective == 'Player1' else 'Player1'
    enemy_corners = state.get('homeCorners', {}).get(opponent, [])
    
    # Get home corner vertices for distance calculations
    home_corner_vertices = [vertices[cid] for cid in home_corners if cid in vertices]
    enemy_corner_vertices = [vertices[cid] for cid in enemy_corners if cid in vertices]
    
    for vid, vertex in vertices.items():
        stack = vertex.get('stack', [])
        if not stack:
            continue
            
        owner = stack[0]['player']
        count = len(stack)
        energy = vertex.get('energy', 0)
        force = count * 10 + energy * 15  # Total force value
        
        if owner == perspective:
            # Base material score
            score += force
            
            # Bonus for high-force stacks (consolidated power)
            if count >= 3:
                score += 50 * count
            elif count >= 2:
                score += 20 * count
            
            # REDUCED defensive proximity bonus (was 0.3, now 0.1)
            if home_corner_vertices:
                min_distance = min(_calculate_distance(vertex, hc) for hc in home_corner_vertices)
                proximity_bonus = max(0, (10 - min_distance) * force * 0.1)
                score += proximity_bonus
            
            # OFFENSIVE: Reward being near enemy corners
            if enemy_corner_vertices:
                min_enemy_distance = min(_calculate_distance(vertex, ec) for ec in enemy_corner_vertices)
                # Closer to enemy = bigger bonus
                if min_enemy_distance <= 6:
                    offensive_bonus = (7 - min_enemy_distance) * force * 0.4
                    score += offensive_bonus
        else:
            # Enemy forces - REDUCED penalty (was 0.5, now 0.3)
            if home_corner_vertices:
                min_distance = min(_calculate_distance(vertex, hc) for hc in home_corner_vertices)
                if min_distance <= 5:
                    proximity_penalty = (6 - min_distance) * force * 0.3
                    score -= proximity_penalty
    
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
