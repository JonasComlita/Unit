from typing import Dict, Any, Optional
from .agent_utils import get_legal_moves, apply_move

def _calculate_distance(v1: Dict[str, Any], v2: Dict[str, Any]) -> float:
    """Calculate Manhattan distance between two vertices."""
    dx = abs(v1['x'] - v2['x'])
    dz = abs(v1['z'] - v2['z'])
    # Add layer difference for 3D distance
    dy = abs(v1['layer'] - v2['layer'])
    return dx + dz + dy

def evaluate_spreader(state: Dict[str, Any], perspective: str) -> float:
    """
    Spreader Heuristic:
    - Maximize the number of occupied vertices (territorial control)
    - Small bonus for proximity to home (maintain supply lines)
    - Moderate penalty for enemies getting close to home corners
    - Small bonus for material to break ties
    - OFFENSIVE: Bonus for expanding toward enemy territory
    """
    occupied_count = 0
    material_score = 0.0
    proximity_score = 0.0
    
    vertices = state.get('vertices', {})
    home_corners = state.get('homeCorners', {}).get(perspective, [])
    
    # Get opponent
    opponent = 'Player2' if perspective == 'Player1' else 'Player1'
    enemy_corners = state.get('homeCorners', {}).get(opponent, [])
    
    # Get home corner vertices for distance calculations
    home_corner_vertices = [vertices[cid] for cid in home_corners if cid in vertices]
    enemy_corner_vertices = [vertices[cid] for cid in enemy_corners if cid in vertices]
    
    for vertex in vertices.values():
        stack = vertex.get('stack', [])
        if not stack:
            continue
            
        owner = stack[0]['player']
        count = len(stack)
        energy = vertex.get('energy', 0)
        force = count * 10 + energy * 15
        
        if owner == perspective:
            occupied_count += 1
            material_score += count * 10 + energy * 1
            
            # REDUCED proximity bonus (was 5, now 2)
            # Spreader wants to expand but maintain connection to home
            if home_corner_vertices:
                min_distance = min(_calculate_distance(vertex, hc) for hc in home_corner_vertices)
                # Gentle bonus for being within reasonable distance (not too far)
                if min_distance <= 8:
                    proximity_bonus = (9 - min_distance) * 2
                    proximity_score += proximity_bonus
            
            # OFFENSIVE: Reward expanding toward enemy territory
            if enemy_corner_vertices:
                min_enemy_distance = min(_calculate_distance(vertex, ec) for ec in enemy_corner_vertices)
                # Bonus for being closer to enemy (encourages expansion)
                if min_enemy_distance <= 8:
                    expansion_bonus = (9 - min_enemy_distance) * 15
                    proximity_score += expansion_bonus
        else:
            # Enemy forces - REDUCED penalty (was 0.4, now 0.25)
            if home_corner_vertices:
                min_distance = min(_calculate_distance(vertex, hc) for hc in home_corner_vertices)
                # Significant penalty for enemies near home
                if min_distance <= 4:
                    threat_penalty = (5 - min_distance) * force * 0.25
                    proximity_score -= threat_penalty
    
    # Weighted combination: prioritize spreading, but consider positioning
    score = (
        occupied_count * 100.0 +  # Primary goal: occupy many vertices
        material_score * 0.5 +     # Secondary: have some material
        proximity_score            # Tertiary: maintain defensive positioning + offensive expansion
    )
    
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
