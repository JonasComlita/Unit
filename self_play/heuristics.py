from typing import Dict, Any, List, Optional
import math
from .agent_utils import get_force, LAYER_GRAVITY, FORCE_CAP_MAX

def get_position_coord(vertex: Dict[str, Any], coord: str) -> float:
    """Safely extract position coordinate from vertex (handles both dict and object formats)."""
    pos = vertex.get('position', {})
    # Try direct access first (dict format)
    if isinstance(pos, dict):
        return pos.get(coord, 0.0)
    # Try attribute access (object format from TypeScript)
    return getattr(pos, coord, 0.0)

def get_distance(v1: Dict[str, Any], v2: Dict[str, Any]) -> float:
    """Calculate Euclidean distance between two vertices."""
    dx = get_position_coord(v1, 'x') - get_position_coord(v2, 'x')
    dz = get_position_coord(v1, 'z') - get_position_coord(v2, 'z')
    # Simple layer difference approximation (assuming layer spacing is consistent enough for relative distance)
    dy = (v1.get('layer', 0) - v2.get('layer', 0)) * 2.0 # Arbitrary vertical scaling
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def get_manhattan_distance(v1: Dict[str, Any], v2: Dict[str, Any]) -> int:
    """Calculate Manhattan-like distance (grid steps) - simplified."""
    return abs(get_position_coord(v1, 'x') - get_position_coord(v2, 'x')) + \
           abs(get_position_coord(v1, 'z') - get_position_coord(v2, 'z')) + \
           abs(v1.get('layer', 0) - v2.get('layer', 0))

def h_material(state: Dict[str, Any], perspective: str) -> float:
    """
    Heuristic 1: Material Advantage
    (P_my - P_opp) * 10 + (E_my - E_opp) * 15
    """
    my_pieces = 0
    my_energy = 0
    opp_pieces = 0
    opp_energy = 0
    
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
            opp_pieces += count
            opp_energy += energy
            
    return (my_pieces - opp_pieces) * 10.0 + (my_energy - opp_energy) * 15.0

def h_force(state: Dict[str, Any], perspective: str) -> float:
    """
    Heuristic 2: Effective Force
    Sum(MyForce) - Sum(OppForce)
    """
    my_force = 0.0
    opp_force = 0.0
    
    vertices = state.get('vertices', {})
    for vertex in vertices.values():
        stack = vertex.get('stack', [])
        if not stack:
            continue
            
        force = get_force(vertex)
        owner = stack[0]['player']
        
        if owner == perspective:
            my_force += force
        else:
            opp_force += force
            
    return my_force - opp_force

def h_territory(state: Dict[str, Any], perspective: str) -> float:
    """
    Heuristic 3: Territory Control
    MyVertices - OppVertices
    """
    my_count = 0
    opp_count = 0
    
    vertices = state.get('vertices', {})
    for vertex in vertices.values():
        stack = vertex.get('stack', [])
        if not stack:
            continue
        
        owner = stack[0]['player']
        if owner == perspective:
            my_count += 1
        else:
            opp_count += 1
            
    return float(my_count - opp_count)

def h_home_defense(state: Dict[str, Any], perspective: str) -> float:
    """
    Heuristic 4: Home Defense
    Sum(Distance(Home, NearestEnemy) * DefenseForce)
    We want to MAXIMIZE distance to enemies and MAXIMIZE defense force.
    
    Simplified: 
    - Penalty for enemies close to home.
    - Bonus for friendly force on/near home.
    """
    score = 0.0
    home_corners = state['homeCorners'][perspective]
    vertices = state['vertices']
    
    # Find all enemy locations
    enemy_locs = []
    for v in vertices.values():
        if v['stack'] and v['stack'][0]['player'] != perspective:
            enemy_locs.append(v)
            
    if not enemy_locs:
        return 100.0 # Safe
        
    for corner_id in home_corners:
        corner = vertices[corner_id]
        
        # Bonus for defending the corner itself
        if corner['stack'] and corner['stack'][0]['player'] == perspective:
            score += get_force(corner) * 2.0
            
        # Penalty for nearby enemies
        min_dist = float('inf')
        nearest_enemy_force = 0.0
        
        for enemy in enemy_locs:
            dist = get_distance(corner, enemy)
            if dist < min_dist:
                min_dist = dist
                nearest_enemy_force = get_force(enemy)
        
        # If enemy is very close, big penalty
        if min_dist < 0.1: # On top of corner (game over usually, but good to penalize)
            score -= 1000.0
        else:
            # Penalty inversely proportional to distance, scaled by enemy threat
            score -= (nearest_enemy_force * 10.0) / (min_dist + 0.1)
            
    return score

def h_offensive_threat(state: Dict[str, Any], perspective: str) -> float:
    """
    Heuristic 5: Offensive Threat
    Sum(MyForce / DistanceToEnemyHome)
    """
    score = 0.0
    opponent = 'Player2' if perspective == 'Player1' else 'Player1'
    opp_home_corners = state['homeCorners'][opponent]
    vertices = state['vertices']
    
    my_units = []
    for v in vertices.values():
        if v['stack'] and v['stack'][0]['player'] == perspective:
            my_units.append(v)
            
    for unit in my_units:
        force = get_force(unit)
        if force <= 0:
            continue
            
        # Find distance to nearest enemy home
        min_dist = float('inf')
        for corner_id in opp_home_corners:
            corner = vertices[corner_id]
            dist = get_distance(unit, corner)
            if dist < min_dist:
                min_dist = dist
                
        if min_dist < 0.1: # Occupying enemy home!
            score += 1000.0
        else:
            score += (force * 10.0) / (min_dist + 0.1)
            
    return score

def h_pincer_potential(state: Dict[str, Any], perspective: str) -> float:
    """
    Heuristic 6: Pincer Potential
    Count of enemy units with >= 2 friendly neighbors.
    """
    score = 0.0
    vertices = state['vertices']
    
    for v in vertices.values():
        # Check if enemy occupied
        if v['stack'] and v['stack'][0]['player'] != perspective:
            friendly_neighbors = 0
            for adj_id in v['adjacencies']:
                neighbor = vertices[adj_id]
                if neighbor['stack'] and neighbor['stack'][0]['player'] == perspective:
                    friendly_neighbors += 1
            
            if friendly_neighbors >= 2:
                score += 1.0
                
    return score

def evaluate_combined(state: Dict[str, Any], perspective: str, weights: Dict[str, float]) -> float:
    """
    Calculate weighted sum of all heuristics.
    """
    score = 0.0
    
    if weights.get('h_mat', 0):
        score += weights['h_mat'] * h_material(state, perspective)
        
    if weights.get('h_force', 0):
        score += weights['h_force'] * h_force(state, perspective)
        
    if weights.get('h_terr', 0):
        score += weights['h_terr'] * h_territory(state, perspective)
        
    if weights.get('h_def', 0):
        score += weights['h_def'] * h_home_defense(state, perspective)
        
    if weights.get('h_off', 0):
        score += weights['h_off'] * h_offensive_threat(state, perspective)
        
    if weights.get('h_pin', 0):
        score += weights['h_pin'] * h_pincer_potential(state, perspective)
        
    return score
