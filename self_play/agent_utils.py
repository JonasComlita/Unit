from typing import Dict, List, Any, Optional
import copy


def _get_board_layout() -> List[int]:
    # Mirror default used in SelfPlayGenerator
    return [3, 5, 7, 5, 3]


# Occupancy requirement constants (mirror TypeScript constants.ts)
OCCUPATION_REQUIREMENTS = [
    {'minPieces': 1, 'minEnergy': 1, 'minForce': 1},  # 3x3 layers (0, 4)
    {'minPieces': 1, 'minEnergy': 1, 'minForce': 4},  # 5x5 layers (1, 3)
    {'minPieces': 1, 'minEnergy': 1, 'minForce': 9},  # 7x7 layer (2)
]

LAYER_GRAVITY = [1.0, 2.0, 3.0, 2.0, 1.0]
FORCE_CAP_MAX = 10


def get_occupation_requirement(layer: int) -> Dict[str, int]:
    """Get occupation requirements for a given layer."""
    if layer == 0 or layer == 4:
        return OCCUPATION_REQUIREMENTS[0]
    if layer == 1 or layer == 3:
        return OCCUPATION_REQUIREMENTS[1]
    return OCCUPATION_REQUIREMENTS[2]


def get_force(vertex: Dict[str, Any]) -> float:
    """Calculate force for a vertex (mirrors TypeScript getForce)."""
    if not vertex or not vertex.get('stack'):
        return 0.0
    
    layer = vertex.get('layer', 0)
    gravity_divider = LAYER_GRAVITY[layer]
    pieces = len(vertex['stack'])
    energy = vertex.get('energy', 0)
    
    force = (pieces * energy) / gravity_divider
    return min(force, FORCE_CAP_MAX)


def is_occupied(vertex: Dict[str, Any], requirement_layer: Optional[int] = None) -> bool:
    """Check if a vertex meets occupation requirements for a given layer."""
    layer_to_check = requirement_layer if requirement_layer is not None else vertex.get('layer', 0)
    req = get_occupation_requirement(layer_to_check)
    force = get_force(vertex)
    
    pieces = len(vertex.get('stack', []))
    energy = vertex.get('energy', 0)
    
    return (pieces >= req['minPieces'] and 
            energy >= req['minEnergy'] and 
            force >= req['minForce'])


def initialize_game() -> Dict[str, Any]:
    board_layout = _get_board_layout()
    vertices = {}
    vertex_id = 0
    for layer_idx, size in enumerate(board_layout):
        for x in range(size):
            for z in range(size):
                vid = f"v{vertex_id}"
                vertices[vid] = {
                    'id': vid,
                    'layer': layer_idx,
                    'x': x,
                    'z': z,
                    'stack': [],
                    'energy': 0,
                    'adjacencies': []
                }
                vertex_id += 1

    # set adjacencies (4-connected grid per layer)
    for vid, vertex in vertices.items():
        adj = []
        layer_verts = [v for v in vertices.values() if v['layer'] == vertex['layer']]
        for other in layer_verts:
            if other['id'] != vid:
                dx = abs(other['x'] - vertex['x'])
                dz = abs(other['z'] - vertex['z'])
                if (dx == 1 and dz == 0) or (dx == 0 and dz == 1):
                    adj.append(other['id'])
        vertex['adjacencies'] = adj

    # corners for home positions (simplified)
    layer0 = [v for v in vertices.values() if v['layer'] == 0]
    corners_p1 = [v['id'] for v in layer0 if v['x'] == 0 and v['z'] == 0]
    corners_p2 = [v['id'] for v in layer0 if v['x'] == max([lv['x'] for lv in layer0]) and v['z'] == max([lv['z'] for lv in layer0])]

    return {
        'vertices': vertices,
        'currentPlayerId': 'Player1',
        'winner': None,
        'turn': {
            'hasPlaced': False,
            'hasInfused': False,
            'hasMoved': False,
            'turnNumber': 1
        },
        'players': {
            'Player1': {'reinforcements': 3},
            'Player2': {'reinforcements': 3}
        },
        'homeCorners': {
            'Player1': corners_p1 if corners_p1 else [list(vertices.keys())[0]],
            'Player2': corners_p2 if corners_p2 else [list(vertices.keys())[-1]]
        }
    }


def get_legal_moves(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reimplementation of SelfPlayGenerator.get_legal_moves for standalone
    agent use. Returns a list of move dicts using the same keys as the
    generator's implementation.
    """
    moves: List[Dict[str, Any]] = []
    current_player = state['currentPlayerId']
    turn = state['turn']
    vertices = state['vertices']

    if not turn['hasPlaced'] and state['players'][current_player]['reinforcements'] > 0:
        for corner_id in state['homeCorners'][current_player]:
            vertex = vertices[corner_id]
            # Can place if empty OR if occupied by self
            if not vertex['stack'] or vertex['stack'][0]['player'] == current_player:
                moves.append({'type': 'place', 'vertexId': corner_id})

    # Infusion
    if not turn['hasInfused']:
        for vid, vertex in vertices.items():
            if vertex['stack'] and vertex['stack'][0]['player'] == current_player:
                # Check if infusion would exceed force cap
                layer = vertex.get('layer', 0)
                gravity_divider = LAYER_GRAVITY[layer]
                pieces = len(vertex['stack'])
                current_energy = vertex.get('energy', 0)
                
                # Calculate potential force after infusion
                potential_force = (pieces * (current_energy + 1)) / gravity_divider
                
                # Only allow infusion if it won't exceed force cap
                if potential_force <= FORCE_CAP_MAX:
                    moves.append({'type': 'infuse', 'vertexId': vid})

    # Movement
    if not turn['hasMoved']:
        # Check if any home corners are at max force (forced move rule)
        forced_move_origins = []
        for corner_id in state['homeCorners'][current_player]:
            corner = vertices[corner_id]
            if corner['stack'] and corner['stack'][0]['player'] == current_player:
                force = get_force(corner)
                if force >= FORCE_CAP_MAX:
                    forced_move_origins.append(corner_id)
        
        # If forced moves exist, only generate moves from those origins
        valid_origins = forced_move_origins if forced_move_origins else vertices.keys()

        for vid in valid_origins:
            vertex = vertices[vid]
            # Standard check: must be owned by current player (redundant for forced moves but safe)
            if vertex['stack'] and vertex['stack'][0]['player'] == current_player:
                for target_id in vertex['adjacencies']:
                    target = vertices[target_id]
                    # Can't move onto enemy pieces
                    if target['stack'] and target['stack'][0]['player'] != current_player:
                        continue
                    
                    # Check if source meets occupation requirements for target layer
                    if is_occupied(vertex, target['layer']):
                        moves.append({'type': 'move', 'fromId': vid, 'toId': target_id})

    # Attack
    if not turn['hasMoved']:
        for vid, vertex in vertices.items():
            if vertex['stack'] and vertex['stack'][0]['player'] == current_player:
                if len(vertex['stack']) >= 1 and vertex.get('energy', 0) >= 1:
                    for target_id in vertex['adjacencies']:
                        target = vertices[target_id]
                        if target['stack'] and target['stack'][0]['player'] != current_player:
                            moves.append({'type': 'attack', 'vertexId': vid, 'targetId': target_id})

        # Pincer
        # Find all enemy vertices that have >= 2 friendly neighbors
        for vid, vertex in vertices.items():
            # Check if vertex is enemy-occupied
            if vertex['stack'] and vertex['stack'][0]['player'] != current_player:
                # Find friendly neighbors
                friendly_neighbors = []
                for adj_id in vertex['adjacencies']:
                    neighbor = vertices[adj_id]
                    if neighbor['stack'] and neighbor['stack'][0]['player'] == current_player:
                        friendly_neighbors.append(adj_id)
                
                # If >= 2 friendly neighbors, it's a valid pincer target
                if len(friendly_neighbors) >= 2:
                    # We pass the targetId and the list of originIds
                    moves.append({
                        'type': 'pincer', 
                        'targetId': vid, 
                        'originIds': friendly_neighbors
                    })

    if turn['hasPlaced'] and turn['hasInfused'] and turn['hasMoved']:
        moves.append({'type': 'endTurn'})

    if not moves:
        moves.append({'type': 'endTurn'})

    return moves


def apply_move(state: Dict[str, Any], move: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a move to a copy of state and return the new state. Mirrors the
    simplified rules used in SelfPlayGenerator.apply_move.
    """
    new_state = copy.deepcopy(state)
    move_type = move.get('type')
    current_player = new_state['currentPlayerId']

    if move_type == 'place':
        vertex_id = move['vertexId']
        vertex = new_state['vertices'][vertex_id]
        # create a simple piece id based on stack length
        # Insert at 0 (top) to match gameLogic.ts
        vertex['stack'].insert(0, {'player': current_player, 'id': f"p{len(vertex['stack'])}"})
        new_state['players'][current_player]['reinforcements'] -= 1
        new_state['turn']['hasPlaced'] = True

    elif move_type == 'infuse':
        vertex_id = move['vertexId']
        new_state['vertices'][vertex_id]['energy'] += 1
        new_state['turn']['hasInfused'] = True

    elif move_type == 'move':
        from_id = move['fromId']
        to_id = move['toId']
        source = new_state['vertices'][from_id]
        target = new_state['vertices'][to_id]
        target['stack'] = source['stack']
        target['energy'] = source['energy']
        source['stack'] = []
        source['energy'] = 0
        new_state['turn']['hasMoved'] = True

    elif move_type == 'attack':
        attacker_id = move['vertexId']
        defender_id = move['targetId']
        attacker = new_state['vertices'][attacker_id]
        defender = new_state['vertices'][defender_id]
        
        # Use get_force for strength comparison
        attacker_strength = get_force(attacker)
        defender_strength = get_force(defender)
        
        att_pieces = len(attacker['stack'])
        att_energy = attacker.get('energy', 0)
        def_pieces = len(defender['stack'])
        def_energy = defender.get('energy', 0)

        if attacker_strength > defender_strength:
            # Attacker wins
            new_pieces = abs(att_pieces - def_pieces)
            new_energy = abs(att_energy - def_energy)
            
            # Move attacker to defender vertex (trimmed)
            defender['stack'] = attacker['stack'][:new_pieces]
            defender['energy'] = new_energy
            
            attacker['stack'] = []
            attacker['energy'] = 0
            
        elif defender_strength > attacker_strength:
            # Defender wins
            new_pieces = abs(def_pieces - att_pieces)
            new_energy = abs(def_energy - att_energy)
            
            # Defender remains (trimmed)
            defender['stack'] = defender['stack'][:new_pieces]
            defender['energy'] = new_energy
            
            attacker['stack'] = []
            attacker['energy'] = 0
            
        else:
            # Draw / Equal Force
            new_att_pieces = max(0, att_pieces - def_pieces)
            new_att_energy = max(0, att_energy - def_energy)
            
            new_def_pieces = max(0, def_pieces - att_pieces)
            new_def_energy = max(0, def_energy - att_energy)
            
            attacker['stack'] = attacker['stack'][:new_att_pieces]
            attacker['energy'] = new_att_energy
            
            defender['stack'] = defender['stack'][:new_def_pieces]
            defender['energy'] = new_def_energy
            
            # Return early to avoid common cleanup (which clears attacker)
            new_state['currentPlayerId'] = 'Player2' if current_player == 'Player1' else 'Player1'
            new_state['players'][new_state['currentPlayerId']]['reinforcements'] += 1
            new_state['turn'] = {
                'hasPlaced': False,
                'hasInfused': False,
                'hasMoved': False,
                'turnNumber': new_state['turn']['turnNumber'] + 1
            }
            return new_state

        # Common cleanup for Win/Loss cases (turn end)
        new_state['currentPlayerId'] = 'Player2' if current_player == 'Player1' else 'Player1'
        new_state['players'][new_state['currentPlayerId']]['reinforcements'] += 1
        new_state['turn'] = {
            'hasPlaced': False,
            'hasInfused': False,
            'hasMoved': False,
            'turnNumber': new_state['turn']['turnNumber'] + 1
        }
        return new_state

    elif move_type == 'pincer':
        target_id = move.get('targetId')
        origin_ids = move.get('originIds', [])
        
        defender = new_state['vertices'].get(target_id)
        origin_verts = [new_state['vertices'][oid] for oid in origin_ids if oid in new_state['vertices']]
        
        if not defender or not origin_verts:
            return new_state

        # Calculate combined attacker stats
        attacker_force = 1.0
        for v in origin_verts:
            attacker_force *= get_force(v)
        attacker_force = min(attacker_force, FORCE_CAP_MAX)
        
        defender_force = get_force(defender)
        
        attacker_pieces = sum(len(v['stack']) for v in origin_verts)
        attacker_energy = sum(v.get('energy', 0) for v in origin_verts)
        
        defender_pieces = len(defender['stack'])
        defender_energy = defender.get('energy', 0)
        
        new_pieces = abs(attacker_pieces - defender_pieces)
        new_energy = abs(attacker_energy - defender_energy)
        
        if attacker_force > defender_force:
            # Attacker wins
            defender['stack'] = []
            for i in range(new_pieces):
                defender['stack'].append({
                    'id': f"p-conquer-{i}",
                    'player': current_player
                })
            defender['energy'] = new_energy
        else:
            # Defender wins/holds
            defender_owner = defender['stack'][0]['player'] if defender['stack'] else current_player
            defender['stack'] = []
            for i in range(new_pieces):
                defender['stack'].append({
                    'id': f"p-defend-{i}",
                    'player': defender_owner
                })
            defender['energy'] = new_energy
            
        # Clear all origin vertices
        for v in origin_verts:
            v['stack'] = []
            v['energy'] = 0

        # End turn
        new_state['currentPlayerId'] = 'Player2' if current_player == 'Player1' else 'Player1'
        new_state['players'][new_state['currentPlayerId']]['reinforcements'] += 1
        new_state['turn'] = {
            'hasPlaced': False,
            'hasInfused': False,
            'hasMoved': False,
            'turnNumber': new_state['turn']['turnNumber'] + 1
        }

    elif move_type == 'endTurn':
        new_state['currentPlayerId'] = 'Player2' if current_player == 'Player1' else 'Player1'
        new_state['players'][new_state['currentPlayerId']]['reinforcements'] += 1
        new_state['turn'] = {
            'hasPlaced': False,
            'hasInfused': False,
            'hasMoved': False,
            'turnNumber': new_state['turn']['turnNumber'] + 1
        }

    # Check winner (control opponent corners)
    for player in ['Player1', 'Player2']:
        opponent = 'Player2' if player == 'Player1' else 'Player1'
        opponent_corners = new_state['homeCorners'][opponent]
        if opponent_corners and all(
            new_state['vertices'][cid]['stack'] and new_state['vertices'][cid]['stack'][0]['player'] == player
            for cid in opponent_corners
        ):
            new_state['winner'] = player
            break

    return new_state


def evaluate_position(state: Dict[str, Any], perspective: Optional[str] = None) -> float:
    """
    Simple evaluation used by the greedy agent and as a heuristic for A*.
    Positive favors perspective (defaults to state's currentPlayerId).
    """
    score = 0.0
    current_player = perspective if perspective is not None else state.get('currentPlayerId')
    vertices = state.get('vertices', {})
    for vertex in vertices.values():
        if vertex.get('stack'):
            owner = vertex['stack'][0]['player']
            piece_count = len(vertex['stack'])
            energy = vertex.get('energy', 0)
            value = piece_count * 10 + energy * 15
            if owner == current_player:
                score += value
            else:
                score -= value
    return score
