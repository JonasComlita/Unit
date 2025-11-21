from typing import Dict, Any, Optional, Tuple
from .agent_utils import get_legal_moves, apply_move
from .heuristics import evaluate_combined
from .dynamic_agent import select_weights

MAX_DEPTH = 2  # Default depth (can be increased for stronger play but slower)

def alpha_beta(state: Dict[str, Any], depth: int, alpha: float, beta: float, maximizing_player: bool, perspective: str) -> float:
    """
    Alpha-Beta pruning algorithm.
    """
    # Leaf node or terminal state
    winner = state.get('winner')
    if winner:
        if winner == perspective:
            return 100000.0 + depth # Prefer winning sooner
        else:
            return -100000.0 - depth # Prefer losing later
            
    if depth == 0:
        # Use dynamic weights for evaluation
        weights = select_weights(state, perspective)
        return evaluate_combined(state, perspective, weights)

    legal_moves = get_legal_moves(state)
    if not legal_moves:
        # No moves = end turn (effectively a pass, or game over if no moves possible)
        # If it's a pass, we just evaluate the state as is, or simulate a pass.
        # For simplicity, treat as leaf.
        weights = select_weights(state, perspective)
        return evaluate_combined(state, perspective, weights)

    # Move ordering could go here (e.g. try attacks first)
    
    if maximizing_player:
        max_eval = -float('inf')
        for move in legal_moves:
            new_state = apply_move(state, move)
            # Next level is minimizing (opponent)
            # Note: apply_move updates currentPlayerId, so we check if it changed
            # Usually it changes, but sometimes (like extra turn mechanics) it might not.
            # In this game, turns alternate strictly except for some edge cases? 
            # Actually apply_move handles turn switching.
            # If next player is still us (e.g. bonus turn), we are still maximizing.
            # But standard game is alternating.
            
            next_player = new_state['currentPlayerId']
            is_next_maximizing = (next_player == perspective)
            
            eval_val = alpha_beta(new_state, depth - 1, alpha, beta, is_next_maximizing, perspective)
            max_eval = max(max_eval, eval_val)
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            new_state = apply_move(state, move)
            next_player = new_state['currentPlayerId']
            is_next_maximizing = (next_player == perspective)
            
            eval_val = alpha_beta(new_state, depth - 1, alpha, beta, is_next_maximizing, perspective)
            min_eval = min(min_eval, eval_val)
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval

def select_move(game_state: Dict[str, Any], perspective: Optional[str] = None) -> Dict:
    mover = perspective or game_state.get('currentPlayerId')
    legal = get_legal_moves(game_state)
    if not legal:
        return {'type': 'endTurn'}
        
    best_move = None
    best_val = -float('inf')
    alpha = -float('inf')
    beta = float('inf')
    
    # Root level search
    for move in legal:
        new_state = apply_move(game_state, move)
        
        # Check if next state is still our turn (rare but possible in some games)
        next_player = new_state['currentPlayerId']
        is_next_maximizing = (next_player == mover)
        
        # Start recursive search
        val = alpha_beta(new_state, MAX_DEPTH - 1, alpha, beta, is_next_maximizing, mover)
        
        if val > best_val:
            best_val = val
            best_move = move
        
        alpha = max(alpha, val)
        
    return best_move if best_move is not None else {'type': 'endTurn'}
