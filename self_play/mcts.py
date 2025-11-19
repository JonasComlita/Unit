import math
import random
import copy
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .agent_utils import get_legal_moves, apply_move, evaluate_position

logger = logging.getLogger(__name__)

class MCTSNode:
    """Node in the MCTS tree."""
    __slots__ = ('parent', 'move', 'children', 'visits', 'value_sum', 'is_expanded', 'player', 'prior')

    def __init__(self, parent=None, move=None, player=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.player = player
        self.children: Dict[str, MCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.prior = prior

    def q(self) -> float:
        return (self.value_sum / self.visits) if self.visits > 0 else 0.0

    def uct(self, c_param: float = 1.414) -> float:
        if self.visits == 0:
            # If unvisited, use prior to break ties or encourage exploration
            # Standard PUCT: Q + c * P * sqrt(N) / (1 + n)
            # If Q is 0, it relies entirely on prior.
            # To ensure unvisited nodes are tried, we can return a large value or rely on the formula.
            # With c=1.414 and small visits, this might be small.
            # AlphaZero uses Q + U.
            pass
        
        # AlphaZero PUCT formula
        # U = c_param * prior * sqrt(parent_visits) / (1 + visits)
        u = c_param * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return self.q() + u


class MCTSAgent:
    """
    Monte Carlo Tree Search Agent.
    
    Supports configurable simulations and rollout depth.
    Can use a custom evaluation function (e.g. neural network) or default heuristic.
    """
    def __init__(self, simulations: int = 100, rollout_depth: int = 3, c_param: float = 1.414, eval_fn=None):
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.c_param = c_param
        self.eval_fn = eval_fn

    def select_move(self, state: Dict[str, Any], temperature: float = 0.0) -> Dict[str, Any]:
        """
        Select the best move for the given state using MCTS.
        
        Args:
            state: The current game state.
            temperature: Controls diversity. 
                         0.0 = Greedy (Max visits).
                         1.0 = Sample proportional to visits.
                         >1.0 = More random.
        """
        root = MCTSNode(player=state['currentPlayerId'])
        
        # If only one move is possible, take it immediately
        legal_moves = get_legal_moves(state)
        if not legal_moves:
            return {'type': 'endTurn'}
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Expand root immediately to add noise
        self._expand(root, state)
        self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.simulations):
            node = root
            sim_state = copy.deepcopy(state)

            # 1. Selection
            while node.is_expanded and node.children:
                node = self._select_child(node)
                sim_state = apply_move(sim_state, node.move)

            # 2. Expansion
            if not node.is_expanded and not sim_state.get('winner'):
                self._expand(node, sim_state)
                if node.children: 
                    # Pick a child to rollout from (based on priors?)
                    # For simple MCTS, just pick random or first.
                    # Let's pick based on priors (which are uniform here unless model used)
                    node = list(node.children.values())[0] 
                    sim_state = apply_move(sim_state, node.move)

            # 3. Simulation (Rollout) & Evaluation
            value = self._simulate(sim_state)

            # 4. Backpropagation
            self._backpropagate(node, value, root.player)

        # Select move based on temperature
        children = list(root.children.values())
        if not children:
            return {'type': 'endTurn'} # Should not happen if legal_moves > 0

        if temperature == 0.0:
            # Greedy selection (max visits)
            best_child = max(children, key=lambda c: c.visits)
            return best_child.move
        else:
            # Stochastic selection
            visits = np.array([c.visits for c in children])
            # Avoid divide by zero if visits are 0 (shouldn't happen for all if sims > 0)
            if visits.sum() == 0:
                return random.choice(children).move
            
            # Exponentiate by 1/temp
            visits_powered = visits ** (1.0 / temperature)
            probs = visits_powered / visits_powered.sum()
            
            selected_child = np.random.choice(children, p=probs)
            return selected_child.move

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCT value."""
        return max(node.children.values(), key=lambda c: c.uct(self.c_param))

    def _expand(self, node: MCTSNode, state: Dict[str, Any]):
        """Expand the node by adding all legal moves as children."""
        legal_moves = get_legal_moves(state)
        num_moves = len(legal_moves)
        for move in legal_moves:
            move_key = str(move) 
            if move_key not in node.children:
                # Initialize with uniform prior if no policy provided
                # If we had a policy network, we'd set it here.
                prior = 1.0 / num_moves if num_moves > 0 else 0.0
                child = MCTSNode(parent=node, move=move, prior=prior)
                node.children[move_key] = child
        node.is_expanded = True

    def _add_dirichlet_noise(self, node: MCTSNode, epsilon: float = 0.25, alpha: float = 0.3):
        """
        Add Dirichlet noise to the priors of the children nodes.
        This encourages exploration of different moves at the root.
        """
        children = list(node.children.values())
        if not children:
            return
        
        noise = np.random.dirichlet([alpha] * len(children))
        
        for i, child in enumerate(children):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def _simulate(self, state: Dict[str, Any]) -> float:
        """
        Simulate/Evaluate the state.
        If eval_fn is provided, use it directly on the leaf (AlphaZero style).
        Otherwise, perform a rollout and use heuristic.
        """
        if self.eval_fn:
            # Custom evaluator (e.g. Neural Net)
            # Expected to return value in [-1, 1] from perspective of current player?
            # Or absolute? Let's assume eval_fn returns value for 'Player1'.
            return self.eval_fn(state)
        
        # Default Rollout + Heuristic
        curr_state = copy.deepcopy(state)
        for _ in range(self.rollout_depth):
            if curr_state.get('winner'):
                break
            
            legal_moves = get_legal_moves(curr_state)
            if not legal_moves:
                break 
            
            import random
            move = random.choice(legal_moves)
            curr_state = apply_move(curr_state, move)

        # Heuristic evaluation (Player1 perspective)
        score = evaluate_position(curr_state, perspective='Player1')
        return math.tanh(score / 100.0) 

    def _backpropagate(self, node: MCTSNode, value: float, root_player: str):
        """Update node statistics up the tree."""
        while node is not None:
            node.visits += 1
            
            # Value is assumed to be "Advantage for Player1"
            if node.parent:
                if node.parent.player == 'Player1':
                    node.value_sum += value
                else:
                    node.value_sum -= value
            else:
                pass
                
            node = node.parent
