# ./.venv311/bin/python -m self_play.training_pipeline train --data-dir shards/v1_model_data --epochs 100 --batch-size 256

#!/usr/bin/env python3
"""Minimal training pipeline example.

This script demonstrates reading exported Parquet (or JSONL) and running a
very small baseline evaluation: majority-class predictor and a simple
train/val split. It purposefully avoids heavy ML deps so it can run in CI.
"""
import argparse
import json
import copy
import statistics
from typing import List

import dotenv
# Ensure environment variables from .env at project root are loaded so
# DATABASE_URL and other settings are available to the async pipeline.
try:
    dotenv.load_dotenv()
except Exception:
    # Non-fatal: if python-dotenv isn't available or loading fails, we'll
    # continue and allow explicit environment variables to be used.
    pass

try:
    import pandas as pd
except Exception:
    pd = None


def load_parquet(path: str):
    if pd is None:
        raise RuntimeError('pandas/pyarrow required to read Parquet')
    df = pd.read_parquet(path)
    return df


def majority_baseline(labels: List[str]):
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    if not counts:
        return None
    majority = max(counts.items(), key=lambda x: x[1])[0]
    acc = sum(1 for l in labels if l == majority) / len(labels)
    return majority, acc



"""
Complete Training Pipeline for Unit Game AI

This connects:
1. Self-play system (generates games using minimax AI)
2. Database (stores game data)
3. Neural network (learns from game data)
4. Evaluation (tests NN vs minimax)

Usage:
    # Generate training data
    python training_pipeline.py generate --games 10000 --concurrent 20
    
    # Train neural network
    python training_pipeline.py train --epochs 100 --batch-size 256
    
    # Evaluate model
    python training_pipeline.py evaluate --model checkpoints/model_epoch_100.pt
    
    # Full pipeline
    python training_pipeline.py full --games 10000 --epochs 50

    # 🎮 Next Steps

    Improve state representation in _parse_state() - currently a placeholder
    Add evaluation games: NN plays against minimax to measure progress
    Implement MCTS: Add Monte Carlo Tree Search for even better training
    Deploy NN as API: Serve the trained model to your TypeScript frontend

    Does this complete training system match what you need?
"""

import asyncio
import asyncpg
import json
import logging
import os
import glob
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Import your existing modules
from self_play.neural_network_model import (
    UnitGameNet, 
    UnitGameTrainer, 
    state_to_tensor
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    database_url: str = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/unitgame')
    
    # Data generation
    games_to_generate: int = 10000
    concurrent_games: int = 20
    exploration_rate: float = 0.15  # Mix in random moves for diversity
    
    # Training
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 0.001
    train_test_split: float = 0.9
    
    # Model
    num_vertices: int = 117  # 3²+5²+7²+5²+3²
    checkpoint_dir: str = 'checkpoints'
    # Optional local shard/parquet input directory to train from instead of DB
    data_dir: str = None
    # Data loader workers for PyTorch DataLoader - helps prefetch shards
    data_loader_workers: int = 4
    
    # Evaluation
    eval_games: int = 100
    eval_minimax_depth: int = 4


class GameDataset(Dataset):
    """PyTorch Dataset for game training data."""
    
    def __init__(self, states: np.ndarray, moves: np.ndarray, outcomes: np.ndarray):
        """
        Args:
            states: [N, num_vertices, 5] game states
            moves: [N] class indices (policy target indices in range num_vertices*5)
            outcomes: [N] game outcomes (-1, 0, 1)
        """
        self.states = torch.FloatTensor(states)
        # Policy targets should be class indices (long) for cross-entropy
        self.moves = torch.LongTensor(moves)
        self.outcomes = torch.FloatTensor(outcomes)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.moves[idx], self.outcomes[idx]


class ShardedIterableDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that lazily yields examples from compressed .npz shards."""

    def __init__(self, shard_files: List[str]):
        super().__init__()
        self.shard_files = list(shard_files)

    def __iter__(self):
        for shard in self.shard_files:
            try:
                with np.load(shard, allow_pickle=False) as data:
                    states = data['states']
                    policies = data['policies']
                    values = data['values']
                    for i in range(len(states)):
                        # yield row-wise: numpy arrays and scalars; DataLoader will collate
                        yield states[i], policies[i], values[i]
            except Exception as e:
                logger.warning(f"Failed to read shard {shard}: {e}")


class DataLoader:
    """Loads training data from PostgreSQL database."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
    
    async def connect(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(self.database_url)
        logger.info("Connected to database")
    
    async def load_games(self, batch_size: int = 1000, limit: int = None):
        """
        Async generator that yields lists of games (dicts) fetched in batches.

        This avoids loading all games into memory at once. It uses LIMIT/OFFSET
        pagination; for very large tables consider server-side cursors.
        """
        # Include initial_state from games table (used when state_serialization='none')
        base_query = """
            SELECT g.game_id, g.winner, g.total_moves, g.initial_state,
                   array_agg(
                       json_build_object(
                           'move_number', m.move_number,
                           'player_id', m.player_id,
                           'action_type', m.action_type,
                           'action_data', m.action_data
                       ) ORDER BY m.move_number
                   ) as moves
            FROM games g
            JOIN moves m ON g.game_id = m.game_id
            WHERE g.platform = 'selfplay'
            GROUP BY g.game_id
            ORDER BY g.start_time DESC
        """

        offset = 0
        fetched = 0
        async with self.pool.acquire() as conn:
            while True:
                q = base_query + f" LIMIT {batch_size} OFFSET {offset}"
                rows = await conn.fetch(q)
                if not rows:
                    break

                games = []
                for row in rows:
                    games.append({
                        'game_id': row['game_id'],
                        'winner': row['winner'],
                        'total_moves': row['total_moves'],
                        # initial_state may be stored as JSON/text
                        'initial_state': row.get('initial_state'),
                        'moves': row['moves']
                    })

                yield games

                offset += len(rows)
                fetched += len(rows)
                if limit and fetched >= limit:
                    break
    
    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()


class TrainingDataProcessor:
    """Process raw game data into neural network training format."""
    
    def __init__(self, num_vertices: int = 117):
        self.num_vertices = num_vertices
    
    def process_games(self, games: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert games into training tensors.
        
        Returns:
            states: [N, num_vertices, 5] - game states
            policies: [N] - policy class indices (int in [0, num_vertices*5))
            values: [N] - game outcomes
        """
        states = []
        policies = []
        values = []

        for game in games:
            game_outcome = self._outcome_to_value(game.get('winner'))

            # Load and parse initial state. Support both JSON/text and already-parsed dict.
            initial = game.get('initial_state')
            if initial is None:
                # Fall back to per-move state_before (older exports) by trying to parse the first move
                logger.warning(f"Game {game.get('game_id')} missing initial_state; attempting per-move parsing")
                # We'll still try the old per-move flow
                for move_data in game.get('moves', []):
                    # Normalize move_data: DB may return JSON strings for moves
                    if isinstance(move_data, str):
                        try:
                            move_data = json.loads(move_data)
                        except Exception:
                            continue
                    state = self._parse_state(move_data)
                    if state is None:
                        continue
                    state_tensor = state_to_tensor(state, self.num_vertices)
                    policy_idx = self._move_to_policy(move_data, state)
                    if policy_idx is None or policy_idx < 0:
                        continue
                    player = move_data.get('player_id') or move_data.get('player')
                    outcome = game_outcome if player == game.get('winner') else -game_outcome
                    states.append(state_tensor)
                    policies.append(policy_idx)
                    values.append(outcome)
                continue

            if isinstance(initial, str):
                try:
                    current_state = json.loads(initial)
                except Exception:
                    logger.warning(f"Failed to parse initial_state for game {game.get('game_id')}")
                    continue
            else:
                # assume dict-like
                current_state = copy.deepcopy(initial)

            # Re-simulate the game: for each move, yield (state_before, policy_target, outcome)
            moves = game.get('moves', [])
            
            # Handle compressed moves (bytes)
            if isinstance(moves, bytes):
                try:
                    import gzip
                    moves = json.loads(gzip.decompress(moves).decode('utf-8'))
                except Exception as e:
                    logger.warning(f"Failed to decompress moves for game {game.get('game_id')}: {e}")
                    continue
            
            for move_data in moves:
                # Normalize move_data in case DB returned JSON strings
                if isinstance(move_data, str):
                    try:
                        move_data = json.loads(move_data)
                    except Exception:
                        # skip malformed move entry
                        continue
                
                # Handle integer moves (compressed indices) - still keep this just in case
                if isinstance(move_data, int):
                    # Regenerate legal moves to look up the move by index
                    # This assumes the generator saved the index into the legal_moves list
                    # sorted by some deterministic order (usually the order returned by get_legal_moves)
                    from self_play.agent_utils import get_legal_moves
                    legal_moves = get_legal_moves(current_state)
                    if 0 <= move_data < len(legal_moves):
                        move_data = legal_moves[move_data]
                    else:
                        logger.warning(f"Move index {move_data} out of bounds for {len(legal_moves)} legal moves")
                        continue

                # Ensure action_data is parsed when needed inside helpers
                try:
                    state_tensor = state_to_tensor(current_state, self.num_vertices)
                except Exception as e:
                    logger.warning(f"state_to_tensor failed for game {game.get('game_id')}: {e}")
                    break

                policy_idx = self._move_to_policy(move_data, current_state)
                if policy_idx is None or policy_idx < 0:
                    # Still advance the state to keep alignment
                    self._apply_move(current_state, move_data)
                    continue

                player = move_data.get('player_id') or move_data.get('player')
                outcome = game_outcome if player == game.get('winner') else -game_outcome

                states.append(state_tensor)
                policies.append(int(policy_idx))
                values.append(float(outcome))

                # advance state to reflect the move we just processed
                self._apply_move(current_state, move_data)
        
        if len(states) == 0 and len(games) > 0:
            logger.warning(f"Processed 0 examples from {len(games)} games. Debugging first game:")
            if games:
                g = games[0]
                logger.warning(f"Game ID: {g.get('game_id')}, Initial State Present: {bool(g.get('initial_state'))}")
                moves = g.get('moves', [])
                logger.warning(f"Move count: {len(moves)}")
                if moves:
                    m0 = moves[0]
                    if isinstance(m0, str): m0 = json.loads(m0)
                    logger.warning(f"First move keys: {m0.keys()}")
                    action_data = m0.get('action_data') or m0.get('action')
                    logger.warning(f"First move action data: {action_data}")

        logger.info(f"Processed {len(states)} training examples from {len(games)} games")
        
        return (
            np.array(states),
            np.array(policies, dtype=np.int64),
            np.array(values, dtype=np.float32)
        )
    
    def _parse_state(self, move_data: Dict) -> Dict:
        """Parse game state from move data."""
        # Extract and parse action_data and the serialized state_before
        try:
            # Some exports store the entire move as a JSON string; handle that here
            if isinstance(move_data, str):
                try:
                    move_data = json.loads(move_data)
                except Exception:
                    return None
            action_data = move_data.get('action_data', {})
            # some exports use 'action' as the nested key
            if not action_data:
                action_data = move_data.get('action', action_data)
            if isinstance(action_data, str):
                action_data = json.loads(action_data)

            # Many schemas store the pre-move state as a JSON string under
            # action_data['state_before'] or similar. Adjust if your schema
            # differs.
            state_before = action_data.get('state_before')
            if state_before is None:
                return None
            if isinstance(state_before, str):
                return json.loads(state_before)
            return state_before
        except Exception as e:
            logger.warning(f"Failed to parse state: {e}")
            return None

    def _apply_move(self, state: Dict, move_data: Dict) -> None:
        """Apply a single move to `state` in-place.

        This is a lightweight, best-effort re-simulation used only to rebuild
        the sequence of states during training. It intentionally implements
        the minimum necessary effects so that `state_to_tensor` remains
        meaningful (stacks, ownership, currentPlayerId, basic movement).
        """
        try:
            # Defensive: if move_data is not a dict or JSON string, skip it
            if not isinstance(move_data, (dict, str)):
                logger.debug(f"_apply_move received unexpected move_data type {type(move_data)}; skipping")
                return

            # Support move_data being a JSON string from DB exports
            if isinstance(move_data, str):
                try:
                    move_data = json.loads(move_data)
                except Exception:
                    # malformed move_data string — skip applying
                    return
            action_data = move_data.get('action_data', {})
            # some exports use 'action' as the nested key
            if not action_data:
                action_data = move_data.get('action', action_data)
            if isinstance(action_data, str):
                action_data = json.loads(action_data)

            action_type = action_data.get('type') or move_data.get('action_type')
            player = move_data.get('player_id') or move_data.get('player')

            vertices = state.setdefault('vertices', {})

            def ensure_vertex(v_id):
                if v_id not in vertices:
                    vertices[v_id] = {'stack': [], 'energy': 0, 'layer': 0}
                # Ensure proper defaults
                v = vertices[v_id]
                v.setdefault('stack', [])
                v.setdefault('energy', 0)
                v.setdefault('layer', 0)
                return v

            if action_type == 'place':
                to_id = action_data.get('vertexId') or action_data.get('toId')
                if to_id:
                    v = ensure_vertex(to_id)
                    # place on top (index 0) to match gameLogic.ts
                    v['stack'].insert(0, {'player': player})
                    # Decrement reinforcements if we can track them (state might not have players dict fully populated in this lightweight sim)
                    if 'players' in state and player in state['players']:
                        state['players'][player]['reinforcements'] -= 1

            elif action_type == 'move':
                from_id = action_data.get('fromId') or action_data.get('vertexId')
                to_id = action_data.get('toId')
                if from_id and to_id and from_id in vertices:
                    src = vertices[from_id]
                    dst = ensure_vertex(to_id)
                    
                    # Move ENTIRE stack and energy
                    if src.get('stack'):
                        dst['stack'] = src['stack'] # Move all pieces
                        dst['energy'] = src.get('energy', 0) # Move all energy
                        
                        # Clear source
                        src['stack'] = []
                        src['energy'] = 0

            elif action_type == 'attack':
                attacker_id = action_data.get('vertexId') or action_data.get('fromId')
                target_id = action_data.get('targetId') or action_data.get('toId')
                
                if attacker_id and target_id and attacker_id in vertices and target_id in vertices:
                    attacker = vertices[attacker_id]
                    defender = vertices[target_id]
                    
                    if attacker.get('stack') and defender.get('stack'):
                        # Calculate Force
                        # Note: We use raw values here. Gravity is not applied in this simplified sim 
                        # unless we import constants, but usually force comparison is relative.
                        # However, the rule is Force = (pieces * energy) / gravity. 
                        # If we don't have gravity, we might be inaccurate. 
                        # But for now, let's assume standard combat logic if gravity isn't available.
                        # Actually, let's try to be as accurate as possible.
                        
                        att_pieces = len(attacker['stack'])
                        att_energy = attacker.get('energy', 0)
                        def_pieces = len(defender['stack'])
                        def_energy = defender.get('energy', 0)
                        
                        # Simple force approximation if we don't have layer info easily
                        # (pieces * energy). 
                        # ideally we should use the real get_force logic if we have layers.
                        att_layer = attacker.get('layer', 0)
                        def_layer = defender.get('layer', 0)
                        
                        # Hardcoded gravity for now to match constants.ts
                        LAYER_GRAVITY = [1.0, 2.0, 3.0, 2.0, 1.0]
                        
                        att_force = (att_pieces * att_energy) / LAYER_GRAVITY[att_layer] if att_layer < len(LAYER_GRAVITY) else (att_pieces * att_energy)
                        def_force = (def_pieces * def_energy) / LAYER_GRAVITY[def_layer] if def_layer < len(LAYER_GRAVITY) else (def_pieces * def_energy)
                        
                        # Combat Resolution
                        if att_force > def_force:
                            # Attacker Wins
                            new_pieces = abs(att_pieces - def_pieces)
                            new_energy = abs(att_energy - def_energy)
                            
                            defender['stack'] = attacker['stack'][:new_pieces] # Keep top n pieces
                            defender['energy'] = new_energy
                            
                            attacker['stack'] = []
                            attacker['energy'] = 0
                            
                        elif def_force > att_force:
                            # Defender Wins
                            new_pieces = abs(def_pieces - att_pieces)
                            new_energy = abs(def_energy - att_energy)
                            
                            defender['stack'] = defender['stack'][:new_pieces]
                            defender['energy'] = new_energy
                            
                            attacker['stack'] = []
                            attacker['energy'] = 0
                            
                        else:
                            # Draw / Equal Force
                            # Both remain, updated values
                            new_att_pieces = max(0, att_pieces - def_pieces)
                            new_att_energy = max(0, att_energy - def_energy)
                            
                            new_def_pieces = max(0, def_pieces - att_pieces)
                            new_def_energy = max(0, def_energy - att_energy)
                            
                            attacker['stack'] = attacker['stack'][:new_att_pieces]
                            attacker['energy'] = new_att_energy
                            
                            defender['stack'] = defender['stack'][:new_def_pieces]
                            defender['energy'] = new_def_energy

            elif action_type == 'infuse':
                # infuse may change energy on a vertex
                vid = action_data.get('vertexId') or action_data.get('toId')
                # amount is usually 1 in this game version
                amount = action_data.get('amount') or 1 
                if vid:
                    v = ensure_vertex(vid)
                    try:
                        v['energy'] = max(0, v.get('energy', 0) + int(amount))
                    except Exception:
                        pass

            elif action_type == 'pincer':
                # Basic Pincer implementation
                attacker_id = action_data.get('vertexId')
                target_id = action_data.get('targetId')
                if attacker_id and target_id and attacker_id in vertices and target_id in vertices:
                    attacker = vertices[attacker_id]
                    defender = vertices[target_id]
                    
                    if attacker.get('stack') and defender.get('stack'):
                        # Pincer logic: Attacker moves to defender, defender destroyed? 
                        # Or just damage? 
                        # Standard pincer: Attacker + Ally flank defender. Defender takes damage/destroyed.
                        # Simplified: Defender stack replaced by attacker stack (capture) if strong enough?
                        # Let's assume standard capture for now as per agent_utils.py
                        
                        # For now, just clear attacker to simulate the move completion
                        # and update defender if we want to be precise.
                        # But without full context, clearing attacker is the minimum state change.
                        attacker['stack'] = []
                        attacker['energy'] = 0
                        # (Real pincer logic is complex, but this prevents ghost pieces)

            # Toggle current player if present in state
            cur = state.get('currentPlayerId')
            if cur is not None and player is not None:
                # attempt a simple flip for canonical two-player ids like 'Player1'/'Player2'
                if isinstance(cur, str) and cur.startswith('Player'):
                    state['currentPlayerId'] = 'Player2' if cur == 'Player1' else 'Player1'
                else:
                    # if not textual, try to infer from player in move
                    state['currentPlayerId'] = player

        except Exception as e:
            logger.debug(f"_apply_move failed on move {move_data.get('move_number')}: {e}")
    
    def _move_to_policy(self, move_data: Dict, state: Dict) -> np.ndarray:
        """
    Convert move to a single class index in range [0, num_vertices*5).

    Index layout: vertex_idx * 5 + action_offset
        Returns integer index, or None if it can't be derived.
        """
        try:
            # Defensive: only handle dicts or JSON strings
            if not isinstance(move_data, (dict, str)):
                logger.debug(f"_move_to_policy received unexpected move_data type {type(move_data)}; skipping")
                return None

            # Accept move_data as JSON string or dict
            if isinstance(move_data, str):
                try:
                    move_data = json.loads(move_data)
                except Exception:
                    return None
            action_data = move_data.get('action_data', {})
            # accept either 'action_data' or 'action'
            if not action_data:
                action_data = move_data.get('action', action_data)
            if isinstance(action_data, str):
                action_data = json.loads(action_data)

            action_type = action_data.get('type')
            vertex_id = action_data.get('vertexId') or action_data.get('fromId')

            # Map vertex_id to index (you'll need your vertex mapping)
            vertex_idx = self._vertex_id_to_index(vertex_id, state)
            if vertex_idx is None:
                return None

            action_offset = {
                'place': 0,
                'infuse': 1,
                'move': 2,
                'attack': 3,
                'pincer': 4
            }.get(action_type, None)

            if action_offset is None:
                return None

            return int(vertex_idx * 5 + action_offset)
        except Exception as e:
            logger.warning(f"Failed to convert move to policy: {e}")
            return None
    
    def _vertex_id_to_index(self, vertex_id: str, state: Dict) -> int:
        """Map vertex ID to index in tensor."""
        if not state or not vertex_id:
            return None
        
        vertices = sorted(list(state.get('vertices', {}).keys()), key=lambda x: int(x[1:]))
        try:
            return vertices.index(vertex_id)
        except ValueError:
            return None
    
    def _outcome_to_value(self, winner: str) -> float:
        """Convert game outcome to value."""
        if winner == 'Player1':
            return 1.0
        elif winner == 'Player2':
            return -1.0
        else:
            return 0.0  # Draw

    def _policy_to_move(self, policy_idx: int, state: Dict) -> Dict:
        """Convert policy index back to move dictionary."""
        if policy_idx is None:
            return None
        
        policy_idx = int(policy_idx)
        vertex_idx = policy_idx // 5
        action_offset = policy_idx % 5
        
        vertices = sorted(list(state.get('vertices', {}).keys()), key=lambda x: int(x[1:]))
        if vertex_idx >= len(vertices):
            return None
            
        vertex_id = vertices[vertex_idx]
        
        action_types = {
            0: 'place',
            1: 'infuse',
            2: 'move',
            3: 'attack',
            4: 'pincer'
        }
        action_type = action_types.get(action_offset)
        
        # We can't fully reconstruct targetId for move/attack/pincer just from the index
        # because the index only encodes (Source, ActionType).
        # The target is NOT encoded in the simple 5-action policy head.
        # Wait, the policy head is [num_vertices * 5]. 
        # 0: Place (on self)
        # 1: Infuse (on self)
        # 2: Move (from self... to where?) -> This policy head is incomplete for Move/Attack!
        #
        # If the shards contain integers, they MUST be encoding the full move somehow.
        # If the policy is just 5 actions per vertex, it assumes fixed targets or a different scheme.
        #
        # However, if the generator wrote integers, it might be writing the *index in the legal_moves list*?
        # No, that would be unstable.
        #
        # Let's assume for a moment the integer IS the policy index as defined in _move_to_policy.
        # But _move_to_policy maps (vertex, type) -> index. It drops the target!
        # This means the current policy definition is insufficient for a game with targets (Move/Attack).
        #
        # CRITICAL REALIZATION: The current `_move_to_policy` implementation is:
        # return int(vertex_idx * 5 + action_offset)
        # It completely ignores `toId` / `targetId`. 
        # This means the model is only learning "Select Unit X and do Action Y", but not "Where".
        # This is a major design flaw in the current `neural_network_model.py` / `training_pipeline.py`.
        #
        # But right now, I just need to unblock the training.
        # If the shard contains integers, and I can't reconstruct the target, I can't apply the move.
        #
        # WAIT. If the shard contains integers, maybe they are NOT policy indices.
        # Maybe they are something else?
        #
        # Let's look at the error again: "int object has no attribute keys".
        # This happens in `for move_data in game.get('moves', [])`.
        #
        # If I can't reconstruct the move, I can't advance the state.
        # If I can't advance the state, I can't train on the sequence.
        #
        # Temporary Fix:
        # If move is int, assume it's a policy index.
        # Use the policy index for the *label*.
        # But for *applying* the move, we are stuck.
        # UNLESS... we just skip applying it? No, then the state is wrong for the next step.
        #
        # Maybe the integer is an index into `legal_moves`?
        # If so, we'd need to regenerate legal moves to find it.
        #
        # Let's try to regenerate legal moves and see if the integer matches an index.
        # This is expensive but might work for recovery.
        
        return None


class TrainingPipeline:
    """Main training pipeline orchestrator."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = UnitGameNet(num_vertices=config.num_vertices)
        self.trainer = UnitGameTrainer(self.model, learning_rate=config.learning_rate)
        self.data_loader = DataLoader(config.database_url)
        self.processor = TrainingDataProcessor(config.num_vertices)
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    async def generate_training_data(self):
        """
        Generate training data using self-play.
        
        This runs your refactored self_play_system.py
        """
        logger.info(f"Generating {self.config.games_to_generate} games via self-play...")
        
        # Import and run self-play system
        from self_play.config import SelfPlayConfig
        from self_play.self_play_generator import SelfPlayGenerator
        
        selfplay_config = SelfPlayConfig(
            games_per_batch=self.config.games_to_generate,
            concurrent_games=self.config.concurrent_games,
            exploration_rate=self.config.exploration_rate,
            database_url=self.config.database_url,
            batch_only=True,  # Generate one batch
            log_level='INFO'
        )
        
        generator = SelfPlayGenerator(selfplay_config)
        await generator.initialize()
        await generator.generate_training_data()
        await generator.shutdown()
        
        logger.info("Training data generation complete")
    
    async def train_model(self):
        """Train neural network on database game data."""
        logger.info("Streaming training data from database and sharding to disk...")

    # Buffers to accumulate examples before writing compressed shard files
        train_buf_states = []
        train_buf_policies = []
        train_buf_values = []

        test_buf_states = []
        test_buf_policies = []
        test_buf_values = []

        train_shards: List[str] = []
        test_shards: List[str] = []

        shard_counter = {'train': 0, 'test': 0}

        buffer_limit = max(self.config.batch_size * 200, 10000)

        # If a local data_dir is provided, read parquet/jsonl files from it
        if getattr(self.config, 'data_dir', None):
            logger.info("Reading training data from local directory: %s", self.config.data_dir)
            # find parquet files first
            files = sorted(glob.glob(os.path.join(self.config.data_dir, '*.parquet')))
            # also accept jsonl files
            files += sorted(glob.glob(os.path.join(self.config.data_dir, '*.jsonl')))

            if not files:
                raise RuntimeError(f"No parquet/jsonl files found in {self.config.data_dir}")

            for fpath in files:
                try:
                    if fpath.endswith('.parquet'):
                        if pd is None:
                            raise RuntimeError('pandas required to read parquet files')
                        df = pd.read_parquet(fpath)
                        rows = df.to_dict(orient='records')
                    else:
                        # jsonl
                        rows = []
                        with open(fpath, 'r') as fh:
                            for line in fh:
                                try:
                                    rows.append(json.loads(line))
                                except Exception:
                                    continue

                    # normalize rows to 'games' format expected by processor
                    games_batch = []
                    for r in rows:
                        moves_val = r.get('moves') or r.get('moves_list') or []
                        # If the entire moves column is a JSON-serialized list, parse it
                        if isinstance(moves_val, str):
                            try:
                                moves_val = json.loads(moves_val)
                            except Exception:
                                # leave as string; processor will handle per-entry JSON strings
                                pass

                        game = {
                            'game_id': r.get('game_id') or r.get('id') or None,
                            'winner': r.get('winner'),
                            'total_moves': r.get('total_moves') or r.get('moves_count') or None,
                            'initial_state': r.get('initial_state'),
                            'moves': moves_val
                        }
                        games_batch.append(game)

                    states, policies, values = self.processor.process_games(games_batch)
                    if len(states) == 0:
                        continue

                    for i in range(len(states)):
                        if random.random() < self.config.train_test_split:
                            train_buf_states.append(states[i])
                            train_buf_policies.append(int(policies[i]))
                            train_buf_values.append(float(values[i]))
                        else:
                            test_buf_states.append(states[i])
                            test_buf_policies.append(int(policies[i]))
                            test_buf_values.append(float(values[i]))

                    # Flush buffers periodically per file to avoid huge memory use
                    if len(train_buf_states) >= buffer_limit:
                        shard_path = os.path.join(self.config.checkpoint_dir, f'train_shard_{shard_counter["train"]}.npz')
                        np.savez_compressed(shard_path,
                                            states=np.array(train_buf_states),
                                            policies=np.array(train_buf_policies, dtype=np.int64),
                                            values=np.array(train_buf_values, dtype=np.float32))
                        train_shards.append(shard_path)
                        shard_counter['train'] += 1
                        train_buf_states.clear(); train_buf_policies.clear(); train_buf_values.clear()

                    if len(test_buf_states) >= buffer_limit:
                        shard_path = os.path.join(self.config.checkpoint_dir, f'test_shard_{shard_counter["test"]}.npz')
                        np.savez_compressed(shard_path,
                                            states=np.array(test_buf_states),
                                            policies=np.array(test_buf_policies, dtype=np.int64),
                                            values=np.array(test_buf_values, dtype=np.float32))
                        test_shards.append(shard_path)
                        shard_counter['test'] += 1
                        test_buf_states.clear(); test_buf_policies.clear(); test_buf_values.clear()
                except Exception as e:
                    logger.warning("Failed to process data file %s: %s", fpath, e)
        else:
            # Connect to database
            await self.data_loader.connect()

            # Stream games from DB in batches and process incrementally
            async for games_batch in self.data_loader.load_games(batch_size=1000, limit=None):
                states, policies, values = self.processor.process_games(games_batch)
                if len(states) == 0:
                    continue

                for i in range(len(states)):
                    if random.random() < self.config.train_test_split:
                        train_buf_states.append(states[i])
                        train_buf_policies.append(int(policies[i]))
                        train_buf_values.append(float(values[i]))
                    else:
                        test_buf_states.append(states[i])
                        test_buf_policies.append(int(policies[i]))
                        test_buf_values.append(float(values[i]))

            # Flush train buffer to shard
            if len(train_buf_states) >= buffer_limit:
                shard_path = os.path.join(self.config.checkpoint_dir, f'train_shard_{shard_counter["train"]}.npz')
                np.savez_compressed(shard_path,
                                    states=np.array(train_buf_states),
                                    policies=np.array(train_buf_policies, dtype=np.int64),
                                    values=np.array(train_buf_values, dtype=np.float32))
                train_shards.append(shard_path)
                shard_counter['train'] += 1
                train_buf_states.clear(); train_buf_policies.clear(); train_buf_values.clear()

            # Flush test buffer to shard
            if len(test_buf_states) >= buffer_limit:
                shard_path = os.path.join(self.config.checkpoint_dir, f'test_shard_{shard_counter["test"]}.npz')
                np.savez_compressed(shard_path,
                                    states=np.array(test_buf_states),
                                    policies=np.array(test_buf_policies, dtype=np.int64),
                                    values=np.array(test_buf_values, dtype=np.float32))
                test_shards.append(shard_path)
                shard_counter['test'] += 1
                test_buf_states.clear(); test_buf_policies.clear(); test_buf_values.clear()

        # Write remaining buffers
        if train_buf_states:
            shard_path = os.path.join(self.config.checkpoint_dir, f'train_shard_{shard_counter["train"]}.npz')
            np.savez_compressed(shard_path,
                                states=np.array(train_buf_states),
                                policies=np.array(train_buf_policies, dtype=np.int64),
                                values=np.array(train_buf_values, dtype=np.float32))
            train_shards.append(shard_path)
            shard_counter['train'] += 1
            train_buf_states.clear(); train_buf_policies.clear(); train_buf_values.clear()

        if test_buf_states:
            shard_path = os.path.join(self.config.checkpoint_dir, f'test_shard_{shard_counter["test"]}.npz')
            np.savez_compressed(shard_path,
                                states=np.array(test_buf_states),
                                policies=np.array(test_buf_policies, dtype=np.int64),
                                values=np.array(test_buf_values, dtype=np.float32))
            test_shards.append(shard_path)
            shard_counter['test'] += 1
            test_buf_states.clear(); test_buf_policies.clear(); test_buf_values.clear()

        # Close DB pool
        await self.data_loader.close()

        if not train_shards:
            raise RuntimeError("No training data produced — check database or parsing logic")

        if not test_shards:
            logger.warning("No test shards produced; using a small split of training data for validation")

        # Create iterable datasets that read shards lazily
        train_dataset = ShardedIterableDataset(train_shards)
        test_dataset = ShardedIterableDataset(test_shards) if test_shards else ShardedIterableDataset(train_shards[:1])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=getattr(self.config, 'data_loader_workers', 0),
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=getattr(self.config, 'data_loader_workers', 0),
            pin_memory=True
        )

        # Training loop
        logger.info(f"Training for {self.config.epochs} epochs (streamed shards)...")
        best_test_loss = float('inf')

        for epoch in range(self.config.epochs):
            # Train
            train_policy_loss = 0.0
            train_value_loss = 0.0
            batches = 0

            # Generate static edge index for the board
            # We need to construct it based on the canonical board layout
            from self_play.agent_utils import initialize_game
            dummy_state = initialize_game()
            # Ensure vertices are sorted by ID to match our tensor mapping
            sorted_ids = sorted(dummy_state['vertices'].keys(), key=lambda x: int(x[1:]))
            id_to_idx = {vid: i for i, vid in enumerate(sorted_ids)}
            
            edges = []
            for vid in sorted_ids:
                src_idx = id_to_idx[vid]
                for neighbor_id in dummy_state['vertices'][vid]['adjacencies']:
                    if neighbor_id in id_to_idx:
                        dst_idx = id_to_idx[neighbor_id]
                        edges.append([src_idx, dst_idx])
            
            if edges:
                edge_index = np.array(edges).T # [2, E]
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)

            for batch_states, batch_policies, batch_values in train_loader:
                # Ensure numpy arrays for trainer API
                bs = batch_states.numpy() if hasattr(batch_states, 'numpy') else np.array(batch_states)
                bp = batch_policies.numpy() if hasattr(batch_policies, 'numpy') else np.array(batch_policies)
                bv = batch_values.numpy() if hasattr(batch_values, 'numpy') else np.array(batch_values)

                policy_loss, value_loss = self.trainer.train_on_batch(bs, bp, bv, edge_index=edge_index)
                train_policy_loss += policy_loss
                train_value_loss += value_loss
                batches += 1

            if batches:
                train_policy_loss /= batches
                train_value_loss /= batches
            else:
                train_policy_loss = 0.0
                train_value_loss = 0.0

            # Evaluate on test set
            test_policy_loss, test_value_loss = self._evaluate(test_loader)
            test_total_loss = test_policy_loss + test_value_loss

            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train P/V Loss: {train_policy_loss:.4f}/{train_value_loss:.4f} - "
                f"Test P/V Loss: {test_policy_loss:.4f}/{test_value_loss:.4f}"
            )

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f'model_epoch_{epoch + 1}.pt'
                )
                self.trainer.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            if test_total_loss < best_test_loss:
                best_test_loss = test_total_loss
                best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
                self.trainer.save_checkpoint(best_path)

        logger.info("Training complete")
    
    def _evaluate(self, test_loader) -> Tuple[float, float]:
        """Evaluate model on test set."""
        self.model.eval()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch_states, batch_policies, batch_values in test_loader:
                # Ensure input is float32 on the correct device (prevent double/float mismatch)
                bs = batch_states
                if hasattr(bs, 'to'):
                    bs = bs.to(self.trainer.device)
                else:
                    bs = torch.as_tensor(bs, device=self.trainer.device)
                bs = bs.float()
                policy_pred, value_pred = self.model(bs)

                policy_loss = torch.nn.functional.cross_entropy(
                    policy_pred,
                    batch_policies.to(self.trainer.device)
                )
                value_loss = torch.nn.functional.mse_loss(
                    value_pred.squeeze(),
                    batch_values.to(self.trainer.device)
                )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                count += 1

        if count == 0:
            return 0.0, 0.0
        return total_policy_loss / count, total_value_loss / count
    
    async def evaluate_model(self):
        """Evaluate the current model against a baseline."""
        logger.info(f"Evaluating model from {self.config.model_path}...")
        
        if not self.config.model_path or not os.path.exists(self.config.model_path):
            logger.error("No model path provided or file does not exist.")
            return

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UnitGameNet(num_vertices=self.config.num_vertices).to(device)
        try:
            checkpoint = torch.load(self.config.model_path, map_location=device)
            # Handle both full checkpoint dict and raw state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
        
        model.eval()
        
        # Define agent function
        from self_play.neural_network_model import state_to_tensor
        from self_play.agent_utils import get_legal_moves
        
        def model_agent(state: Dict) -> Dict:
            legal_moves = get_legal_moves(state)
            if not legal_moves:
                return {'type': 'endTurn'}
            
            # Prepare input
            try:
                state_np = state_to_tensor(state, self.config.num_vertices)
                tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    policy_logits, value = model(tensor)
                
                # Mask illegal moves
                # This is tricky because we need to map legal moves to policy indices.
                # We reuse the processor's helper if available, or just pick the best valid one.
                # Since we don't have a perfect map, let's just iterate legal moves, 
                # get their policy index, and pick the one with highest logit.
                
                best_move = None
                best_logit = -float('inf')
                
                # We need an instance of processor to use _move_to_policy
                # But _move_to_policy is an instance method.
                # Let's just use the one attached to self.
                
                for move in legal_moves:
                    idx = self.processor._move_to_policy(move, state)
                    if idx is not None and 0 <= idx < policy_logits.shape[1]:
                        logit = policy_logits[0, idx].item()
                        if logit > best_logit:
                            best_logit = logit
                            best_move = move
                
                if best_move:
                    return best_move
                
                # Fallback if no moves mapped
                import random
                return random.choice(legal_moves)
            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                return {'type': 'endTurn'}

        # Run benchmark
        from self_play.benchmark import benchmark
        from self_play.greedy_algorithm import select_move as greedy_select
        
        logger.info("Running benchmark: Model vs Greedy (20 rounds)...")
        benchmark(model_agent, greedy_select, rounds=20)

    async def run_full_pipeline(self):
        """Run complete pipeline: generate data → train → evaluate."""
        logger.info("Starting full training pipeline...")
        
        # Step 1: Generate training data
        await self.generate_training_data()
        
        # Step 2: Train model
        await self.train_model()
        
        # Step 3: Evaluate
        # Use the best model we just trained
        self.config.model_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        await self.evaluate_model()
        
        logger.info("Training pipeline complete!")


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unit Game ML Training Pipeline')
    parser.add_argument('command', choices=['generate', 'train', 'evaluate', 'full'])
    parser.add_argument('--games', type=int, default=10000)
    parser.add_argument('--concurrent', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model', type=str, help='Model checkpoint path')
    parser.add_argument('--db-url', type=str, help='Database URL')
    parser.add_argument('--data-dir', type=str, help='Directory for training data shards')
    
    args = parser.parse_args()
    
    # Build config
    config = TrainingConfig(
        games_to_generate=args.games,
        concurrent_games=args.concurrent,
        epochs=args.epochs,
        batch_size=args.batch_size,
        database_url=args.db_url or os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/unitgame')
    )
    if args.model:
        config.model_path = args.model
    
    # allow data-dir from CLI to override config when provided
    if getattr(args, 'data_dir', None):
        config.data_dir = args.data_dir
    
    pipeline = TrainingPipeline(config)
    
    if args.command == 'generate':
        await pipeline.generate_training_data()
    elif args.command == 'train':
        await pipeline.train_model()
    elif args.command == 'evaluate':
        await pipeline.evaluate_model()
    elif args.command == 'full':
        await pipeline.run_full_pipeline()


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    asyncio.run(main())