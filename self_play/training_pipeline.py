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
from neural_network_model import (
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
    num_vertices: int = 83
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
    
    def __init__(self, num_vertices: int = 83):
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
            for move_data in game.get('moves', []):
                # Normalize move_data in case DB returned JSON strings
                if isinstance(move_data, str):
                    try:
                        move_data = json.loads(move_data)
                    except Exception:
                        # skip malformed move entry
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
                    # place a minimal unit record (ownership is key for state_to_tensor)
                    v['stack'].insert(0, {'player': player})

            elif action_type == 'move':
                from_id = action_data.get('fromId') or action_data.get('vertexId')
                to_id = action_data.get('toId')
                if from_id and to_id and from_id in vertices:
                    src = vertices[from_id]
                    if src.get('stack'):
                        piece = src['stack'].pop(0)
                        dst = ensure_vertex(to_id)
                        dst['stack'].insert(0, piece)

            elif action_type == 'attack':
                # Simplified: remove top opposing piece at target if exists
                target_id = action_data.get('targetId') or action_data.get('vertexId') or action_data.get('toId')
                if target_id and target_id in vertices and vertices[target_id].get('stack'):
                    top = vertices[target_id]['stack'][0]
                    if top.get('player') != player:
                        vertices[target_id]['stack'].pop(0)

            elif action_type == 'infuse':
                # infuse may change energy on a vertex
                vid = action_data.get('vertexId') or action_data.get('toId')
                amount = action_data.get('amount') or 0
                if vid:
                    v = ensure_vertex(vid)
                    try:
                        v['energy'] = max(0, v.get('energy', 0) + int(amount))
                    except Exception:
                        pass

            # pincer and other types: best-effort no-op to keep simulation deterministic

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
        
        vertices = list(state.get('vertices', {}).keys())
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

            for batch_states, batch_policies, batch_values in train_loader:
                # Ensure numpy arrays for trainer API
                bs = batch_states.numpy() if hasattr(batch_states, 'numpy') else np.array(batch_states)
                bp = batch_policies.numpy() if hasattr(batch_policies, 'numpy') else np.array(batch_policies)
                bv = batch_values.numpy() if hasattr(batch_values, 'numpy') else np.array(batch_values)

                policy_loss, value_loss = self.trainer.train_on_batch(bs, bp, bv)
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
    
    async def run_full_pipeline(self):
        """Run complete pipeline: generate data → train → evaluate."""
        logger.info("Starting full training pipeline...")
        
        # Step 1: Generate training data
        await self.generate_training_data()
        
        # Step 2: Train model
        await self.train_model()
        
        # Step 3: Evaluate (placeholder - would run games NN vs minimax)
        logger.info("Training pipeline complete!")
        logger.info(f"Best model saved to: {self.config.checkpoint_dir}/best_model.pt")


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
    parser.add_argument('--data-dir', type=str, help='Directory of Parquet/JSONL shards to train from (bypass DB)')
    
    args = parser.parse_args()
    
    # Build config
    config = TrainingConfig(
        games_to_generate=args.games,
        concurrent_games=args.concurrent,
        epochs=args.epochs,
        batch_size=args.batch_size,
        database_url=args.db_url or os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/unitgame')
    )
    # allow data-dir from CLI to override config when provided
    if getattr(args, 'data_dir', None):
        config.data_dir = args.data_dir
    
    pipeline = TrainingPipeline(config)
    
    if args.command == 'generate':
        await pipeline.generate_training_data()
    elif args.command == 'train':
        await pipeline.train_model()
    elif args.command == 'evaluate':
        # TODO: Implement evaluation (NN vs minimax games)
        logger.info("Evaluation not yet implemented")
    elif args.command == 'full':
        await pipeline.run_full_pipeline()


if __name__ == '__main__':
    asyncio.run(main())