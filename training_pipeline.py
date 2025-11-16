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
    
    # Evaluation
    eval_games: int = 100
    eval_minimax_depth: int = 4


class GameDataset(Dataset):
    """PyTorch Dataset for game training data."""
    
    def __init__(self, states: np.ndarray, moves: np.ndarray, outcomes: np.ndarray):
        """
        Args:
            states: [N, num_vertices, 5] game states
            moves: [N, num_vertices * 4] move probabilities (policy targets)
            outcomes: [N] game outcomes (-1, 0, 1)
        """
        self.states = torch.FloatTensor(states)
        self.moves = torch.FloatTensor(moves)
        self.outcomes = torch.FloatTensor(outcomes)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.moves[idx], self.outcomes[idx]


class DataLoader:
    """Loads training data from PostgreSQL database."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
    
    async def connect(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(self.database_url)
        logger.info("Connected to database")
    
    async def load_games(self, limit: int = None) -> List[Dict]:
        """
        Load games from database.
        
        Returns:
            List of game dictionaries with moves
        """
        query = """
            SELECT g.game_id, g.winner, g.total_moves,
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
        
        if limit:
            query += f" LIMIT {limit}"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
        
        games = []
        for row in rows:
            games.append({
                'game_id': row['game_id'],
                'winner': row['winner'],
                'total_moves': row['total_moves'],
                'moves': row['moves']
            })
        
        logger.info(f"Loaded {len(games)} games from database")
        return games
    
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
            policies: [N, num_vertices * 4] - move distributions
            values: [N] - game outcomes
        """
        states = []
        policies = []
        values = []
        
        for game in games:
            game_outcome = self._outcome_to_value(game['winner'])
            
            for move_data in game['moves']:
                # Parse state_before from move data
                state = self._parse_state(move_data)
                if state is None:
                    continue
                
                # Convert state to tensor
                state_tensor = state_to_tensor(state, self.num_vertices)
                states.append(state_tensor)
                
                # Convert move to policy target (one-hot encoding)
                policy = self._move_to_policy(move_data, state)
                policies.append(policy)
                
                # Flip outcome based on whose turn it was
                player = move_data['player_id']
                outcome = game_outcome if player == game['winner'] else -game_outcome
                values.append(outcome)
        
        logger.info(f"Processed {len(states)} training examples from {len(games)} games")
        
        return (
            np.array(states),
            np.array(policies),
            np.array(values)
        )
    
    def _parse_state(self, move_data: Dict) -> Dict:
        """Parse game state from move data."""
        # Assuming your moves table stores state_before in action_data
        # Adjust based on your actual schema
        try:
            action_data = move_data.get('action_data', {})
            if isinstance(action_data, str):
                action_data = json.loads(action_data)
            
            # You'll need to reconstruct game state from your data
            # This is a placeholder - adjust to your schema
            return action_data.get('state_before')
        except Exception as e:
            logger.warning(f"Failed to parse state: {e}")
            return None
    
    def _move_to_policy(self, move_data: Dict, state: Dict) -> np.ndarray:
        """
        Convert move to policy target vector.
        
        Policy vector format: [num_vertices * 4]
        For each vertex: [place_prob, infuse_prob, move_prob, attack_prob]
        """
        policy = np.zeros(self.num_vertices * 4)
        
        try:
            action_data = move_data.get('action_data', {})
            if isinstance(action_data, str):
                action_data = json.loads(action_data)
            
            action_type = action_data.get('type')
            vertex_id = action_data.get('vertexId') or action_data.get('fromId')
            
            # Map vertex_id to index (you'll need your vertex mapping)
            vertex_idx = self._vertex_id_to_index(vertex_id, state)
            if vertex_idx is None:
                return policy
            
            # Set probability to 1.0 for the action taken
            action_offset = {
                'place': 0,
                'infuse': 1,
                'move': 2,
                'attack': 3
            }.get(action_type, 0)
            
            policy[vertex_idx * 4 + action_offset] = 1.0
            
        except Exception as e:
            logger.warning(f"Failed to convert move to policy: {e}")
        
        return policy
    
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
        from self_play_system import SelfPlayConfig, SelfPlayGenerator
        
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
        logger.info("Loading training data from database...")
        
        # Connect to database
        await self.data_loader.connect()
        
        # Load games
        games = await self.data_loader.load_games()
        
        # Process into training format
        states, policies, values = self.processor.process_games(games)
        
        # Split train/test
        split_idx = int(len(states) * self.config.train_test_split)
        train_states, test_states = states[:split_idx], states[split_idx:]
        train_policies, test_policies = policies[:split_idx], policies[split_idx:]
        train_values, test_values = values[:split_idx], values[split_idx:]
        
        # Create datasets
        train_dataset = GameDataset(train_states, train_policies, train_values)
        test_dataset = GameDataset(test_states, test_policies, test_values)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size
        )
        
        # Training loop
        logger.info(f"Training for {self.config.epochs} epochs...")
        best_test_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Train
            train_policy_loss = 0.0
            train_value_loss = 0.0
            
            for batch_states, batch_policies, batch_values in train_loader:
                policy_loss, value_loss = self.trainer.train_on_batch(
                    batch_states.numpy(),
                    batch_policies.numpy(),
                    batch_values.numpy()
                )
                train_policy_loss += policy_loss
                train_value_loss += value_loss
            
            train_policy_loss /= len(train_loader)
            train_value_loss /= len(train_loader)
            
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
        
        await self.data_loader.close()
        logger.info("Training complete")
    
    def _evaluate(self, test_loader) -> Tuple[float, float]:
        """Evaluate model on test set."""
        self.model.eval()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        with torch.no_grad():
            for batch_states, batch_policies, batch_values in test_loader:
                policy_pred, value_pred = self.model(batch_states.to(self.trainer.device))
                
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
        
        return total_policy_loss / len(test_loader), total_value_loss / len(test_loader)
    
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
    
    args = parser.parse_args()
    
    # Build config
    config = TrainingConfig(
        games_to_generate=args.games,
        concurrent_games=args.concurrent,
        epochs=args.epochs,
        batch_size=args.batch_size,
        database_url=args.db_url or os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/unitgame')
    )
    
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