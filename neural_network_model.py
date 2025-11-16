"""
Neural network for Unit game - inspired by AlphaZero
Uses policy network (what move to make) + value network (who's winning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

class UnitGameNet(nn.Module):
    """
    Neural network that outputs:
    1. Policy: Probability distribution over all possible moves
    2. Value: Estimated win probability for current player
    """
    
    def __init__(self, 
                 num_vertices: int = 83,  # Total vertices in game
                 num_layers: int = 5,
                 board_channels: int = 32,
                 policy_channels: int = 4,  # place, infuse, move, attack
                 fc_size: int = 256):
        super().__init__()
        
        self.num_vertices = num_vertices
        
        # Input: State representation (vertices × features)
        # Features: [has_piece, piece_count, energy, player_ownership, layer]
        input_features = 5
        
        # Convolutional-like layers for board processing
        # Since our board is irregular (not a grid), we use FC layers
        # In a production system, you might use Graph Neural Networks (GNNs)
        
        self.input_layer = nn.Linear(num_vertices * input_features, board_channels * num_vertices)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(board_channels * num_vertices) 
            for _ in range(10)
        ])
        
        # Policy head: What move should we make?
        self.policy_head = nn.Sequential(
            nn.Linear(board_channels * num_vertices, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_vertices * policy_channels)  # Output for each action type per vertex
        )
        
        # Value head: Who is winning?
        self.value_head = nn.Sequential(
            nn.Linear(board_channels * num_vertices, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output between -1 (losing) and +1 (winning)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Batch of game states [batch_size, num_vertices, features]
        
        Returns:
            policy: [batch_size, num_vertices * policy_channels]
            value: [batch_size, 1]
        """
        batch_size = x.shape[0]
        
        # Flatten input
        x = x.view(batch_size, -1)
        
        # Process through input layer
        x = F.relu(self.input_layer(x))
        
        # Process through residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        policy = self.policy_head(x)
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = self.value_head(x)
        
        return policy, value

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, size: int):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.bn1 = nn.BatchNorm1d(size)
        self.bn2 = nn.BatchNorm1d(size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x += residual
        x = F.relu(x)
        return x

class UnitGameTrainer:
    """Training loop for the neural network"""
    
    def __init__(self, model: UnitGameNet, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_on_batch(self, 
                      states: np.ndarray, 
                      policy_targets: np.ndarray, 
                      value_targets: np.ndarray) -> Tuple[float, float]:
        """
        Train on a batch of data
        
        Args:
            states: Game states [batch_size, num_vertices, features]
            policy_targets: Target move probabilities from MCTS
            value_targets: Actual game outcomes (-1, 0, +1)
        
        Returns:
            policy_loss, value_loss
        """
        self.model.train()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        policy_targets = torch.FloatTensor(policy_targets).to(self.device)
        value_targets = torch.FloatTensor(value_targets).to(self.device)
        
        # Forward pass
        policy_pred, value_pred = self.model(states)
        
        # Calculate losses
        policy_loss = F.cross_entropy(policy_pred, policy_targets)
        value_loss = F.mse_loss(value_pred.squeeze(), value_targets)
        
        # Combined loss
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def state_to_tensor(game_state: dict, num_vertices: int = 83) -> np.ndarray:
    """
    Convert game state dictionary to neural network input tensor
    
    Features per vertex:
    - has_piece (0/1)
    - piece_count (0-10)
    - energy (0-10)
    - player_ownership (-1 for enemy, 0 for neutral, +1 for friendly)
    - layer (0-4 normalized)
    """
    tensor = np.zeros((num_vertices, 5))
    
    current_player = game_state['currentPlayerId']
    
    for i, (vertex_id, vertex) in enumerate(game_state['vertices'].items()):
        if i >= num_vertices:
            break
        
        # has_piece
        tensor[i, 0] = 1.0 if vertex['stack'] else 0.0
        
        # piece_count (normalized)
        tensor[i, 1] = len(vertex['stack']) / 10.0
        
        # energy (normalized)
        tensor[i, 2] = vertex['energy'] / 10.0
        
        # player_ownership
        if vertex['stack']:
            owner = vertex['stack'][0]['player']
            tensor[i, 3] = 1.0 if owner == current_player else -1.0
        else:
            tensor[i, 3] = 0.0
        
        # layer (normalized)
        tensor[i, 4] = vertex['layer'] / 4.0
    
    return tensor

# Example usage
if __name__ == "__main__":
    # Create model
    model = UnitGameNet()
    trainer = UnitGameTrainer(model)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example training loop (pseudo-code)
    # for epoch in range(num_epochs):
    #     states, policies, values = load_training_data()
    #     policy_loss, value_loss = trainer.train_on_batch(states, policies, values)
    #     print(f"Epoch {epoch}: Policy Loss = {policy_loss:.4f}, Value Loss = {value_loss:.4f}")
