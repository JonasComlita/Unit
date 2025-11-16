"""
Neural network for Unit game - inspired by AlphaZero
Uses policy network (what move to make) + value network (who's winning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional

# Optional import: prefer PyTorch Geometric for GCN layers. If unavailable,
# raise a clear error explaining how to install it.
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as e:  # pragma: no cover - environment dependent
    GCNConv = None
    global_mean_pool = None
    _pyg_import_error = e

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

        # Ensure PyG is available
        if GCNConv is None:
            raise ImportError(
                "PyTorch Geometric is required for the GCN-based model. "
                f"Original import error: {_pyg_import_error!r}.\n"
                "Install instructions: follow https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html"
            )

        # Graph convolutional layers (process node features respecting adjacency)
        # We'll process batched graphs by replicating a single-graph edge_index across the batch
        self.node_embed_dim = board_channels
        self.gcn_layers = nn.ModuleList([
            GCNConv(input_features if i == 0 else self.node_embed_dim, self.node_embed_dim)
            for i in range(num_layers)
        ])

        # Per-node policy head: maps node embedding -> action logits per node
        self.node_policy_head = nn.Sequential(
            nn.Linear(self.node_embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, policy_channels)  # logits per action type for each node
        )

        # Value head: pool node embeddings to a graph embedding then predict scalar value
        self.value_head = nn.Sequential(
            nn.Linear(self.node_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x: Optional[torch.Tensor] = None, data: Optional[object] = None, edge_index: Optional[torch.Tensor] = None, batch_vec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Batch of game states [batch_size, num_vertices, features]
        
        Returns:
            policy: [batch_size, num_vertices * policy_channels]
            value: [batch_size, 1]
        """
        # Two input modes supported:
        # 1) x provided as tensor of shape [batch, num_nodes, features] (legacy)
        # 2) data provided as a PyG Data or Batch object (preferred) - supports variable node counts

        if data is not None:
            # Expect a PyG Data or Batch object
            # Import here to avoid top-level dependency in environments not using PyG
            try:
                from torch_geometric.data import Batch as PyGBatch
            except Exception:
                PyGBatch = None

            if PyGBatch is None:
                raise ImportError("torch_geometric is required to pass Data/Batch objects to forward().")

            # If a list of Data objects was passed, convert to Batch
            if isinstance(data, list):
                data = PyGBatch.from_data_list(data)

            # Now `data` should be a Batch
            # Move batch data to the same device as model parameters
            device = next(self.parameters()).device
            node_feats = data.x.to(device) if hasattr(data, 'x') else None
            edge_index = data.edge_index.to(device) if hasattr(data, 'edge_index') else None
            batch_vec = data.batch.to(device) if hasattr(data, 'batch') else None

            h = node_feats
            for conv in self.gcn_layers:
                h = conv(h, edge_index)
                h = F.relu(h)

            # Node-level policy logits
            node_logits = self.node_policy_head(h)  # shape [total_nodes, policy_channels]

            # Attempt to reshape back to per-graph flattened logits when possible using self.num_vertices
            total_nodes = node_logits.size(0)
            if self.num_vertices is not None and total_nodes % self.num_vertices == 0:
                batch_size = total_nodes // self.num_vertices
                policy = node_logits.view(batch_size, self.num_vertices * node_logits.size(-1))
                policy = F.softmax(policy, dim=1)
            else:
                # Fallback: return node_logits directly (caller can handle per-node logits)
                policy = node_logits

            # Graph-level value pooling
            if global_mean_pool is None:
                # manual pooling
                batch_size = int(batch_vec.max().item()) + 1 if batch_vec is not None and batch_vec.numel() > 0 else 1
                pooled = torch.zeros((batch_size, h.size(-1)), device=h.device)
                counts = torch.zeros(batch_size, device=h.device)
                for n_idx in range(h.size(0)):
                    b = int(batch_vec[n_idx].item()) if batch_vec is not None else 0
                    pooled[b] += h[n_idx]
                    counts[b] += 1
                counts = counts.clamp_min(1.0).unsqueeze(-1)
                pooled = pooled / counts
            else:
                pooled = global_mean_pool(h, batch_vec)

            value = self.value_head(pooled)
            return policy, value

        # Legacy tensor input mode: x tensor [batch, num_nodes, features]
        if x is None:
            raise ValueError("Either `x` tensor or `data` (PyG Data/Batch) must be provided to forward().")

        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        device = x.device

        # Flatten node dimensions so GCNConv sees (total_nodes, features)
        node_feats = x.view(batch_size * num_nodes, -1)

        # If edge_index is None, create empty edge_index (no edges)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        if edge_index.device != device:
            edge_index = edge_index.to(device)

        if batch_vec is None:
            batch_vec = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)

        # Detect whether edge_index is for a single graph (max index < num_nodes)
        if edge_index.numel() > 0 and int(edge_index.max()) < num_nodes:
            # replicate edges for each graph, offsetting node indices
            copies = []
            for i in range(batch_size):
                offset = i * num_nodes
                copies.append(edge_index + offset)
            edge_index = torch.cat(copies, dim=1)

        # Run through GCN layers
        h = node_feats
        for conv in self.gcn_layers:
            h = conv(h, edge_index)
            h = F.relu(h)

        node_logits = self.node_policy_head(h)
        policy = node_logits.view(batch_size, num_nodes * node_logits.size(-1))
        policy = F.softmax(policy, dim=1)

        if global_mean_pool is None:
            pooled = torch.zeros((batch_size, h.size(-1)), device=device)
            counts = torch.zeros(batch_size, device=device)
            for n_idx in range(h.size(0)):
                b = int(batch_vec[n_idx].item())
                pooled[b] += h[n_idx]
                counts[b] += 1
            counts = counts.clamp_min(1.0).unsqueeze(-1)
            pooled = pooled / counts
        else:
            pooled = global_mean_pool(h, batch_vec)

        value = self.value_head(pooled)
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
    
    def __init__(self, model: UnitGameNet, learning_rate: float = 0.001, device: Optional[str] = None, use_amp: Optional[bool] = None):
        """Create a trainer.

        Args:
            model: UnitGameNet instance
            learning_rate: optimizer lr
            device: device string e.g. 'cuda', 'cuda:0' or 'cpu'. If None, auto-selects CUDA when available.
            use_amp: enable torch.cuda.amp mixed precision for training (only used when CUDA is available)
        """
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        # If use_amp is None, enable AMP when CUDA is available; otherwise respect caller
        if use_amp is None:
            self.use_amp = torch.cuda.is_available()
        else:
            self.use_amp = bool(use_amp) and torch.cuda.is_available()
        self.model.to(self.device)
        # GradScaler only needed when using AMP and CUDA
        self._scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def train_on_batch(self, 
                      states: np.ndarray, 
                      policy_targets: np.ndarray, 
                      value_targets: np.ndarray,
                      edge_index: Optional[np.ndarray] = None,
                      batch_vec: Optional[np.ndarray] = None,
                      pyg_batch: Optional[object] = None) -> Tuple[float, float]:
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
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        # policy targets are class indices for cross-entropy loss
        policy_targets = torch.as_tensor(policy_targets, dtype=torch.long, device=self.device)
        value_targets = torch.as_tensor(value_targets, dtype=torch.float32, device=self.device)

        # Convert adjacency if provided
        edge_index_tensor = None
        if edge_index is not None:
            # Expect shape [2, E]
            edge_index_tensor = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        batch_vec_tensor = None
        if batch_vec is not None:
            batch_vec_tensor = torch.as_tensor(batch_vec, dtype=torch.long, device=self.device)
        # If a PyG Batch (or list of Data) is provided, convert/move it to device
        pyg_batch_obj = None
        if pyg_batch is not None:
            try:
                from torch_geometric.data import Batch as PyGBatch
            except Exception:
                PyGBatch = None
            if PyGBatch is None:
                raise ImportError("torch_geometric is required to pass `pyg_batch` to train_on_batch().")

            if isinstance(pyg_batch, list):
                pyg_batch_obj = PyGBatch.from_data_list(pyg_batch)
            elif isinstance(pyg_batch, PyGBatch):
                pyg_batch_obj = pyg_batch
            else:
                # assume a single Data object or already-batched object
                pyg_batch_obj = pyg_batch

            # move to device
            pyg_batch_obj = pyg_batch_obj.to(self.device)

        # Training with optional AMP
        self.optimizer.zero_grad()
        if self.use_amp:
            with torch.cuda.amp.autocast():
                if pyg_batch_obj is not None:
                    policy_pred, value_pred = self.model(data=pyg_batch_obj)
                else:
                    policy_pred, value_pred = self.model(states, edge_index=edge_index_tensor, batch_vec=batch_vec_tensor)
                policy_loss = F.cross_entropy(policy_pred, policy_targets)
                value_loss = F.mse_loss(value_pred.squeeze(), value_targets)
                total_loss = policy_loss + value_loss

            # scale gradients
            self._scaler.scale(total_loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            if pyg_batch_obj is not None:
                policy_pred, value_pred = self.model(data=pyg_batch_obj)
            else:
                policy_pred, value_pred = self.model(states, edge_index=edge_index_tensor, batch_vec=batch_vec_tensor)
            policy_loss = F.cross_entropy(policy_pred, policy_targets)
            value_loss = F.mse_loss(value_pred.squeeze(), value_targets)
            total_loss = policy_loss + value_loss
            total_loss.backward()
            self.optimizer.step()

        return float(policy_loss.item()), float(value_loss.item())
    
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
    # Small sanity-check example using PyG Data objects and DataLoader
    num_vertices = 5
    batch_size = 2
    model = UnitGameNet(num_vertices=num_vertices, num_layers=3, board_channels=16)
    trainer = UnitGameTrainer(model)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Dummy per-vertex features for two graphs
    dummy_states_list = [torch.as_tensor(np.random.rand(num_vertices, 5).astype(np.float32)) for _ in range(batch_size)]

    # Create a small ring adjacency for a single graph
    edges = torch.tensor([
        [0,1,1,2,2,3,3,4,4,0],
        [1,0,2,1,3,2,4,3,0,4]
    ], dtype=torch.long)

    # Build PyG Data objects
    try:
        from torch_geometric.data import Data, DataLoader
    except Exception as e:  # pragma: no cover - environment dependent
        print("torch_geometric is required for the Data/DataLoader example:", e)
    else:
        data_list = []
        for i in range(batch_size):
            d = Data(x=dummy_states_list[i], edge_index=edges)
            data_list.append(d)

        loader = DataLoader(data_list, batch_size=batch_size)
        batch = next(iter(loader))

        # Ensure batch is on correct device and run forward
        batch = batch.to(trainer.device)
        policy_out, value_out = model(data=batch)
        print("policy_out shape:", policy_out.shape)
        print("value_out shape:", value_out.shape)
