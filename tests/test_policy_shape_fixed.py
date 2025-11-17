import pytest
import numpy as np
import torch

from self_play.neural_network_model import UnitGameNet
from self_play.training_pipeline import TrainingDataProcessor


def test_unit_game_net_policy_shape():
    num_vertices = 9
    model = UnitGameNet(num_vertices=num_vertices)
    # Create a dummy input batch as a torch tensor
    batch_np = np.random.rand(2, num_vertices, 5).astype(np.float32)
    batch = torch.from_numpy(batch_np)
    policy, value = model(batch)
    # Policy should be shape [batch, num_vertices * policy_channels]
    assert policy.shape[0] == batch.shape[0]
    # node_policy_head's final Linear out_features is the per-node policy channel count
    per_node_channels = model.node_policy_head[-1].out_features
    assert policy.shape[1] == num_vertices * per_node_channels


def test_move_to_policy_range():
    processor = TrainingDataProcessor(num_vertices=9)
    # craft a minimal state with vertices v0..v8
    state = {'vertices': {f'v{i}': {'stack': [], 'energy': 0, 'layer': 0} for i in range(9)}, 'currentPlayerId': 'Player1'}
    # valid move example
    move_data = {'player_id': 'Player1', 'action_data': {'type': 'place', 'vertexId': 'v3'}}
    idx = processor._move_to_policy(move_data, state)
    assert idx is not None
    assert 0 <= idx < processor.num_vertices * 5

    # invalid vertex
    move_data2 = {'player_id': 'Player1', 'action_data': {'type': 'place', 'vertexId': 'vx'}}
    idx2 = processor._move_to_policy(move_data2, state)
    assert idx2 is None
