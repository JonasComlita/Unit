"""
Unit and integration tests for self-play system.

Run with: pytest -v test_self_play_system.py
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import from the refactored module
# Assuming the refactored code is in self_play_system.py
from self_play_system import (
    DatabaseWriter,
    Metrics,
    SelfPlayConfig,
    SelfPlayGenerator,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return SelfPlayConfig(
        games_per_batch=10,
        concurrent_games=2,
        database_url='postgresql://test:test@localhost/test',
        dry_run=True,
        log_level='DEBUG'
    )


@pytest.fixture
def metrics():
    """Create metrics instance."""
    return Metrics()


@pytest.fixture
def generator(config):
    """Create generator instance."""
    return SelfPlayGenerator(config)


# --- Unit Tests ---

class TestEvaluatePosition:
    """Test position evaluation heuristics."""

    def test_evaluate_balanced_position(self, generator):
        """Test that balanced position returns near-zero score."""
        state = {
            'currentPlayerId': 'Player1',
            'vertices': {
                'v1': {
                    'stack': [{'player': 'Player1'}],
                    'energy': 5
                },
                'v2': {
                    'stack': [{'player': 'Player2'}],
                    'energy': 5
                }
            }
        }
        score = generator.evaluate_position(state)
        # Both players have 1 piece and 5 energy, should be balanced
        assert abs(score) < 10, "Balanced position should have near-zero score"

    def test_evaluate_player_advantage(self, generator):
        """Test evaluation when current player has advantage."""
        state = {
            'currentPlayerId': 'Player1',
            'vertices': {
                'v1': {
                    'stack': [{'player': 'Player1'}, {'player': 'Player1'}],
                    'energy': 10
                },
                'v2': {
                    'stack': [{'player': 'Player2'}],
                    'energy': 2
                }
            }
        }
        score = generator.evaluate_position(state)
        assert score > 0, "Player1 advantage should yield positive score"

    def test_evaluate_empty_board(self, generator):
        """Test evaluation of empty board."""
        state = {
            'currentPlayerId': 'Player1',
            'vertices': {}
        }
        score = generator.evaluate_position(state)
        assert score == 0.0, "Empty board should have zero score"

    def test_evaluate_opponent_advantage(self, generator):
        """Test evaluation when opponent has advantage."""
        state = {
            'currentPlayerId': 'Player1',
            'vertices': {
                'v1': {
                    'stack': [{'player': 'Player1'}],
                    'energy': 2
                },
                'v2': {
                    'stack': [
                        {'player': 'Player2'},
                        {'player': 'Player2'},
                        {'player': 'Player2'}
                    ],
                    'energy': 15
                }
            }
        }
        score = generator.evaluate_position(state)
        assert score < 0, "Opponent advantage should yield negative score"


class TestGetRandomMove:
    """Test random move selection."""

    def test_get_random_move_with_moves(self, generator):
        """Test random move selection when moves available."""
        state = {'test': 'state'}
        moves = [
            {'type': 'place', 'vertexId': 'v1'},
            {'type': 'infuse', 'vertexId': 'v2'},
            {'type': 'move', 'fromId': 'v1', 'toId': 'v2'}
        ]
        
        with patch.object(generator, 'get_legal_moves', return_value=moves):
            for _ in range(10):  # Run multiple times to test randomness
                move = generator.get_random_move(state)
                assert move in moves, "Should return one of the legal moves"

    def test_get_random_move_no_moves(self, generator):
        """Test random move when no moves available."""
        state = {'test': 'state'}
        
        with patch.object(generator, 'get_legal_moves', return_value=[]):
            move = generator.get_random_move(state)
            assert move == {'type': 'endTurn'}, \
                "Should return endTurn when no moves available"


class TestGetEngineMove:
    """Test engine move selection."""

    def test_get_engine_move_selects_best(self, generator):
        """Test that engine selects best evaluated move."""
        state = {'currentPlayerId': 'Player1', 'vertices': {}}
        moves = [
            {'type': 'place', 'vertexId': 'v1'},
            {'type': 'infuse', 'vertexId': 'v2'},
        ]
        
        # Mock to return different scores
        def mock_evaluate(s):
            if 'v1' in str(s):
                return 100.0  # Higher score
            return 50.0
        
        with patch.object(generator, 'get_legal_moves', return_value=moves):
            with patch.object(generator, 'evaluate_position', side_effect=mock_evaluate):
                with patch.object(generator, 'apply_move', side_effect=lambda s, m: {**s, 'move': m}):
                    move = generator.get_engine_move(state, depth=1)
                    assert move['vertexId'] == 'v1', "Should select move with highest score"

    def test_get_engine_move_no_moves(self, generator):
        """Test engine move when no legal moves."""
        state = {'test': 'state'}
        
        with patch.object(generator, 'get_legal_moves', return_value=[]):
            move = generator.get_engine_move(state, depth=2)
            assert move == {'type': 'endTurn'}, \
                "Should return endTurn when no moves available"


class TestSerializeState:
    """Test state serialization."""

    def test_serialize_simple_state(self, generator):
        """Test serialization of simple state."""
        state = {
            'currentPlayerId': 'Player1',
            'vertices': {'v1': {'stack': [], 'energy': 0}}
        }
        serialized = generator.serialize_state(state)
        
        # Should be valid JSON
        deserialized = json.loads(serialized)
        assert deserialized == state, "Serialized state should deserialize correctly"

    def test_serialize_compact(self, generator):
        """Test that serialization is compact (no whitespace)."""
        state = {'key': 'value', 'nested': {'a': 1}}
        serialized = generator.serialize_state(state)
        
        assert ' ' not in serialized, "Should not contain spaces"
        assert '\n' not in serialized, "Should not contain newlines"


# --- Database Writer Tests ---

@pytest.mark.asyncio
class TestDatabaseWriter:
    """Test database writer functionality."""

    async def test_initialize_pool_success(self, config, metrics):
        """Test successful pool initialization."""
        config.dry_run = False
        writer = DatabaseWriter(config, metrics)
        
        mock_pool = AsyncMock()
        with patch('asyncpg.create_pool', return_value=mock_pool):
            await writer.initialize()
            assert writer.pool is mock_pool, "Pool should be initialized"

    async def test_initialize_dry_run(self, config, metrics):
        """Test initialization in dry run mode."""
        config.dry_run = True
        writer = DatabaseWriter(config, metrics)
        
        await writer.initialize()
        assert writer.pool is None, "Pool should not be initialized in dry run"

    async def test_initialize_with_retry(self, config, metrics):
        """Test pool initialization with retries."""
        config.dry_run = False
        config.db_retry_attempts = 3
        config.db_retry_delay = 0.01  # Fast for testing
        writer = DatabaseWriter(config, metrics)
        
        mock_pool = AsyncMock()
        call_count = 0
        
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection failed")
            return mock_pool
        
        with patch('asyncpg.create_pool', side_effect=side_effect):
            await writer.initialize()
            assert call_count == 2, "Should retry once before succeeding"
            assert writer.pool is mock_pool

    async def test_enqueue_game_dry_run(self, config, metrics):
        """Test enqueuing game in dry run mode."""
        config.dry_run = True
        writer = DatabaseWriter(config, metrics)
        
        game = {'game_id': 'test123', 'moves': []}
        await writer.enqueue_game(game)
        
        assert writer.write_queue.empty(), \
            "Queue should remain empty in dry run mode"

    async def test_save_game_to_db_called(self, config, metrics):
        """Test that save_game_to_db executes SQL correctly."""
        config.dry_run = False
        writer = DatabaseWriter(config, metrics)
        
        # Mock pool and connection
        mock_conn = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire = AsyncMock(return_value=mock_conn)
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        # Mock transaction
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock()
        mock_transaction.__aexit__ = AsyncMock()
        mock_conn.transaction = MagicMock(return_value=mock_transaction)
        
        writer.pool = mock_pool
        
        game = {
            'game_id': 'test_game_1',
            'winner': 'Player1',
            'total_moves': 10,
            'moves': [
                {
                    'move_number': 0,
                    'player': 'Player1',
                    'action': {'type': 'place', 'vertexId': 'v1'}
                }
            ]
        }
        
        await writer._save_game_to_db(game)
        
        # Verify execute was called for game insert
        assert mock_conn.execute.called, "Should call execute for game insert"
        # Verify executemany was called for moves
        assert mock_conn.executemany.called, "Should call executemany for moves"

    async def test_shutdown_drains_queue(self, config, metrics):
        """Test that shutdown waits for queue to drain."""
        config.dry_run = False
        writer = DatabaseWriter(config, metrics)
        
        # Mock pool
        mock_pool = AsyncMock()
        writer.pool = mock_pool
        
        # Add items to queue
        await writer.write_queue.put({'game_id': 'test1', 'moves': []})
        await writer.write_queue.put({'game_id': 'test2', 'moves': []})
        
        # Start writer
        writer.writer_task = asyncio.create_task(writer._writer_worker())
        
        # Mock save to just mark tasks done
        async def mock_save(game):
            pass
        
        with patch.object(writer, '_save_game_with_retry', side_effect=mock_save):
            await writer.shutdown()
        
        assert writer.write_queue.empty(), "Queue should be drained"
        assert mock_pool.close.called, "Pool should be closed"


# --- Integration Tests ---

@pytest.mark.asyncio
class TestSelfPlayGenerator:
    """Integration tests for self-play generator."""

    async def test_play_single_game_completes(self, generator):
        """Test that single game plays to completion."""
        # Mock game logic to end quickly
        def mock_apply_move(state, move):
            state = state.copy()
            state['winner'] = 'Player1'  # End game
            return state
        
        with patch.object(generator, 'get_legal_moves', return_value=[{'type': 'endTurn'}]):
            with patch.object(generator, 'apply_move', side_effect=mock_apply_move):
                game = await generator.play_single_game(game_id=1)
                
                assert 'game_id' in game, "Should have game_id"
                assert 'moves' in game, "Should have moves"
                assert 'winner' in game, "Should have winner"
                assert game['winner'] == 'Player1', "Game should end with winner"

    async def test_play_single_game_max_moves(self, generator):
        """Test that game ends at max moves."""
        # Never set winner to test max moves limit
        with patch.object(generator, 'get_legal_moves', return_value=[{'type': 'endTurn'}]):
            with patch.object(generator, 'apply_move', return_value={'winner': None, 'currentPlayerId': 'Player1', 'vertices': {}}):
                game = await generator.play_single_game(game_id=1)
                
                assert game['total_moves'] == 500, "Should stop at max moves"

    async def test_generate_batch(self, generator):
        """Test batch generation."""
        # Mock game to complete quickly
        async def mock_play_game(game_id):
            return {
                'game_id': f'test_{game_id}',
                'moves': [],
                'winner': 'Player1',
                'total_moves': 10,
                'timestamp': '2024-01-01T00:00:00'
            }
        
        with patch.object(generator, 'play_single_game', side_effect=mock_play_game):
            await generator._generate_batch()
            
            assert generator.metrics.games_generated == generator.config.concurrent_games, \
                "Should generate concurrent_games number of games"

    async def test_initialize_and_shutdown(self, generator):
        """Test initialization and shutdown sequence."""
        with patch.object(generator.db_writer, 'initialize', new_callable=AsyncMock):
            with patch.object(generator.db_writer, 'start_writer', new_callable=AsyncMock):
                with patch.object(generator.db_writer, 'shutdown', new_callable=AsyncMock):
                    await generator.initialize()
                    await generator.shutdown()
                    
                    assert generator.db_writer.initialize.called
                    assert generator.db_writer.start_writer.called
                    assert generator.db_writer.shutdown.called


# --- Metrics Tests ---

class TestMetrics:
    """Test metrics tracking."""

    def test_metrics_initialization(self, metrics):
        """Test metrics start at zero."""
        assert metrics.games_generated == 0
        assert metrics.games_saved == 0
        assert metrics.db_errors == 0
        assert metrics.game_errors == 0

    def test_metrics_log_summary(self, metrics, caplog):
        """Test metrics logging."""
        metrics.games_generated = 100
        metrics.games_saved = 95
        metrics.db_errors = 5
        
        with caplog.at_level('INFO'):
            metrics.log_summary()
            
        assert 'Games Generated: 100' in caplog.text
        assert 'Games Saved: 95' in caplog.text


if __name__ == '__main__':
    pytest.main([__file__, '-v'])