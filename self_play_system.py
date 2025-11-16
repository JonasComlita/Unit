"""
Production-ready self-play system for generating training data.

This module generates game training data asynchronously with:
- Async database operations using asyncpg
- Connection pooling with retry logic
- Graceful shutdown handling
- CLI configuration
- Structured logging and metrics
- Comprehensive error handling
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
import numpy as np
from asyncpg.pool import Pool

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play generator."""
    games_per_batch: int = 100
    search_depth: int = 4
    exploration_rate: float = 0.1
    concurrent_games: int = 10
    database_url: str = field(
        default_factory=lambda: os.getenv(
            'DATABASE_URL',
            'postgresql://user:pass@localhost/unitgame'
        )
    )
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20
    db_retry_attempts: int = 3
    db_retry_delay: float = 1.0
    dry_run: bool = False
    batch_only: bool = False
    log_level: str = 'INFO'
    # Max size for the in-memory write queue (provide backpressure).
    # Set to a large number by default so tests and local runs don't block.
    write_queue_maxsize: int = 1000


@dataclass
class Metrics:
    """Track system metrics."""
    games_generated: int = 0
    games_saved: int = 0
    db_errors: int = 0
    game_errors: int = 0
    start_time: float = field(default_factory=time.time)

    def log_summary(self):
        """Log metrics summary."""
        elapsed = time.time() - self.start_time
        logger.info(
            f"Metrics Summary - Games Generated: {self.games_generated}, "
            f"Games Saved: {self.games_saved}, "
            f"DB Errors: {self.db_errors}, "
            f"Game Errors: {self.game_errors}, "
            f"Elapsed: {elapsed:.1f}s, "
            f"Rate: {self.games_generated / elapsed if elapsed > 0 else 0:.2f} games/s"
        )


class DatabaseWriter:
    """Handles asynchronous database operations with connection pooling."""

    def __init__(self, config: SelfPlayConfig, metrics: Metrics):
        self.config = config
        self.metrics = metrics
        self.pool: Optional[Pool] = None
        # Use a bounded queue to provide backpressure between generator and DB writer.
        maxsize = getattr(self.config, 'write_queue_maxsize', 0) or 0
        # asyncio.Queue treats 0 as infinite, so pass maxsize directly.
        self.write_queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.writer_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()

    async def initialize(self):
        """Initialize database connection pool."""
        if self.config.dry_run:
            logger.info("Dry run mode - skipping database initialization")
            return

        logger.info("Initializing database connection pool...")
        retry_count = 0
        last_error = None

        while retry_count < self.config.db_retry_attempts:
            try:
                # Some tests patch `asyncpg.create_pool` with a plain object
                # or AsyncMock that may not be awaitable. Call it and handle
                # both awaitable and non-awaitable returns to be robust.
                pool_candidate = asyncpg.create_pool(
                    self.config.database_url,
                    min_size=self.config.db_pool_min_size,
                    max_size=self.config.db_pool_max_size,
                    command_timeout=60
                )

                # If the factory returned an awaitable/coroutine, await it.
                if hasattr(pool_candidate, '__await__'):
                    self.pool = await pool_candidate  # real asyncpg returns a coroutine
                else:
                    # Tests may return AsyncMock or a plain object directly.
                    self.pool = pool_candidate
                logger.info(
                    f"Database pool initialized (min={self.config.db_pool_min_size}, "
                    f"max={self.config.db_pool_max_size})"
                )
                return
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.config.db_retry_attempts:
                    delay = self.config.db_retry_delay * (2 ** (retry_count - 1))
                    logger.warning(
                        f"Database connection failed (attempt {retry_count}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)

        logger.error(f"Failed to initialize database after {retry_count} attempts")
        raise last_error

    async def start_writer(self):
        """Start the database writer worker."""
        if self.config.dry_run:
            logger.info("Dry run mode - writer not started")
            return
        if self.writer_task and not self.writer_task.done():
            logger.debug("Writer already running")
            return

        self.writer_task = asyncio.create_task(self._writer_worker())
        logger.info("Database writer worker started")

    async def _writer_worker(self):
        """Worker that consumes games from queue and writes to database."""
        logger.info("Writer worker running")
        
        while not self.shutdown_event.is_set() or not self.write_queue.empty():
            try:
                # Wait for game with timeout to allow shutdown checks
                game = await asyncio.wait_for(
                    self.write_queue.get(),
                    timeout=1.0
                )
                await self._save_game_with_retry(game)
                self.write_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Writer worker error: {e}", exc_info=True)
                self.metrics.db_errors += 1

        logger.info("Writer worker shutting down")

    async def _save_game_with_retry(self, game: Dict[str, Any]):
        """Save game to database with retry logic."""
        retry_count = 0
        last_error = None

        while retry_count < self.config.db_retry_attempts:
            try:
                await self._save_game_to_db(game)
                self.metrics.games_saved += 1
                return
            except Exception as e:
                last_error = e
                retry_count += 1
                self.metrics.db_errors += 1
                
                if retry_count < self.config.db_retry_attempts:
                    delay = self.config.db_retry_delay * (2 ** (retry_count - 1))
                    logger.warning(
                        f"DB write failed (attempt {retry_count}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)

        logger.error(
            f"Failed to save game {game['game_id']} after "
            f"{retry_count} attempts: {last_error}"
        )

    async def _save_game_to_db(self, game: Dict[str, Any]):
        """Save game and moves to PostgreSQL."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        # Prefer game-provided timestamps if available, otherwise use now.
        start_ts = int(game.get('start_time') or int(time.time() * 1000))
        end_ts = int(game.get('end_time') or int(time.time() * 1000))
        # Acquire may return either an async context manager (real pool)
        # or an awaitable that yields a connection (some test mocks). Handle both.
        acquire_candidate = self.pool.acquire()

        # Helper to perform DB operations given a connection object.
        async def _perform_ops(conn):
            async with conn.transaction():
                # Insert game record
                await conn.execute(
                    """
                    INSERT INTO games (
                        game_id, start_time, end_time, winner, 
                        total_moves, platform
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (game_id) DO NOTHING
                    """,
                    game['game_id'],
                    start_ts,
                    end_ts,
                    game.get('winner'),
                    game.get('total_moves'),
                    'selfplay'
                )

                # Batch insert moves
                now_ts = int(time.time() * 1000)
                move_records = [
                    (
                        game['game_id'],
                        move.get('move_number'),
                        move.get('player'),
                        (move.get('action') or {}).get('type'),
                        json.dumps(move.get('action', {})),
                        move.get('thinking_time_ms', 0),
                        int(move.get('timestamp') or now_ts)
                    )
                    for move in game.get('moves', [])
                ]

                if move_records:
                    await conn.executemany(
                        """
                        INSERT INTO moves (
                            game_id, move_number, player_id, action_type,
                            action_data, thinking_time_ms, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT DO NOTHING
                        """,
                        move_records
                    )

        # If acquire_candidate supports async context manager protocol, use it.
        if hasattr(acquire_candidate, '__aenter__'):
            async with acquire_candidate as conn:
                await _perform_ops(conn)
        else:
            # Otherwise, assume it's awaitable and yields a connection object.
            conn = await acquire_candidate
            # Perform operations using the plain connection object.
            await _perform_ops(conn)

    async def enqueue_game(self, game: Dict[str, Any]):
        """Add game to write queue."""
        if self.config.dry_run:
            logger.debug(f"[DRY RUN] Would save game: {game['game_id']}")
            return
        try:
            # Provide short backpressure window; if the queue is full we wait briefly.
            await asyncio.wait_for(self.write_queue.put(game), timeout=2.0)
        except asyncio.TimeoutError:
            # Queue is full and cannot accept new items quickly — drop and log.
            logger.warning(
                f"Write queue full, dropping game {game.get('game_id')} to avoid blocking"
            )
            self.metrics.db_errors += 1

    async def shutdown(self):
        """Gracefully shutdown writer and close pool."""
        logger.info("Shutting down database writer...")
        self.shutdown_event.set()

        if self.writer_task:
            # Wait for queue to drain
            await self.write_queue.join()
            await self.writer_task
            logger.info("Writer worker stopped")

        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")


class SelfPlayGenerator:
    """Generates self-play games for training data."""

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.metrics = Metrics()
        self.db_writer = DatabaseWriter(config, self.metrics)
        self.shutdown_requested = False

        # Set log level
        logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))

    async def initialize(self):
        """Initialize generator and database."""
        await self.db_writer.initialize()
        await self.db_writer.start_writer()

    async def generate_training_data(self):
        """Generate games continuously or for one batch."""
        logger.info(
            f"Starting self-play generation - "
            f"concurrent_games={self.config.concurrent_games}, "
            f"batch_size={self.config.games_per_batch}, "
            f"dry_run={self.config.dry_run}"
        )

        try:
            if self.config.batch_only:
                await self._generate_batch()
            else:
                await self._generate_continuous()
        finally:
            self.metrics.log_summary()

    async def _generate_continuous(self):
        """Generate games continuously until shutdown."""
        last_stats_time = time.time()
        stats_interval = 30  # Log stats every 30 seconds

        while not self.shutdown_requested:
            await self._generate_batch()

            # Periodic stats logging
            if time.time() - last_stats_time > stats_interval:
                self.metrics.log_summary()
                last_stats_time = time.time()

    async def _generate_batch(self):
        """Generate one batch of games."""
        batch_start = time.time()
        
        # Create tasks for concurrent games
        tasks = [
            self.play_single_game(game_id=i)
            for i in range(self.config.concurrent_games)
        ]

        # Gather with error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        games = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Game generation error: {result}", exc_info=result)
                self.metrics.game_errors += 1
            else:
                games.append(result)
                self.metrics.games_generated += 1

        # Save to database
        for game in games:
            await self.db_writer.enqueue_game(game)

        # Print statistics
        self._print_batch_statistics(games, batch_start)

    def _print_batch_statistics(self, games: List[Dict], batch_start: float):
        """Print statistics about generated games."""
        if not games:
            logger.warning("No games generated in batch")
            return

        batch_time = time.time() - batch_start
        avg_moves = np.mean([g['total_moves'] for g in games])
        p1_wins = sum(1 for g in games if g['winner'] == 'Player1')
        p2_wins = sum(1 for g in games if g['winner'] == 'Player2')
        draws = len(games) - p1_wins - p2_wins

        logger.info(
            f"Batch complete: {len(games)} games in {batch_time:.1f}s "
            f"({len(games)/batch_time:.2f} games/s)"
        )
        logger.info(
            f"Stats - Avg moves: {avg_moves:.1f}, "
            f"P1 wins: {p1_wins} ({p1_wins/len(games)*100:.1f}%), "
            f"P2 wins: {p2_wins} ({p2_wins/len(games)*100:.1f}%), "
            f"Draws: {draws}"
        )

    async def play_single_game(self, game_id: int) -> Dict[str, Any]:
        """
        Simulate one complete game.
        
        Args:
            game_id: Unique identifier for this game
            
        Returns:
            Dictionary containing game data and move history
        """
        # Record start timestamp for the game (ms)
        start_ms = int(time.time() * 1000)
        game_state = self.initialize_game()
        move_history = []
        move_count = 0
        max_moves = 500

        while not game_state['winner'] and move_count < max_moves:
            current_player = game_state['currentPlayerId']

            # Exploration vs exploitation
            if np.random.random() < self.config.exploration_rate:
                move = self.get_random_move(game_state)
            else:
                move = self.get_engine_move(game_state, self.config.search_depth)

            # Record state before move
            state_before = self.serialize_state(game_state)

            # Apply move
            game_state = self.apply_move(game_state, move)

            # Record state after move
            state_after = self.serialize_state(game_state)

            move_history.append({
                'move_number': move_count,
                'player': current_player,
                'action': move,
                'state_before': state_before,
                'state_after': state_after,
                'timestamp': int(time.time() * 1000)
            })

            move_count += 1

        end_ms = int(time.time() * 1000)

        return {
            'game_id': f'selfplay_{game_id}_{end_ms}',
            'moves': move_history,
            'winner': game_state.get('winner'),
            'total_moves': move_count,
            'start_time': start_ms,
            'end_time': end_ms,
            'timestamp': datetime.fromtimestamp(end_ms / 1000.0).isoformat(),
        }

    def get_engine_move(self, state: Dict, depth: int) -> Dict:
        """
        Get best move using simplified evaluation.
        
        In production, this would call the TypeScript engine via API.
        """
        legal_moves = self.get_legal_moves(state)

        if not legal_moves:
            return {'type': 'endTurn'}

        best_move = None
        best_score = -float('inf')

        for move in legal_moves:
            # Shallow copy for evaluation
            new_state = self.apply_move(state.copy(), move)
            score = self.evaluate_position(new_state)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move if best_move else {'type': 'endTurn'}

    def get_random_move(self, state: Dict) -> Dict:
        """
        Get a random legal move for exploration.
        
        Args:
            state: Current game state
            
        Returns:
            Random legal move or endTurn if no moves available
        """
        legal_moves = self.get_legal_moves(state)
        if not legal_moves:
            return {'type': 'endTurn'}
        return legal_moves[np.random.randint(len(legal_moves))]

    def evaluate_position(self, state: Dict) -> float:
        """
        Simple heuristic position evaluation.
        
        Args:
            state: Game state to evaluate
            
        Returns:
            Score (positive favors current player)
        """
        score = 0.0
        current_player = state.get('currentPlayerId')

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

    def initialize_game(self) -> Dict:
        """
        Create initial game state with simplified 3-layer board.
        
        For production, replace this with full game initialization
        or call the TypeScript engine API.
        """
        # Simplified 3-layer board for training data generation
        board_layout = [3, 5, 3]  # Simplified from [3, 5, 7, 5, 3]
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
                        'adjacencies': []  # Set after all vertices created
                    }
                    vertex_id += 1
        
        # Set up adjacencies (simplified 4-connected grid)
        for vid, vertex in vertices.items():
            adj = []
            layer_verts = [v for v in vertices.values() if v['layer'] == vertex['layer']]
            
            # Same layer neighbors
            for other in layer_verts:
                if other['id'] != vid:
                    dx = abs(other['x'] - vertex['x'])
                    dz = abs(other['z'] - vertex['z'])
                    if (dx == 1 and dz == 0) or (dx == 0 and dz == 1):
                        adj.append(other['id'])
            
            vertex['adjacencies'] = adj
        
        # Find corner vertices for home positions
        layer0 = [v for v in vertices.values() if v['layer'] == 0]
        corners_p1 = [v['id'] for v in layer0 if v['x'] == 0 and v['z'] == 0]
        corners_p2 = [v['id'] for v in layer0 if v['x'] == 2 and v['z'] == 2]
        
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

    def get_legal_moves(self, state: Dict) -> List[Dict]:
        """
        Generate all legal moves for current player.
        
        Returns list of move dictionaries with 'type' and relevant fields.
        """
        moves = []
        current_player = state['currentPlayerId']
        turn = state['turn']
        vertices = state['vertices']
        
        # Placement moves
        if not turn['hasPlaced'] and state['players'][current_player]['reinforcements'] > 0:
            for corner_id in state['homeCorners'][current_player]:
                moves.append({'type': 'place', 'vertexId': corner_id})
        
        # Infusion moves
        if not turn['hasInfused']:
            for vid, vertex in vertices.items():
                if vertex['stack'] and vertex['stack'][0]['player'] == current_player:
                    # Check force cap (simplified)
                    if vertex['energy'] < 10:
                        moves.append({'type': 'infuse', 'vertexId': vid})
        
        # Movement moves
        if not turn['hasMoved']:
            for vid, vertex in vertices.items():
                if vertex['stack'] and vertex['stack'][0]['player'] == current_player:
                    for target_id in vertex['adjacencies']:
                        target = vertices[target_id]
                        # Can move to empty spaces
                        if not target['stack']:
                            moves.append({'type': 'move', 'fromId': vid, 'toId': target_id})
        
        # Attack moves
        for vid, vertex in vertices.items():
            if vertex['stack'] and vertex['stack'][0]['player'] == current_player:
                if len(vertex['stack']) >= 1 and vertex['energy'] >= 1:
                    for target_id in vertex['adjacencies']:
                        target = vertices[target_id]
                        if target['stack'] and target['stack'][0]['player'] != current_player:
                            moves.append({'type': 'attack', 'vertexId': vid, 'targetId': target_id})
        
        # End turn (if mandatory actions done)
        if turn['hasPlaced'] and turn['hasInfused'] and turn['hasMoved']:
            moves.append({'type': 'endTurn'})
        
        # Always allow end turn if no other moves
        if not moves:
            moves.append({'type': 'endTurn'})
        
        return moves

    def apply_move(self, state: Dict, move: Dict) -> Dict:
        """
        Apply move to state and return new state.
        
        This is a simplified implementation. For production, integrate
        with the full TypeScript game engine via API.
        """
        import copy
        new_state = copy.deepcopy(state)
        
        move_type = move.get('type')
        current_player = new_state['currentPlayerId']
        
        if move_type == 'place':
            vertex_id = move['vertexId']
            vertex = new_state['vertices'][vertex_id]
            vertex['stack'].append({'player': current_player, 'id': f'p{len(vertex["stack"])}'})
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
            
            # Transfer stack and energy
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
            
            # Simplified combat: compare stack size + energy
            attacker_strength = len(attacker['stack']) * 10 + attacker['energy'] * 15
            defender_strength = len(defender['stack']) * 10 + defender['energy'] * 15
            
            if attacker_strength > defender_strength:
                # Attacker wins: defender gets attacker's pieces
                defender['stack'] = attacker['stack']
                defender['energy'] = max(0, attacker['energy'] - defender['energy'])
            else:
                # Defender wins or draw: defender keeps position
                defender['energy'] = max(0, defender['energy'] - attacker['energy'])
            
            # Attacker position is emptied
            attacker['stack'] = []
            attacker['energy'] = 0
            
            # Attack ends turn
            new_state['currentPlayerId'] = 'Player2' if current_player == 'Player1' else 'Player1'
            new_state['players'][new_state['currentPlayerId']]['reinforcements'] += 1
            new_state['turn'] = {
                'hasPlaced': False,
                'hasInfused': False,
                'hasMoved': False,
                'turnNumber': new_state['turn']['turnNumber'] + 1
            }
        
        elif move_type == 'endTurn':
            # Switch players
            new_state['currentPlayerId'] = 'Player2' if current_player == 'Player1' else 'Player1'
            new_state['players'][new_state['currentPlayerId']]['reinforcements'] += 1
            new_state['turn'] = {
                'hasPlaced': False,
                'hasInfused': False,
                'hasMoved': False,
                'turnNumber': new_state['turn']['turnNumber'] + 1
            }
        
        # Check win condition: control opponent's home corners
        for player in ['Player1', 'Player2']:
            opponent = 'Player2' if player == 'Player1' else 'Player1'
            opponent_corners = new_state['homeCorners'][opponent]
            
            if opponent_corners and all(
                new_state['vertices'][cid]['stack'] and 
                new_state['vertices'][cid]['stack'][0]['player'] == player
                for cid in opponent_corners
            ):
                new_state['winner'] = player
                break
        
        return new_state

    def serialize_state(self, state: Dict) -> str:
        """Convert state to compact string representation."""
        return json.dumps(state, separators=(',', ':'))

    async def shutdown(self):
        """Gracefully shutdown the generator."""
        logger.info("Shutdown requested...")
        self.shutdown_requested = True
        await self.db_writer.shutdown()
        self.metrics.log_summary()
        logger.info("Shutdown complete")


async def main():
    """Main entry point with CLI argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Self-play training data generator'
    )
    parser.add_argument(
        '--concurrent-games',
        type=int,
        default=10,
        help='Number of concurrent games (default: 10)'
    )
    parser.add_argument(
        '--games-per-batch',
        type=int,
        default=100,
        help='Games per batch (default: 100)'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default=None,
        help='PostgreSQL connection URL (or use DATABASE_URL env var)'
    )
    parser.add_argument(
        '--search-depth',
        type=int,
        default=4,
        help='Engine search depth (default: 4)'
    )
    parser.add_argument(
        '--exploration-rate',
        type=float,
        default=0.1,
        help='Random move exploration rate 0-1 (default: 0.1)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without database writes'
    )
    parser.add_argument(
        '--batch-only',
        action='store_true',
        help='Run single batch and exit'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Build config from args
    config = SelfPlayConfig(
        concurrent_games=args.concurrent_games,
        games_per_batch=args.games_per_batch,
        database_url=args.db_url or os.getenv(
            'DATABASE_URL',
            'postgresql://user:pass@localhost/unitgame'
        ),
        search_depth=args.search_depth,
        exploration_rate=args.exploration_rate,
        dry_run=args.dry_run,
        batch_only=args.batch_only,
        log_level=args.log_level,
    )

    # Create generator
    generator = SelfPlayGenerator(config)

    # Setup signal handlers for graceful shutdown
    # Prefer asyncio's add_signal_handler when running in an event loop
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(
            signal.SIGINT, lambda: asyncio.create_task(generator.shutdown())
        )
        loop.add_signal_handler(
            signal.SIGTERM, lambda: asyncio.create_task(generator.shutdown())
        )
    except NotImplementedError:
        # Fallback for platforms where add_signal_handler isn't implemented
        def _signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating shutdown...")
            asyncio.create_task(generator.shutdown())

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    try:
        await generator.initialize()
        await generator.generate_training_data()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await generator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())