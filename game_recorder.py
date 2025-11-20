"""
Game Recording Module

Records human gameplay for training data collection.
Stores games in the same format as self-play for seamless integration.
"""

import sqlite3
import json
import gzip
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

DB_PATH = 'game.db'


def init_game_tables():
    """Initialize tables for recording human games."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Games table - mirrors self_play schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            platform TEXT NOT NULL DEFAULT 'human',
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            winner TEXT,
            total_moves INTEGER,
            initial_state TEXT,
            player1_id TEXT,
            player2_id TEXT,
            player1_device TEXT,
            player2_device TEXT,
            game_mode TEXT DEFAULT 'pvp',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Moves table - stores each move
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            action

_type TEXT NOT NULL,
            action_data TEXT NOT NULL,
            state_before TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    ''')
    
    # Create indexes for efficient queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_games_platform 
        ON games(platform)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_games_start_time 
        ON games(start_time DESC)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_moves_game_id 
        ON moves(game_id, move_number)
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Game recording tables initialized")


def create_game(game_id: str, initial_state: Dict, player1_device: str, player2_device: str = 'AI', game_mode: str = 'pvp') -> Dict:
    """
    Create a new game record.
    
    Args:
        game_id: Unique game identifier
        initial_state: Initial game state dict
        player1_device: Player 1's device ID
        player2_device: Player 2's device ID or 'AI'
        game_mode: 'pvp' (player vs player) or 'pva' (player vs AI)
    
    Returns:
        Game record dict
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO games (
            game_id, platform, initial_state, 
            player1_device, player2_device, game_mode
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        game_id,
        'human',
        json.dumps(initial_state),
        player1_device,
        player2_device,
        game_mode
    ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Created game record: {game_id} ({game_mode})")
    
    return {
        'game_id': game_id,
        'platform': 'human',
        'player1_device': player1_device,
        'player2_device': player2_device,
        'game_mode': game_mode
    }


def record_move(game_id: str, move_number: int, player_id: str, action: Dict, state_before: Optional[Dict] = None):
    """
    Record a single move.
    
    Args:
        game_id: Game identifier
        move_number: Sequential move number (0-indexed)
        player_id: Player who made the move ('Player1' or 'Player2')
        action: Move action dict with 'type' and other fields
        state_before: Optional game state before this move (for training)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO moves (
            game_id, move_number, player_id, action_type, action_data, state_before
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        game_id,
        move_number,
        player_id,
        action.get('type', 'unknown'),
        json.dumps(action),
        json.dumps(state_before) if state_before else None
    ))
    
    conn.commit()
    conn.close()


def complete_game(game_id: str, winner: Optional[str], total_moves: int):
    """
    Mark a game as complete.
    
    Args:
        game_id: Game identifier
        winner: 'Player1', 'Player2', or None for draw
        total_moves: Total number of moves
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE games 
        SET end_time = ?, winner = ?, total_moves = ?
        WHERE game_id = ?
    ''', (
        datetime.now().isoformat(),
        winner,
        total_moves,
        game_id
    ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Game completed: {game_id}, Winner: {winner}, Moves: {total_moves}")


def get_game_for_training(game_id: str) -> Optional[Dict]:
    """
    Retrieve a game in training format (compatible with training_pipeline.py).
    
    Returns dict with:
        - game_id
        - winner
        - total_moves
        - initial_state
        - moves (list of move dicts or compressed bytes)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get game
    cursor.execute('''
        SELECT game_id, winner, total_moves, initial_state
        FROM games
        WHERE game_id = ? AND end_time IS NOT NULL
    ''', (game_id,))
    
    game_row = cursor.fetchone()
    if not game_row:
        conn.close()
        return None
    
    # Get moves
    cursor.execute('''
        SELECT move_number, player_id, action_type, action_data, state_before
        FROM moves
        WHERE game_id = ?
        ORDER BY move_number
    ''', (game_id,))
    
    move_rows = cursor.fetchall()
    conn.close()
    
    # Parse moves
    moves = []
    for row in move_rows:
        move_dict = json.loads(row['action_data'])
        move_dict['player_id'] = row['player_id']
        move_dict['move_number'] = row['move_number']
        moves.append(move_dict)
    
    return {
        'game_id': game_row['game_id'],
        'winner': game_row['winner'],
        'total_moves': game_row['total_moves'],
        'initial_state': json.loads(game_row['initial_state']) if game_row['initial_state'] else None,
        'moves': moves
    }


def export_games_to_training_format(limit: int = 1000, compress_moves: bool = True) -> List[Dict]:
    """
    Export completed human games in training format.
    
    Args:
        limit: Max number of games to export
        compress_moves: If True, compress moves as gzip JSON bytes
    
    Returns:
        List of game dicts ready for training_pipeline.process_games()
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get completed games
    cursor.execute('''
        SELECT game_id
        FROM games
        WHERE platform = 'human' AND end_time IS NOT NULL
        ORDER BY start_time DESC
        LIMIT ?
    ''', (limit,))
    
    game_ids = [row['game_id'] for row in cursor.fetchall()]
    conn.close()
    
    games = []
    for game_id in game_ids:
        game = get_game_for_training(game_id)
        if game:
            if compress_moves and game['moves']:
                # Compress moves to match self-play format
                moves_json = json.dumps(game['moves'])
                game['moves'] = gzip.compress(moves_json.encode('utf-8'))
            games.append(game)
    
    logger.info(f"Exported {len(games)} human games for training")
    return games


if __name__ == '__main__':
    # Initialize tables
    logging.basicConfig(level=logging.INFO)
    init_game_tables()
    print("Game recording tables initialized!")
