"""
Multiplayer Server Module

WebSocket-based multiplayer system with matchmaking and real-time game synchronization.
Integrates with game recording for automatic training data collection.
"""

from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import logging
import uuid
from datetime import datetime
from game_recorder import create_game, record_move, complete_game

logger = logging.getLogger(__name__)

# Type aliases
PlayerId = str  # 'Player1' or 'Player2'
SocketId = str
DeviceId = str


@dataclass
class Player:
    """Represents a player in matchmaking or game"""
    sid: SocketId
    device_id: DeviceId
    player_id: Optional[PlayerId] = None  # Assigned when matched


@dataclass
class GameRoom:
    """Represents an active multiplayer game"""
    room_id: str
    player1: Player
    player2: Player
    current_state: dict
    current_turn: PlayerId
    recording_game_id: str
    move_count: int = 0
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class MultiplayerManager:
    """Manages matchmaking and active games"""
    
    def __init__(self):
        self.matchmaking_queue: List[Player] = []
        self.active_rooms: Dict[str, GameRoom] = {}
        self.player_to_room: Dict[SocketId, str] = {}  # sid -> room_id
        
    def add_to_queue(self, sid: SocketId, device_id: DeviceId) -> Optional[GameRoom]:
        """
        Add player to matchmaking queue.
        Returns GameRoom if a match is found, None otherwise.
        """
        player = Player(sid=sid, device_id=device_id)
        
        # Check if already in queue
        if any(p.sid == sid for p in self.matchmaking_queue):
            logger.warning(f"Player {sid} already in matchmaking queue")
            return None
        
        # If queue is empty, add and wait
        if not self.matchmaking_queue:
            self.matchmaking_queue.append(player)
            logger.info(f"Player {sid} added to matchmaking queue (waiting)")
            return None
        
        # Match found! Pop first player from queue
        opponent = self.matchmaking_queue.pop(0)
        
        # Create game room
        room = self._create_room(player, opponent)
        logger.info(f"Match found! Room {room.room_id}: {player.sid} vs {opponent.sid}")
        
        return room
    
    def remove_from_queue(self, sid: SocketId):
        """Remove player from matchmaking queue"""
        self.matchmaking_queue = [p for p in self.matchmaking_queue if p.sid != sid]
        logger.info(f"Player {sid} removed from matchmaking queue")
    
    def _create_room(self, p1: Player, p2: Player) -> GameRoom:
        """Create a new game room with two players"""
        room_id = str(uuid.uuid4())
        
        # Assign player IDs
        p1.player_id = 'Player1'
        p2.player_id = 'Player2'
        
        # Initialize game state (to be sent from client)
        # For now, empty. Client will send initial state
        initial_state = {}
        
        # Create room
        room = GameRoom(
            room_id=room_id,
            player1=p1,
            player2=p2,
            current_state=initial_state,
            current_turn='Player1',  # Player1 always starts
            recording_game_id=''  # Will be set when initial state received
        )
        
        self.active_rooms[room_id] = room
        self.player_to_room[p1.sid] = room_id
        self.player_to_room[p2.sid] = room_id
        
        return room
    
    def get_room_for_player(self, sid: SocketId) -> Optional[GameRoom]:
        """Get the room a player is in"""
        room_id = self.player_to_room.get(sid)
        if room_id:
            return self.active_rooms.get(room_id)
        return None
    
    def apply_move(self, sid: SocketId, action: dict, state_before: dict) -> Optional[dict]:
        """
        Apply a move to the game.
        Returns error dict if invalid, None if successful.
        """
        room = self.get_room_for_player(sid)
        if not room:
            return {"error": "Not in a game"}
        
        # Determine player
        if sid == room.player1.sid:
            player = room.player1
        elif sid == room.player2.sid:
            player = room.player2
        else:
            return {"error": "Player not in this room"}
        
        # Validate turn
        if player.player_id != room.current_turn:
            return {"error": "Not your turn"}
        
        # Record move (async, don't block on failure)
        try:
            if room.recording_game_id:
                record_move(
                    game_id=room.recording_game_id,
                    move_number=room.move_count,
                    player_id=player.player_id,
                    action=action,
                    state_before=state_before
                )
        except Exception as e:
            logger.error(f"Failed to record move: {e}")
        
        # Update room state
        room.move_count += 1
        
        # Switch turn
        room.current_turn = 'Player2' if room.current_turn == 'Player1' else 'Player1'
        
        return None  # Success
    
    def initialize_game_state(self, sid: SocketId, initial_state: dict) -> Optional[str]:
        """Initialize the game state and start recording"""
        room = self.get_room_for_player(sid)
        if not room:
            return None
        
        room.current_state = initial_state
        
        # Start game recording
        try:
            recording_id = create_game(
                game_id=room.room_id,
                initial_state=initial_state,
                player1_device=room.player1.device_id,
                player2_device=room.player2.device_id,
                game_mode='pvp'
            )
            room.recording_game_id = recording_id['game_id']
            logger.info(f"Started recording for room {room.room_id}")
            return room.recording_game_id
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return None
    
    def complete_game(self, sid: SocketId, winner: Optional[str], total_moves: int):
        """Complete game recording and clean up room"""
        room = self.get_room_for_player(sid)
        if not room:
            return
        
        # Complete recording
        try:
            if room.recording_game_id:
                complete_game(
                    game_id=room.recording_game_id,
                    winner=winner,
                    total_moves=total_moves
                )
                logger.info(f"Game completed: {room.recording_game_id}, Winner: {winner}")
        except Exception as e:
            logger.error(f"Failed to complete recording: {e}")
        
        # Clean up
        self._cleanup_room(room.room_id)
    
    def handle_disconnect(self, sid: SocketId) -> Optional[GameRoom]:
        """
        Handle player disconnect.
        Returns the room if opponent should be notified.
        """
        # Remove from queue if waiting
        self.remove_from_queue(sid)
        
        # Get room
        room = self.get_room_for_player(sid)
        if not room:
            return None
        
        # Notify opponent and clean up
        logger.info(f"Player {sid} disconnected from room {room.room_id}")
        
        # For now, immediately end the game
        # TODO: Add reconnection grace period
        return room
    
    def _cleanup_room(self, room_id: str):
        """Remove room and player mappings"""
        room = self.active_rooms.get(room_id)
        if room:
            # Remove player mappings
            self.player_to_room.pop(room.player1.sid, None)
            self.player_to_room.pop(room.player2.sid, None)
            
            # Remove room
            del self.active_rooms[room_id]
            logger.info(f"Cleaned up room {room_id}")


# Singleton instance
multiplayer_manager = MultiplayerManager()


def init_multiplayer(socketio: SocketIO):
    """Initialize multiplayer WebSocket event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'sid': request.sid})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")
        
        # Handle disconnect in game
        room = multiplayer_manager.handle_disconnect(request.sid)
        if room:
            # Notify opponent
            opponent_sid = room.player2.sid if request.sid == room.player1.sid else room.player1.sid
            emit('opponent_disconnected', {}, room=opponent_sid)
    
    @socketio.on('join_matchmaking')
    def handle_join_matchmaking(data):
        """Player wants to find a match"""
        device_id = data.get('deviceId')
        if not device_id:
            emit('error', {'message': 'Missing deviceId'})
            return
        
        logger.info(f"Player {request.sid} joining matchmaking")
        
        # Add to queue
        room = multiplayer_manager.add_to_queue(request.sid, device_id)
        
        if room:
            # Match found! Notify both players
            join_room(room.room_id)
            
            # Notify player 1
            emit('match_found', {
                'roomId': room.room_id,
                'playerId': room.player1.player_id,
                'opponentId': room.player2.player_id
            }, room=room.player1.sid)
            
            # Notify player 2
            emit('match_found', {
                'roomId': room.room_id,
                'playerId': room.player2.player_id,
                'opponentId': room.player1.player_id
            }, room=room.player2.sid)
            
            logger.info(f"Match created: Room {room.room_id}")
        else:
            # Waiting for opponent
            emit('matchmaking_waiting', {})
    
    @socketio.on('cancel_matchmaking')
    def handle_cancel_matchmaking():
        """Player cancels matchmaking"""
        multiplayer_manager.remove_from_queue(request.sid)
        emit('matchmaking_cancelled', {})
    
    @socketio.on('init_game_state')
    def handle_init_game_state(data):
        """Initialize game state (sent by Player1)"""
        initial_state = data.get('initialState')
        if not initial_state:
            emit('error', {'message': 'Missing initialState'})
            return
        
        recording_id = multiplayer_manager.initialize_game_state(request.sid, initial_state)
        
        if recording_id:
            room = multiplayer_manager.get_room_for_player(request.sid)
            if room:
                # Broadcast to both players
                emit('game_initialized', {
                    'initialState': initial_state,
                    'currentTurn': room.current_turn
                }, room=room.room_id)
    
    @socketio.on('make_move')
    def handle_make_move(data):
        """Player makes a move"""
        action = data.get('action')
        state_before = data.get('stateBefore')
        
        if not action:
            emit('error', {'message': 'Missing action'})
            return
        
        # Apply move
        error = multiplayer_manager.apply_move(request.sid, action, state_before)
        
        if error:
            emit('move_rejected', error)
            return
        
        # Get room to broadcast
        room = multiplayer_manager.get_room_for_player(request.sid)
        if not room:
            return
        
        # Confirm to sender
        emit('move_confirmed', {'action': action})
        
        # Notify opponent
        opponent_sid = room.player2.sid if request.sid == room.player1.sid else room.player1.sid
        emit('opponent_move', {
            'action': action,
            'currentTurn': room.current_turn,
            'moveNumber': room.move_count
        }, room=opponent_sid)
    
    @socketio.on('game_over')
    def handle_game_over(data):
        """Game ended"""
        winner = data.get('winner')
        total_moves = data.get('totalMoves', 0)
        
        multiplayer_manager.complete_game(request.sid, winner, total_moves)
        
        room = multiplayer_manager.get_room_for_player(request.sid)
        if room:
            # Notify both players
            emit('game_ended', {
                'winner': winner,
                'totalMoves': total_moves
            }, room=room.room_id)
    
    @socketio.on('leave_game')
    def handle_leave_game():
        """Player voluntarily leaves game"""
        room = multiplayer_manager.get_room_for_player(request.sid)
        if room:
            # Determine opponent
            opponent_sid = room.player2.sid if request.sid == room.player1.sid else room.player1.sid
            
            # Notify opponent
            emit('opponent_left', {}, room=opponent_sid)
            
            # Clean up
            multiplayer_manager._cleanup_room(room.room_id)
    
    logger.info("Multiplayer WebSocket handlers initialized")


# Import request for socket.io
from flask import request
