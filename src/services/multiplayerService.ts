// src/services/multiplayerService.ts
/**
 * Multiplayer Service
 * 
 * WebSocket client for real-time multiplayer gameplay.
 * Handles matchmaking, move synchronization, and opponent state.
 */

import { io, Socket } from 'socket.io-client';
import { GameState, PlayerAction, PlayerId } from '../game/types';

const API_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:3000';

export interface MatchInfo {
    roomId: string;
    playerId: PlayerId;
    opponentId: PlayerId;
}

export interface OpponentMove {
    action: PlayerAction;
    currentTurn: PlayerId;
    moveNumber: number;
}

type EventCallback = (...args: any[]) => void;

class MultiplayerService {
    private socket: Socket | null = null;
    private roomId: string | null = null;
    private myPlayerId: PlayerId | null = null;
    private deviceId: string;
    private eventHandlers: Map<string, EventCallback[]> = new Map();

    constructor() {
        // Get or create device ID
        this.deviceId = this.getOrCreateDeviceId();
    }

    private getOrCreateDeviceId(): string {
        let deviceId = localStorage.getItem('deviceId');
        if (!deviceId) {
            deviceId = crypto.randomUUID();
            localStorage.setItem('deviceId', deviceId);
        }
        return deviceId;
    }

    /**
     * Connect to multiplayer server
     */
    connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            if (this.socket?.connected) {
                resolve();
                return;
            }

            this.socket = io(API_URL, {
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionAttempts: 5,
                reconnectionDelay: 1000
            });

            this.socket.on('connect', () => {
                console.log('🔌 Connected to multiplayer server');
                this.setupEventListeners();
                resolve();
            });

            this.socket.on('connect_error', (error) => {
                console.error('Connection error:', error);
                reject(error);
            });

            this.socket.on('disconnect', () => {
                console.log('❌ Disconnected from multiplayer server');
            });
        });
    }

    /**
     * Disconnect from server
     */
    disconnect(): void {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        this.roomId = null;
        this.myPlayerId = null;
    }

    /**
     * Join matchmaking queue
     */
    async joinMatchmaking(): Promise<MatchInfo> {
        if (!this.socket?.connected) {
            await this.connect();
        }

        return new Promise((resolve, reject) => {
            if (!this.socket) {
                reject(new Error('Not connected'));
                return;
            }

            // Set up one-time listeners
            const matchFoundHandler = (data: MatchInfo) => {
                this.roomId = data.roomId;
                this.myPlayerId = data.playerId;
                console.log(`🎮 Match found! You are ${data.playerId}`);
                resolve(data);
            };

            const errorHandler = (data: { message: string }) => {
                reject(new Error(data.message));
            };

            this.socket.once('match_found', matchFoundHandler);
            this.socket.once('error', errorHandler);

            // Send matchmaking request
            this.socket.emit('join_matchmaking', {
                deviceId: this.deviceId
            });

            console.log('🔍 Searching for opponent...');
        });
    }

    /**
     * Cancel matchmaking
     */
    cancelMatchmaking(): void {
        if (this.socket?.connected) {
            this.socket.emit('cancel_matchmaking');
            console.log('❌ Matchmaking cancelled');
        }
    }

    /**
     * Initialize game state (called by Player1)
     */
    initializeGameState(initialState: GameState): void {
        if (!this.socket?.connected || !this.roomId) {
            console.error('Cannot initialize game: not in a room');
            return;
        }

        this.socket.emit('init_game_state', {
            initialState
        });
        console.log('📤 Sent initial game state');
    }

    /**
     * Send a move to opponent
     */
    makeMove(action: PlayerAction, stateBefore: GameState): void {
        if (!this.socket?.connected || !this.roomId) {
            console.error('Cannot make move: not in a room');
            return;
        }

        this.socket.emit('make_move', {
            action,
            stateBefore
        });
    }

    /**
     * Leave current game
     */
    leaveGame(): void {
        if (this.socket?.connected && this.roomId) {
            this.socket.emit('leave_game');
            this.roomId = null;
            this.myPlayerId = null;
            console.log('👋 Left game');
        }
    }

    /**
     * Notify server that game is over
     */
    gameOver(winner: PlayerId | null, totalMoves: number): void {
        if (this.socket?.connected && this.roomId) {
            this.socket.emit('game_over', {
                winner,
                totalMoves
            });
            console.log(`🏁 Game over: ${winner || 'Draw'}`);
        }
    }

    // ==================== Event Listeners ====================

    private setupEventListeners(): void {
        if (!this.socket) return;

        // Matchmaking events
        this.socket.on('matchmaking_waiting', () => {
            this.triggerEvent('matchmaking_waiting');
        });

        this.socket.on('matchmaking_cancelled', () => {
            this.triggerEvent('matchmaking_cancelled');
        });

        // Game events
        this.socket.on('game_initialized', (data: { initialState: GameState; currentTurn: PlayerId }) => {
            console.log('🎮 Game initialized');
            this.triggerEvent('game_initialized', data);
        });

        this.socket.on('opponent_move', (data: OpponentMove) => {
            console.log(`👤 Opponent moved (turn: ${data.currentTurn})`);
            this.triggerEvent('opponent_move', data);
        });

        this.socket.on('move_confirmed', (data: { action: PlayerAction }) => {
            console.log('✅ Move confirmed');
            this.triggerEvent('move_confirmed', data);
        });

        this.socket.on('move_rejected', (data: { error: string }) => {
            console.warn('❌ Move rejected:', data.error);
            this.triggerEvent('move_rejected', data);
        });

        this.socket.on('opponent_disconnected', () => {
            console.log('⚠️ Opponent disconnected');
            this.triggerEvent('opponent_disconnected');
        });

        this.socket.on('opponent_left', () => {
            console.log('👋 Opponent left the game');
            this.triggerEvent('opponent_left');
        });

        this.socket.on('game_ended', (data: { winner: PlayerId | null; totalMoves: number }) => {
            console.log(`🏁 Game ended: ${data.winner || 'Draw'}`);
            this.triggerEvent('game_ended', data);
        });

        this.socket.on('error', (data: { message: string }) => {
            console.error('Server error:', data.message);
            this.triggerEvent('error', data);
        });
    }

    /**
     * Register event listener
     */
    on(event: string, callback: EventCallback): void {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event)!.push(callback);
    }

    /**
     * Remove event listener
     */
    off(event: string, callback: EventCallback): void {
        const handlers = this.eventHandlers.get(event);
        if (handlers) {
            const index = handlers.indexOf(callback);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    /**
     * Trigger event to all registered handlers
     */
    private triggerEvent(event: string, ...args: any[]): void {
        const handlers = this.eventHandlers.get(event);
        if (handlers) {
            handlers.forEach(handler => handler(...args));
        }
    }

    // ==================== Getters ====================

    isConnected(): boolean {
        return this.socket?.connected || false;
    }

    isInGame(): boolean {
        return this.roomId !== null;
    }

    getMyPlayerId(): PlayerId | null {
        return this.myPlayerId;
    }

    getRoomId(): string | null {
        return this.roomId;
    }
}

// Export singleton instance
export const multiplayerService = new MultiplayerService();
